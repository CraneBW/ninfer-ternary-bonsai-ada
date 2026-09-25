// zatfung 模板抓取器 —— 只下载模板里真正有用的 3.11 GiB，跳过 15.92 GiB 无用区段。
//
// 为什么需要它
// ------------
// `pack.py`（三元 Bonsai → .ninfer 的打包器）需要一个"模板 artifact"来借用
// vision / mtp / dflash2 / frontend / draft_head 共 419 个对象的载荷。这份模板的
// 完整文件是 19.03 GiB，但其中 771 个 text/* 对象的载荷（15.92 GiB）**从不被读取**
// —— 它们由打包器自己 produce() 生成。
//
// 更巧的是，那 419 个对象在文件布局上**恰好集中在两端**：
//
//	文件偏移 0                 .. 13,025,792          header + frontend(6)
//	文件偏移 13,025,792        .. 17,106,516,480       771 个 text/*（无用）
//	文件偏移 17,106,516,480    .. 20,437,336,576       draft_head+mtp+vision+dflash2
//
// 所以本程序用 HTTP Range 精确抓取这两段（3,343,845,888 B = 3.114 GiB，省 83.6%），
// 再按需要还原成可直接使用的模板。仓库因此只需存这份源码，不必存 3 GB 二进制。
//
// 用法
// ----
//
//	template-fetch                       # 抓取并还原出完整模板（下载 3.11 GiB）
//	template-fetch -slim                 # 只产出 3.11 GiB 的 extract（需配 restore 步骤）
//	template-fetch -out D:\mytpl.ninfer  # 指定输出路径
//	template-fetch -j 8                  # 并发下载块数（默认 4）
//	template-fetch -no-resume            # 丢弃已有进度，重新下载
//
// 断点续传：中断后重跑即可，已完成的块会被跳过（状态记在 <out>.tplfetch.json）。
// 校验：全部块就绪后对整份数据算 SHA-256，与内置基准比对，不符则报错并要求重下。
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// 内置下载计划
// ---------------------------------------------------------------------------

const (
	defaultURL = "https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/" +
		"dc370fb6295a/qwen3_8_27b.ninfer" // revision 固定在 v2（main 已是 v3，不兼容）

	sourceBytes  = 20437336576 // 源文件总大小（用于确认 revision 没变）
	extractBytes = 3343845888  // 两段合计

	// 整份 extract 的 SHA-256 基准。抓完必须与它一致。
	extractSHA256 = "97437007c19310a5f3204e594a6fb1acc190e2ffade9acc568b8c9310ea2a72c"
)

// segment 是源文件里一段"有用"的连续字节。
type segment struct {
	BlobOffset int64 `json:"blob_offset"` // 在 extract（拼接结果）中的位置
	FileOffset int64 `json:"file_offset"` // 在源文件中的位置
	Bytes      int64 `json:"bytes"`
}

var segments = []segment{
	{BlobOffset: 0, FileOffset: 0, Bytes: 13025792},                    // header + frontend
	{BlobOffset: 13025792, FileOffset: 17106516480, Bytes: 3330820096}, // draft_head..dflash2
}

// ---------------------------------------------------------------------------
// 输出（与 builder.go 同一套配色）
// ---------------------------------------------------------------------------

const (
	colorRed    = "\033[0;31m"
	colorGreen  = "\033[0;32m"
	colorYellow = "\033[0;33m"
	colorBlue   = "\033[0;34m"
	colorNC     = "\033[0m"
)

func printInfo(msg string)    { fmt.Print(colorBlue + "[INFO] " + msg + colorNC + "\n") }
func printSuccess(msg string) { fmt.Print(colorGreen + "[ OK ] " + msg + colorNC + "\n") }
func printWarning(msg string) { fmt.Print(colorYellow + "[WARN] " + msg + colorNC + "\n") }
func printError(msg string)   { fmt.Print(colorRed + "[FAIL] " + msg + colorNC + "\n") }

// ---------------------------------------------------------------------------
// 分块
// ---------------------------------------------------------------------------

// task 是一个待下载的块。块不跨段，因此到 URL 的映射是单段线性关系。
type task struct {
	Index      int   `json:"index"`
	BlobOffset int64 `json:"blob_offset"` // 在 extract 中的写入位置
	FileOffset int64 `json:"file_offset"` // 在源文件中的读取位置（也是 Range 起点）
	Bytes      int64 `json:"bytes"`
	Done       bool  `json:"done"`
}

// buildTasks 按段切块。块不跨段，避免 Range 映射复杂化。
func buildTasks(chunk int64) []task {
	var out []task
	idx := 0
	for _, s := range segments {
		for off := int64(0); off < s.Bytes; off += chunk {
			n := chunk
			if off+n > s.Bytes {
				n = s.Bytes - off
			}
			out = append(out, task{
				Index:      idx,
				BlobOffset: s.BlobOffset + off,
				FileOffset: s.FileOffset + off,
				Bytes:      n,
			})
			idx++
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// 续传状态
// ---------------------------------------------------------------------------

type stateFile struct {
	URL       string `json:"url"`
	ChunkSize int64  `json:"chunk_size"`
	Done      []int  `json:"done"` // 已完成的块序号
}

func statePathFor(out string) string { return out + ".tplfetch.json" }

func loadState(p string) (*stateFile, error) {
	b, err := os.ReadFile(p)
	if err != nil {
		return nil, err
	}
	var st stateFile
	if err := json.Unmarshal(b, &st); err != nil {
		return nil, err
	}
	return &st, nil
}

func saveState(p, url string, chunk int64, tasks []task) error {
	var done []int
	for _, t := range tasks {
		if t.Done {
			done = append(done, t.Index)
		}
	}
	sort.Ints(done)
	b, err := json.MarshalIndent(stateFile{URL: url, ChunkSize: chunk, Done: done}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(p, b, 0o644)
}

// ---------------------------------------------------------------------------
// 下载
// ---------------------------------------------------------------------------

// fetchRange 抓取源文件的 [off, off+n) 并写到 dst 的 writeAt 位置。
// 用 io.NewOffsetWriter 而不是 ReadAll，避免把整块读进内存。
func fetchRange(client *http.Client, url string, off, n int64, dst *os.File, writeAt int64) error {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", off, off+n-1))

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// 206 = 服务端接受 Range；200 表示它忽略了 Range 并返回整个文件
	// （那样会把 19 GiB 全灌进来，必须当作错误处理）。
	if resp.StatusCode != http.StatusPartialContent {
		return fmt.Errorf("expected 206 Partial Content, got %d (服务端可能不支持 Range)", resp.StatusCode)
	}

	w := io.NewOffsetWriter(dst, writeAt)
	written, err := io.CopyN(w, resp.Body, n)
	if err != nil {
		return fmt.Errorf("copied %d/%d bytes: %w", written, n, err)
	}
	return nil
}

// runTasks 并发执行块下载，带重试。返回实际下载的字节数。
func runTasks(client *http.Client, url string, tasks []task, dst *os.File, jobs int,
	stateOut string, chunk int64) (int64, error) {

	ch := make(chan int)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var total int64
	var firstErr error
	attempts := 5

	worker := func() {
		defer wg.Done()
		for i := range ch {
			t := tasks[i]
			if t.Done {
				continue
			}
			var lastErr error
			for a := 1; a <= attempts; a++ {
				lastErr = fetchRange(client, url, t.FileOffset, t.Bytes, dst, t.BlobOffset)
				if lastErr == nil {
					break
				}
				printWarning(fmt.Sprintf("块 %d/%d 第 %d 次失败: %v",
					t.Index+1, len(tasks), a, lastErr))
				time.Sleep(time.Duration(a) * 2 * time.Second)
			}
			if lastErr != nil {
				mu.Lock()
				if firstErr == nil {
					firstErr = fmt.Errorf("块 %d 下载失败: %w", t.Index, lastErr)
				}
				mu.Unlock()
				return
			}
			mu.Lock()
			tasks[i].Done = true
			total += t.Bytes
			done := 0
			for _, x := range tasks {
				if x.Done {
					done++
				}
			}
			pct := 100.0 * float64(done) / float64(len(tasks))
			fmt.Printf("\r%sprogress: %d/%d chunks (%.1f%%)%s   ",
				colorBlue, done, len(tasks), pct, colorNC)
			_ = saveState(stateOut, url, chunk, tasks)
			mu.Unlock()
		}
	}

	wg.Add(jobs)
	for i := 0; i < jobs; i++ {
		go worker()
	}
	for i := range tasks {
		ch <- i
	}
	close(ch)
	wg.Wait()
	fmt.Println()
	return total, firstErr
}

// sha256File 计算文件摘要。
func sha256File(p string) (string, int64, error) {
	f, err := os.Open(p)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

// expandTemplate 把紧凑 extract 展开成"完整模板"布局。
//
// 中间被跳过的那段 text/* 区段**保持为零** —— 这是可以的：`Artifact.open()` 的
// `_validate_ranges` 只校验每个对象的 offset 落在文件范围内，不校验内容；而 pack.py
// 对这些对象走 `produce()` 自己生成，从不读取模板里的它们。
func expandTemplate(extractPath, templatePath string, sourceBytes int64) error {
	src, err := os.Open(extractPath)
	if err != nil {
		return err
	}
	defer src.Close()

	dst, err := os.Create(templatePath)
	if err != nil {
		return err
	}
	defer dst.Close()

	// 先建立最终尺寸，让并发/跳跃写入可以落在任意偏移
	if err := dst.Truncate(sourceBytes); err != nil {
		return err
	}
	for _, s := range segments {
		if _, err := src.Seek(s.BlobOffset, io.SeekStart); err != nil {
			return err
		}
		w := io.NewOffsetWriter(dst, s.FileOffset)
		if _, err := io.CopyN(w, src, s.Bytes); err != nil {
			return err
		}
	}
	return dst.Sync()
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

func main() {
	out := flag.String("out", "qwen3_8_27b_v2.ninfer", "输出路径（默认=完整模板）")
	slim := flag.Bool("slim", false, "只产出 3.11 GiB 的 extract（两段拼接），不还原成完整模板")
	jobs := flag.Int("j", 4, "并发下载块数")
	chunkMB := flag.Int64("chunk-mb", 32, "分块大小（MB），决定续传粒度")
	url := flag.String("url", defaultURL, "源文件 URL（默认固定在 v2 revision）")
	noResume := flag.Bool("no-resume", false, "丢弃已有进度重新下载")
	fromExtract := flag.String("from-extract", "",
		"跳过下载，直接用已有的 extract（3.11 GiB）展开成完整模板")
	flag.Parse()

	chunk := *chunkMB << 20
	tasks := buildTasks(chunk)
	var totalNeeded int64
	for _, t := range tasks {
		totalNeeded += t.Bytes
	}

	printInfo("=== zatfung 模板抓取器 ===")
	printInfo(fmt.Sprintf("源: %s", *url))
	printInfo(fmt.Sprintf("源文件大小: %s B", formatInt(sourceBytes)))
	printInfo(fmt.Sprintf("需要下载: %s B = %.3f GiB（跳过 %.3f GiB 无用区段）",
		formatInt(totalNeeded), float64(totalNeeded)/(1<<30),
		float64(sourceBytes-totalNeeded)/(1<<30)))
	printInfo(fmt.Sprintf("分块: %d 块 × %d MB   并发: %d", len(tasks), *chunkMB, *jobs))
	printInfo(fmt.Sprintf("输出: %s（%s）", *out, map[bool]string{true: "slim extract",
		false: "完整模板"}[*slim]))

	client := &http.Client{
		Timeout: 0, // 大块传输不设总超时，靠 Range 分块与重试保证进度
	}

	// ---- 确认源文件身份：HEAD 拿 Content-Length，避免 revision 漂移 ----
	{
		req, _ := http.NewRequest(http.MethodHead, *url, nil)
		resp, err := client.Do(req)
		if err != nil {
			printWarning(fmt.Sprintf("HEAD 失败（%v），跳过大小校验直接下载", err))
		} else {
			resp.Body.Close()
			if got := resp.ContentLength; got > 0 && got != sourceBytes {
				printError(fmt.Sprintf("源文件大小变了：期望 %d，实际 %d。"+
					"HF 上的 artifact 可能已被替换 —— 请核对 revision 是否仍是 v2。",
					sourceBytes, got))
				os.Exit(1)
			}
			printSuccess("源文件大小与基准一致（revision 未漂移）")
		}
	}

	// ---- 快速路径：直接把已有的 extract 展开成模板，不重新下载 ----
	// 典型用法：先 `-slim` 抓一次（3.11 GiB），之后想换输出位置/格式时反复复用，
	// 不必再花一次下载。
	if *fromExtract != "" {
		extractPath := *fromExtract
		printInfo("跳过下载，直接使用已有 extract: " + extractPath)
		sum, size, err := sha256File(extractPath)
		if err != nil {
			printError("读取 extract 失败: " + err.Error())
			os.Exit(1)
		}
		if size != extractBytes {
			printError(fmt.Sprintf("extract 大小不符：期望 %s，实际 %s",
				formatInt(extractBytes), formatInt(size)))
			os.Exit(1)
		}
		if sum != extractSHA256 {
			printError(fmt.Sprintf("SHA-256 不匹配！\n  expect %s\n  got    %s", extractSHA256, sum))
			os.Exit(1)
		}
		printSuccess(fmt.Sprintf("extract 校验通过（%s B，%s）", formatInt(size), sum[:16]+"…"))
		printInfo(fmt.Sprintf("展开为完整模板（中段 %.3f GiB 留零）…",
			float64(sourceBytes-totalNeeded)/(1<<30)))
		if err := expandTemplate(extractPath, *out, sourceBytes); err != nil {
			printError("展开失败: " + err.Error())
			os.Exit(1)
		}
		st, _ := os.Stat(*out)
		printSuccess("产出完整模板: " + *out)
		printSuccess(fmt.Sprintf("  %s B = %.3f GiB", formatInt(st.Size()),
			float64(st.Size())/(1<<30)))
		return
	}

	// ---- 准备输出文件 ----
	// 下载数据一律先落成 extract 布局（两段拼接，3.11 GiB）：
	//   -slim 时它就是最终产物，全程只占 3.11 GiB；
	//   默认模式再多一步展开成完整模板（19.03 GiB），展开后删掉临时 extract。
	extractPath := *out
	if !*slim {
		extractPath = *out + ".extract.tmp"
	}
	f, err := os.OpenFile(extractPath, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		printError("无法打开输出文件: " + err.Error())
		os.Exit(1)
	}
	defer f.Close()

	// 预先建立 extract 尺寸；各块用 WriteAt 落到自己的 blob 偏移，不会越界。
	if err := f.Truncate(extractBytes); err != nil {
		printError("无法分配文件尺寸: " + err.Error())
		os.Exit(1)
	}

	stateP := statePathFor(extractPath)
	if *noResume {
		os.Remove(stateP)
	} else if st, err := loadState(stateP); err == nil {
		if st.ChunkSize == chunk && st.URL == *url {
			done := map[int]bool{}
			for _, i := range st.Done {
				done[i] = true
			}
			n := 0
			for i := range tasks {
				if done[i] {
					tasks[i].Done = true
					n++
				}
			}
			if n > 0 {
				printSuccess(fmt.Sprintf("续传：已有 %d/%d 块完成，跳过它们", n, len(tasks)))
			}
		} else {
			printWarning("已有状态文件的分块/URL 与本次不一致，忽略它重新开始")
		}
	}

	n, err := runTasks(client, *url, tasks, f, *jobs, stateP, chunk)
	if err != nil {
		printError(err.Error())
		printWarning("已完成的块已记录，重跑本程序可继续")
		os.Exit(1)
	}
	printSuccess(fmt.Sprintf("下载完成，本次写入 %s B", formatInt(n)))

	if err := f.Sync(); err != nil {
		printWarning("fsync 失败: " + err.Error())
	}

	// ---- 校验（一律对 extract 算 SHA-256）----
	printInfo("校验（读回整份 extract 计算 SHA-256）…")
	sum, size, err := sha256File(extractPath)
	if err != nil {
		printError("校验读取失败: " + err.Error())
		os.Exit(1)
	}
	if sum != extractSHA256 {
		printError(fmt.Sprintf("SHA-256 不匹配！\n  expect %s\n  got    %s\n"+
			"请用 -no-resume 重下。", extractSHA256, sum))
		os.Exit(1)
	}
	printSuccess(fmt.Sprintf("SHA-256 校验通过：%s（%s B）", sum[:16]+"…", formatInt(size)))

	if *slim {
		os.Remove(stateP)
		printSuccess("产出 extract: " + extractPath)
		printInfo("这是 pack.py 需要的「精简模板」；要直接可用的完整模板，去掉 -slim 重跑")
		return
	}

	// ---- 展开为完整模板 ----
	printInfo(fmt.Sprintf("展开为完整模板（中段 %.3f GiB 留零）…",
		float64(sourceBytes-totalNeeded)/(1<<30)))
	if err := expandTemplate(extractPath, *out, sourceBytes); err != nil {
		printError("展开失败: " + err.Error())
		os.Exit(1)
	}
	os.Remove(extractPath)
	os.Remove(stateP)

	st, err := os.Stat(*out)
	if err != nil {
		printError("统计产出失败: " + err.Error())
		os.Exit(1)
	}
	printSuccess("产出完整模板: " + *out)
	printSuccess(fmt.Sprintf("  %s B = %.3f GiB（实际只下载 %.3f GiB，省 %.1f%%）",
		formatInt(st.Size()), float64(st.Size())/(1<<30), float64(totalNeeded)/(1<<30),
		100.0*float64(sourceBytes-totalNeeded)/float64(sourceBytes)))
	printInfo("可直接作为 pack.py 的 ZATFUNG_TEMPLATE 使用")
}

// formatInt 千分位分隔，便于读大数。
func formatInt(n int64) string {
	s := fmt.Sprintf("%d", n)
	var b strings.Builder
	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteRune(c)
	}
	return b.String()
}
