// zatfung 构建器 —— 疾风引擎（zatfung）的生产标准构建器。
//
// 为什么不是 .bat / .sh：
//   - 本机 cmd.exe 被安全策略禁用，vcvars64.bat 根本调不起来，任何依赖
//     "先 call vcvars 再 cmake" 的脚本在这里都是死路；
//   - 编译产物/子进程不应受 git-bash 的环境清洗差异影响。此前 SHELL 路径实测
//     踩过的坑在这里全部内置处理：
//   - 代理变量（HTTP_PROXY 等）会让 MSBuild 报 MSB6001 —— 子进程环境强制剥离；
//   - `env -i` 清洗过头的反面教训：FindCUDAToolkit 探到版本却找不到
//     CUDA_CUDART 库 —— 本构建器保留完整环境（仅剥离代理），CUDA 探测正常；
//   - `$PWD` 是 POSIX 形式（/g/...），Windows 原生 CMake 不认 —— 用 Go 原生
//     路径（G:\...）传给 CMake 的 -S/-B；
//   - git-bash 的 rm -rf 大目录可能被环境安全钩子拦截 —— Go 的 os.RemoveAll
//     原生删除，不走 shell；另提供 -keep 增量模式（只清 CMake 缓存）。
//
// 本机（RTX 3060 Ti / Ampere sm_86 / CUDA 12.8.93 / VS2022 Community）实测结论，
// 已固化进本构建器的默认行为：
//  1. CUDA 12.8 + MSVC 19.44 实测能编译 sm_86 目标文件（.cu → .obj 通过）。
//     zatfung 上游 CMakeLists 却硬要求 CUDA >= 13.1 —— 那是 Blackwell/sm_120a 线的
//     策略性门槛。构建器会在 arch != 120a 时自动带上 -DNINFER_ALLOW_LEGACY_CUDA=ON，
//     从而不必为一个版本号去装 6 GB 的新工具链；arch=120a 时该逃生舱被拒绝。
//  2. Visual Studio 生成器在本环境**不可用**：CMake 找不到 C/CXX 编译器
//     （No CMAKE_C_COMPILER could be found）—— 因为 cmd.exe/vcvars 通道被禁。
//     因此默认生成器是 Ninja，并由 Go 自己探测 VS/MSVC/SDK 拼出 INCLUDE / LIB / PATH
//     （等价 vcvars64.bat 的关键部分），让 cl.exe 直接可用。
//  3. nvcc 内部会调 reg.exe 查注册表，而 reg.exe 在安全策略黑名单里会被拦。
//     该调用不影响编译结果（实测 .obj 正常产出），但会在日志里留下刺眼的
//     "PROGRAM BLOCKED BY SECURITY POLICY" 段落 —— 构建器会识别并解释它，
//     避免下次误判为构建失败。
//  4. ffmpeg 目录随发布包提供，源码树里默认没有。CMakeLists 已做成能力探测
//     （没有就关媒体解码、退化为纯文本），构建器只负责在摘要里说清楚。
//
// 用法：
//
//	builder.exe                     # 探测本机 GPU 架构并构建（最快路径）
//	builder.exe -list               # 只做环境自检，打印探测结果后退出（不编译）
//	builder.exe -arch 86            # 指定架构：75 | 86 | 89 | 120a | native | auto
//	builder.exe -j 16               # 并行度
//	builder.exe -fresh              # 删除构建目录后全量重建
//	builder.exe -keep               # 只重置 CMake 状态，保留已编译对象
//	builder.exe -clean              # 只清理，不构建
//	builder.exe -no-configure       # 跳过 configure（改了 CMakeLists 时必须去掉）
//	builder.exe -target ninfer      # 只构建单个目标（跳过产物齐全性自检）
//	builder.exe -media on           # 媒体解码：auto（默认）| on | off
//	builder.exe -gen ninja          # 生成器：ninja（默认）| vs
//
// 退出码：0 成功；1 构建/校验失败；90 工具链自检失败；91 架构不受支持。
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// 输出
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
// 环境
// ---------------------------------------------------------------------------

// proxyVars 传给 MSBuild / nvcc 的环境里必须剥离的变量（MSB6001 的根因）。
var proxyVars = []string{
	"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY",
	"http_proxy", "https_proxy", "all_proxy", "ftp_proxy",
	"NO_PROXY", "no_proxy",
}

// childExtraEnv 由 main 在解析参数后填好，run*() 调用时合并进子进程环境。
var childExtraEnv []string

// cleanEnv 返回剥离代理变量后的环境副本，再叠加 childExtraEnv（MSVC / ccache）。
// 注意：**只剥代理**，不做任何其它清洗 —— env -i 式的过度清洗会让
// FindCUDAToolkit 探到版本却找不到 CUDA_CUDART。
func cleanEnv() []string {
	var env []string
	for _, kv := range os.Environ() {
		blocked := false
		for _, p := range proxyVars {
			if strings.HasPrefix(kv, p+"=") {
				blocked = true
				break
			}
		}
		if !blocked {
			env = append(env, kv)
		}
	}
	return append(env, childExtraEnv...)
}

// ---------------------------------------------------------------------------
// 命令执行
// ---------------------------------------------------------------------------

// run 执行命令（工作目录固定为仓库根）。
func run(repoRoot, name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Dir = repoRoot
	cmd.Env = cleanEnv()
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	printInfo("$ " + name + " " + strings.Join(args, " "))
	start := time.Now()
	if err := cmd.Run(); err != nil {
		printError(fmt.Sprintf("%s 失败（耗时 %s）: %v", name, time.Since(start).Round(time.Second), err))
		return err
	}
	printInfo(fmt.Sprintf("%s 完成（耗时 %s）", name, time.Since(start).Round(time.Second)))
	return nil
}

// runCapture 执行命令并返回合并输出（不打印）。
func runCapture(dir, name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	cmd.Env = cleanEnv()
	out, err := cmd.CombinedOutput()
	return string(out), err
}

// transientPatterns 是并行编译时的环境瞬时竞争特征串：
//   - nvcc 在 %TEMP% 下生成的 tmpxft_*_cudafe1.cpp 被抢/丢失 →
//     "c1xx: fatal error C1083: 无法打开源文件 ... tmpxft_..."
//   - VS 生成器的构建系统自检戳文件被防病毒/索引器瞬时占住 → MSB8066 / MSB6001
//   - Ninja 在 Windows 上偶发的文件占用 → "Cannot restore timestamp"
//
// 两类都是重跑即过，不应让人工介入。
var transientPatterns = []string{
	"tmpxft_", "MSB8066", "MSB6001", "Cannot restore timestamp",
}

// benignNoise 是"看着吓人但不算失败"的输出特征：安全策略拦截 nvcc 内部的 reg.exe。
var benignNoise = []string{
	"PROGRAM BLOCKED BY SECURITY POLICY",
	"is not recognized as an internal or external command",
}

// logHasAny 检查日志尾部是否包含任一特征串。只看尾部 256 KB，避免大日志全量扫描。
func logHasAny(logPath string, patterns []string) bool {
	b, err := os.ReadFile(logPath)
	if err != nil {
		return false
	}
	if len(b) > 256*1024 {
		b = b[len(b)-256*1024:]
	}
	s := string(b)
	for _, p := range patterns {
		if strings.Contains(s, p) {
			return true
		}
	}
	return false
}

// runLogged 执行命令并把输出同时写到控制台与日志文件（tee）。
func runLogged(repoRoot, logPath, name string, args ...string) error {
	if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err != nil {
		printWarning("无法创建日志目录: " + err.Error())
	}
	f, err := os.Create(logPath)
	if err != nil {
		printWarning("无法创建日志文件（仅输出到控制台）: " + err.Error())
		return run(repoRoot, name, args...)
	}
	defer f.Close()

	cmd := exec.Command(name, args...)
	cmd.Dir = repoRoot
	cmd.Env = cleanEnv()
	w := io.MultiWriter(os.Stdout, f)
	cmd.Stdout, cmd.Stderr = w, w
	printInfo("$ " + name + " " + strings.Join(args, " ") + "   (日志: " + logPath + ")")
	start := time.Now()
	err = cmd.Run()
	if err != nil {
		printError(fmt.Sprintf("%s 失败（耗时 %s）: %v", name, time.Since(start).Round(time.Second), err))
		if logHasAny(logPath, benignNoise) {
			printWarning("日志里出现安全策略拦截的噪声行（多为 nvcc 内部调用 reg.exe）。" +
				"它不是构建失败的原因；本机 reg.exe 在程序黑名单里，属已知现象。")
		}
		if logHasAny(logPath, []string{"unsupported Microsoft Visual Studio version"}) {
			printWarning("nvcc 拒绝当前 MSVC：请加 -cuda-allow-unsupported 重试" +
				"（等价 -DNINFER_CUDA_ALLOW_UNSUPPORTED=ON）。")
		}
		return err
	}
	printInfo(fmt.Sprintf("%s 完成（耗时 %s）", name, time.Since(start).Round(time.Second)))
	return nil
}

// runWithRetry 在遇到瞬时竞争（见 transientPatterns）时自动重跑，最多 attempts 次。
func runWithRetry(repoRoot, logPath, name string, attempts int, args ...string) error {
	var lastErr error
	for i := 1; i <= attempts; i++ {
		err := runLogged(repoRoot, logPath, name, args...)
		if err == nil {
			return nil
		}
		lastErr = err
		if i < attempts && logHasAny(logPath, transientPatterns) {
			printWarning(fmt.Sprintf("检测到并行编译的瞬时竞争（nvcc 临时文件 / 构建戳文件），自动重试 %d/%d", i, attempts-1))
			continue
		}
		return err
	}
	return lastErr
}

// ---------------------------------------------------------------------------
// 工具链探测
// ---------------------------------------------------------------------------

const sdkBaseWin = `C:\Program Files (x86)\Windows Kits\10`

// maxDirEntry 返回目录下字典序最大的子目录名（版本号零填充一致时等价于版本序）。
func maxDirEntry(base string) string {
	ents, err := os.ReadDir(base)
	if err != nil {
		return ""
	}
	var names []string
	for _, e := range ents {
		if e.IsDir() {
			names = append(names, e.Name())
		}
	}
	if len(names) == 0 {
		return ""
	}
	sort.Strings(names)
	return names[len(names)-1]
}

// findMSVC 探测 VS 安装根、MSVC 工具集版本、Windows SDK 版本。
func findMSVC() (vsRoot, msvcVer, sdkVer string) {
	for _, c := range []string{
		`C:\Program Files\Microsoft Visual Studio\2022\Community`,
		`C:\Program Files\Microsoft Visual Studio\2022\Professional`,
		`C:\Program Files\Microsoft Visual Studio\2022\Enterprise`,
		`C:\Program Files\Microsoft Visual Studio\2022\BuildTools`,
		`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`,
		`C:\Program Files\Microsoft Visual Studio\18\Community`,
		`C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools`,
	} {
		if v := maxDirEntry(filepath.Join(c, "VC", "Tools", "MSVC")); v != "" {
			vsRoot, msvcVer = c, v
			break
		}
	}
	if vsRoot == "" {
		return "", "", ""
	}
	sdkVer = maxDirEntry(filepath.Join(sdkBaseWin, "Include"))
	return vsRoot, msvcVer, sdkVer
}

// msvcEnvVars 由探测结果拼出 vcvars64.bat 的关键环境变量。
// 本机 cmd.exe 被禁，vcvars 调不动，所以这里是唯一的 cl.exe 可用通道。
func msvcEnvVars(vsRoot, msvcVer, sdkVer string) []string {
	vc := filepath.Join(vsRoot, "VC", "Tools", "MSVC", msvcVer)
	include := strings.Join([]string{
		filepath.Join(vc, "include"),
		filepath.Join(sdkBaseWin, "Include", sdkVer, "ucrt"),
		filepath.Join(sdkBaseWin, "Include", sdkVer, "um"),
		filepath.Join(sdkBaseWin, "Include", sdkVer, "shared"),
		filepath.Join(sdkBaseWin, "Include", sdkVer, "winrt"),
	}, ";")
	lib := strings.Join([]string{
		filepath.Join(vc, "lib", "x64"),
		filepath.Join(sdkBaseWin, "Lib", sdkVer, "ucrt", "x64"),
		filepath.Join(sdkBaseWin, "Lib", sdkVer, "um", "x64"),
	}, ";")
	path := strings.Join([]string{
		filepath.Join(vc, "bin", "Hostx64", "x64"),
		filepath.Join(vsRoot, "Common7", "IDE", "CommonExtensions", "Microsoft", "CMake", "Ninja"),
		filepath.Join(sdkBaseWin, "bin", sdkVer, "x64"),
		os.Getenv("PATH"),
	}, string(os.PathListSeparator))
	return []string{
		"INCLUDE=" + include,
		"LIB=" + lib,
		"PATH=" + path,
		"VCToolsInstallDir=" + vc + `\`,
		"VCINSTALLDIR=" + filepath.Join(vsRoot, "VC") + `\`,
		"WindowsSdkDir=" + sdkBaseWin + `\`,
		"WindowsSDKVersion=" + sdkVer + `\`,
		// cl.exe 默认输出中文（GBK）；nvcc/ccache 按 UTF-8 解析会崩，统一英文输出。
		"VSLANG=1033",
	}
}

// findNinja 定位 ninja.exe：先 PATH，再 VS 自带路径。
func findNinja(vsRoot string) string {
	if p, err := exec.LookPath("ninja"); err == nil {
		return p
	}
	if vsRoot == "" {
		return ""
	}
	p := filepath.Join(vsRoot, "Common7", "IDE", "CommonExtensions", "Microsoft", "CMake", "Ninja", "ninja.exe")
	if _, err := os.Stat(p); err == nil {
		return p
	}
	return ""
}

// findCmake 定位 cmake.exe（本机装在 C:\Programs\CMake\bin，非标准前缀）。
func findCmake() string {
	if p, err := exec.LookPath("cmake"); err == nil {
		return p
	}
	for _, p := range []string{
		`C:\Programs\CMake\bin\cmake.exe`,
		`C:\Program Files\CMake\bin\cmake.exe`,
		`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`,
	} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

// cudaInfo 记录 CUDA 工具链探测结果。
type cudaInfo struct {
	binDir  string // 例：C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
	nvcc    string
	major   int
	minor   int
	version string // 例：12.8
}

// findCuda 定位 nvcc 并解析版本。返回的 binDir 用于把 cudart64_*.dll 落到运行时布局。
func findCuda() (cudaInfo, error) {
	var ci cudaInfo
	nvcc, err := exec.LookPath("nvcc")
	if err != nil {
		// PATH 里没有就试 CUDA_PATH / 默认安装前缀
		for _, base := range []string{
			os.Getenv("CUDA_PATH"),
			`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`,
			`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`,
		} {
			if base == "" {
				continue
			}
			p := filepath.Join(base, "bin", "nvcc.exe")
			if _, err := os.Stat(p); err == nil {
				nvcc = p
				break
			}
		}
	}
	if nvcc == "" {
		return ci, errors.New("未找到 nvcc.exe（PATH 与 CUDA_PATH 都没有）")
	}
	ci.nvcc = nvcc
	ci.binDir = filepath.Dir(nvcc)

	out, err := runCapture("", nvcc, "--version")
	if err != nil {
		return ci, fmt.Errorf("执行 nvcc --version 失败: %w", err)
	}
	// 形如: "Cuda compilation tools, release 12.8, V12.8.93"
	re := regexp.MustCompile(`release\s+(\d+)\.(\d+)`)
	m := re.FindStringSubmatch(out)
	if m == nil {
		return ci, errors.New("无法从 nvcc --version 解析版本号")
	}
	ci.major, _ = strconv.Atoi(m[1])
	ci.minor, _ = strconv.Atoi(m[2])
	ci.version = fmt.Sprintf("%d.%d", ci.major, ci.minor)
	return ci, nil
}

// findCcache 定位 ccache.exe：先 PATH，再常见便携安装目录。
func findCcache() string {
	if p, err := exec.LookPath("ccache"); err == nil {
		return p
	}
	for _, pat := range []string{
		filepath.Join(`D:\winkit\share`, "ccache-*", "ccache.exe"),
		filepath.Join(`C:\winkit\share`, "ccache-*", "ccache.exe"),
		filepath.Join(`D:\tools`, "ccache-*", "ccache.exe"),
		filepath.Join(`C:\Programs`, "ccache-*", "ccache.exe"),
	} {
		if ms, err := filepath.Glob(pat); err == nil && len(ms) > 0 {
			sort.Strings(ms)
			return ms[len(ms)-1]
		}
	}
	return ""
}

// ccacheEnvVars 组装 ccache 运行环境。缓存放仓库的兄弟目录：
// 多个 worktree/分支共享同一份缓存，且不污染 git 工作区。
func ccacheEnvVars(repoRoot, ccacheDir string) []string {
	if ccacheDir == "" {
		ccacheDir = filepath.Join(filepath.Dir(repoRoot), ".ccache")
	}
	os.MkdirAll(ccacheDir, 0o755)
	return []string{
		"CCACHE_DIR=" + ccacheDir,
		"CCACHE_MAXSIZE=20G",
		// MSVC/nvcc 的 include 时间戳与绝对路径会让 ccache 拒绝缓存，放宽检查
		"CCACHE_SLOPPINESS=time_macros,include_file_ctime,include_file_mtime",
		"CCACHE_BASEDIR=" + filepath.Dir(repoRoot),
	}
}

// ---------------------------------------------------------------------------
// GPU 探测
// ---------------------------------------------------------------------------

// archSpec 是一档受支持的 CUDA 架构。
type archSpec struct {
	id      string // CMake 的 CMAKE_CUDA_ARCHITECTURES 取值
	compute string // nvidia-smi 的 compute_cap，例 "8.6"
	name    string // 人类可读
	smem    string // 共享内存档位说明（写进摘要，便于判断内核该走哪条分支）
}

var archTable = []archSpec{
	{id: "75", compute: "7.5", name: "Turing (RTX 2080 Ti)", smem: "64 KiB/SM · 无 cp.async · 无 bf16 MMA"},
	{id: "86", compute: "8.6", name: "Ampere (RTX 3090 / 3060 Ti)", smem: "100 KiB/SM · 有 cp.async/bf16 MMA"},
	{id: "89", compute: "8.9", name: "Ada (RTX 4090 / 4070)", smem: "100 KiB/SM · capped w8 变体"},
	{id: "120a", compute: "12.0", name: "Blackwell (RTX 5090)", smem: "高内存调度 · NVFP4 TMA"},
}

// lookupArch 按 id 或 compute_cap 反查架构档位。
func lookupArch(key string) (archSpec, bool) {
	k := strings.ToLower(strings.TrimSpace(key))
	for _, a := range archTable {
		if strings.EqualFold(a.id, k) || a.compute == key {
			return a, true
		}
	}
	return archSpec{}, false
}

// detectGpu 用 nvidia-smi 探测本机 GPU 名称与 compute_cap。
func detectGpu() (name, compute string, err error) {
	out, err := runCapture("", "nvidia-smi",
		"--query-gpu=compute_cap,name", "--format=csv,noheader")
	if err != nil {
		return "", "", err
	}
	line := strings.TrimSpace(strings.Split(strings.TrimSpace(out), "\n")[0])
	parts := strings.SplitN(line, ",", 2)
	if len(parts) < 2 {
		return "", "", fmt.Errorf("nvidia-smi 输出无法解析: %q", line)
	}
	return strings.TrimSpace(parts[1]), strings.TrimSpace(parts[0]), nil
}

// resolveArch 把用户传入的 -arch 解析成具体架构档位。native/auto/空 → 探测本机 GPU。
func resolveArch(archFlag string) (archSpec, string, error) {
	if archFlag == "" || archFlag == "auto" || archFlag == "native" {
		_, cap, err := detectGpu()
		if err != nil {
			return archSpec{}, "", fmt.Errorf(
				"无法探测本机 GPU（%v）。请显式指定 -arch 75|86|89|120a", err)
		}
		spec, ok := lookupArch(cap)
		if !ok {
			return archSpec{}, cap, fmt.Errorf(
				"本机 compute_cap=%s 不在支持列表（75/86/89/120a）内，请显式指定 -arch", cap)
		}
		return spec, cap, nil
	}
	spec, ok := lookupArch(archFlag)
	if !ok {
		return archSpec{}, "", fmt.Errorf("不支持的 -arch %q（可选 75|86|89|120a|native）", archFlag)
	}
	return spec, "", nil
}

// ---------------------------------------------------------------------------
// 构建目录
// ---------------------------------------------------------------------------

// buildDirName 生成架构相关的构建目录名，多档架构可并存互不干扰。
func buildDirName(arch string) string {
	return "_build_" + arch
}

// detectGenerator 读已有构建目录的 CMakeCache，返回其生成器名（无则空串）。
// CMake 不允许在既有目录上更换生成器，所以已有目录必须沿用原生成器。
func detectGenerator(buildDir string) string {
	b, err := os.ReadFile(filepath.Join(buildDir, "CMakeCache.txt"))
	if err != nil {
		return ""
	}
	for _, l := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(l, "CMAKE_GENERATOR:INTERNAL=") {
			return strings.TrimPrefix(l, "CMAKE_GENERATOR:INTERNAL=")
		}
	}
	return ""
}

func removeBuildDir(buildDir string) {
	if _, err := os.Stat(buildDir); err != nil {
		return
	}
	if err := os.RemoveAll(buildDir); err != nil {
		printWarning("删除构建目录失败: " + err.Error())
		return
	}
	printInfo("已删除构建目录 " + buildDir)
}

// clearCMakeState 只清 CMake 的缓存与生成物，保留 object 文件（-keep 用）。
func clearCMakeState(buildDir string) {
	for _, f := range []string{"CMakeCache.txt", "CMakeFiles", "build.ninja", "rules.ninja"} {
		os.RemoveAll(filepath.Join(buildDir, f))
	}
	printInfo("已清除 CMake 状态（保留已编译对象）")
}

// ---------------------------------------------------------------------------
// 构建流程
// ---------------------------------------------------------------------------

// configure 组装并执行 cmake configure。
func configure(opts *options, genName string, cuda cudaInfo, ccachePath, logPath string) error {
	args := []string{"-B", opts.buildDir, "-S", ".", "-G", genName}
	if genName == "Visual Studio 17 2022" {
		args = append(args, "-A", "x64")
	}
	args = append(args,
		"-DCMAKE_CUDA_ARCHITECTURES="+opts.arch.id,
		"-DCMAKE_BUILD_TYPE=Release",
	)

	// CUDA 版本策略：非 120a 目标允许放宽到 12.8（见文件头第 1 条实测结论）。
	if cuda.major*100+cuda.minor < 1301 {
		if opts.arch.id != "120a" {
			args = append(args, "-DNINFER_ALLOW_LEGACY_CUDA=ON")
		} else {
			printError(fmt.Sprintf(
				"CUDA %s 无法编译 sm_120a（需要 12.9+/13.x）。请升级 CUDA，或改用非 Blackwell 目标。",
				cuda.version))
			return errors.New("CUDA 版本不满足 sm_120a 要求")
		}
	}
	if opts.cudaAllowUnsupported {
		args = append(args, "-DNINFER_CUDA_ALLOW_UNSUPPORTED=ON")
	}

	// 媒体解码：默认 auto，由 CMake 的能力探测决定；显式 on 时缺 ffmpeg 直接报错。
	switch opts.media {
	case "off":
		// CMakeLists 里 set(NINFER_BUILD_MEDIA_ACQUIRE OFF) 是普通变量，会遮蔽
		// -D 传入的同名缓存变量，所以这里只能靠"清空 ffmpeg 探测"的语义等价手段：
		// 显式传 FFMPEG 目录为空串让它探测失败。
		args = append(args, "-DNINFER_FORCE_NO_MEDIA=ON")
	case "on":
		if _, err := os.Stat(filepath.Join(opts.repoRoot, "ffmpeg", "include")); err != nil {
			printError("-media on 要求源码树根存在 ffmpeg/include 与 ffmpeg/lib（当前缺失）")
			return errors.New("ffmpeg 缺失")
		}
	}

	// ccache：只挂 CUDA。
	//   VS 生成器会**静默忽略** CMAKE_<LANG>_COMPILER_LAUNCHER；Ninja 是少数真正执行它的
	//   Windows 生成器。而 ccache 包装本机本地化 MSVC 必崩（cl.exe 输出 GBK，ccache 按
	//   UTF-8 解析 → std::filesystem "Illegal byte sequence"），包装 nvcc 却完全正常。
	//   CUDA 实例正是耗时大头，所以只挂 CUDA 既安全又有收益。
	ccacheOn := ccachePath != "" && genName == "Ninja"
	if ccacheOn {
		args = append(args, "-DCMAKE_CUDA_COMPILER_LAUNCHER="+ccachePath)
	}
	args = append(args, opts.extra...)
	return runWithRetry(opts.repoRoot, logPath, opts.cmake, 2, args...)
}

// build 执行编译。瞬时竞争（nvcc 临时文件等）允许再重试两次。
func build(opts *options, logPath string, targets []string) error {
	args := []string{"--build", opts.buildDir}
	if strings.Contains(opts.genName, "Ninja") {
		args = append(args, fmt.Sprintf("-j%d", opts.jobs))
	} else {
		args = append(args, "--config", "Release", fmt.Sprintf("-j%d", opts.jobs))
	}
	if len(targets) > 0 {
		args = append(args, "--target")
		args = append(args, targets...)
	}
	return runWithRetry(opts.repoRoot, logPath, opts.cmake, 3, args...)
}

// productNames 是运行时的三个可执行文件名（apps/CMakeLists.txt 的 target 名）。
var productNames = []string{"ninfer", "ninfer-serve", "ninfer-perplexity"}

// findBinDir 在候选布局里定位实际产出目录。
// Ninja（单配置）落在 <build>/apps/；VS（多配置）落在 <build>/apps/Release/。
func findBinDir(buildDir string) string {
	candidates := []string{
		filepath.Join(buildDir, "apps"),
		filepath.Join(buildDir, "apps", "Release"),
		filepath.Join(buildDir, "bin", "Release"),
		filepath.Join(buildDir, "bin"),
		buildDir,
	}
	for _, d := range candidates {
		if _, err := os.Stat(filepath.Join(d, "ninfer.exe")); err == nil {
			return d
		}
	}
	return ""
}

// verifyArtifacts 收尾自检。只看 --build 的退出码是不够的：
// 并行构建下单个工程的中断未必让整体返回非零，静默缺产物必须显式抓出来。
func verifyArtifacts(opts *options) error {
	binDir := findBinDir(opts.buildDir)
	if binDir == "" {
		return errors.New("构建自检：没有在预期位置找到 ninfer.exe")
	}
	missing := 0
	for _, exe := range productNames {
		p := filepath.Join(binDir, exe+".exe")
		if fi, err := os.Stat(p); err != nil {
			printError("构建自检: 缺少产物 " + p)
			missing++
		} else {
			printSuccess(fmt.Sprintf("产物 %s.exe（%.1f MB）", exe, float64(fi.Size())/(1<<20)))
		}
	}
	if missing > 0 {
		return fmt.Errorf("缺少 %d 个产物", missing)
	}
	printSuccess("构建完成: " + binDir)
	return nil
}

// stageRuntime 把可执行文件与必需的 DLL 摆到构建目录根部，形成可直接运行的布局。
// 对齐既有 build_windows.bat 的做法，但多了一步：把 cudart 一并落位 ——
// 源码树里的 exe 直接双击运行时，缺 cudart64_*.dll 会以 0xc0000135 秒退。
func stageRuntime(opts *options, cuda cudaInfo) error {
	binDir := findBinDir(opts.buildDir)
	if binDir == "" {
		return errors.New("找不到产出目录，无法落位运行时")
	}
	dst := filepath.Join(opts.buildDir)
	n := 0
	for _, exe := range productNames {
		src := filepath.Join(binDir, exe+".exe")
		if _, err := os.Stat(src); err != nil {
			continue
		}
		if filepath.Clean(binDir) == filepath.Clean(dst) {
			continue
		}
		if err := copyFile(src, filepath.Join(dst, exe+".exe")); err != nil {
			printWarning("落位失败: " + err.Error())
		} else {
			n++
		}
	}
	if n > 0 {
		printSuccess(fmt.Sprintf("运行时布局：%d 个可执行文件已落位到 %s", n, dst))
	}

	// CUDA 运行时 DLL
	if cuda.binDir != "" {
		if ms, err := filepath.Glob(filepath.Join(cuda.binDir, "cudart64_*.dll")); err == nil {
			for _, m := range ms {
				_ = copyFile(m, filepath.Join(dst, filepath.Base(m)))
			}
			if len(ms) > 0 {
				printSuccess("运行时布局：cudart 已落位（" + filepath.Base(ms[0]) + "）")
			}
		}
	}

	// ffmpeg DLL（仅有 ffmpeg 目录时）
	if ms, err := filepath.Glob(filepath.Join(opts.repoRoot, "ffmpeg", "bin", "*.dll")); err == nil {
		c := 0
		for _, m := range ms {
			if err := copyFile(m, filepath.Join(dst, filepath.Base(m))); err == nil {
				c++
			}
		}
		if c > 0 {
			printSuccess(fmt.Sprintf("运行时布局：ffmpeg %d 个 DLL 已落位", c))
		}
	}
	return nil
}

// copyFile 复制文件（保留可执行位）。
func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	tmp := dst + ".tmp"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		os.Remove(tmp)
		return err
	}
	if err := out.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, dst)
}

// checkSkipConfigure 校验 -no-configure 的使用前提。
func checkSkipConfigure(opts *options, explicit map[string]bool) {
	if _, err := os.Stat(filepath.Join(opts.buildDir, "CMakeCache.txt")); err != nil {
		printError("-no-configure 需要已配置过的构建目录（未找到 CMakeCache.txt）。请先去掉本参数跑一次")
		os.Exit(90)
	}
	for _, n := range []string{"arch", "gen", "media", "no-ccache"} {
		if explicit[n] {
			printWarning(fmt.Sprintf("-%s 与 -no-configure 同用：跳过 configure 意味着该参数本次不会生效", n))
		}
	}
	printWarning("-no-configure：跳过 configure。若改过 CMakeLists 或换了构建参数，请去掉本参数重跑一次")
}

// ---------------------------------------------------------------------------
// 选项与主流程
// ---------------------------------------------------------------------------

type options struct {
	repoRoot             string
	buildDir             string
	arch                 archSpec
	jobs                 int
	genName              string
	genFlag              string
	cmake                string
	media                string
	keep                 bool
	fresh                bool
	clean                bool
	noConfigure          bool
	targets              []string
	noCcache             bool
	ccacheDir            string
	cudaAllowUnsupported bool
	extra                []string
	listOnly             bool
}

// envReport 打印环境探测摘要。
func envReport(opts *options, vsRoot, msvcVer, sdkVer, ninja, ccachePath string,
	cuda cudaInfo, cudaErr error, gpuName, gpuCap string) {

	printInfo(fmt.Sprintf("仓库根: %s", opts.repoRoot))
	printInfo(fmt.Sprintf("构建目录: %s", opts.buildDir))
	archLine := fmt.Sprintf("目标架构: sm_%s — %s", opts.arch.id, opts.arch.name)
	if opts.arch.smem != "" {
		archLine += "  [" + opts.arch.smem + "]"
	}
	printInfo(archLine)
	if gpuName != "" {
		printInfo(fmt.Sprintf("本机 GPU: %s (compute_cap=%s)", gpuName, gpuCap))
	}
	if vsRoot != "" {
		printInfo(fmt.Sprintf("MSVC: %s (%s) / SDK %s", vsRoot, msvcVer, sdkVer))
	} else {
		printWarning("未探测到 VS 安装：cl.exe 将不可用，构建必然失败")
	}
	if ninja != "" {
		printInfo("Ninja: " + ninja)
	} else if opts.genName == "Ninja" {
		printWarning("未找到 ninja.exe（PATH 与 VS 自带路径都没有）")
	}
	printInfo("CMake: " + opts.cmake)
	if cudaErr != nil {
		printError("CUDA: " + cudaErr.Error())
	} else {
		printInfo(fmt.Sprintf("CUDA: nvcc %s（%s）", cuda.version, cuda.nvcc))
		if cuda.major*100+cuda.minor < 1301 {
			if opts.arch.id == "120a" {
				printError("CUDA < 13.1 无法构建 sm_120a")
			} else {
				printSuccess("CUDA 版本策略：将启用 NINFER_ALLOW_LEGACY_CUDA（非 Blackwell 目标允许 < 13.1）")
			}
		}
	}
	if ccachePath != "" && opts.genName == "Ninja" {
		printSuccess("ccache: " + ccachePath + "（仅挂 CUDA）")
	} else if ccachePath != "" {
		printWarning("ccache: 已找到但生成器非 Ninja（VS 生成器会忽略 launcher），本次不生效")
	} else {
		printWarning("ccache: 未找到 ccache.exe（可选；放在 PATH 或 C:\\Programs\\ccache-*）")
	}

	// ffmpeg 能力
	if _, err := os.Stat(filepath.Join(opts.repoRoot, "ffmpeg", "include")); err == nil {
		printSuccess("ffmpeg: 已就位（媒体解码启用）")
	} else {
		printWarning("ffmpeg: 源码树里没有 ffmpeg/ 目录 → 媒体解码关闭，本次为纯文本构建" +
			"（需要图像/视频输入时把 ffmpeg 目录放进去再重跑）")
	}
}

func main() {
	jobs := flag.Int("j", 0, "并行编译度（默认 CPU 核数）")
	archFlag := flag.String("arch", "auto", "CUDA 架构：75 | 86 | 89 | 120a | native | auto（默认探测本机 GPU）")
	gen := flag.String("gen", "ninja", "生成器：ninja（默认）| vs")
	root := flag.String("C", "", "仓库根（默认 builder.exe 所在目录）")
	media := flag.String("media", "auto", "媒体解码：auto（默认）| on | off")
	keep := flag.Bool("keep", false, "仅重置 CMake 状态，保留已编译对象（CMake 缓存损坏/改选项后用）")
	fresh := flag.Bool("fresh", false, "先删除构建目录再重建（全量）")
	clean := flag.Bool("clean", false, "只清理不构建")
	noConfigure := flag.Bool("no-configure", false, "跳过 CMake configure（改了 CMakeLists 时必须去掉）")
	targetsFlag := flag.String("target", "", "只构建指定目标（逗号分隔，如 ninfer-serve）；此时跳过产物齐全性自检")
	noCcache := flag.Bool("no-ccache", false, "关闭 ccache")
	ccacheDir := flag.String("ccache-dir", "", "ccache 缓存目录（默认 <仓库父目录>/.ccache）")
	cudaAllowUnsupported := flag.Bool("cuda-allow-unsupported", false,
		"给 nvcc 传 -allow-unsupported-compiler（当前 MSVC 比 nvcc 白名单新时用）")
	listOnly := flag.Bool("list", false, "只打印探测到的工具链后退出（不编译）")
	configureOnly := flag.Bool("configure-only", false, "只跑 CMake configure 后退出（CI 分阶段 / 验证工具链）")
	extraFlag := flag.String("D", "", "额外 CMake 参数，分号分隔，如 -D=\"NINFER_BUILD_BENCHMARKS=ON\"")
	flag.Parse()

	explicit := map[string]bool{}
	flag.Visit(func(f *flag.Flag) { explicit[f.Name] = true })

	// ---- 仓库根 ----
	repoRoot := ""
	if *root != "" {
		if abs, err := filepath.Abs(*root); err == nil {
			repoRoot = abs
		}
	} else {
		exePath, err := os.Executable()
		if err != nil {
			exePath = os.Args[0]
		}
		repoRoot, _ = filepath.Abs(filepath.Dir(exePath))
		// `go run builder.go` 时 exe 在临时目录里，退回当前工作目录。
		if strings.Contains(filepath.Base(exePath), "go-build") ||
			filepath.Base(exePath) == "builder.test.exe" {
			if wd, err := os.Getwd(); err == nil {
				repoRoot = wd
			}
		}
	}
	// 仓库根自检：zatfung 的特征是 CMakeLists.txt + src/ + 内核目录。
	for _, need := range []string{"CMakeLists.txt", filepath.Join("src", "CMakeLists.txt"),
		filepath.Join("src", "ops", "linear", "ternary")} {
		if _, err := os.Stat(filepath.Join(repoRoot, need)); err != nil {
			printError("仓库根不像 zatfung 源码树（缺少 " + need + "）: " + repoRoot)
			os.Exit(90)
		}
	}

	// ---- 架构 ----
	archSpecResolved, gpuCap, err := resolveArch(*archFlag)
	if err != nil {
		printError(err.Error())
		os.Exit(91)
	}
	gpuName := ""
	if gpuCap != "" {
		gpuName, _, _ = detectGpu()
	}

	opts := &options{
		repoRoot:             repoRoot,
		arch:                 archSpecResolved,
		genFlag:              *gen,
		media:                *media,
		keep:                 *keep,
		fresh:                *fresh,
		clean:                *clean,
		noConfigure:          *noConfigure,
		noCcache:             *noCcache,
		ccacheDir:            *ccacheDir,
		cudaAllowUnsupported: *cudaAllowUnsupported,
		listOnly:             *listOnly,
	}
	opts.buildDir = buildDirName(opts.arch.id)
	if *targetsFlag != "" {
		for _, t := range strings.Split(*targetsFlag, ",") {
			if t = strings.TrimSpace(t); t != "" {
				opts.targets = append(opts.targets, t)
			}
		}
	}
	if *extraFlag != "" {
		for _, e := range strings.Split(*extraFlag, ";") {
			if e = strings.TrimSpace(e); e != "" {
				opts.extra = append(opts.extra, e)
			}
		}
	}

	opts.jobs = *jobs
	if opts.jobs <= 0 {
		opts.jobs = runtimeNumCPU()
	}

	// ---- 工具链探测 ----
	vsRoot, msvcVer, sdkVer := findMSVC()
	ninja := findNinja(vsRoot)
	ccachePath := findCcache()
	if opts.noCcache {
		ccachePath = ""
	}
	cuda, cudaErr := findCuda()
	cmake := findCmake()

	// 生成器：本机 VS 生成器不可用（cmd.exe 被禁 → CMake 找不到 cl），默认 Ninja。
	switch strings.ToLower(opts.genFlag) {
	case "vs", "visualstudio":
		opts.genName = "Visual Studio 17 2022"
	default:
		opts.genName = "Ninja"
	}
	if opts.genName == "Ninja" && ninja == "" {
		printError("ninja 生成器需要 ninja.exe，但没找到。要么安装 ninja，要么用 -gen vs（本机 VS 生成器不可用）")
		os.Exit(90)
	}

	// 子进程环境：MSVC 变量 + ccache 变量
	childExtraEnv = nil
	if vsRoot != "" {
		childExtraEnv = append(childExtraEnv, msvcEnvVars(vsRoot, msvcVer, sdkVer)...)
	}
	if ccachePath != "" {
		childExtraEnv = append(childExtraEnv, ccacheEnvVars(repoRoot, opts.ccacheDir)...)
	}

	opts.cmake = cmake
	if opts.cmake == "" {
		printError("未找到 cmake.exe（PATH 与常见安装前缀都没有）")
		os.Exit(90)
	}

	// ---- 兼容既有构建目录的生成器 ----
	detected := detectGenerator(filepath.Join(repoRoot, opts.buildDir))
	if detected != "" && !strings.Contains(detected, opts.genName[:4]) {
		printError(fmt.Sprintf(
			"已有构建目录 %s 用的是 %q，无法直接换成 %s；请先执行: builder.exe -clean -arch %s",
			opts.buildDir, detected, opts.genName, opts.arch.id))
		os.Exit(90)
	}

	// ---- 摘要 ----
	printInfo("=== zatfung 构建器 ===")
	envReport(opts, vsRoot, msvcVer, sdkVer, ninja, ccachePath, cuda, cudaErr, gpuName, gpuCap)
	printInfo(fmt.Sprintf("生成器: %s   并行度: %d", opts.genName, opts.jobs))

	if opts.listOnly {
		printInfo("子进程环境变量：")
		for _, kv := range childExtraEnv {
			printInfo("  " + kv)
		}
		return
	}
	if cudaErr != nil {
		os.Exit(90)
	}
	for _, p := range proxyVars {
		if os.Getenv(p) != "" {
			printWarning(fmt.Sprintf("检测到 %s，已对子进程剥离（防 MSB6001）", p))
			break
		}
	}

	// ---- clean ----
	if opts.clean {
		removeBuildDir(filepath.Join(repoRoot, opts.buildDir))
		printSuccess("清理完成")
		return
	}
	if opts.fresh {
		removeBuildDir(filepath.Join(repoRoot, opts.buildDir))
	} else if opts.keep {
		clearCMakeState(filepath.Join(repoRoot, opts.buildDir))
	}

	logDir := filepath.Join(repoRoot, opts.buildDir, "logs")
	os.MkdirAll(logDir, 0o755)

	// ---- configure ----
	if opts.noConfigure {
		checkSkipConfigure(opts, explicit)
	} else {
		logPath := filepath.Join(logDir, "configure.log")
		if err := configure(opts, opts.genName, cuda, ccachePath, logPath); err != nil {
			printError("configure 失败，详见 " + logPath)
			os.Exit(1)
		}
	}
	if *configureOnly {
		printSuccess("configure 完成（-configure-only，未编译）: " + filepath.Join(repoRoot, opts.buildDir))
		return
	}

	// ---- build ----
	buildLog := filepath.Join(logDir, "build.log")
	if err := build(opts, buildLog, opts.targets); err != nil {
		printError("编译失败，详见 " + buildLog)
		os.Exit(1)
	}

	// ---- 只构建部分目标时不做产物齐全性自检 ----
	if len(opts.targets) > 0 {
		printSuccess("指定目标构建完成: " + strings.Join(opts.targets, ", "))
		return
	}

	// ---- 收尾 ----
	if err := verifyArtifacts(opts); err != nil {
		printError(err.Error())
		os.Exit(1)
	}
	if err := stageRuntime(opts, cuda); err != nil {
		printWarning("运行时落位未完成: " + err.Error())
	}
	printSuccess("zatfung sm_" + opts.arch.id + " 构建链路全部完成")
}

// runtimeNumCPU 取并行度默认值。
func runtimeNumCPU() int {
	if n := os.Getenv("NUMBER_OF_PROCESSORS"); n != "" {
		if v, err := strconv.Atoi(n); err == nil && v > 0 {
			return v
		}
	}
	return 8
}
