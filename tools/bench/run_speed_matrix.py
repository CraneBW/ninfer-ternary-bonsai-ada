#!/usr/bin/env python3
"""decode/prefill 速度矩阵：任务 x 上下文长度 x 思考 x 驻留模式。

READ ME FIRST —— 这份脚本存在的理由是两个已经踩过的坑：

1. **K 必须按任务取各自最优点。** `--draft-tokens` 的收益随任务的可预测性移动：
   结构化输出（纯 tool_call）在 K=7 上比 K=3 快 26%，而自然语言写作在 K=3 上最好、
   K=7 反而慢 30%。用一个固定的 K 跑遍所有任务，等于让一部分任务跑在它们的次优点上，
   量出来的"极限速度"不是极限。BEST_K 表就是这么来的（见 run_k_sweep.py）。

2. **"工具查询"有两种任务，速度差一倍。** 纯 tool_call 是结构化输出、接受率 80~95%；
   "查到结果之后写建议"是自然语言、接受率 40% 上下。为了让输出够长（短输出的 decode
   测不稳）而把前者改成后者，会把任务性质一起换掉 —— 那样量到的不是同一个任务。

用法：
    python3 tools/bench/run_speed_matrix.py 短 [轮数]        # 短上下文
    python3 tools/bench/run_speed_matrix.py 长 [轮数]        # 长上下文
    ONLY_MODE=KVMem-int8 python3 tools/bench/run_speed_matrix.py 长 1

环境变量：
    NINFER_BIN    可执行文件（默认 <repo>/build/apps/ninfer）
    NINFER_MODEL  .ninfer 制品（必填，或用 --model）
结果追加到 bench/fixtures/speed-matrix/results.jsonl。
"""
import json, os, re, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FX = os.path.join(ROOT, "bench/fixtures/speed-matrix")
OUT = os.path.join(FX, "results.jsonl")

BIN = os.environ.get("NINFER_BIN", os.path.join(ROOT, "build/apps/ninfer"))
MODEL = os.environ.get("NINFER_MODEL", "")

SPEED = re.compile(r"summary\s+decode speed\s+([\d.]+)")
ACC = re.compile(r"summary\s+mtp acceptance rate\s+([\d.]+)")
PREF = re.compile(r"summary\s+prefill speed\s+([\d.]+)([kM]?)")
PTOK = re.compile(r"summary\s+prompt tokens\s+(\d+)")
GEN = re.compile(r"summary\s+generated tokens\s+(\d+)")
HOST = re.compile(r"host KV pinned\s+\|\s+([\d.]+) GiB")


def mode_args(mode, maxctx):
    """(KV 格式, 额外参数)。KVMem 的池与窗口跟着 max-context 走：短上下文给一个刚好装下
    context 的池（不触发换出，量的是机制本身的固定开销），长上下文才真换出。"""
    if mode == "KVMem-int8":
        if maxctx <= 32768:
            return "int8", ["--kvmem", "--kv-device-tokens", str(maxctx),
                            "--kvmem-budget", str(maxctx), "--kvmem-gen-reserve", "0",
                            "--host-kv-mib", "8192"]
        return "int8", ["--kvmem", "--kv-device-tokens", "131072",
                        "--kvmem-budget", "98304", "--host-kv-mib", "8192"]
    if mode == "resident-fp8":
        return "fp8", []
    return "int8", []


MODES = ["resident-fp8", "resident-int8", "KVMem-int8"]

# 长度 → (任务 → 夹具, max-context)
LENS = {"短": ({"pelican": f"{FX}/pelican.json",
                "novel": f"{FX}/novel.json",
                "tool": f"{FX}/tool-weather.json"}, 8192),
        "长": ({"pelican": f"{FX}/long-pelican.json",
                "novel": f"{FX}/long-novel.json",
                "tool": f"{FX}/long-tool-call.json"}, 131072)}

THINKS = {"think": True, "nothink": False}
MAXNEW = "256"

# run_k_sweep.py 扫出来的最优 K（短上下文）。长上下文沿用它 —— 124k 上重扫的代价是每点
# 90 秒、三个 K 就是 4.5 分钟/任务，本轮没做，这是本表的一个已知缺口。
BEST_K = {("pelican", "think"): 3, ("pelican", "nothink"): 5,
          ("novel", "think"): 3, ("novel", "nothink"): 3,
          ("tool", "think"): 7, ("tool", "nothink"): 7}


def run(fixture, kv, extra, thinking, maxctx, k):
    args = [BIN, MODEL, "--messages", fixture,
            "--max-new", MAXNEW, "--spec", "mtp", "--draft-tokens", str(k),
            "--greedy", "--max-context", str(maxctx), "--kv-dtype", kv] + extra
    if not thinking:
        args.append("--no-thinking")
    t0 = time.time()
    p = subprocess.run(args, capture_output=True, text=True, timeout=2400)
    wall = time.time() - t0
    t = p.stdout + p.stderr
    m = SPEED.search(t)
    if m is None:
        return {"error": (t.strip().splitlines() or ["?"])[-1][:200], "wall": wall}
    mult = {"k": 1e3, "M": 1e6}.get(PREF.search(t).group(2), 1) if PREF.search(t) else 1
    return {"decode": float(m.group(1)),
            "acc": float(ACC.search(t).group(1)) if ACC.search(t) else None,
            "prefill": float(PREF.search(t).group(1)) * mult if PREF.search(t) else None,
            "prompt": int(PTOK.search(t).group(1)) if PTOK.search(t) else None,
            "gen": int(GEN.search(t).group(1)) if GEN.search(t) else None,
            "host_gib": float(HOST.search(t).group(1)) if HOST.search(t) else 0.0,
            "wall": wall}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in LENS:
        sys.exit(__doc__)
    if not MODEL:
        sys.exit("set NINFER_MODEL to the .ninfer artifact")
    length = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    fixtures, maxctx = LENS[length]

    print(f"===== {length}上下文（max-context {maxctx}）· {reps} 轮・各任务用各自最优 K =====",
          flush=True)
    print(f"{'模式':<14}{'任务':<9}{'思考':<9}{'K':>3}{'prompt':>9}{'gen':>6}"
          f"{'prefill':>10}{'decode':>9}{'接受率':>8}{'host':>7}", flush=True)
    print("-" * 88, flush=True)

    only = os.environ.get("ONLY_MODE")
    for mode in MODES:
        if only and mode != only:
            continue
        kv, extra = mode_args(mode, maxctx)
        # 长上下文：两种模式的设备池都是 131072 token（2048 页）。全驻留的逻辑 context 就是
        # 它 —— 再大显存装不下（262144 要 9.65 GB，16 GB 卡只有 8.87 GB 可用）；KVMem 的逻辑
        # context 是它的两倍，差额落 host，而这正是 KVMem 的价值所在。prompt 两边相同，可比。
        mode_maxctx = 262144 if (mode == "KVMem-int8" and maxctx > 32768) else maxctx
        for task, fixture in fixtures.items():
            for tname, thinking in THINKS.items():
                k = BEST_K[(task, tname)]
                rows = []
                for _ in range(reps):
                    r = run(fixture, kv, extra, thinking, mode_maxctx, k)
                    if "error" in r:
                        print(f"{mode:<14}{task:<9}{tname:<9}{k:>3}  失败：{r['error']}", flush=True)
                        break
                    rows.append(r)
                if not rows:
                    continue
                med = lambda key: statistics.median([x[key] for x in rows if x[key] is not None]) \
                    if any(x[key] is not None for x in rows) else None
                rec = {"len": length, "mode": mode, "task": task, "think": tname, "k": k,
                       "kv": kv, "reps": len(rows), "decode": med("decode"),
                       "acc": med("acc"), "prefill": med("prefill"),
                       "prompt": rows[0]["prompt"], "gen": rows[0]["gen"],
                       "host_gib": rows[0]["host_gib"], "wall": med("wall")}
                with open(OUT, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                pf = f"{rec['prefill']/1000:.2f}k" if rec["prefill"] else "?"
                ac = f"{rec['acc']:.1f}%" if rec["acc"] is not None else "?"
                print(f"{mode:<14}{task:<9}{tname:<9}{k:>3}{rec['prompt']:>9}{rec['gen']:>6}"
                      f"{pf:>10}{rec['decode']:>9.1f}{ac:>8}{rec['host_gib']:>7.1f}", flush=True)


if __name__ == "__main__":
    main()
