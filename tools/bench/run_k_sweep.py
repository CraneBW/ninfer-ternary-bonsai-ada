#!/usr/bin/env python3
"""K 扫描：同一任务换 --draft-tokens，找 decode 的最优点。

为什么需要它：`--draft-tokens` 的最优值随任务移动，而且移动幅度很大 ——
本机实测结构化输出（纯 tool_call）K=3 → K=7 是 +26%，自然语言写作 K=3 → K=7 是 −30%。
run_speed_matrix.py 的 BEST_K 表就是这份扫描的产物；换任务、换机器、换 KV 档之后要重扫。

判据是 decode speed，同时记 MTP 接受率 —— 接受率若在臂间不同，说明任务漂了而不是
K 值钱（同一个 `--greedy` 下，KV 量化会改变输出，接受率跟着变，这是真实差异不是噪声）。

用法：
    python3 tools/bench/run_k_sweep.py [轮数]      # 默认 1 轮，每格取较好值

环境变量：NINFER_BIN / NINFER_MODEL（同 run_speed_matrix.py）
明细写到 bench/fixtures/speed-matrix/ksweep.json。
"""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FX = os.path.join(ROOT, "bench/fixtures/speed-matrix")
OUT = os.path.join(FX, "ksweep.json")

BIN = os.environ.get("NINFER_BIN", os.path.join(ROOT, "build/apps/ninfer"))
MODEL = os.environ.get("NINFER_MODEL", "")

SPEED = re.compile(r"summary\s+decode speed\s+([\d.]+)")
ACC = re.compile(r"summary\s+mtp acceptance rate\s+([\d.]+)")
LEN = re.compile(r"summary\s+mtp acceptance length\s+([\d.]+)")
GEN = re.compile(r"summary\s+generated tokens\s+(\d+)")

# tool-weather-long 是"查到结果之后写建议"，与纯 tool_call 是**两种任务**，两者都留着 ——
# 它们的接受率差一倍，正好说明"工具查询"这个词本身有多含混。
TASKS = [("pelican", f"{FX}/pelican.json"),
         ("novel", f"{FX}/novel.json"),
         ("tool-advice", f"{FX}/tool-weather-long.json"),
         ("tool-call", f"{FX}/tool-weather.json")]
KS = [3, 5, 7]
THINKS = [("think", True), ("nothink", False)]


def run(fixture, k, thinking, reps, kv):
    args = [BIN, MODEL, "--messages", fixture, "--max-new", "256",
            "--spec", "mtp", "--draft-tokens", str(k), "--greedy",
            "--max-context", "8192", "--kv-dtype", kv]
    if not thinking:
        args.append("--no-thinking")
    best = None
    for _ in range(reps):
        p = subprocess.run(args, capture_output=True, text=True, timeout=600)
        t = p.stdout + p.stderr
        m = SPEED.search(t)
        if m is None:
            return None
        row = {"decode": float(m.group(1)),
               "acc": float(ACC.search(t).group(1)) if ACC.search(t) else None,
               "len": float(LEN.search(t).group(1)) if LEN.search(t) else None,
               "gen": int(GEN.search(t).group(1)) if GEN.search(t) else None}
        if best is None or row["decode"] > best["decode"]:
            best = row
    return best


def main():
    if not MODEL:
        sys.exit("set NINFER_MODEL to the .ninfer artifact")
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    kv = os.environ.get("K_SWEEP_KV", "fp8")
    print(f"===== K 扫描（短上下文 8192，{kv}，驻留，每组取 {reps} 次较好值）=====", flush=True)
    print(f"{'任务':<14}{'思考':<9}" + "".join(f"{'K=' + str(k):>22}" for k in KS), flush=True)
    print("-" * 92, flush=True)

    out = []
    for task, fixture in TASKS:
        for tname, thinking in THINKS:
            cells, best_k, best_v = [], None, -1
            for k in KS:
                r = run(fixture, k, thinking, reps, kv)
                if r is None:
                    cells.append("  失败")
                    continue
                cells.append(f"{r['decode']:.1f} ({r['acc']:.0f}%)")
                out.append({"task": task, "think": tname, "k": k, **r})
                if r["decode"] > best_v:
                    best_v, best_k = r["decode"], k
            cells = [c + ("*" if i == KS.index(best_k) else " ") if best_k else c
                     for i, c in enumerate(cells)]
            print(f"{task:<14}{tname:<9}" + "".join(f"{c:>22}" for c in cells), flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n* = 该行最优 K。明细写到 {os.path.relpath(OUT, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
