"""Same-binary 200W AR shared-vector pairs at exact pp4K/32K, tg128.

Reuses ninfer_bench's fixed token corpus, graph prime and measured timing.
The report contains token counts, not output IDs; real-weight CPU oracle and
the frozen long Serve questions qualify the changed paths separately.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from tools.bench.prefix_checkpoint import PROFILE, ROOT, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = ROOT / "build/bench/ninfer_bench"
    corpus = ROOT / "bench/fixtures/bench_corpus.ids"
    power = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"], text=True)
    assert len(power.splitlines()) == 1 and float(power.strip()) == 200, power
    result = dict(power_w=200, binary_sha256=sha(binary), model=str(args.artifact.resolve()),
                  corpus=str(corpus), environment=PROFILE, runs=[])
    for i, policy in enumerate([0, 1, 1, 0, 0, 1]):
        pair = i // 2 + 1
        label = f"pair{pair}-shared{policy}"
        report = out / f"{label}.json"
        command = [
            str(binary), "--weights", str(args.artifact.resolve()),
            "--corpus", str(corpus), "-pg", "4096,128;32768,128",
            "--max-ctx", "36864", "-r", "1", "--warmup", "1",
            "--kv-dtype", "int8", "--prefill-chunk", "512",
            "--output", "json", "--output-file", str(report)]
        env = dict(os.environ, **PROFILE)
        env.update(NINFER_SM89_SHARED_VECTOR=str(policy),
                   NINFER_SM89_SHARED_VECTOR_TRACE=str(policy))
        print(f"START {label} pp4K/32K tg128 ordinary-AR", flush=True)
        began = time.perf_counter()
        with (out / f"{label}.log").open("w") as log:
            subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           check=True)
        raw = json.loads(report.read_text())
        config = raw["config"]
        assert config["draft_tokens"] == 0 and config["max_context"] == 36864
        assert config["use_cuda_graph"] and config["decode_graph_prime"]["primed"]
        assert [(t["n_prompt"], t["n_gen"]) for t in raw["tests"]] == [(4096,128),(32768,128)]
        for test in raw["tests"]:
            assert test["reps"][0]["generated_output_tokens"] == 129
            assert test["reps"][0]["decode_output_tokens"] == 128
        row = dict(label=label, pair=pair, policy=policy, report=str(report), command=command,
                   wall_s=time.perf_counter()-began, tests=raw["tests"])
        result["runs"].append(row)
        (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"label": label, "tests": [
            {"n_prompt": t["n_prompt"], "prefill": t["prefill_tok_s_mean"],
             "decode": t["decode_output_tok_s_mean"]} for t in raw["tests"]]}), flush=True)
    result["paired"] = []
    for j, context in enumerate([4096,32768]):
        diffs = {name: [] for name in ["prefill", "decode", "total_reduction"]}
        for pair in range(1,4):
            baseline = next(r["tests"][j] for r in result["runs"]
                            if r["pair"] == pair and r["policy"] == 0)
            candidate = next(r["tests"][j] for r in result["runs"]
                             if r["pair"] == pair and r["policy"] == 1)
            diffs["prefill"].append(candidate["prefill_tok_s_mean"]/baseline["prefill_tok_s_mean"]-1)
            diffs["decode"].append(candidate["decode_output_tok_s_mean"]/baseline["decode_output_tok_s_mean"]-1)
            diffs["total_reduction"].append(1-candidate["total_seconds_mean"]/baseline["total_seconds_mean"])
        result["paired"].append(dict(context=context, per_pair=diffs,
                                     medians={k:statistics.median(v) for k,v in diffs.items()}))
    result["state"] = "PASS_3_PAIRS_4K_32K"
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["paired"]), flush=True)


if __name__ == "__main__":
    main()
