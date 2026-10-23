"""Same-binary public Engine pairs for SM89 candidates and the selected profile.

Uses the existing frozen GSQ v3 model, C1/Graph/INT8, chunk1024 and MTP7.
Runs three alternating pairs at pp4096/32768+tg128, or pp256/4352+tg128
for the isolated silu256 extension (existing wide fusion stays on in both arms).
Complete Graph-prime, warmup and measured token IDs are captured outside timing;
raw reports retain the actual speculative work. The final profile compares
chunk512 against qualified chunk1024 with both S4 experiments disabled.
Routine runs require 200W; --release-acceptance requires an externally set 300W
and runs one final acceptance pair. This tool never changes power itself.
The chunk2048 candidate keeps the qualified internal MMQ cap at 1024 and
changes only the outer prefill batch. --screen runs one pair before deciding
whether to complete the remaining pairs; it is not a stable speed claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

from tools.bench.kvmem_async import signature
from tools.bench.prefix_checkpoint import PROFILE, ROOT, WORK
from tools.bench.sm89_profiles import dump


def read(path: Path):
    return json.loads(path.read_text())


def require(condition, message: str):
    if not condition:
        raise RuntimeError(message)


def source_signatures():
    files = [Path(__file__).resolve(), ROOT / "src/ops/linear/gguf/gguf_dispatch.cpp",
             ROOT / "src/ops/linear/gguf/ggml_bridge.cu",
             ROOT / "src/targets/qwen3_6_27b/impl/variant.cpp"]
    for directory in (
        ROOT / "src/ops/softmax_attention/dense/causal_cache",
        ROOT / "src/ops/linear_attention/gated_delta_net",
    ):
        files.extend(p for p in directory.iterdir()
                     if p.suffix in (".cu", ".cuh", ".cpp", ".h"))
    return {str(p.relative_to(ROOT)): dict(signature(p),
                sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)}


def power_limit(expected=200):
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"],
        text=True,
    )
    lines = raw.splitlines()
    require(len(lines) == 1 and float(lines[0]) == expected,
            f"this run requires one GPU at {expected}W; observed {raw.strip()}")
    return float(lines[0])


def capture_ids(path: Path, raw):
    records = []
    for line in path.read_text().splitlines():
        label, count, tokens = line.split("\t")
        ids = [int(token) for token in tokens.split()]
        require(len(ids) == int(count) and len(ids) > 0,
                f"incomplete full token capture in {path}: {label}")
        records.append(dict(label=label, count=int(count), ids=ids))
    expected_labels = ["decode-graph-prime"] + [
        f"pp{test["n_prompt"]}+tg128" for test in raw["tests"] for _ in range(2)]
    require([r["label"] for r in records] == expected_labels,
            f"missing Graph-prime/warmup/measured token capture: {path}")
    require(records[0]["count"] == raw["config"]["decode_graph_prime"]["output_tokens"],
            "Graph-prime capture count disagrees with the raw report")
    require(all(r["count"] == 129 for r in records[1:]),
            "warmup/measured workload did not produce the complete 129 output tokens")
    for record, phase in zip(records, ("graph-prime", "warmup", "measured",
                                       "warmup", "measured"), strict=True):
        record["phase"] = phase
    return records


def qualify_report(raw, chunk=1024, contexts=(4096, 32768)):
    config = raw["config"]
    require(config["draft_tokens"] == 7 and config["proposal_head"] == "optimized"
            and config["speculative_backend"] == "mtp", "MTP7 optimized-head profile changed")
    require(config["max_context"] == 36864 and config["prefill_chunk"] == chunk
            and config["kv_cache"] == "int8-group64", "context/chunk/KV profile changed")
    require(config["use_cuda_graph"] and config["decode_graph_prime"]["primed"]
            and config["repetitions"] == 1 and config["warmup"] == 1,
            "Graph/prime/warmup/repetition profile changed")
    require([(t["n_prompt"], t["n_gen"]) for t in raw["tests"]]
            == [(tokens, 128) for tokens in contexts], "public Engine workload changed")
    for test in raw["tests"]:
        require(test["requested_output_tokens"] == 129 and len(test["reps"]) == 1,
                "requested output/repetition count changed")
        rep = test["reps"][0]
        require(rep["generated_output_tokens"] == 129 and rep["decode_output_tokens"] == 128
                and rep["decode_engine_tokens"] == 128, "incomplete decode work")
        work = rep["speculative"]
        require(work["backend"] == "mtp" and work["enabled"] and work["draft_window"] == 7
                and work["rounds"] > 0, "missing actual speculative-work receipt")


def compare(reference, candidate):
    require(reference["output_ids"] == candidate["output_ids"],
            f"full Graph-prime/warmup/measured token IDs changed in pair {reference['pair']}")
    contexts = []
    for a, b in zip(reference["report"]["tests"], candidate["report"]["tests"], strict=True):
        contexts.append(dict(
            tokens=a["n_prompt"],
            prefill_speedup_pct=100 * (b["prefill_tok_s_mean"] / a["prefill_tok_s_mean"] - 1),
            decode_speedup_pct=100 * (b["decode_output_tok_s_mean"]
                                      / a["decode_output_tok_s_mean"] - 1),
            total_reduction_pct=100 * (1 - b["total_seconds_mean"] / a["total_seconds_mean"]),
            reference_speculative=a["speculative"], candidate_speculative=b["speculative"],
            speculative_work_equal=a["speculative"] == b["speculative"],
        ))
    return dict(pair=reference["pair"], output_ids_equal=True, contexts=contexts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("both", "attention", "gdn", "final", "chunk2048", "silu", "silu256"), default="both")
    parser.add_argument("--screen", action="store_true", help="one preliminary 200W pair only")
    parser.add_argument("--release-acceptance", action="store_true",
                        help="final profile only: one acceptance pair at externally set 300W")
    parser.add_argument("--output", type=Path, required=True, help="fresh receipt directory")
    args = parser.parse_args()
    if args.release_acceptance and args.candidate != "final":
        parser.error("--release-acceptance is only valid for --candidate final")
    if args.screen and args.release_acceptance:
        parser.error("--screen and --release-acceptance are separate scopes")
    contexts = (256, 4352) if args.candidate == "silu256" else (4096, 32768)
    expected_power = 300 if args.release_acceptance else 200
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = dict(state="RUNNING", candidate=args.candidate, power_w=expected_power, adopted=False,
                  release_acceptance=args.release_acceptance,
                  screening=args.screen, contexts=contexts,
                  schedule=[0, 1] if args.release_acceptance or args.screen else [0, 1, 1, 0, 0, 1],
                  arms=[], pairs=[])
    receipt = out / "comparison.json"
    dump(receipt, result)
    try:
        frozen = read(WORK / "benchmarks/frozen-long-context/213k/manifest.json")
        model = Path(frozen["model"]["path"])
        require(signature(model) == frozen["model"], "frozen GSQ model changed")
        binary = ROOT / "build/bench/ninfer_bench"
        corpus = ROOT / "bench/fixtures/bench_corpus.ids"
        identity, corpus_identity, sources = signature(binary), signature(corpus), source_signatures()
        result.update(binary=identity, model=frozen["model"], corpus=corpus_identity,
                      sources=sources, power_before_w=power_limit(expected_power))
        dump(receipt, result)
        for index, mode in enumerate(result["schedule"]):
            pair = index // 2 + 1
            label = f"pair-{pair}-{'candidate' if mode else 'reference'}"
            require(signature(binary) == identity, "binary changed during pairs")
            require(signature(model) == frozen["model"] and signature(corpus) == corpus_identity,
                    "frozen model/corpus changed during pairs")
            require(source_signatures() == sources, "production/tool source changed during pairs")
            power_before = power_limit(expected_power)
            chunk = 512 if args.candidate == "final" and not mode else 1024
            if args.candidate == "chunk2048" and mode:
                chunk = 2048
            wide = int(args.candidate != "final" or bool(mode))
            report = out / f"{label}.json"
            capture = out / f"{label}.ids.tsv"
            command = [str(binary), "--weights", str(model), "--corpus", str(corpus),
                       "-pg", ";".join(f"{tokens},128" for tokens in contexts), "--max-ctx", "36864",
                       "--prefill-chunk", str(chunk), "--kv-dtype", "int8", "--spec", "mtp",
                       "--draft-tokens", "7", "--lm-head-draft", "-r", "1", "--warmup", "1",
                       "--output", "json", "--output-file", str(report)]
            env = dict(os.environ, **PROFILE)
            for key in list(env):
                if key.startswith("NINFER_") and key not in PROFILE:
                    del env[key]
            env.update(NINFER_R29_BUNDLE="0", NINFER_GGUF_LARGE_PREFILL=str(wide),
                       NINFER_SM89_SHARED_VECTOR="0",
                       NINFER_GGUF_SILU_MMQ=str(int(args.candidate == "silu256" or (bool(mode) and args.candidate == "silu"))),
                       NINFER_GGUF_SILU_MMQ_T256=str(int(bool(mode) and args.candidate == "silu256")),
                       NINFER_SM89_ATTN_SPLIT_VECTOR=str(int(bool(mode)
                                                          and args.candidate in ("both", "attention"))),
                       NINFER_SM89_GDN_PRENORM=str(int(bool(mode)
                                                     and args.candidate in ("both", "gdn"))),
                       NINFER_BENCH_TOKEN_IDS_PATH=str(capture))
            row = dict(pair=pair, mode=mode, chunk=chunk, label=label, command=command,
                       profile={key: value for key, value in env.items()
                                if key.startswith("NINFER_")}, power_before_w=power_before)
            result["arms"].append(row)
            dump(receipt, result)
            print(f"SM89_ATTN_GDN begin {args.candidate} {label}", flush=True)
            with (out / f"{label}.runner.log").open("w") as log:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT)
            row["returncode"] = process.returncode
            if report.exists():
                row["report"] = read(report)
            dump(receipt, result)
            require(process.returncode == 0, f"benchmark failed in {label}: {process.returncode}")
            require(signature(binary) == identity and source_signatures() == sources,
                    "binary/production/tool source changed while an arm ran")
            row["power_after_w"] = power_limit(expected_power)
            qualify_report(row["report"], chunk, contexts)
            row["output_ids"] = capture_ids(capture, row["report"])
            if index % 2 == 1:
                arms = {arm["mode"]: arm for arm in result["arms"] if arm["pair"] == pair}
                result["pairs"].append(compare(arms[0], arms[1]))
            dump(receipt, result)
            print(f"SM89_ATTN_GDN end {label} " + str([
                dict(tokens=test["n_prompt"], prefill=test["prefill_tok_s_mean"],
                     decode=test["decode_output_tok_s_mean"], work=test["speculative"])
                for test in row["report"]["tests"]]), flush=True)
        medians = []
        for index, tokens in enumerate(contexts):
            medians.append(dict(tokens=tokens, **{
                key: statistics.median(pair["contexts"][index][key] for pair in result["pairs"])
                for key in ("prefill_speedup_pct", "decode_speedup_pct", "total_reduction_pct")
            }))
        result.update(state="PASS", medians=medians, power_after_w=power_limit(expected_power))
        dump(receipt, result)
        print(json.dumps(dict(state="PASS", candidate=args.candidate, medians=medians)), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        result.update(state="STOPPED", error=f"{type(error).__name__}: {error}")
        dump(receipt, result)
        print(json.dumps(dict(state="STOPPED", error=result["error"])), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
