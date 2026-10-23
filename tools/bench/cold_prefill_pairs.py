"""Same-binary 200W chunk512 vs chunk1024 with the existing I64 wide route.

engine: exact pp4K/32K, tg128 with complete public Engine token captures.
long: unchanged frozen original 213K six questions, cold first and continuation
TTFT separately. No archive loading, question rewriting or power changes.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from tools.bench.prefix_checkpoint import PROFILE, ROOT, WORK
from tools.bench.kvmem_async import signature
from tools.bench.sm89_profiles import dump


def read(path):
    return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=("engine","long"),required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    frozen=read(WORK/"benchmarks/frozen-long-context/213k/manifest.json")
    model=Path(frozen["model"]["path"])
    assert signature(model)==frozen["model"]
    name="ninfer_bench" if args.phase=="engine" else "ninfer-serve"
    binary=ROOT/("build/bench" if args.phase=="engine" else "build/apps")/name
    identity=signature(binary)
    power=subprocess.check_output(["nvidia-smi","--query-gpu=power.limit",
        "--format=csv,noheader,nounits"],text=True)
    assert len(power.splitlines())==1 and float(power.strip())==200
    modes=(0,1,1,0,0,1) if args.phase=="engine" else (1,0,0,1,1,0)
    result=dict(state="RUNNING",phase=args.phase,binary=identity,model=frozen["model"],
                power_w=200,arms=[])
    for i,mode in enumerate(modes):
        pair=i//2+1;chunk=1024 if mode else 512
        label=f"pair-{pair}-chunk-{chunk}"
        assert signature(binary)==identity,"binary changed during pairs"
        if args.phase=="long":
            arm=out/label
            command=[sys.executable,"-m","tools.bench.local_long_quality",
                "--backend","mtp","--draft-tokens","7","--original",
                "--original-causal-prefill","--original-content-anchors","--case","213k",
                "--prefill-chunk",str(chunk),"--large-prefill",str(mode),"--output",str(arm)]
            env=None
        else:
            report=out/f"{label}.json"
            capture=out/f"{label}.ids.tsv"
            command=[str(binary),"--weights",str(model),"--corpus",
                str(ROOT/"bench/fixtures/bench_corpus.ids"),"-pg","4096,128;32768,128",
                "--max-ctx","36864","--prefill-chunk",str(chunk),"--kv-dtype","int8",
                "--spec","mtp","--draft-tokens","7","--lm-head-draft",
                "-r","1","--warmup","1","--output","json","--output-file",str(report)]
            env=dict(os.environ,**PROFILE)
            for key in list(env):
                if key.startswith(("NINFER_TEST_","NINFER_BENCH_")):del env[key]
            env.update(NINFER_R29_BUNDLE="0",NINFER_GGUF_LARGE_PREFILL=str(mode),
                       NINFER_BENCH_TOKEN_IDS_PATH=str(capture),NINFER_SM89_SHARED_VECTOR="0")
        print(f"COLD_PREFILL begin {args.phase} {label}",flush=True)
        with (out/f"{label}.runner.log").open("w") as log:
            process=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        raw=read(arm/"report.json") if args.phase=="long" else read(report)
        row=dict(pair=pair,mode=mode,chunk=chunk,label=label,command=command,report=raw)
        if args.phase=="engine":
            row["output_ids"]=capture.read_text()
            assert raw["config"]["draft_tokens"]==7 and raw["config"]["max_context"]==36864
            assert raw["config"]["use_cuda_graph"]
            assert [(t["n_prompt"],t["n_gen"]) for t in raw["tests"]]==[(4096,128),(32768,128)]
        result["arms"].append(row)
        dump(out/"comparison.json",result)
        if process.returncode or (args.phase=="long" and raw["state"]!="PASS"):
            result["state"]="STOPPED";dump(out/"comparison.json",result);return 1
        print(f"COLD_PREFILL end {label} " + (f"first_ttft={raw['requests'][0]['ttft_s']:.6f}"
            if args.phase=="long" else str([t["prefill_tok_s_mean"] for t in raw["tests"]])),flush=True)
    pairs=[]
    for pair in range(1,4):
        arms={r["mode"]:r for r in result["arms"] if r["pair"]==pair}
        a,b=arms[0]["report"],arms[1]["report"]
        if args.phase=="long":
            assert [(r["sha256"],r["actual"]) for r in a["requests"]]==[
                (r["sha256"],r["actual"]) for r in b["requests"]]
            pairs.append(dict(pair=pair,first_ttft_reduction=1-b["requests"][0]["ttft_s"]/a["requests"][0]["ttft_s"],
                session_reduction=1-b["session_wall_s"]/a["session_wall_s"],
                reference_first_s=a["requests"][0]["ttft_s"],candidate_first_s=b["requests"][0]["ttft_s"]))
        else:
            pairs.append(dict(pair=pair,output_ids_equal=arms[0]["output_ids"]==arms[1]["output_ids"],
                contexts=[dict(tokens=x["n_prompt"],prefill_speedup=y["prefill_tok_s_mean"]/x["prefill_tok_s_mean"]-1,
                    decode_speedup=y["decode_output_tok_s_mean"]/x["decode_output_tok_s_mean"]-1,
                    total_reduction=1-y["total_seconds_mean"]/x["total_seconds_mean"])
                    for x,y in zip(a["tests"],b["tests"],strict=True)]))
    result.update(state="PASS",pairs=pairs)
    if args.phase=="long":
        result["median_first_ttft_reduction"]=statistics.median(x["first_ttft_reduction"] for x in pairs)
        result["median_session_reduction"]=statistics.median(x["session_reduction"] for x in pairs)
    dump(out/"comparison.json",result)
    print(json.dumps(dict(state=result["state"],pairs=pairs)),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
