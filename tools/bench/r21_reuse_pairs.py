"""200W R21 TTFT pairs: unchanged frozen ACK then known Topaz continuation.

The first candidate completes all six frozen questions as the quality gate.
Timing compares only the identical first two requests in each arm; later
questions have quality coverage, not three-pair latency coverage. No saved KV.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
from tools.bench.prefix_checkpoint import ROOT
from tools.bench.kvmem_async import signature
from tools.bench.sm89_profiles import dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = signature(ROOT/"build/apps/ninfer-serve")
    reports = {}
    schedule = ((0,"content",6),(0,"root",2),(1,"root",2),
                (1,"content",2),(2,"content",2),(2,"root",2))
    for pair, mode, count in schedule:
        assert signature(ROOT/"build/apps/ninfer-serve") == binary
        path = out/f"pair-{pair+1}-{mode}"
        command = [sys.executable,"-m","tools.bench.local_long_quality",
                   "--backend","mtp","--original","--draft-tokens","7",
                   "--case","213k","--request-count",str(count),"--output",str(path)]
        if mode=="content":
            command += ["--original-causal-prefill","--original-content-anchors"]
        print(f"R21_PAIR begin pair={pair+1} mode={mode} requests={count}",flush=True)
        # Complete raw logs stay per-arm; emit one compact result at each boundary.
        with (out/f"pair-{pair+1}-{mode}.runner.log").open("w") as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads((path/"report.json").read_text())
        assert report["binary"]==binary and report["server_exit"]==0
        assert report["state"]==("PASS" if count==6 else "DIAGNOSTIC_PASS")
        rows=[json.loads(line) for line in (path/"requests.jsonl").read_text().splitlines()]
        done=[row for row in rows if row.get("event")=="request_done"]
        assert len(done)==count
        reports[pair,mode]=dict(path=str(path),first_two=report["requests"][:2],
                               done=[row["result"] for row in done[:2]],
                               summaries=report["layer_summaries"],
                               startup=report["startup"])
        print(f"R21_PAIR end pair={pair+1} mode={mode} "
              f"ttft={[r['ttft_s'] for r in report['requests'][:2]]}",flush=True)
    pairs=[]
    for pair in range(3):
        root,content=reports[pair,"root"],reports[pair,"content"]
        for before,after in zip(root["first_two"],content["first_two"],strict=True):
            assert before["sha256"]==after["sha256"] and before["actual"]==after["actual"]
            for key in ("prompt_tokens","completion_tokens","total_tokens"):
                assert before["usage"][key]==after["usage"][key]
        rt,ct=root["first_two"][1]["ttft_s"],content["first_two"][1]["ttft_s"]
        rw=sum(r["wall_s"] for r in root["first_two"])
        cw=sum(r["wall_s"] for r in content["first_two"])
        assert root["done"][1]["prefix_reuse_path"]=="root"
        assert content["done"][1]["prefix_cache_hit_tokens"]>200000
        pairs.append(dict(pair=pair+1,root=root,content=content,
                          continuation_ttft_reduction=1-ct/rt,
                          two_request_wall_sum_reduction=1-cw/rw))
    result=dict(state="PASS",power_w=200,binary=binary,
                scope="six-question quality; three pairs of identical frozen ACK->Topaz requests",
                qualification=str(out/"pair-1-content/report.json"),pairs=pairs,
                median_continuation_ttft_reduction=statistics.median(
                    p["continuation_ttft_reduction"] for p in pairs),
                median_two_request_wall_sum_reduction=statistics.median(
                    p["two_request_wall_sum_reduction"] for p in pairs))
    dump(out/"comparison.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="pairs"}),flush=True)


if __name__=="__main__":
    main()
