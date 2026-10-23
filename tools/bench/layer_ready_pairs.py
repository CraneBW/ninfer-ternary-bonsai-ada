"""Three bounded full-prefill 213K pairs for the opt-in SM89 page pipeline.

Reuses the frozen original requests and local_long_quality runner. No snapshots,
changed questions, retries, or cache eviction. Candidate's first arm is also its
quality gate; timing uses the same low-overhead exit summary in all arms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
from tools.bench.kvmem_async import signature
from tools.bench.prefix_checkpoint import ROOT, WORK
from tools.bench.sm89_profiles import dump


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--bulk",action="store_true",
                        help="Include fixed source/Cold workers and large Cold H2D in async arms")
    args=parser.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    frozen=json.loads((WORK/"benchmarks/frozen-long-context/213k/manifest.json").read_text())
    binary=ROOT/"build/apps/ninfer-serve"
    identity=signature(binary)
    schedule=[(pair,mode) for pair in range(3)
              for mode in (("async","sync") if pair%2==0 else ("sync","async"))]
    manifest=dict(binary=identity,model=frozen["model"],requests=frozen["requests"],
                  power_w=200,host_pages=128,draft_tokens=15,
                  schedule=schedule,canonical_query_reuse=True,
                  scope="ready-to-last-request; fresh process/full cold prefill, no saved KV; "
                        "NVMe-backed files can be OS-cache hot; /proc IO distinguishes physical reads",
                  bulk=args.bulk,
                  async_scope="source-event + async D2H/NVMe prefetch + Main16 layer-ready; " +
                      ("fixed two-worker source/Cold reads + bounded bulk Cold H2D; "
                       "whole-batch source rebuild and unsupported consumers retain waits"
                       if args.bulk else "multi-page source/Cold restores keep synchronous fallback"),
                  adoption_gate="all three session pairs improve; median reduction >=10%")
    dump(out/"manifest.json",manifest)
    reports=[]
    for pair,mode in schedule:
        assert signature(binary)==identity,"binary changed during pairs"
        arm=out/f"pair-{pair}-{mode}"
        command=[sys.executable,"-m","tools.bench.local_long_quality",
                 "--backend","mtp","--draft-tokens","15","--case","213k",
                 "--layer-ready",str(int(mode=="async")),"--transfer-mode",mode,
                 "--canonical-query-reuse","1","--host-pages","128","--output",str(arm)]
        command += ["--fixed-io-workers",str(int(args.bulk and mode=="async")),
                    "--cold-copy-pipeline",str(int(args.bulk and mode=="async"))]
        result=subprocess.run(command,cwd=ROOT)
        report=json.loads((arm/"report.json").read_text())
        reports.append(dict(pair=pair,mode=mode,path=str(arm),report=report))
        dump(out/"results.json",dict(state="RUNNING",arms=reports))
        if result.returncode or report["state"]!="PASS":
            dump(out/"results.json",dict(state="STOPPED",arms=reports))
            return 1
    reductions=[]
    for pair in range(3):
        arms={a["mode"]:a["report"] for a in reports if a["pair"]==pair}
        assert [r["actual"] for r in arms["sync"]["requests"]]==[r["actual"] for r in arms["async"]["requests"]]
        assert [r["usage"] for r in arms["sync"]["requests"]]==[r["usage"] for r in arms["async"]["requests"]]
        reductions.append(1-arms["async"]["session_wall_s"]/arms["sync"]["session_wall_s"])
    median=statistics.median(reductions)
    dump(out/"results.json",dict(state="PASS",arms=reports,session_reductions=reductions,
          median_session_reduction=median,adopt_async=all(x>0 for x in reductions) and median>=.10))
    print(json.dumps(dict(state="PASS",session_reductions=reductions,
                         median_session_reduction=median)),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
