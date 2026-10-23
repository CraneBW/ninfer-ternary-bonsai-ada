"""Directed three-query Main-only canonical continuation, no long prefill."""
import argparse
import copy
import json
import os
from pathlib import Path
import urllib.request
from tools.bench.prefix_checkpoint import ROOT, PROFILE
from tools.bench.sm89_profiles import LocalServer, dump

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=json.loads((ROOT/"tmp-nvcc/qualification-20261020/long-resident/32768-0.json").read_text())
    source["messages"][1]["content"]="\n".join(source["messages"][1]["content"].splitlines()[:550])
    source.update(model="qualification",max_tokens=8,temperature=0,stream=False)
    source.pop("stream_options",None)
    source["messages"].pop()
    os.environ.update(PROFILE,NINFER_KVMEM_CANONICAL_QUERY_REUSE="1",
                      NINFER_KVMEM_CANONICAL_QUERY_TRACE="1",NINFER_SM89_SHARED_VECTOR="1")
    command=[str(ROOT/"build/apps/ninfer-serve"),
        "/home/ran/ninfer-work/real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer",
        "--model-id","qualification","--host","127.0.0.1","--port","18176",
        "--max-context","12288","--kv-device-tokens","8192","--prefill-chunk","128",
        "--host-kv-mib","256","--max-concurrency","1","--kv-dtype","int8","--greedy",
        "--no-thinking","--kvmem","--kvmem-budget","4096","--kvmem-gen-reserve","256",
        "--kvmem-action-lrfu","--kvmem-lrfu-lambda","0.5","--kvmem-lrfu-top-frac","0.3",
        "--kvmem-transfer-mode","sync","--request-log-jsonl",str(out/"requests.jsonl")]
    report={"requests":[],"command":command}
    with LocalServer(command,"127.0.0.1",18176,out/"requests.jsonl") as server:
        report["startup"]=server.wait_until_ready()
        for expected in ["ACK","BETA","ALPHA"]:
            source["messages"].append({"role":"user","content":f"Reply exactly {expected}."})
            request=urllib.request.Request("http://127.0.0.1:18176/v1/chat/completions",
                data=json.dumps(source).encode(),headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(request,timeout=120) as response:
                body=json.loads(response.read())
                actual=body["choices"][0]["message"]["content"]
                report["requests"].append({"expected":expected,"actual":actual,"usage":body["usage"],
                                           "pass":actual==expected})
            source["messages"].append({"role":"assistant","content":expected})
    text=(out/"requests.console.log").read_text()
    report["reuse_hits"]=text.count("KVMEM_CANONICAL_QUERY_REUSE q0=")
    report["server_exit"]=server.process.returncode
    report["quality_pass"]=all(r["pass"] for r in report["requests"]) and report["server_exit"]==0
    report["state"]=("REUSE_HIT" if report["reuse_hits"] else "SAFE_DECLINE") if report["quality_pass"] else "QUALITY_FAILED"
    dump(out/"report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ["startup","command"]}),flush=True)
    return 0 if report["quality_pass"] else 1

if __name__=="__main__":
    raise SystemExit(main())
