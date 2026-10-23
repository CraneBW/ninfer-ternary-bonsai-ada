"""Qualify existing frozen six-question long inputs for AR, MTP or DFlash2.

Uses existing owned-server, protocol, exact-answer and metrics parsers. Always
uses real cold prefill without saved KV. Capacity rejection precedes the first
expensive prefill; 256K requires a passing 213K receipt for the same profile.
Single-request diagnostics cannot satisfy the full six-question gate.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from tools.bench.prefix_checkpoint import PROFILE, ROOT, WORK
from tools.bench.kvmem_async import signature
from tools.bench import kvmem_qualification as quality
from tools.bench.sm89_profiles import LocalServer, dump
from tools.ninfer_serve.client import NInferServeClient, ProtocolRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("ar","mtp","dflash2"), required=True)
    parser.add_argument("--original", action="store_true")
    parser.add_argument("--original-causal-prefill", action="store_true")
    parser.add_argument("--original-content-anchors", action="store_true")
    parser.add_argument("--yarn-factor", type=int, choices=(1,2,4))
    parser.add_argument("--draft-tokens", type=int, default=7)
    parser.add_argument("--prefill-chunk", type=int, choices=(512,1024), default=512)
    parser.add_argument("--large-prefill", type=int, choices=(0,1), default=0)
    parser.add_argument("--silu-mmq", type=int, choices=(0,1), default=0)
    parser.add_argument("--request-index", type=int,
                        help="Run one unchanged frozen request in a fresh process; diagnostic only")
    parser.add_argument("--request-count", type=int, choices=range(1,7), default=6,
                        help="Run only the first N unchanged requests; subsets are diagnostics")
    parser.add_argument("--layer-ready", type=int, choices=(0,1), default=0)
    parser.add_argument("--canonical-query-reuse", type=int, choices=(0,1), default=0)
    parser.add_argument("--transfer-mode", choices=("sync","async"), default="sync")
    parser.add_argument("--fixed-io-workers", type=int, choices=(0,1), default=0)
    parser.add_argument("--cold-copy-pipeline", type=int, choices=(0,1), default=0)
    parser.add_argument("--host-pages", type=int,
                        help="Bound the KVMem Host tier to exercise NVMe-backed Cold reads")
    parser.add_argument("--case", choices=("213k","256k"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--passed-213k", type=Path)
    args = parser.parse_args()
    if args.yarn_factor is not None and not args.original:
        parser.error("fixed YaRN diagnostic requires --original")
    if args.original and args.layer_ready:
        parser.error("layer-ready applies to canonical ordinary prefill only")
    if args.original_causal_prefill and not args.original:
        parser.error("causal history prefill requires --original")
    if args.original_content_anchors and not args.original_causal_prefill:
        parser.error("content anchors require --original-causal-prefill")
    binary = ROOT/"build/apps/ninfer-serve"
    if args.case == "256k":
        assert args.passed_213k
        prior = json.loads((args.passed_213k/"report.json").read_text())
        assert prior["state"] == "PASS" and prior["backend"] == args.backend
        assert prior["binary"] == signature(binary)
        assert prior["original_positions"] == args.original
        assert prior.get("original_causal_prefill",False) == args.original_causal_prefill
        assert prior.get("original_content_anchors",False) == args.original_content_anchors
        assert prior["yarn_factor"] == args.yarn_factor
        assert prior.get("draft_tokens",7) == args.draft_tokens
        assert prior.get("prefill_chunk",512) == args.prefill_chunk
        assert prior.get("large_prefill",0) == args.large_prefill
        assert prior.get("silu_mmq",0) == args.silu_mmq
        assert prior.get("layer_ready",0) == args.layer_ready
        assert prior.get("canonical_query_reuse",args.backend=="ar") == bool(args.canonical_query_reuse or args.backend=="ar")
        assert prior.get("transfer_mode","sync") == args.transfer_mode
        assert prior.get("host_pages") == args.host_pages
        assert prior.get("fixed_io_workers",0) == args.fixed_io_workers
        assert prior.get("cold_copy_pipeline",0) == args.cold_copy_pipeline
    power = subprocess.check_output(
        ["nvidia-smi","--query-gpu=power.limit","--format=csv,noheader,nounits"],text=True)
    assert len(power.splitlines()) == 1 and float(power.strip()) == 200, power
    source = json.loads((WORK/"benchmarks/frozen-long-context"/args.case/"manifest.json").read_text())
    if args.request_index is not None and not 0 <= args.request_index < len(source["requests"]):
        parser.error("request index outside frozen request list")
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/"backing").mkdir()
    model = Path(source["model"]["path"])
    assert signature(model) == source["model"]
    for name in list(os.environ):
        if name.startswith(("NINFER_BENCH_","NINFER_TEST_")):
            del os.environ[name]
    os.environ.update(PROFILE, NINFER_SM89_SHARED_VECTOR="1" if args.backend=="ar" else "0",
                      NINFER_SM89_SHARED_VECTOR_TRACE="1" if args.backend=="ar" else "0",
                      NINFER_KVMEM_CANONICAL_QUERY_REUSE=str(int(args.canonical_query_reuse or args.backend=="ar")),
                      NINFER_KVMEM_COLD_DIR=str(out/"backing"))
    os.environ.update(NINFER_KVMEM_LAYER_READY=str(args.layer_ready),
                      NINFER_KVMEM_ORIGINAL_CAUSAL_PREFILL=str(int(args.original_causal_prefill)),
                      NINFER_KVMEM_ORIGINAL_CONTENT_ANCHORS=str(int(args.original_content_anchors)),
                      NINFER_KVMEM_FIXED_IO_WORKERS=str(args.fixed_io_workers),
                      NINFER_KVMEM_COLD_COPY_PIPELINE=str(args.cold_copy_pipeline),
                      NINFER_KVMEM_IO_WORKER_SUMMARY="1",
                      NINFER_GGUF_LARGE_PREFILL=str(args.large_prefill),
                      NINFER_GGUF_SILU_MMQ=str(args.silu_mmq),
                      NINFER_KVMEM_LAYER_READY_SUMMARY="1",
                      NINFER_KVMEM_LAYER_READY_TRACE="0")
    if args.transfer_mode=="async":
        os.environ.update(NINFER_KVMEM_TRANSFER_PIPELINE="1",
                          NINFER_KVMEM_SOURCE_EVENT_PIPELINE="1")
    if args.host_pages is not None:
        assert args.host_pages > 0
        os.environ["NINFER_KVMEM_HOST_PAGES"]=str(args.host_pages)
    if args.original:
        os.environ.update(NINFER_KVMEM_ORIGINAL_POSITIONS="1", NINFER_R29_BUNDLE="0")
    if args.yarn_factor is not None:
        os.environ["NINFER_KVMEM_ORIGINAL_YARN_FACTOR"]=str(args.yarn_factor)
    command = [str(binary),str(model),"--model-id","qualification","--host","127.0.0.1",
        "--port","18118","--max-context",str(source["context_limit"]),"--kv-device-tokens","36864",
        "--prefill-chunk",str(args.prefill_chunk),"--host-kv-mib","0","--max-concurrency","1","--kv-dtype","int8",
        "--greedy","--no-thinking","--kvmem","--kvmem-budget","32768",
        "--kvmem-gen-reserve","512","--kvmem-action-lrfu","--kvmem-lrfu-lambda","0.5",
        "--kvmem-lrfu-top-frac","0.3","--kvmem-transfer-mode",args.transfer_mode,
        "--request-log-jsonl",str(out/"requests.jsonl")]
    if args.backend=="dflash2":
        command += ["--spec","dflash2","--draft-tokens","3","--lm-head-draft"]
    elif args.backend=="mtp":
        command += ["--spec","mtp","--draft-tokens",str(args.draft_tokens),"--lm-head-draft"]
    report = dict(state="RUNNING",backend=args.backend,case=args.case,model=signature(model),
                  binary=signature(binary),power_w=200,command=command,requests=[],
                  input_manifest=str(WORK/"benchmarks/frozen-long-context"/args.case/"manifest.json"),
                  original_positions=args.original,yarn_factor=args.yarn_factor,
                  original_causal_prefill=args.original_causal_prefill,
                  original_content_anchors=args.original_content_anchors,
                  draft_tokens=args.draft_tokens,diagnostic_request_index=args.request_index,
                  prefill_chunk=args.prefill_chunk,large_prefill=args.large_prefill,
                  silu_mmq=args.silu_mmq,
                  diagnostic_request_count=args.request_count,
                  layer_ready=args.layer_ready,
                  canonical_query_reuse=bool(args.canonical_query_reuse or args.backend=="ar"),
                  transfer_mode=args.transfer_mode,
                  host_pages=args.host_pages,
                  fixed_io_workers=args.fixed_io_workers,cold_copy_pipeline=args.cold_copy_pipeline,
                  snapshot="not loaded: real cold prefill",environment={
                      k:v for k,v in os.environ.items() if k.startswith("NINFER_")})
    dump(out/"report.json",report)
    began=time.monotonic()
    try:
        with LocalServer(command,"127.0.0.1",18118,out/"requests.jsonl") as server:
            start=server.wait_until_ready()
            session_began=time.monotonic()
            def process_io():
                return {key:int(value) for key,value in
                        (line.split(":",1) for line in
                         Path(f"/proc/{server.process.pid}/io").read_text().splitlines())}
            io_before=process_io()
            report["startup"]=start
            free=start["memory"]["available_after_startup_bytes"]
            if free<256*1024**2:
                report.update(state="CAPACITY_REJECTED",free_bytes=free)
                print(f"{args.backend} {args.case} CAPACITY_REJECTED free={free}",flush=True)
            else:
                client=NInferServeClient("http://127.0.0.1:18118",600)
                pending_receipts=[]
                last_response_end=session_began
                for i,entry in enumerate(source["requests"][:args.request_count]):
                    if args.request_index is not None and i != args.request_index:
                        continue
                    raw=Path(entry["file"]).read_bytes()
                    assert hashlib.sha256(raw).hexdigest()==entry["sha256"]
                    exchange=client.prepare(ProtocolRequest("openai_chat","/v1/chat/completions",
                                                            json.loads(raw))).execute()
                    last_response_end=time.monotonic()
                    content="".join(e.output for e in exchange.events if e.kind=="model_output")
                    terminal=any(e.kind=="terminal" for e in exchange.events)
                    usage=next((e.payload["usage"] for e in reversed(exchange.events)
                                if e.payload and e.payload.get("usage")),None)
                    row=dict(file=entry["file"],sha256=entry["sha256"],expected=entry["expected"],
                             actual=content,correct=quality.correct(content,entry),
                             terminal=terminal,usage=usage,http_status=exchange.http.status)
                    first=next((e.received_ns for e in exchange.events
                                if e.kind=="model_output" and e.output),None)
                    row["ttft_s"]=(first-exchange.http.sent_ns)/1e9 if first else None
                    row["wall_s"]=(exchange.http.ended_ns-exchange.http.sent_ns)/1e9
                    report["requests"].append(row)
                    receipt=dataclasses.asdict(exchange)
                    receipt["http"].pop("body")
                    pending_receipts.append((i,exchange.http.body,receipt))
                    print(f"{args.backend} {args.case} {i+1}/6 {content!r} correct={row['correct']}",flush=True)
                    if exchange.http.status!=200 or exchange.error_code or exchange.http.error or exchange.protocol_error or not terminal or not usage or not row["correct"]:
                        report["state"]="QUALITY_FAILED"
                        break
                else:
                    import urllib.request
                    with urllib.request.urlopen("http://127.0.0.1:18118/metrics",timeout=5) as response:
                        metrics=response.read().decode()
                    (out/"metrics.txt").write_text(metrics)
                    report["metrics"]=quality.metric_values(metrics)
                    assert all(v==0 for k,v in report["metrics"].items()
                               if "kvmem_inflight_pages" in k or "kvmem_inflight_bytes" in k)
                    report["state"]="PASS" if args.request_index is None and args.request_count==6 else "DIAGNOSTIC_PASS"
                report["session_wall_s"]=last_response_end-session_began
                report["request_wall_sum_s"]=sum(row["wall_s"] for row in report["requests"])
                report["session_client_gap_s"]=report["session_wall_s"]-report["request_wall_sum_s"]
            report["process_io_delta"]={k:v-io_before[k] for k,v in process_io().items()}
            # Outside all request/session timing; VmHWM is the process peak,
            # not a sample that could miss a transient allocation.
            status=Path(f"/proc/{server.process.pid}/status").read_text()
            report["process_memory_bytes"]={key:int(value)*1024 for key,value in
                re.findall(r"^(VmRSS|VmHWM):\s+(\d+)\s+kB$",status,re.MULTILINE)}
            if free>=256*1024**2:
                # Receipt disk IO is outside the session timing and outside
                # the measured process-IO interval. Six short responses remain
                # bounded in RAM; no duplicate prompt/KV copy is retained.
                for i,body,receipt in pending_receipts:
                    (out/f"request-{i}.sse").write_bytes(body)
                    dump(out/f"request-{i}.receipt.json",receipt)
        report["server_exit"]=server.process.returncode
        summaries=[]
        io_workers=[]
        for line in (out/"requests.console.log").read_text().splitlines():
            if "KVMEM_LAYER_SUMMARY " in line:
                summaries.append({key:int(value) for key,value in
                                  re.findall(r"(\w+)=(\d+)",line)})
            if "KVMEM_IO_WORKERS " in line:
                io_workers.append({key:int(value) for key,value in
                                   re.findall(r"(\w+)=(\d+)",line)})
        report["layer_summaries"]=summaries
        report["io_worker_summaries"]=io_workers
        if args.layer_ready and report["state"] == "PASS":
            if not any(s["epochs"] and s["waits"] and s["releases"] for s in summaries):
                report["state"]="PATH_NOT_HIT"
            elif any(s["pending_pinned_pages"] or s["inflight_pages"] or
                     s["producer_active"] for s in summaries):
                report["state"]="DRAIN_FAILED"
        if args.cold_copy_pipeline and report["state"] == "PASS":
            if not any(s.get("cold_copy_pages",0)>2 and s.get("cold_copy_batches",0)
                       for s in summaries):
                report["state"]="COLD_COPY_PATH_NOT_HIT"
        if args.fixed_io_workers and report["state"] == "PASS":
            if not io_workers or not all(s["reads"]>0 and s["pending"]==0 for s in io_workers):
                report["state"]="IO_WORKER_PATH_NOT_HIT_OR_DRAIN_FAILED"
        if report["server_exit"]!=0:
            report["state"]="SERVER_EXIT_FAILED"
    except Exception as error:
        log=out/"requests.console.log"
        text=log.read_text() if log.exists() else ""
        capacity=re.search(r"requested Engine runtime reservation requires (\d+) bytes, "
                           r"but only (\d+) bytes are available for runtime capacity",text)
        if capacity and not report["requests"]:
            report.update(state="CAPACITY_REJECTED",startup_exception=repr(error),
                runtime_required_bytes=int(capacity[1]),runtime_available_bytes=int(capacity[2]),
                server_exit=server.process.returncode,
                quality="NOT_RUN: rejected before weights/prefill; dependent 256K not run")
        else:
            report.update(state="ERROR",exception=repr(error))
            raise
    finally:
        report["wall_s"]=time.monotonic()-began
        dump(out/"report.json",report)
    print(json.dumps({k:v for k,v in report.items()
                      if k not in ("requests","startup","environment","metrics")}),flush=True)
    return 0 if report["state"] in ("PASS","DIAGNOSTIC_PASS","CAPACITY_REJECTED") else 1


if __name__=="__main__":
    raise SystemExit(main())
