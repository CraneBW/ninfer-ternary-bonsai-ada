"""Bounded local GSQ profile pairs; shared Engine, corpus logging and SSE receipts.

Daily 200W only. No model downloads, global cache eviction, or process KILL.
A wave rate includes prefill/queue; request TBT is the mean per decoded token,
not a token-level tail estimate or an SSE-frame interval.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import http.client
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.bench import run_serve_corpus as corpus
from tools.bench.prefix_checkpoint import PROFILE

PROFILES = {"ar": ("none", 0), "mtp4": ("mtp", 4),
            "mtp15": ("mtp", 15), "dflash2": ("dflash2", 3)}
TASKS = [
    ("conversation", "CONVERSATION:",
     "Begin exactly with CONVERSATION: then explain why unit tests and integration tests serve "
     "different purposes. Give many concrete software examples; continue until stopped."),
    ("tool-code", "TOOL-CODE:14",
     "A Python tool ran def f(x): return x*2 and printed f(7), giving 14. "
     "Begin exactly with TOOL-CODE:14 then give a detailed code review, edge cases and "
     "deterministic tests. Continue listing checks until stopped."),
    ("high-acceptance", "RECORDS:",
     "Begin exactly with RECORDS: then continue numbered records, one per line, without a "
     "conclusion, until stopped. Format: item 001: verified; item 002: verified; item 003: verified."),
]


class LocalServer(corpus.RunningServer):
    """Use existing log-tail/readiness logic; only own PID, graceful stop."""
    def __enter__(self):
        self.console = self.log_path.with_suffix(".console.log").open("wb")
        self.process = subprocess.Popen(self.command, cwd=ROOT, stdout=self.console,
                                        stderr=subprocess.STDOUT, start_new_session=True)
        self.tail = corpus.ServerLogTail(self.log_path, self.process, 0)
        return self

    def stop(self):
        try:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=40)
                except subprocess.TimeoutExpired as error:
                    raise corpus.CampaignError(
                        f"Serve did not stop after TERM; preserve PID {self.process.pid}"
                    ) from error
        finally:
            self.console.close()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    at = (len(ordered) - 1) * q
    low = int(at)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] * (high - at) + ordered[high] * (at - low) if high != low else ordered[low]


def fixture(index, lines, tokens, unique=False):
    name, expected, question = TASKS[index % len(TASKS)]
    if unique:
        unique_prefix = f"CASE-{index:03d}:"
        # Replace the task's existing first-prefix instruction; prepending a
        # second incompatible "begin exactly" instruction is ambiguous.
        question = question.replace(expected, unique_prefix)
        expected = unique_prefix
    messages = [{"role": "system", "content": "Follow the requested output format exactly."}]
    if lines:
        document = "\n".join(
            f"record {i:04d}: baseline component observed; independent verification complete."
            for i in range(lines))
        messages.append({"role": "user", "content": "Background log for this task:\n" + document})
        messages.append({"role": "assistant", "content": "I have read the background log."})
    messages.append({"role": "user", "content": question})
    return corpus.Fixture(f"{name}-{index}", messages, False, tokens, "local-profile", name), expected


def stream_request(port, payload, barrier=None):
    if barrier:
        barrier.wait()
    begin = time.perf_counter()
    first = None
    frames, content, finish, usage, identifiers = [], "", None, None, set()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=180)
    try:
        corpus.send_json(connection, dict(payload, stream=True,
                                          stream_options={"include_usage": True}))
        response = connection.getresponse()
        if response.status != 200:
            raise corpus.CampaignError(f"HTTP {response.status}: {response.read()[:800]!r}")
        done = False
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            text = line[5:].strip()
            if text == "[DONE]":
                done = True
                continue
            frame = json.loads(text)
            frames.append(frame)
            if frame.get("id"):
                identifiers.add(frame["id"])
            if frame.get("usage") is not None:
                usage = frame["usage"]
            for choice in frame.get("choices", []):
                delta = choice.get("delta", {}).get("content") or ""
                if delta and first is None:
                    first = time.perf_counter()
                content += delta
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        end = time.perf_counter()
        if not done or usage is None or len(identifiers) != 1 or not finish:
            raise corpus.CampaignError("incomplete SSE terminal, identity or usage")
        return {
            "response": {"id": next(iter(identifiers)), "usage": usage,
                         "choices": [{"message": {"content": content}, "finish_reason": finish}]},
            "frames": frames, "started": begin, "finished": end,
            "client_wall_s": end - begin, "client_ttft_s": None if first is None else first - begin,
        }
    finally:
        connection.close()


def run_arm(args, out, label, profile, concurrency, policy, jobs, pair):
    backend, width = PROFILES[profile]
    log = out / f"{label}.jsonl"
    command = [str(args.serve), str(args.artifact),
               "--model-id", "qualification", "--host", "127.0.0.1", "--port", str(args.port),
               "--max-context", str(args.max_context), "--kv-capacity", str(args.kv_tokens),
               "--prefill-chunk", str(args.prefill_chunk),
               "--max-concurrency", str(concurrency), "--max-pending-requests", "16",
               "--pending-timeout-ms", "120000", "--kv-dtype", "int8",
               "--greedy", "--no-thinking", "--presence-penalty", "0",
               "--no-prefix-reuse", "--request-log-jsonl", str(log),
               "--log-stats-interval-ms", "1000"]
    if backend != "none":
        command += ["--spec", backend, "--draft-tokens", str(width), "--lm-head-draft"]
    os.environ.update(PROFILE)
    os.environ.update(NINFER_KVMEM_ORIGINAL_POSITIONS="0", NINFER_KVMEM_LAYER_READY="0",
                      NINFER_SM89_SHARED_VECTOR=str(policy),
                      NINFER_SM89_SHARED_VECTOR_TRACE=str(policy))
    row = dict(label=label, profile=profile, concurrency=concurrency, policy=policy,
               pair=pair, command=command, environment=dict(PROFILE, NINFER_SM89_SHARED_VECTOR=str(policy)), records=[])
    with LocalServer(command, "127.0.0.1", args.port, log) as server:
        start = server.wait_until_ready()
        row["startup"] = start
        free = start["memory"]["available_after_startup_bytes"]
        if free < 256 * 1024**2:
            row.update(state="CAPACITY_REJECTED", free_bytes=free)
        else:
            # Same short warmup for each arm; no context/prefix reuse.
            warm, _ = fixture(0, 0, 8)
            connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=60)
            try:
                corpus.post_json(connection, corpus.request_payload("qualification", warm, 7))
            finally:
                connection.close()
            server.wait_for_request_done(start["server_instance_id"])
            # Concurrency comparisons share the same eight-request arrival
            # wave even at C1; otherwise its client TTFT would omit queueing.
            burst = args.phase == "concurrency"
            barrier = threading.Barrier(len(jobs)) if burst else None
            if not burst:
                results = [stream_request(args.port, corpus.request_payload("qualification", job, seed))
                           for job, expected, seed in jobs]
            else:
                with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
                    futures = [executor.submit(
                        stream_request, args.port,
                        corpus.request_payload("qualification", job, seed), barrier)
                        for job, expected, seed in jobs]
                    results = [future.result() for future in futures]
            events = {}
            for _ in jobs:
                event = server.wait_for_request_done(start["server_instance_id"])
                events[event["request"]["sampling"]["seed"]] = event
            for (job, expected, seed), client in zip(jobs, results):
                spec = corpus.RunSpec("qwen3_8_27b", "qualification", args.artifact, profile,
                                      backend, width, "greedy", job, seed)
                payload = corpus.request_payload("qualification", job, seed)
                result = corpus.build_result_record(spec, start["artifact"]["weights_id"],
                                                     payload, client["response"], events[seed])
                result.update(client=client, expected_prefix=expected,
                              quality_pass=client["response"]["choices"][0]["message"]["content"]
                              .lstrip().startswith(expected))
                row["records"].append(result)
            totals = [r["metrics"]["completion_tokens"] for r in row["records"]]
            wave = max(r["finished"] for r in results) - min(r["started"] for r in results)
            means_tbt = [1000 * r["metrics"]["decode_seconds"] /
                         max(r["metrics"]["decode_tokens"], 1) for r in row["records"]]
            row["metrics"] = {
                "request_count": len(jobs), "completion_tokens": sum(totals),
                "wave_wall_s": wave, "wave_output_tok_s": sum(totals) / wave,
                "prefill_tok_s_median": statistics.median(
                    r["metrics"]["prefill_tok_s"] for r in row["records"]),
                "decode_tok_s_median": statistics.median(
                    r["metrics"]["decode_tok_s"] for r in row["records"]),
                "ttft_client_p50_s": percentile([r["client_ttft_s"] for r in results], .5),
                "ttft_client_p95_s": percentile([r["client_ttft_s"] for r in results], .95),
                "mean_request_tbt_p50_ms": percentile(means_tbt, .5),
                "mean_request_tbt_p95_ms": percentile(means_tbt, .95),
                "queue_wait_p50_s": percentile(
                    [r["metrics"]["queue_wait_seconds"] for r in row["records"]], .5),
                "queue_wait_p95_s": percentile(
                    [r["metrics"]["queue_wait_seconds"] for r in row["records"]], .95),
            }
            row["state"] = "PASS" if all(r["quality_pass"] for r in row["records"]) else "QUALITY_FAILED"
        # Process accounting only; VmPin is the kernel counter, not a claim
        # that it measures all CUDA pinned allocations. Host KV/state capacity
        # remains independently reported in the startup receipt.
        status = Path(f"/proc/{server.process.pid}/status").read_text()
        row["host_process_kib"] = {
            line.split(":",1)[0]: int(line.split()[1]) for line in status.splitlines()
            if line.startswith(("VmRSS:", "VmHWM:", "VmLck:", "VmPin:"))}
    row["server_exit"] = server.process.returncode
    if row["server_exit"] != 0:
        raise corpus.CampaignError(f"Serve exited {row['server_exit']}")
    dump(out / f"{label}.json", row)
    print(json.dumps({k: row[k] for k in ("label", "state", "metrics") if k in row}), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("spec", "concurrency", "one"))
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--serve", type=Path, default=ROOT / "build/apps/ninfer-serve")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", default="ar,mtp4,mtp15,dflash2")
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--lines", type=int, default=280)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=36864)
    parser.add_argument("--kv-tokens", type=int, default=36864)
    parser.add_argument("--prefill-chunk", type=int, default=512)
    parser.add_argument("--port", type=int, default=18174)
    args = parser.parse_args()
    if args.pairs < 1 or args.pairs > 3:
        parser.error("--pairs must be 1..3")
    args.serve, args.artifact = args.serve.resolve(), args.artifact.resolve()
    profiles = args.profiles.split(",")
    if not profiles or any(p not in PROFILES for p in profiles):
        parser.error("unknown profile")
    levels = [int(x) for x in args.concurrency.split(",")] if args.phase == "concurrency" else [1]
    if any(c not in (1,2,4,8) for c in levels):
        parser.error("concurrency must be selected from 1,2,4,8")
    if args.kv_tokens > args.max_context * min(levels):
        parser.error("the fixed shared pool must fit the smallest selected concurrency; "
                     "kv-capacity cannot exceed max-context * C")
    power = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"], text=True)
    if len(power.splitlines()) != 1 or float(power.strip()) != 200:
        raise corpus.CampaignError(f"one 200W GPU required, got {power!r}")
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise corpus.CampaignError(f"occupied port {args.port}")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import hashlib
    report = {"phase": args.phase, "power_w": 200, "arms": [],
              "binary_sha256": hashlib.sha256(args.serve.read_bytes()).hexdigest(),
              "logical_context_per_request": args.max_context,
              "shared_physical_kv_tokens": args.kv_tokens,
              "pre_measurement_budget": {
                  "kv_bytes_per_token": 33792,
                  "physical_kv_bytes": args.kv_tokens * 33792,
                  "gdn_image_bytes": 153954304,
                  "state_images_per_lane_upper_bound": 2,
                  "device_state_upper_bound_by_c": {
                      str(c): 2*c*153954304 for c in (1,2,4,8)},
                  "admission": "shared pool, no uniform per-lane split; actual prompt/output claims; "
                               "startup runtime/graph/workspace and 256 MiB floor checked before requests",
                  "host": "no-prefix-reuse: host KV and cached Host StateImage slots zero"},
              "scope": "bounded local paired evidence; wave rates include prefill/queue; "
                       "TBT percentiles are across request means, not token-level tails"}
    jobs = []
    for i in range(8 if args.phase == "concurrency" else 3):
        job, expected = fixture(i, args.lines if args.phase != "concurrency" else
                                (16 + 8 * (i % 3)), args.tokens,
                                unique=args.phase == "concurrency")
        jobs.append((job, expected, 42 + i))
    if args.phase == "one":
        jobs = jobs[:1]
    report["frozen_jobs"] = [dict(expected_prefix=expected,seed=seed,
        request=corpus.request_payload("qualification",job,seed)) for job,expected,seed in jobs]
    dump(out / "summary.json", report)
    try:
        for c in levels:
            rejected = False
            for pair in range(1, args.pairs + 1):
                order = list(profiles) if pair % 2 else list(reversed(profiles))
                for profile in order:
                    policies = ([0, 1] if pair % 2 else [1, 0]) if args.phase == "concurrency" else [0]
                    for policy in policies:
                        label = f"pair{pair}-{profile}-c{c}-shared{policy}"
                        row = run_arm(args, out, label, profile, c, policy, jobs, pair)
                        report["arms"].append(row)
                        dump(out / "summary.json", report)
                        if row["state"] != "PASS":
                            rejected = True
                            break
                    if rejected:
                        break
                if rejected:
                    break
            if rejected:
                report["stopped_at_concurrency"] = c
                break
    except Exception as error:
        report["exception"] = repr(error)
        dump(out / "summary.json", report)
        raise
    report["state"] = "PASS" if all(r["state"] == "PASS" for r in report["arms"]) else "STOPPED"
    dump(out / "summary.json", report)


if __name__ == "__main__":
    main()
