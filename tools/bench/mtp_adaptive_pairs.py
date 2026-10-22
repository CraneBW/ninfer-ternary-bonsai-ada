"""OpenSpec 6.4: 200W C1 fixed-MTP4 vs adaptive-MTP4, 4K/32K, three same-binary pairs.

Launch only after the S3 worker releases sole ownership of the GPU and Serve.
This script never signals a process it did not launch and never escalates to SIGKILL.
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import signal
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "tmp-nvcc/opsx-s5-20261022"
SOURCE = ROOT / "tmp-nvcc/qualification-20261020/long-resident/32768-0.json"
MODEL = "/home/ran/ninfer-work/real-models/qwen3_8_gsq_rco_iq3_s_v3.ninfer"
SERVE = ROOT / "build/apps/ninfer-serve"
PORT = 18159
URL = f"http://127.0.0.1:{PORT}"


def make_request(profile: str) -> dict:
    source = json.loads(SOURCE.read_text())
    if profile in ("4k", "4k128"):
        notes = source["messages"][1]["content"].splitlines()
        source["messages"][1]["content"] = "\n".join(notes[:263])
    source["messages"][-1]["content"] = (
        "Write the next 200 numbered checklist lines with this exact format, one line per item: "
        "check 001: verified; check 002: verified; and so on through check 200. "
        "Do not stop until check 200. No introductory or final sentence."
        if profile == "4k128" else
        "Write a numbered list of small deterministic software testing checks. "
        "Use one short sentence per item and continue until stopped."
    )
    source["max_tokens"] = 128
    source["temperature"] = 0
    source["stream"] = False
    source.pop("stream_options", None)
    source["model"] = "qualification"
    return source


def wait_ready(process: subprocess.Popen) -> bool:
    deadline = time.monotonic() + 70
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(URL + "/v1/models", timeout=0.5) as response:
                if response.status == 200:
                    return True
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.2)
    return False


def serve_one(case: str, arm: str, pair: int) -> dict:
    payload = make_request(case)
    name = f"{case}-{pair}-{arm}"
    logpath = OUT / (name + ".log")
    jsonl = OUT / (name + ".jsonl")
    command = [
        str(SERVE), MODEL, "--model-id", "qualification", "--host", "127.0.0.1",
        "--port", str(PORT), "--max-context", "36864", "--kv-device-tokens", "36864",
        "--prefill-chunk", "512", "--host-kv-mib", "256", "--max-concurrency", "1",
        "--kv-dtype", "int8", "--greedy", "--no-thinking", "--kvmem",
        "--kvmem-budget", "32768", "--kvmem-gen-reserve", "512",
        "--kvmem-transfer-mode", "sync",
        "--spec", "mtp", "--draft-tokens", "4", "--lm-head-draft",
        "--default-max-tokens", "128", "--request-log-jsonl", str(jsonl),
    ]
    if arm == "adaptive":
        command.append("--adaptive-mtp")
    result = {
        "case": case, "arm": arm, "pair": pair, "command": command,
        "source": str(SOURCE), "max_tokens": 128,
        "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "status": None,
    }
    begin = time.monotonic()
    env = dict(os.environ, NINFER_KVMEM_LAYER_READY="0",
               NINFER_KVMEM_ORIGINAL_POSITIONS="0",
               NINFER_KVMEM_CANONICAL_QUERY_REUSE="0")
    with logpath.open("wb") as log:
        proc = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            ready = wait_ready(proc)
            result["startup_seconds"] = round(time.monotonic() - begin, 4)
            if not ready:
                raise RuntimeError(f"Serve not ready, exit={proc.poll()}")
            request = urllib.request.Request(
                URL + "/v1/chat/completions", data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            started = time.monotonic()
            with urllib.request.urlopen(request, timeout=180) as response:
                result["status"] = response.status
                data = json.loads(response.read())
                result["usage"] = data.get("usage")
                choice = data.get("choices", [{}])[0]
                text = choice.get("message", {}).get("content") or ""
                result["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
                result["finish_reason"] = choice.get("finish_reason")
            result["client_request_wall_s"] = round(time.monotonic() - started, 4)
        except Exception as error:
            result["error"] = repr(error)
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=25)
                result["serve_exit_code"] = proc.returncode
            except subprocess.TimeoutExpired:
                result["serve_exit_code"] = None
                result["own_serve_still_running"] = proc.pid
                # Caller must stop; leave the live owner untouched.
    result["total_wall_s"] = round(time.monotonic() - begin, 4)
    logfile = logpath.read_text(errors="replace")
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "request_done":
                for field in ("timings_seconds", "engine_timing", "speculative",
                              "adaptive_speculative", "result"):
                    if field in event:
                        result[field] = event[field]
    finished = [line for line in logfile.splitlines() if " req#" in line and " done " in line]
    result["request_log"] = finished[-1:] if finished else []
    if finished:
        line = finished[-1]
        for field, regex in (
            ("server_wall_s", r"total ([0-9.]+)s"),
            ("ttft_s", r"TTFT ([0-9.]+)s"),
            ("prefill_toks", r"prefill ([0-9.]+) tok/s"),
            ("decode_toks", r"decode ([0-9.]+) tok/s"),
        ):
            m = re.search(regex, line)
            if m:
                result[field] = float(m.group(1))
        m = re.search(r"mtp accepted (\d+)/(\d+)", line)
        if m:
            result["accepted"] = int(m.group(1))
            result["proposed"] = int(m.group(2))
    return result


def summarize(results: dict) -> None:
    result_pairs = []
    for pair in range(1, 4):
        runs = [row for row in results["arm_results"] if row["pair"] == pair]
        fixed = next(row for row in runs if row["arm"] == "fixed")
        adaptive = next(row for row in runs if row["arm"] == "adaptive")
        same_output = (fixed.get("text_sha256") == adaptive.get("text_sha256")
                       and fixed.get("usage") == adaptive.get("usage"))
        if any(key not in run for run in (fixed, adaptive)
               for key in ("engine_timing", "speculative", "timings_seconds")):
            raise RuntimeError("request_done JSONL record missing; cannot verify host/device")
        fields = {
            "request_wall": lambda r: r["client_request_wall_s"],
            "ttft": lambda r: r["timings_seconds"]["ttft"],
            "session_total": lambda r: r["timings_seconds"]["total"],
            "prefill": lambda r: r["timings_seconds"]["prefill"],
            "decode": lambda r: r["timings_seconds"]["decode"],
            "host_exposed": lambda r: r["engine_timing"]["host_exposed_seconds"]["total"],
            "device_wait_exposed": lambda r: r["engine_timing"]["device_wait_exposed_seconds"],
            "decode_host_exposed": lambda r: r["engine_timing"]["decode"]["host_exposed_seconds"],
            "decode_device_wait_exposed": lambda r: r["engine_timing"]["decode"]["device_wait_exposed_seconds"],
        }
        raw = {field: {"fixed": getter(fixed), "adaptive": getter(adaptive)}
               for field, getter in fields.items()}
        reductions = {field: (1 - values["adaptive"] / values["fixed"])
                      if values["fixed"] > 0 else None
                      for field, values in raw.items()}
        result_pairs.append({"pair": pair, "same_response_text_and_usage": same_output,
                             "raw_seconds": raw, "adaptive_time_reduction": reductions,
                             "fixed_speculative": fixed["speculative"],
                             "adaptive_speculative": adaptive["speculative"],
                             "adaptive_windows": adaptive.get("adaptive_speculative")})
    results["pairs"] = result_pairs
    results["summary_median_time_reduction"] = {
        field: statistics.median(row["adaptive_time_reduction"][field] for row in result_pairs
                                 if row["adaptive_time_reduction"][field] is not None)
        for field in result_pairs[0]["adaptive_time_reduction"]
        if any(row["adaptive_time_reduction"][field] is not None for row in result_pairs)
    }
    results["all_3_pairs_same_response"] = all(p["same_response_text_and_usage"] for p in result_pairs)
    results["state"] = ("THREE_PAIRS_SAME_TEXT_AND_USAGE"
                        if results["all_3_pairs_same_response"] else "THREE_PAIRS_OUTPUT_DIFFERENCE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("4k", "4k128", "32k"), required=True)
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()
    if args.pairs != 3:
        raise ValueError("OpenSpec 6.4 requires three pairs per context")
    OUT.mkdir(parents=True, exist_ok=True)
    power = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if len(power.splitlines()) != 1 or abs(float(power) - 200.0) > 0.1:
        raise RuntimeError(f"single GPU at 200W required, observed {power}")
    port = socket.socket()
    try:
        if port.connect_ex(("127.0.0.1", PORT)) == 0:
            raise RuntimeError(f"port {PORT} is occupied; refusing to affect external Serve")
    finally:
        port.close()
    case_results = {"case": args.case, "model": MODEL,
                    "binary_sha256": hashlib.sha256(SERVE.read_bytes()).hexdigest(),
                    "state": "RUNNING",
                    "expected_watt_cap": 200, "pairs": [], "arm_results": []}
    output_file = OUT / f"results-{args.case}.json"
    for pair in range(args.pairs):
        for arm in (("fixed", "adaptive") if pair % 2 == 0 else ("adaptive", "fixed")):
            result = serve_one(args.case, arm, pair + 1)
            case_results["arm_results"].append(result)
            output_file.write_text(json.dumps(case_results, indent=2, ensure_ascii=False) + "\n")
            print(args.case, pair + 1, arm, result.get("status"),
                  result.get("usage"), result.get("serve_exit_code"), flush=True)
            if result.get("status") != 200 or result.get("serve_exit_code") != 0:
                raise RuntimeError(f"{arm} did not complete normally; see {output_file}")
    summarize(case_results)
    output_file.write_text(json.dumps(case_results, indent=2, ensure_ascii=False) + "\n")
    print(args.case, case_results["state"],
          case_results["summary_median_time_reduction"], flush=True)


if __name__ == "__main__":
    main()
