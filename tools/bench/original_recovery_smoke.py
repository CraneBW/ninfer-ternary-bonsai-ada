"""Qualify one same-request canonical restart, then a different fresh request.

Reuses the frozen short real-model recovery prompt and Program's production
fault boundaries. No long quality suite, answer snapshot, or forced GPU reset.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault", choices=("after_main_rewind", "after_state_h2d"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18172)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError(f"occupied port: {args.port}")
    power = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"], text=True
    ).strip()
    if power != "200.00":
        raise RuntimeError(f"expected 200W; observed {power}")
    fixture = root / "tmp-nvcc/qualification-20261020/long-resident/32768-0.json"
    source = json.loads(fixture.read_text())
    source["messages"][1]["content"] = "\n".join(
        source["messages"][1]["content"].splitlines()[:550])
    source.update(max_tokens=8, temperature=0, stream=True, model="qualification",
                  stream_options={"include_usage": True})
    args_serve = [
        str(root / "build/apps/ninfer-serve"), str(args.artifact.resolve()),
        "--model-id", "qualification", "--host", "127.0.0.1", "--port", str(args.port),
        "--max-context", "12288", "--kv-device-tokens", "8192",
        "--prefill-chunk", "128", "--host-kv-mib", "256", "--max-concurrency", "1",
        "--kv-dtype", "int8", "--greedy", "--no-thinking", "--kvmem",
        "--kvmem-budget", "4096", "--kvmem-gen-reserve", "256",
        "--kvmem-transfer-mode", "sync", "--spec", "mtp", "--draft-tokens", "3",
        "--lm-head-draft", "--default-max-tokens", "8",
        "--request-log-jsonl", str(out / "requests.jsonl"),
    ]
    env = dict(os.environ, NINFER_KVMEM_ORIGINAL_POSITIONS="1",
               NINFER_TEST_ORIGINAL_RECOVERY_FAULT=args.fault, NINFER_KVMEM_LAYER_READY="0")
    report = {"stage": args.fault, "power_w": 200, "command": args_serve,
              "mode": "same-request-canonical-retry-then-different-fresh-request",
              "requests": []}
    started = time.monotonic()
    with (out / "serve.log").open("wb") as log:
        server = subprocess.Popen(args_serve, cwd=root, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        (out / "serve.pid").write_text(str(server.pid))
        try:
            for _ in range(300):
                if server.poll() is not None:
                    raise RuntimeError(f"Serve exited before ready: {server.returncode}")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.port}/v1/models", timeout=0.5
                    ) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(0.2)
            else:
                raise RuntimeError("Serve startup timeout")
            for answer in ("ACK", "BETA"):
                source["messages"][-1]["content"] = f"Reply exactly {answer}."
                (out / f"input-{answer}.json").write_text(json.dumps(source, ensure_ascii=False))
                request = urllib.request.Request(
                    f"http://127.0.0.1:{args.port}/v1/chat/completions",
                    data=json.dumps(source).encode(),
                    headers={"Content-Type": "application/json"})
                began = time.monotonic()
                result = {"expected": answer, "frames": [], "answer": "", "ids": [],
                          "done": False, "usage": None, "finish_reasons": []}
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        result["status"] = response.status
                        for line in response:
                            line = line.decode().strip()
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                result["done"] = True
                                continue
                            frame = json.loads(data)
                            result["frames"].append(frame)
                            if frame.get("id"):
                                result["ids"].append(frame["id"])
                            if frame.get("usage") is not None:
                                result["usage"] = frame["usage"]
                            for choice in frame.get("choices", []):
                                delta = choice.get("delta", {})
                                result["answer"] += delta.get("content") or ""
                                if choice.get("finish_reason"):
                                    result["finish_reasons"].append(choice["finish_reason"])
                except urllib.error.HTTPError as error:
                    result.update(status=error.code, error=error.read().decode())
                result["wall_s"] = time.monotonic() - began
                result["pass"] = (result.get("status") == 200 and result["answer"] == answer
                                  and result["done"] and result["usage"] is not None
                                  and len(set(result["ids"])) == 1
                                  and len(result["finish_reasons"]) == 1)
                report["requests"].append(result)
                if not result["pass"]:
                    break
        except Exception as error:
            report["exception"] = repr(error)
        finally:
            if server.poll() is None:
                server.terminate()
            try:
                server.wait(timeout=40)
            except subprocess.TimeoutExpired:
                report["shutdown_timeout_pid"] = server.pid
                # Preserve a wedged process for diagnosis; never escalate to KILL.
    report["server_exit"] = server.poll()
    report["wall_s"] = time.monotonic() - started
    log = (out / "serve.log").read_text(errors="replace")
    report["fault_markers"] = re.findall(r"KVMEM_ORIGINAL_TEST_FAULT stage=\S+", log)
    report["restart_markers"] = re.findall(
        r"KVMEM_ORIGINAL canonical_restart retired_all=1 graphs_rebuilt=\d+ prompt_tokens=\d+", log)
    report["pass"] = (len(report["requests"]) == 2
                      and all(row["pass"] for row in report["requests"])
                      and len(report["fault_markers"]) == len(report["restart_markers"]) == 1
                      and report["server_exit"] == 0 and "exception" not in report)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "requests"}, ensure_ascii=False))
    print(json.dumps([{k: v for k, v in row.items() if k not in ("frames", "ids")}
                      for row in report["requests"]], ensure_ascii=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
