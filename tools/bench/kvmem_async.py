"""Frozen, bounded sync/async session screening using the existing Serve wire client.

freeze creates inputs and the 18-session protocol before inference. run refuses
changed inputs/binary, preserves failures, and never retries or adds samples.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request

from tools.ninfer_serve.client import NInferServeClient, ProtocolRequest
from tools.ninfer_serve.openai_chat import chat_request


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def signature(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def freeze(folder: Path, model: Path, binary: Path) -> None:
    folder.mkdir(parents=True, exist_ok=False)
    alias = "qwen38-kvmem-async-screen"
    args = [str(binary.resolve()), str(model.resolve()), "--model-id", alias,
            "--host", "127.0.0.1", "--port", "18117", "--managed-weights",
            "--max-context", "8192", "--kv-device-tokens", "2048", "--kvmem",
            "--kvmem-budget", "1024", "--kvmem-gen-reserve", "256",
            "--kvmem-action-lrfu", "--kvmem-lrfu-lambda", "0.5",
            "--kvmem-lrfu-top-frac", "0.3", "--prefill-chunk", "256",
            "--host-kv-mib", "256", "--max-concurrency", "1",
            "--default-max-tokens", "16", "--kv-dtype", "int8", "--greedy",
            "--no-thinking", "--no-cuda-graph", "--log-level", "info"]
    system = {"role": "system", "content": "Follow the user instruction. Be concise."}

    def archive(count: int, facts: dict[int, str] | None = None) -> str:
        facts = facts or {}
        return "\n".join(facts.get(i, f"Note {i}: the record is unchanged and archived.")
                         for i in range(count))

    scenes: dict[str, list[tuple[list[dict], str]]] = {}
    scenes["shallow"] = [([system, {"role": "user", "content": "Reply exactly SHALLOW-OK."}], "SHALLOW-OK")]
    scenes["first_overpool"] = [([system, {"role": "user", "content":
        archive(200) + "\nReply exactly ACK."}], "ACK")]
    labels = [("Quartz Parcel", "cobalt-orchid-73"), ("Topaz Ledger", "silver-maple-41"),
              ("Indigo Archive", "amber-willow-92"), ("Jade Cabinet", "violet-birch-68"),
              ("Saffron Register", "copper-lotus-35"), ("Violet Dossier", "azure-pine-57")]
    facts = {40 + 48 * i: f"The {subject} display label is {value}."
             for i, (subject, value) in enumerate(labels)}
    history = [system, {"role": "user", "content": archive(360, facts) + "\nReply exactly ACK."}]
    deep = [(list(history), "ACK")]
    history.append({"role": "assistant", "content": "ACK"})
    for subject, value in labels:
        history.append({"role": "user", "content":
            f"What is the exact {subject} display label? Reply only with the label."})
        deep.append((list(history), value))
        # Fixed replay histories keep every request byte-identical between arms.
        history.append({"role": "assistant", "content": value})
    scenes["deep"] = deep
    records = {}
    for scene, requests in scenes.items():
        records[scene] = []
        for index, (messages, expected) in enumerate(requests):
            name = f"{scene}-{index}.request.json"
            payload = chat_request(alias, messages, 16).payload
            write_json(folder / name, payload)
            records[scene].append({"file": name, "expected": expected,
                "sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest()})
    schedule = [{"scene": scene, "pair": pair, "mode": mode}
                for scene in scenes for pair in range(3)
                for mode in (["sync", "async"] if pair % 2 == 0 else ["async", "sync"])]
    write_json(folder / "manifest.json", {"model": signature(model), "binary": signature(binary),
        "args": args, "scenes": records, "schedule": schedule, "max_sessions": 18,
        "request_timeout_s": 90, "startup_timeout_s": 60, "session_timeout_s": 240,
        "cache_initial_state": "fresh process; no inference warmup; readiness GET only",
        "wall_scope": "ready server, all replay requests; excludes model startup and receipt writes",
        "gates": {"deep_all_pairs_improve": True, "deep_median_reduction": 0.10,
                  "shallow_max_median_regression": 0.03},
        "quality_gate": "every frozen expected answer correct; complete terminal; equal sync/async usage",
        "scope": "Qwen3.8 GSQ, base/no-Graph, one active text request; fixed assistant replay"})
    print(f"Frozen {len(schedule)} sessions in {folder}", flush=True)


def metrics() -> str:
    with urllib.request.urlopen("http://127.0.0.1:18117/metrics", timeout=5) as response:
        return response.read().decode()


def run(folder: Path) -> None:
    manifest = json.loads((folder / "manifest.json").read_text())
    for key in ("model", "binary"):
        assert signature(Path(manifest[key]["path"])) == manifest[key], f"{key} changed after freeze"
    for requests in manifest["scenes"].values():
        for entry in requests:
            assert hashlib.sha256((folder / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]
    result_path = folder / "results.json"
    if result_path.exists():
        raise RuntimeError("results already exist; no automatic retry or campaign enlargement")
    results = []
    write_json(result_path, results)
    for position, item in enumerate(manifest["schedule"]):
        name = f"{position:02d}-{item['scene']}-p{item['pair']}-{item['mode']}"
        args = manifest["args"] + ["--kvmem-transfer-mode", item["mode"]]
        write_json(folder / f"{name}.command.json", args)
        record = {**item, "name": name, "requests": [], "status": "STARTING"}
        results.append(record)
        write_json(result_path, results)
        exchanges = []
        with (folder / f"{name}.log").open("w") as log:
            environment = os.environ.copy()
            environment.pop("NINFER_KVMEM_TRACE", None)
            proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=environment)
            try:
                client = NInferServeClient("http://127.0.0.1:18117", manifest["request_timeout_s"])
                deadline = time.monotonic() + manifest["startup_timeout_s"]
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"owned server exited {proc.returncode}")
                    try:
                        client.discover_model()
                        break
                    except Exception:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.25)
                started = time.perf_counter()
                for entry in manifest["scenes"][item["scene"]]:
                    payload = json.loads((folder / entry["file"]).read_text())
                    exchange = client.prepare(ProtocolRequest("openai_chat", "/v1/chat/completions", payload)).execute()
                    exchanges.append(exchange)
                    text = "".join(event.output for event in exchange.events if event.kind == "model_output")
                    first = next((e.received_ns for e in exchange.events if e.kind == "model_output"), None)
                    terminal = any(e.kind == "terminal" for e in exchange.events)
                    native = next((e.payload["timings"] for e in reversed(exchange.events)
                                   if e.payload and "timings" in e.payload), None)
                    usage = next((e.payload["usage"] for e in reversed(exchange.events)
                                  if e.payload and e.payload.get("usage")), None)
                    record["requests"].append({"input": entry["file"], "text": text,
                        "correct": text.strip() == entry["expected"], "status": exchange.http.status,
                        "ttft_ms": (first - exchange.http.sent_ns) / 1e6 if first is not None else None,
                        "wall_ms": (exchange.http.ended_ns - exchange.http.sent_ns) / 1e6,
                        "timings": native, "usage": usage})
                    if exchange.http.status != 200 or exchange.http.error or exchange.protocol_error or exchange.error_code or not terminal:
                        raise RuntimeError(f"request failed: {exchange.http.error or exchange.protocol_error or exchange.error_message}")
                    if not record["requests"][-1]["correct"]:
                        raise RuntimeError("frozen quality gate failed; stop performance campaign")
                    if time.perf_counter() - started > manifest["session_timeout_s"]:
                        raise RuntimeError("frozen session time limit exceeded")
                record["wall_ms"] = (time.perf_counter() - started) * 1000
                (folder / f"{name}.metrics.txt").write_text(metrics())
                record["status"] = "PASS"
                print(f"{position+1}/18 {name}: {record['wall_ms']:.1f} ms", flush=True)
            except BaseException as error:
                record["status"] = "FAIL"
                record["error"] = repr(error)
                raise
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                record["server_exit"] = proc.returncode
                for index, exchange in enumerate(exchanges):
                    (folder / f"{name}-r{index}.sse").write_bytes(exchange.http.body)
                    receipt = dataclasses.asdict(exchange)
                    receipt["http"].pop("body")
                    write_json(folder / f"{name}-r{index}.receipt.json", receipt)
                write_json(result_path, results)
    report(folder)


def report(folder: Path) -> None:
    manifest = json.loads((folder / "manifest.json").read_text())
    results = json.loads((folder / "results.json").read_text())
    summary = {"sessions": len(results), "planned": 18, "scenes": {}}
    compatible = len(results) == 18 and all(r["status"] == "PASS" for r in results)
    for scene in manifest["scenes"]:
        pairs = []
        for pair in range(3):
            arms = {r["mode"]: r for r in results if r["scene"] == scene and r["pair"] == pair}
            if set(arms) != {"sync", "async"} or any(r["status"] != "PASS" for r in arms.values()):
                compatible = False
                continue
            same = [r["text"] for r in arms["sync"]["requests"]] == [r["text"] for r in arms["async"]["requests"]]
            same_usage = [r["usage"] for r in arms["sync"]["requests"]] == [r["usage"] for r in arms["async"]["requests"]]
            compatible = compatible and same and same_usage
            pairs.append({"pair": pair, "sync_ms": arms["sync"]["wall_ms"],
                "async_ms": arms["async"]["wall_ms"], "outputs_equal": same,
                "reduction": 1 - arms["async"]["wall_ms"] / arms["sync"]["wall_ms"]})
        summary["scenes"][scene] = {"pairs": pairs,
            "median_reduction": statistics.median(p["reduction"] for p in pairs) if pairs else None}
    deep = summary["scenes"]["deep"]
    shallow = summary["scenes"]["shallow"]
    quality = len(results) == 18 and all(r["status"] == "PASS" and all(q["correct"] for q in r["requests"]) for r in results)
    summary["quality_qualified"] = quality
    adopt = compatible and quality and len(deep["pairs"]) == 3 and all(p["reduction"] > 0 for p in deep["pairs"]) and deep["median_reduction"] >= .10 and shallow["median_reduction"] >= -.03
    summary["outputs_compatible"] = compatible
    summary["decision"] = "eligible_for_deployment_confirmation" if adopt else "keep_sync_default_no_more_samples"
    write_json(folder / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["freeze", "run", "report"])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--binary", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        if not args.model or not args.binary:
            parser.error("freeze requires explicit --model and --binary")
        freeze(args.folder, args.model, args.binary)
    elif args.action == "run":
        run(args.folder)
    else:
        report(args.folder)


if __name__ == "__main__":
    main()
