#!/usr/bin/env python3
"""Dispatch NInfer v2 and v3 artifacts to their registered native engines.

The Ada fork implements the v2 fixed-size models and the v3 Qwen 27B
text, vision, MTP and DFlash2 routes. The optional NInfer-all companion
implements the wider v3 family: Ternary Bonsai 2, Gemma 4, Flash-Next,
hybrid MoE and other model-specific variants.
Every backend is an independently built NInfer binary; no weights are copied.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shlex
import sys

ENTRY_V2 = b"NINFER\0\2"
ENTRY_V3 = b"NINFER\0\3"
MAX_DIRECTORY_BYTES = 64 << 20


def inspect(path: Path) -> tuple[int, str, str]:
    with path.open("rb") as file:
        magic = file.read(8)
        if magic not in (ENTRY_V2, ENTRY_V3):
            raise ValueError(f"unsupported .ninfer magic: {magic!r}")
        length = int.from_bytes(file.read(8), "little")
        if not 0 < length <= MAX_DIRECTORY_BYTES:
            raise ValueError(f"invalid directory length: {length}")
        if magic == ENTRY_V3:
            uuid = file.read(16)
            if len(uuid) != 16:
                raise ValueError("v3 UUID header is truncated")
        data = file.read(length)
        if len(data) != length:
            raise ValueError("artifact directory is incomplete")
        directory = json.load(io.TextIOWrapper(io.BytesIO(data), encoding="utf-8"))
    if magic == ENTRY_V2:
        identity = directory.get("identity", {})
        return 2, str(identity.get("model_id", "")), str(identity.get("weights_id", ""))
    return (
        3,
        str(directory.get("metadata", {}).get("name", "")),
        str(directory.get("components", {}).get("text", {})
            .get("config", {}).get("model_type", "")),
    )


def choose_backend(info: tuple[int, str, str], arguments: list[str]) -> str:
    version, model, model_type = info
    if model == "qwen3.6-27b":
        raise ValueError("Qwen3.6-27B support has been retired; use Qwen3.8-27B")
    if version == 2:
        return "native"
    if model in {"qwen3.8-27b-tp2-rank0", "qwen3.8-27b-tp2-rank1"}:
        raise ValueError(
            "TP2 rank artifact requires a native two-rank runtime with collectives; "
            "the current single-GPU and companion backends cannot serve it"
        )
    # DFlash (v1) is provided only by the companion; native Qwen 27B
    # implements MTP and DFlash2, including their optimized proposal head.
    speculative = [arguments[i + 1] for i, value in enumerate(arguments[:-1])
                   if value == "--spec"]
    if (
        model == "qwen3.8-27b"
        and model_type == "qwen3_5_text"
        and "dflash" not in speculative
    ):
        return "native"
    return "companion"


def default_binary(kind: str) -> Path:
    repository = Path(__file__).resolve().parents[1]
    if kind == "native":
        return Path(os.environ.get("NINFER_NATIVE_SERVE", repository / "build/apps/ninfer-serve"))
    workspace = repository.parents[1]
    return Path(os.environ.get(
        "NINFER_ALL_SERVE",
        workspace / "ecosystem/ninfer-all/build-ada-companion/apps/ninfer-serve",
    ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve NInfer v2/3 models with the matching NInfer backend"
    )
    parser.add_argument("model", type=Path, help=".ninfer artifact entry file")
    parser.add_argument(
        "--backend", choices=("auto", "native", "companion"), default="auto",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the selected backend and executable without starting a server",
    )
    args, flags = parser.parse_known_args(argv)
    if flags and flags[0] == "--":
        flags = flags[1:]
    info = inspect(args.model)
    if info[1] == "qwen3.6-27b":
        raise ValueError("Qwen3.6-27B support has been retired; use Qwen3.8-27B")
    backend = choose_backend(info, flags) if args.backend == "auto" else args.backend
    binary = default_binary(backend)
    command = [str(binary), str(args.model.resolve()), *flags]
    print(
        f"NInfer backend={backend} format=v{info[0]} model={info[1]}",
        file=sys.stderr,
        flush=True,
    )
    if args.dry_run:
        print(shlex.join(command))
        return 0
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(
            f"{backend} executable unavailable: {binary}; build the selected backend first"
        )
    os.execv(str(binary), command)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ninfer model router: {error}", file=sys.stderr)
        raise SystemExit(2) from error
