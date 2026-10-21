# SPDX-License-Identifier: Apache-2.0
"""Gracefully stop only a CUDA child owned by a benchmark launcher.

Adapted from XuanzhengZhou/qwen3.8-27b-ternary-5060ti, Apache-2.0,
104af690541857a5d52f23ffd81ccac7eb82a675: server/stop.sh. A grace-period
expiry leaves the child alive and blocks dependent GPU work; never escalate.
"""
from __future__ import annotations

from pathlib import Path
import subprocess


class CudaProcessStillRunning(RuntimeError):
    """The owned child has not exited; its GPU is not available for reuse."""


def stop_owned_cuda_process(proc: subprocess.Popen, grace_seconds: float = 120) -> int:
    """Send TERM to this launcher's child and require an observed terminal state."""
    if grace_seconds < 0:
        raise ValueError("grace_seconds must be non-negative")
    if proc.poll() is not None:
        return proc.returncode
    try:
        proc.terminate()
    except ProcessLookupError:
        return proc.wait()
    try:
        return proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired as error:
        # Diagnostics only: no process-name lookup, group kill, or PID adoption.
        details = []
        for name in ("status", "wchan"):
            try:
                value = (Path("/proc") / str(proc.pid) / name).read_text()
                if name == "status":
                    value = next((line for line in value.splitlines()
                                  if line.startswith("State:")), "State: unavailable")
                details.append(value.strip())
            except OSError:
                pass
        raise CudaProcessStillRunning(
            f"Owned CUDA child PID {proc.pid} did not exit after SIGTERM and "
            f"{grace_seconds:g}s grace; {'; '.join(details)}. No SIGKILL sent. "
            "Do not start another GPU job until this child has actually exited."
        ) from error
