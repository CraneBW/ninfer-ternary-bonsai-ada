# Local CUDA service operations

These rules apply to the current single-owner 4070 Ti SUPER optimization runs.
They adapt the [5060 Ti donor's shutdown procedure](https://github.com/XuanzhengZhou/qwen3.8-27b-ternary-5060ti/blob/104af690541857a5d52f23ffd81ccac7eb82a675/server/stop.sh)
and memory-floor policy. The donor reports a CUDA Xid 62/reset incident after
forced termination; that report is not a locally reproduced diagnosis.

## Stop and restart

- Stop the exact owned service or process, using SIGTERM. Do not use `kill -9`,
  `SIGKILL`, broad `pkill -f`, or an automatic TERM-to-KILL fallback for CUDA jobs.
- Allow active prefill and pending CUDA work to drain. A long-context request can
  outlast a short shutdown timeout. Report a grace-period expiry as **still live**;
  it is not permission to start another engine, build, or GPU benchmark.
- If a process remains alive, inspect its exact PID with `ps -o pid,stat,wchan:32,etime,args -p PID`.
  A `D` state requires investigation of the driver/kernel wait. Do not queue more
  kill signals, automatically reset the GPU, reboot, or shut down the computer.
- Restart only after the owned process and its children have actually exited.
  Auto-restart is for an exited failed service, not a second copy beside a stuck one.

For a **new** user service, keep systemd from silently escalating a stop to KILL:

```ini
[Service]
KillSignal=SIGTERM
KillMode=control-group
SendSIGKILL=no
TimeoutStopSec=infinity
Restart=on-failure
RestartSec=15
```

This is a template, not an installed or enabled service. Model paths and the
qualified command remain local. The donor's `ninfer-hybrid.service` has
`TimeoutStopSec=200` and no `SendSIGKILL=no`; copying that unit unchanged would
not enforce its own no-KILL comment. systemd's explicit no-escalation setting is
therefore required here; see the official [systemd kill procedure](https://www.freedesktop.org/software/systemd/man/latest/systemd.kill.html#SendSIGKILL=).
Memory cgroup limits remain necessary after the previous
host OOM; the kernel can still kill a process on OOM. Such an event is a failed
run, never a clean shutdown or a passed qualification.

The frozen R22 ABBAAB matrix predates these rules and completed all 36 requests
with normal server exit 0; its legacy forced-stop branches did not run. Its runner
and receipts remain unchanged. After that terminal result, the current qualification
runner was switched to `tools/bench/cuda_process.py`: TERM with 120 seconds grace,
preserved failure receipts on timeout, and no KILL escalation. New R23 services also
explicitly used `SendSIGKILL=no`.

## Free VRAM and readiness

Keep at least **256 MiB free on the selected GPU after model, workspaces, and CUDA
graphs are ready**. Inspect the selected device, not a sum over GPUs:

```bash
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits
```

Below the floor, stop admitting work and gracefully stop the owned service; reduce
the configuration before restarting. Do not kill another application's process.
This floor is an operating rule, not a new compiled admission guarantee. The
current managed-weight planner's initial 512 MiB reservation is a different
measurement point and does not prove the post-warmup floor. Do not introduce a
new polling/restart daemon during a performance comparison.

Use the engine's `warmup complete` and ready record, then its model-discovery
endpoint, as readiness. A bound TCP port or a live PID alone does not establish
readiness. Fresh server startup, graph warmup, initial full prefill, and subsequent
API TTFT are separate timings.

## Cache and measurement boundaries

- A byte-identical request-response replay cache is not an Engine prefix-KV hit.
  Report its response provenance and exclude it from original-question inference
  qualification. Changing model, quantization, decoding, tools, or layout invalidates
  the relevant cache identity.
- Persisting a prefix prompt and submitting it at startup warms Engine state; it
  does not persist or restore GDN/KV binary state. For hybrid attention, restore the
  matching GDN state and KV lineage together before claiming a reusable checkpoint.
- Keep streaming reads frame-responsive. The current Python HTTP client already
  uses `response.read1()`; a donor's fixed buffering delay is not evidence of the
  same local delay.
- Time performance in clean runs. Profiler traces establish mechanism; observer
  overhead, assumed speculative acceptance, or a cache-response hit cannot replace
  measured emitted-token throughput and API TTFT.
