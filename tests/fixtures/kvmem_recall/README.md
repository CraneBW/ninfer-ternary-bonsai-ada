# KVMem recall fixtures

Unchanged requests from the frozen 2026-10-20 P0 Qwen3.8 panel: a label
crossing a 64-token page boundary and a historical tool-result label.
Expected turn-1 content is ACK; turn-2 values are violet-cinder-58 and
teal-jasmine-61 respectively. Full-KV controls answered both correctly.

Run the opt-in native integration target with an explicit registered artifact:

```sh
NINFER_QWEN3_8_KVMEM_WEIGHTS=/absolute/model.ninfer \
  build/tests/ninfer_qwen3_8_kvmem_async_real_test graph tests/fixtures/kvmem_recall
```

Routes are base, graph, mtp and dflash2. Each fixture-mode invocation performs
eight requests (two turns, two cases, sync/async), with no automatic retry.
MTP and DFlash2 use draft width 3 and Graph. The original raw-token lifecycle
fixture remains available by omitting the fixture-directory argument.

Append `shared-boundary` to exercise the single cross-page case with Engine's
automatic shared-prefix captures enabled (four calls, sync/async). This mode
checks bounded capture/materialization and transfer parity, and records label
correctness separately; shared/fixed geometry has no general forced-recall
guarantee. The ordinary matrix matches OpenAI serving by disabling implicit
shared captures and using LRFU top fraction 0.3.


Append `semantic` instead to run the original paraphrase case alone (four calls,
two turns × sync/async). Its question has no qualified lexical candidates and
must return `cobalt-orchid-73`. The fixture compares complete token IDs,
content, finish reason, reused/committed tokens and terminal transfer accounting;
MTP/DFlash2 must exercise speculative drafts. The request bytes come from the
same frozen eight-case panel; no larger answer limit is used.


The store-level deliberately blocked-copy tests retain their existing CTest
`CUDA_MODULE_LOADING=EAGER` environment. Direct invocation must pass that same
environment; otherwise first-use lazy module loading can wait on the test's own
callback gate. This is a fixture execution contract, not a new production flag.
