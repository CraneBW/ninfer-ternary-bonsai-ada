# Companion Bonsai 2B v3 small-shape patch (experimental)

This is an in-progress patch for the **separate** `ninfer-all` Companion repository,
recorded here without committing to or pushing that upstream fork. It is not a claim
of successful end-to-end Bonsai 2B generation.

## Baseline and verified input

- Companion baseline: `b2913b6`, branch `master`.
- Model: `/home/ran/ninfer-work/real-models/bonsai2_2b_v3.ninfer`.
- Whole-file SHA-256: `0601756954b5fc24e9056b56e80bad6f40f8db9985e47ddf421eb833e76d7b63`.
- Original Companion executable loaded 608.4 MiB of model weights, but rejected
  the frontend with `text/layers/0: input projection: unsupported logical projection geometry`.

Apply the patch to a **clean checkout of that Companion baseline** with
`git apply --check <path>/tools/companion/bonsai2b-smallshape-experimental.patch`,
then `git apply` with the same patch. Do not apply it twice to the already-modified
Companion working tree. The patch currently matches the existing dirty working tree
(`git apply --reverse --check` passed).

## Implemented experimental routes

- Weight binding accepts 2048-hidden Bonsai2B GDN controls and input parents.
- T2 rowsplit linear adds small widths including the 2048-input/512-output case.
- GDN query/key/value/z T2 split, 6144-channel projected convolution, and
  16-head BF16 gating use exact Bonsai2B geometry (not 27B shape assertions).
- Full-attention T2 split supports query/gate 2048, key/value 512, input 2048.
- Added oracle-backed Bonsai2B modes to `test_t2_a16`, `test_gdn_input_proj`,
  `test_gdn_gating_proj`, and `test_attn_input_proj`.

## Validation checkpoint

- `git diff --check` and reverse-apply check passed.
- Host C++ `g++ -fsyntax-only` checks passed for the changed wrappers,
  routing/planner sources, and dedicated projection tests.
- Independent `nvcc -arch=sm_89 -c` completed for the modified GDN gating and
  GDN projected-convolution CUDA translation units.
- The existing `build-ada-companion` Ninja job is rebuilding hundreds of CUDA
  objects due to recovery of its state file. **Do not start another concurrent
  Ninja build into the same directory.** At this checkpoint, the new application
  binary and GPU oracle tests have not yet been run against these changes.

When that build is complete, execute focused tests from the Companion checkout
using the generated test executables (target names vary by build configuration):
`--bonsai2b` on GDN input, GDN gating and attention input tests; run the
small-shape T2 numerical tests; then start the actual v3 model with:

```sh
build-ada-companion/apps/ninfer-serve \
  /home/ran/ninfer-work/real-models/bonsai2_2b_v3.ninfer \
  --host 127.0.0.1 --port 18098 \
  --max-context 1024 --kv-capacity 1024 \
  --host-kv-mib 256 --max-concurrency 1 \
  --default-max-tokens 32 --kv-dtype int8 --greedy \
  --no-thinking --no-cuda-graph --log-level info
```

Continue from the **next actual** loader/frontend/operator or numerical failure;
the patch is a checkpoint, not a finished Companion backend.
