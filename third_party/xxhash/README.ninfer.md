# xxHash header reuse

Copied without modifications from this machine's `/usr/include/xxhash.h`, version **0.8.4**, on 2026-10-22. Upstream: https://github.com/Cyan4973/xxHash . BSD 2-Clause copyright/license retained in the header and LICENSE.

Only the portable header-only XXH3-128 API is used by benchmark checkpoint integrity indexes. No new system package, global configuration, model dependency or CUDA kernel is installed. The checkpoint payload format and legacy FNV-1a footer stay unchanged; the derived index is generated only after a full legacy validation.

SHA256: `3dc8d161e867a62d3417f7885b55fdaded8b7b497e2f165d05c94f4fe24f2ca4`
