#pragma once

#include "ninfer/engine.h"
#include <ninfer/targets/qwen3_6/frontend.h>

namespace ninfer {

// Internal fixture entry point: preserve family execution provenance when prefilling a frozen
// generated history. Raw prepare_tokens intentionally does not invent chat/GDN boundaries.
// Program still validates the complete identity; this grants no cache or device-state access.
class PreparedPromptTestAccess {
public:
    [[nodiscard]] static PreparedPrompt adopt(targets::qwen3_6::PreparedPrompt prompt,
                                               SamplingMode mode);
};

} // namespace ninfer
