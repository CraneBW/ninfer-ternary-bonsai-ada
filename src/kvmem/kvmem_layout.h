#pragma once
#include <cstdint>
namespace ninfer::kvmem {
// Four independent raw-K sums per page, 16 tokens per subblock.
inline constexpr std::uint32_t kMeanKeySubblocks = 4;
inline constexpr std::uint32_t kMeanKeyTokensPerSubblock = 16;
} // namespace ninfer::kvmem
