#include "runtime/engine/tp_lockstep.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <string>

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

void run_rank(int rank, bool corrupt) {
    setenv("NINFER_TP_RANK", std::to_string(rank).c_str(), 1);
    setenv("NINFER_TP_LOCKSTEP_TIMEOUT_S", "5", 1);
    if (corrupt && rank == 1) { setenv("NINFER_TP_LOCKSTEP_FAULT_UNIT", "2", 1); }
    auto* lockstep = ninfer::runtime::tp::Lockstep::instance();
    assert(lockstep != nullptr && lockstep->rank() == rank);
    assert(lockstep->agree_admissions(rank == 0 ? 3 : 2) == 2);
    const auto cancels = lockstep->exchange(ninfer::runtime::tp::UnitKind::Prefill,
                                            rank == 0 ? 1 : 2, 7, 0x1234, 55, 1);
    assert(cancels == 3);
    if (!corrupt) {
        const auto decoded = lockstep->exchange(ninfer::runtime::tp::UnitKind::Decode,
                                                 0, 2, 0x5678, 56, 1);
        assert(decoded == 0);
    }
}

void launch_pair(const std::filesystem::path& file, bool corrupt) {
    std::filesystem::remove(file);
    // Avoid dumps from the deliberate cross-rank mismatch test.
    const rlimit no_core{0, 0};
    assert(setrlimit(RLIMIT_CORE, &no_core) == 0);
    pid_t pids[2]{};
    for (int rank = 0; rank < 2; ++rank) {
        pids[rank] = fork();
        assert(pids[rank] >= 0);
        if (pids[rank] == 0) {
            setenv("NINFER_TP_LOCKSTEP_FILE", file.c_str(), 1);
            run_rank(rank, corrupt);
            _exit(0);
        }
    }
    for (int rank = 0; rank < 2; ++rank) {
        int status = 0;
        assert(waitpid(pids[rank], &status, 0) == pids[rank]);
        if (corrupt) {
            assert(!WIFEXITED(status) || WEXITSTATUS(status) != 0);
        } else {
            assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
        }
    }
    std::filesystem::remove(file);
}

} // namespace

int main() {
    alarm(20);
    const auto base = std::filesystem::temp_directory_path() /
                      ("ninfer_tp_lockstep_" + std::to_string(getpid()));
    launch_pair(base.string() + ".ok", false);
    launch_pair(base.string() + ".mismatch", true);
}
