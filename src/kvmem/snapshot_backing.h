#pragma once
// Stable private ownership for checkpoint-backed immutable K extents. The
// extent/generation contract follows KVMem/ActKV spill_io; Linux reflink is a
// local transport optimization, never an alias of the user-owned archive.
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__linux__)
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace ninfer::kvmem {
inline bool checkpoint_load_pipeline_enabled() {
    const char* value=std::getenv("NINFER_BENCH_CHECKPOINT_LOAD_PIPELINE");
    if(!value || !std::strcmp(value,"0")) return false;
    if(!std::strcmp(value,"1")) return true;
    throw std::invalid_argument("NINFER_BENCH_CHECKPOINT_LOAD_PIPELINE must be 0 or 1");
}
struct SnapshotBacking {
    int fd=-1;
    std::uint64_t bytes=0;
    SnapshotBacking()=default;
    SnapshotBacking(const SnapshotBacking&)=delete;
    SnapshotBacking& operator=(const SnapshotBacking&)=delete;
    ~SnapshotBacking() {
#if defined(__linux__)
        if(fd>=0) ::close(fd);
#endif
    }
    static std::shared_ptr<SnapshotBacking> clone(FILE* source,const std::string& path) {
#if defined(__linux__)
        const auto separator=path.find_last_of('/');
        const auto directory=separator==std::string::npos ? "." : path.substr(0,separator);
        std::string pattern=directory+"/.ninfer-checkpoint-XXXXXX";
        std::vector<char> name(pattern.begin(),pattern.end());name.push_back(0);
        auto result=std::make_shared<SnapshotBacking>();
        result->fd=::mkstemp(name.data());
        if(result->fd<0) return {};
        if(::unlink(name.data())!=0) {
            const int error=errno;::close(result->fd);result->fd=-1;
            (void)::unlink(name.data());throw std::runtime_error("checkpoint private unlink: "+std::string(std::strerror(error)));
        }
        // Unsupported filesystem/cross-device clone keeps the original bounded
        // copy path. Never fall back to copying a 15GB archive into RAM.
        if(::ioctl(result->fd,FICLONE,::fileno(source))!=0) return {};
        struct stat status{};
        if(::fstat(result->fd,&status)!=0 || status.st_size<0) throw std::runtime_error("checkpoint private extent stat");
        result->bytes=static_cast<std::uint64_t>(status.st_size);return result;
#else
        (void)source;(void)path;return {};
#endif
    }
};
}
