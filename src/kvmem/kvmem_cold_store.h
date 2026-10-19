#pragma once

// Byte-exact, bounded-RAM backing for cold KVMem pages. The backing file is
// unlinked immediately on POSIX or delete-on-close on Windows. Choose a real
// NVMe-mounted directory with NINFER_KVMEM_COLD_DIR to use actual SSD storage.

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <filesystem>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace ninfer::kvmem {

class ColdKVStore {
public:
    struct Handle {
        std::uint32_t slot = 0;
        std::uint32_t generation = 0;

        [[nodiscard]] bool valid() const noexcept { return generation != 0; }
        [[nodiscard]] friend bool operator==(Handle, Handle) noexcept = default;
    };

    explicit ColdKVStore(std::size_t page_bytes, const std::string& directory = {})
        : page_bytes_(page_bytes) {
        if (page_bytes == 0
#if !defined(_WIN32)
            || page_bytes > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())
#endif
        ) {
            throw std::invalid_argument("KVMem cold page size is invalid");
        }
        const char* configured = std::getenv("NINFER_KVMEM_COLD_DIR");
#if defined(_WIN32)
        std::wstring folder;
        if (!directory.empty() || (configured && *configured)) {
            folder = std::filesystem::u8path(!directory.empty() ? directory : configured).wstring();
        } else {
            wchar_t temp[MAX_PATH + 1]{};
            const DWORD length = ::GetTempPathW(MAX_PATH + 1, temp);
            if (length == 0 || length > MAX_PATH) {
                throw std::runtime_error(win_error("temporary directory", ::GetLastError()));
            }
            folder = temp;
        }
        wchar_t name[MAX_PATH + 1]{};
        if (::GetTempFileNameW(folder.c_str(), L"nkv", 0, name) == 0) {
            throw std::runtime_error(win_error("create", ::GetLastError()));
        }
        file_ = ::CreateFileW(name, GENERIC_READ | GENERIC_WRITE,
                              FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE,
                              nullptr);
        if (file_ == INVALID_HANDLE_VALUE) {
            const DWORD err = ::GetLastError();
            ::DeleteFileW(name);
            throw std::runtime_error(win_error("open", err));
        }
#else
        const std::string path = !directory.empty() ? directory
                                 : configured && *configured ? configured : "/tmp";
        std::string pattern = path + "/ninfer-kvmem-XXXXXX";
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back('\0');
        fd_ = ::mkstemp(name.data());
        if (fd_ < 0) { throw std::runtime_error(io_error("create", errno)); }
        // The file descriptor alone owns the pages. No stale files after a crash.
        if (::unlink(name.data()) != 0) {
            const int err = errno;
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error(io_error("unlink", err));
        }
#endif
    }

    ~ColdKVStore() {
#if defined(_WIN32)
        if (file_ != INVALID_HANDLE_VALUE) { ::CloseHandle(file_); }
#else
        if (fd_ >= 0) { ::close(fd_); }
#endif
    }

    ColdKVStore(const ColdKVStore&) = delete;
    ColdKVStore& operator=(const ColdKVStore&) = delete;
    ColdKVStore(ColdKVStore&&) = delete;
    ColdKVStore& operator=(ColdKVStore&&) = delete;

    [[nodiscard]] std::size_t page_bytes() const noexcept { return page_bytes_; }
    [[nodiscard]] std::size_t occupied_bytes() const noexcept {
        return active_ * page_bytes_;
    }
    [[nodiscard]] std::size_t occupied_pages() const noexcept { return active_; }

    // A failed write never publishes a record. Inactive slots are reused to
    // prevent unbounded file growth during repeated KV re-selection.
    [[nodiscard]] Handle store(std::span<const std::byte> page) {
        require_size(page.size());
        const bool reused = !free_.empty();
        const std::uint32_t slot = reused ? free_.back()
                                          : static_cast<std::uint32_t>(slots_.size());
        if (!reused) {
            if (slots_.size() >= std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("KVMem cold slot count overflow");
            }
            // Reserve the corresponding free-list space now so release is noexcept.
            free_.reserve(slots_.size() + 1);
            slots_.push_back(Slot{});
        }
        try {
            transfer(slot, const_cast<std::byte*>(page.data()), /*writing=*/true);
        } catch (...) {
            if (!reused) { slots_.pop_back(); }
            throw;
        }
        if (reused) { free_.pop_back(); }
        slots_[slot].active = true;
        ++active_;
        return Handle{slot, slots_[slot].generation};
    }

    void load(Handle handle, std::span<std::byte> destination) const {
        require_size(destination.size());
        if (!valid(handle)) { throw std::invalid_argument("KVMem cold handle is stale"); }
        transfer(handle.slot, destination.data(), /*writing=*/false);
    }

    [[nodiscard]] bool valid(Handle handle) const noexcept {
        return handle.valid() && handle.slot < slots_.size() &&
               slots_[handle.slot].active &&
               slots_[handle.slot].generation == handle.generation;
    }

    [[nodiscard]] bool release(Handle handle) noexcept {
        if (!valid(handle)) { return false; }
        Slot& slot = slots_[handle.slot];
        slot.active = false;
        ++slot.generation;
        if (slot.generation == 0) { ++slot.generation; }
        free_.push_back(handle.slot);
        --active_;
        return true;
    }

private:
    struct Slot {
        std::uint32_t generation = 1;
        bool active = false;
    };

#if defined(_WIN32)
    [[nodiscard]] static std::string win_error(const char* action, DWORD err) {
        return std::string("KVMem cold ") + action + ": Win32 error " + std::to_string(err);
    }
#else
    [[nodiscard]] static std::string io_error(const char* action, int err) {
        return std::string("KVMem cold ") + action + ": " + std::strerror(err);
    }
#endif

    void require_size(std::size_t bytes) const {
        if (bytes != page_bytes_) { throw std::invalid_argument("KVMem cold page size mismatch"); }
    }

    void transfer(std::uint32_t slot, std::byte* data, bool writing) const {
#if defined(_WIN32)
        constexpr std::uint64_t limit = static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max());
#else
        const auto limit = static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
#endif
        const auto offset = static_cast<std::uint64_t>(slot) * page_bytes_;
        if (offset > limit || page_bytes_ > limit - offset) {
            throw std::overflow_error("KVMem cold file offset overflow");
        }
        std::size_t done = 0;
        while (done != page_bytes_) {
#if defined(_WIN32)
            LARGE_INTEGER position;
            position.QuadPart = static_cast<LONGLONG>(offset + done);
            if (!::SetFilePointerEx(file_, position, nullptr, FILE_BEGIN)) {
                throw std::runtime_error(win_error("seek", ::GetLastError()));
            }
            const std::size_t remaining = page_bytes_ - done;
            const DWORD count = static_cast<DWORD>(
                remaining > std::numeric_limits<DWORD>::max()
                    ? std::numeric_limits<DWORD>::max() : remaining);
            DWORD transferred = 0;
            const BOOL ok = writing ? ::WriteFile(file_, data + done, count, &transferred, nullptr)
                                    : ::ReadFile(file_, data + done, count, &transferred, nullptr);
            if (!ok || transferred == 0) {
                throw std::runtime_error(
                    win_error(writing ? "write" : "read", ok ? ERROR_HANDLE_EOF : ::GetLastError()));
            }
            done += transferred;
#else
            const ssize_t n = writing
                                  ? ::pwrite(fd_, data + done, page_bytes_ - done,
                                             static_cast<off_t>(offset + done))
                                  : ::pread(fd_, data + done, page_bytes_ - done,
                                            static_cast<off_t>(offset + done));
            if (n < 0 && errno == EINTR) { continue; }
            if (n <= 0) {
                throw std::runtime_error(io_error(writing ? "write" : "read",
                                                  n == 0 ? EIO : errno));
            }
            done += static_cast<std::size_t>(n);
#endif
        }
    }

#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
    std::size_t page_bytes_ = 0;
    std::size_t active_ = 0;
    std::vector<Slot> slots_;
    std::vector<std::uint32_t> free_;
};

} // namespace ninfer::kvmem
