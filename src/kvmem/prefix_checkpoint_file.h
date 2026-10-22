#pragma once
#include "kvmem/snapshot_stream.h"
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
#define XXH_INLINE_ALL
#include "xxhash/xxhash.h"
#include <filesystem>
#include <memory>
#include <string>
#if !defined(_WIN32)
#include <unistd.h>
#endif
namespace ninfer::kvmem {
// Local benchmark archive transport. The streaming codec is reused from KVMem;
// unlike its process-local manifest, this archive includes the complete directory.
class PrefixCheckpointWriter {
public:
    explicit PrefixCheckpointWriter(std::string path) : path_(std::move(path)), temp_(path_+".tmp") {
        if(std::filesystem::exists(path_)) throw std::runtime_error("checkpoint already exists");
        file_.reset(std::fopen(temp_.c_str(),"wbx"));
        if(!file_) throw std::runtime_error("cannot create checkpoint temporary file");
        stream_=std::make_unique<SnapshotWriter>([this](const void* p,std::size_t n) {
            if(std::fwrite(p,1,n,file_.get())!=n) throw std::runtime_error("checkpoint write failed");
        });
    }
    ~PrefixCheckpointWriter() {
        file_.reset();
        if(!complete_) { std::error_code ec;std::filesystem::remove(temp_,ec); }
    }
    SnapshotWriter& stream() { return *stream_; }
    void finish() {
        const std::uint64_t footer[]{stream_->bytes(),stream_->hash()};
        if(std::fwrite(footer,sizeof(footer),1,file_.get())!=1 || std::fflush(file_.get()))
            throw std::runtime_error("checkpoint final write failed");
#if !defined(_WIN32)
        if(::fsync(::fileno(file_.get()))) throw std::runtime_error("checkpoint fsync failed");
#endif
        file_.reset();std::filesystem::rename(temp_,path_);complete_=true;
    }
private:
    std::string path_,temp_;
    std::unique_ptr<FILE,decltype(&std::fclose)> file_{nullptr,&std::fclose};
    std::unique_ptr<SnapshotWriter> stream_;
    bool complete_=false;
};
// XXH3-128 is reused from the installed upstream xxHash 0.8.4 header,
// BSD-2-Clause; vendored under third_party/xxhash with its original license.
// The sidecar is derived integrity data, never a replacement for payload checks.
class PrefixCheckpointReader {
public:
    static bool fast_requested() {
        const char* value = std::getenv("NINFER_BENCH_CHECKPOINT_FAST_VERIFY");
        if (!value || std::strcmp(value, "1") == 0) { return true; }
        if (std::strcmp(value, "0") == 0) { return false; }
        throw std::invalid_argument("NINFER_BENCH_CHECKPOINT_FAST_VERIFY must be 0 or 1");
    }
    explicit PrefixCheckpointReader(const std::string& path, bool fast = fast_requested())
        : index_path_(path + ".xxh3idx"), fast_(fast) {
        const auto began = Clock::now();
        file_.reset(std::fopen(path.c_str(), "rb"));
        if (!file_) { throw SnapshotCorrupt("checkpoint not found"); }
        if(checkpoint_load_pipeline_enabled()) {
            backing_=SnapshotBacking::clone(file_.get(),path);
#if defined(__linux__)
            if(backing_) {
                const int duplicate=::dup(backing_->fd);
                if(duplicate<0) throw SnapshotCorrupt("checkpoint private duplicate failed");
                FILE* private_file=::fdopen(duplicate,"rb");
                if(!private_file) { ::close(duplicate);throw SnapshotCorrupt("checkpoint private stream failed"); }
                file_.reset(private_file);
            }
#endif
        }

        if (std::fseek(file_.get(), 0, SEEK_END)) { throw SnapshotCorrupt("checkpoint seek failed"); }
        const auto size = std::ftell(file_.get());
        if (size < 16 || std::fseek(file_.get(), size - 16, SEEK_SET) ||
            std::fread(footer_, sizeof(footer_), 1, file_.get()) != 1 ||
            footer_[0] != std::uint64_t(size - 16)) {
            throw SnapshotCorrupt("checkpoint footer/length mismatch");
        }
        std::rewind(file_.get());
        const bool hit = fast_ && load_index();
        if (fast_) { buffer_.resize(snapshot_chunk); }
        if (hit) {
            std::uint64_t left = footer_[0];
            for (const auto expected : digests_) {
                const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(left, snapshot_chunk));
                read_file(buffer_.data(), n);
                if (!equal(digest(buffer_.data(), n), expected)) {
                    throw SnapshotCorrupt("checkpoint chunk mismatch before mutation");
                }
                left -= n;
            }
        } else {
            // Legacy FNV proves the source of every newly created sidecar.
            // Combine index construction with this pass; no extra payload read.
            SnapshotReader verify([this](void* p, std::size_t n) { read_file(p, n); }, footer_[0]);
            std::vector<std::uint8_t> legacy_buffer;
            if (!fast_) { legacy_buffer.resize(snapshot_chunk); }
            auto& buffer = fast_ ? buffer_ : legacy_buffer;
            while (verify.remaining()) {
                const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(), verify.remaining()));
                verify.read(buffer.data(), n);
                if (fast_) { digests_.push_back(digest(buffer.data(), n)); }
            }
            if (verify.hash() != footer_[1]) {
                throw SnapshotCorrupt("checkpoint checksum mismatch before mutation");
            }
            if (fast_) { save_index(); }
        }
        std::rewind(file_.get());
        stream_ = std::make_unique<SnapshotReader>(
            [this](void* p, std::size_t n) {
                if (fast_) { read_verified(p, n); } else { read_file(p, n); }
            }, footer_[0], fast_ ? SnapshotReader::Integrity::VerifiedTransport : SnapshotReader::Integrity::Fnv1a, backing_);
        std::fprintf(stderr, "KVMEM_CHECKPOINT_VERIFY mode=%s index=%s seconds=%.6f bytes=%llu\n",
                     fast_ ? "indexed" : "legacy", fast_ ? (hit ? "hit" : "built") : "none",
                     std::chrono::duration<double>(Clock::now() - began).count(),
                     static_cast<unsigned long long>(footer_[0]));
    }
    SnapshotReader& stream() { return *stream_; }
    [[nodiscard]] bool private_backing() const { return backing_ != nullptr; }
    void finish() {
        if (stream_->remaining() || (fast_ ? delivered_ != footer_[0] : stream_->hash() != footer_[1])) {
            throw SnapshotCorrupt("checkpoint payload mismatch");
        }
    }
private:
    using Clock = std::chrono::steady_clock;
    using Digest = XXH128_hash_t;
    struct IndexHeader {
        std::uint64_t magic, version, chunk, bytes, legacy_hash, count;
        Digest digest;
    };
    static constexpr std::uint64_t index_magic = 0x3158444950464B4EULL; // NKFPIDX1
    static Digest digest(const void* p, std::size_t n) { return XXH3_128bits(p, n); }
    static bool equal(Digest a, Digest b) { return a.low64 == b.low64 && a.high64 == b.high64; }
    bool load_index() {
        std::unique_ptr<FILE, decltype(&std::fclose)> index(std::fopen(index_path_.c_str(), "rb"), &std::fclose);
        if (!index) { return false; }
        IndexHeader header{};
        const auto count = footer_[0] / snapshot_chunk + (footer_[0] % snapshot_chunk != 0);
        if (std::fread(&header, sizeof(header), 1, index.get()) != 1 ||
            header.magic != index_magic || header.version != 1 || header.chunk != snapshot_chunk ||
            header.bytes != footer_[0] || header.legacy_hash != footer_[1] || header.count != count ||
            count > SIZE_MAX / sizeof(Digest)) { return false; }
        digests_.resize(static_cast<std::size_t>(count));
        if (std::fread(digests_.data(), sizeof(Digest), digests_.size(), index.get()) != digests_.size() ||
            std::fgetc(index.get()) != EOF || std::ferror(index.get()) ||
            !equal(digest(digests_.data(), digests_.size() * sizeof(Digest)), header.digest)) {
            digests_.clear();
            return false;
        }
        return true;
    }
    void save_index() {
        // Optional derived cache: a read-only archive directory can still use
        // the in-memory digests built by the complete FNV pass above.
        const auto temporary = index_path_ + ".tmp";
        std::unique_ptr<FILE, decltype(&std::fclose)> index(std::fopen(temporary.c_str(), "wbx"), &std::fclose);
        if (!index) { return; }
        const IndexHeader header{index_magic, 1, snapshot_chunk, footer_[0], footer_[1],
            digests_.size(), digest(digests_.data(), digests_.size() * sizeof(Digest))};
        bool ok = std::fwrite(&header, sizeof(header), 1, index.get()) == 1 &&
            std::fwrite(digests_.data(), sizeof(Digest), digests_.size(), index.get()) == digests_.size() &&
            std::fflush(index.get()) == 0;
#if !defined(_WIN32)
        if (ok) { ok = ::fsync(::fileno(index.get())) == 0; }
#endif
        index.reset();
        std::error_code error;
        if (ok) { std::filesystem::rename(temporary, index_path_, error); }
        if (!ok || error) { std::filesystem::remove(temporary, error); }
    }
    void read_file(void* p, std::size_t n) {
        if (std::fread(p, 1, n, file_.get()) != n) { throw SnapshotCorrupt("checkpoint read failed"); }
    }
    void read_verified(void* p, std::size_t n) {
        auto* destination = static_cast<std::uint8_t*>(p);
        while (n) {
            if (buffer_cursor_ == buffer_bytes_) {
                buffer_bytes_ = static_cast<std::size_t>(std::min<std::uint64_t>(snapshot_chunk, footer_[0] - delivered_));
                if (buffer_bytes_ == 0 || next_digest_ >= digests_.size()) {
                    throw SnapshotCorrupt("checkpoint chunk extent mismatch");
                }
                read_file(buffer_.data(), buffer_bytes_);
                if (!equal(digest(buffer_.data(), buffer_bytes_), digests_[next_digest_++])) {
                    throw SnapshotCorrupt("checkpoint chunk changed during restore");
                }
                buffer_cursor_ = 0;
            }
            const auto part = std::min(n, buffer_bytes_ - buffer_cursor_);
            std::memcpy(destination, buffer_.data() + buffer_cursor_, part);
            destination += part; n -= part; buffer_cursor_ += part; delivered_ += part;
        }
    }
    std::unique_ptr<FILE, decltype(&std::fclose)> file_{nullptr, &std::fclose};
    std::unique_ptr<SnapshotReader> stream_;
    std::shared_ptr<const SnapshotBacking> backing_;
    std::uint64_t footer_[2]{};
    std::string index_path_;
    bool fast_;
    std::vector<Digest> digests_;
    std::vector<std::uint8_t> buffer_;
    std::size_t buffer_cursor_ = 0, buffer_bytes_ = 0, next_digest_ = 0;
    std::uint64_t delivered_ = 0;
};
}
