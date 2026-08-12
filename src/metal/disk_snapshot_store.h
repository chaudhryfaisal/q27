#pragma once
// Disk snapshot store (prefix snapshots Phase 2, docs/plans/2026-07-16-
// prefix-snapshots.md). Extracted from metal_server.cpp into its own header
// so the T1 eviction gate (tools/test_snapshot_evict_store.cpp) can drive
// the REAL store offline — this box holds exactly one resident model, so
// the eviction path must be testable without a server.
//
// token-prefix keyed: the server tokenizes every prompt itself, so "the
// snapshot's stored token ids are a prefix of this request's tokens" is
// exact and has no BPE-boundary hazard (recorded design deviation from
// ds4's byte-SHA1, which exists for stateless clients that retokenize).
// Files are named by the SHA1 of the token bytes, so an identical prefix
// overwrites rather than duplicates. LRU by mtime; hits touch the file.
// Directory lookup and eviction are pure file I/O. Snapshot save/load wraps
// each backend transfer in one server lease while keeping file I/O outside it.
//
// The store reads snapshot metadata through a Peek functor injected at
// construction: production passes q27::MetalEngine::peek_snapshot; the
// offline gate passes a tiny stub that parses fabricated Q27SNAP1 fixtures.
// SnapshotInfo mirrors q27::MetalEngine::SnapshotInfo's two fields the
// store uses (position, logits_resident, tokens).

#include "snapshot_evict.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <map>
#include <mutex>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <stdexcept>
#include <vector>

#include <sys/stat.h>
#include <sys/file.h>
#include <unistd.h>

struct SnapPeekInfo {
    uint32_t position = 0;
    bool logits_resident = true;
    std::vector<uint32_t> tokens;
};

class DiskSnapshotStore {
  public:
    // peek: pinned descriptor + diagnostic path -> metadata. The store opens
    // each candidate itself so lookup and restore can share one inode.
    using PeekFn = SnapPeekInfo (*)(int,const std::string&);
    // hash: token bytes -> 40-hex-char key (production: SHA1; the offline
    // gate may use any deterministic hex). Injected so this header carries
    // no platform crypto dependency and remains portable to Linux/CUDA hosts.
    using HashFn = void (*)(const uint32_t* tokens, uint32_t count, char out_hex[41]);

    explicit DiskSnapshotStore(PeekFn peek, HashFn hash) : peek_(peek), hash_(hash) {}

    // tag = artifact/KV identity prefix baked into every filename, so one
    // directory shared by different artifacts or fp16/turbo3 servers never
    // cross-matches or overwrites incompatible snapshots; the deep header
    // identity check at load stays underneath.
    void init(std::string dir, uint64_t max_bytes, std::string tag, bool spine_pin=false) {
        std::lock_guard<std::mutex> lk(m_);
        dir_=std::move(dir); max_bytes_=max_bytes; tag_=std::move(tag); spine_pin_=spine_pin;
        reclaim_stale_temporaries_locked();
    }
    bool enabled() const { return !dir_.empty(); }

    struct Match {
        std::string path;
        SnapPeekInfo info;
        int fd=-1;
        Match() = default;
        Match(const Match&) = delete;
        Match& operator=(const Match&) = delete;
        Match(Match&& other) noexcept
            : path(std::move(other.path)),info(std::move(other.info)),fd(other.fd) {
            other.fd=-1;
        }
        Match& operator=(Match&& other) noexcept {
            if(this!=&other) {
                if(fd>=0) ::close(fd);
                path=std::move(other.path);
                info=std::move(other.info);
                fd=other.fd;
                other.fd=-1;
            }
            return *this;
        }
        ~Match() { if(fd>=0) ::close(fd); }
        explicit operator bool() const { return fd>=0; }
    };

    // Longest stored token prefix of `prompt`. Full-length matches whose
    // logits are stale (mid-prefill saves) are skipped: state would be
    // exact but no pending token could be derived.
    Match best_match(const std::vector<uint32_t>& prompt) {
        Match best;
        if(!enabled()) return best;
        std::lock_guard<std::mutex> lk(m_);
        prune_missing_cached_paths_locked();
        uint32_t best_len=0;
        std::error_code ec;
        for(const auto& e:std::filesystem::directory_iterator(dir_,ec)) {
            if(e.path().extension()!=".q27snap") continue;
            const std::string filename=e.path().filename().string();
            if(filename.rfind(tag_,0)!=0 ||
               filename.size()!=tag_.size()+40+std::string(".q27snap").size()) continue;
            const std::string pstr=e.path().string();
            const int fd=::open(pstr.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
            if(fd<0) continue;
            struct stat st{};
            if(::fstat(fd,&st)!=0 || !S_ISREG(st.st_mode)) {
                ::close(fd);
                continue;
            }
            SnapPeekInfo info;
            if(!peek_fd_current(pstr,fd,st,info)) {
                ::close(fd);
                continue;
            }
            // Saves record exactly the encoded prefix; anything else is not
            // resumable by token matching.
            if(info.position!=info.tokens.size() || info.tokens.empty() ||
               info.tokens.size()>prompt.size() ||
               (info.tokens.size()==prompt.size() && !info.logits_resident) ||
               info.tokens.size()<=best_len ||
               !std::equal(info.tokens.begin(),info.tokens.end(),prompt.begin())) {
                ::close(fd);
                continue;
            }
            if(best.fd>=0) ::close(best.fd);
            best.path=pstr;
            best.info=std::move(info);
            best.fd=fd;
            best_len=(uint32_t)best.info.tokens.size();
        }
        if(best) {
            const timespec now[2]={{0,UTIME_NOW},{0,UTIME_NOW}};
            (void)::futimens(best.fd,now);
        }
        return best;
    }

    std::string path_for(const uint32_t* tokens,uint32_t count) {
        char hex[41];
        hash_(tokens,count,hex); hex[40]='\0';
        return dir_+"/"+tag_+hex+".q27snap";
    }

    // True only when the exact token-key pathname currently contains a
    // logits-resident snapshot with matching position/tokens. Callers use
    // this immediately before a lease-serialized stale-logits save so an
    // older save decision cannot downgrade a newly installed exact prefix.
    bool exact_resident(const uint32_t* tokens,uint32_t count) {
        if(!enabled() || !tokens || !count) return false;
        char hex[41]; hash_(tokens,count,hex); hex[40]='\0';
        const std::string path=dir_+"/"+tag_+hex+".q27snap";
        std::lock_guard<std::mutex> lk(m_);
        const int fd=::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
        if(fd<0) {
            meta_.erase(path);
            tokens_.erase(path);
            return false;
        }
        struct stat st{};
        SnapPeekInfo info;
        const bool current=::fstat(fd,&st)==0 && S_ISREG(st.st_mode) &&
                           peek_fd_current(path,fd,st,info);
        ::close(fd);
        return current && info.logits_resident && info.position==count &&
               info.tokens.size()==count &&
               std::equal(info.tokens.begin(),info.tokens.end(),tokens);
    }

    // Serialize the entire inspect -> save -> atomic-rename publication
    // decision for one destination. The in-process mutex covers concurrent
    // request engines; the bounded lock-file stripe covers other server
    // processes sharing the cache directory. Lock files are intentionally
    // persistent: unlinking one while another process waits on its inode can
    // split future publishers across two different locks.
    template<class PublishFn>
    bool with_publication_lock(const std::string& path,PublishFn&& publish) {
        uint64_t hash=1469598103934665603ull;
        for(const unsigned char c:path) {
            hash^=c;
            hash*=1099511628211ull;
        }
        const std::string lock_path=dir_+"/.q27-publish-"+
            std::to_string(hash%64)+".lock";
        std::shared_ptr<std::mutex> local;
        {
            std::lock_guard<std::mutex> lk(m_);
            auto& slot=publication_locks_[lock_path];
            if(!slot) slot=std::make_shared<std::mutex>();
            local=slot;
        }
        std::lock_guard<std::mutex> local_lk(*local);
        const int fd=::open(lock_path.c_str(),O_RDWR|O_CREAT|O_CLOEXEC,0666);
        if(fd<0) throw std::runtime_error("cannot open snapshot publication lock: "+lock_path);
        struct stat st{};
        if(::fstat(fd,&st)!=0 || !S_ISREG(st.st_mode)) {
            ::close(fd);
            throw std::runtime_error("invalid snapshot publication lock: "+lock_path);
        }
        if(::flock(fd,LOCK_EX)!=0) {
            ::close(fd);
            throw std::runtime_error("cannot acquire snapshot publication lock: "+lock_path);
        }
        try {
            const bool result=publish();
            (void)::flock(fd,LOCK_UN);
            ::close(fd);
            return result;
        } catch(...) {
            (void)::flock(fd,LOCK_UN);
            ::close(fd);
            throw;
        }
    }

    // A candidate that passed the shallow header peek but failed the engine's
    // full structural/configuration load is not usable. The descriptor pins
    // the inode that actually failed. Serialize with publishers and unlink
    // only if the pathname still names that inode; a repaired atomic
    // replacement must survive an old reader finishing late.
    void reject(const std::string& path,int rejected_fd) {
        struct stat rejected{};
        if(rejected_fd<0 || ::fstat(rejected_fd,&rejected)!=0) return;
        with_publication_lock(path,[&] {
            struct stat current{};
            bool erase_cache=false;
            if(::lstat(path.c_str(),&current)!=0) {
                erase_cache=errno==ENOENT;
            } else if(S_ISREG(current.st_mode) && current.st_dev==rejected.st_dev &&
                      current.st_ino==rejected.st_ino) {
                erase_cache=(::unlink(path.c_str())==0 || errno==ENOENT);
            }
            if(erase_cache) {
                std::lock_guard<std::mutex> lk(m_);
                meta_.erase(path);
                tokens_.erase(path);
            }
            return erase_cache;
        });
    }

    // Register tokens only after save_state has returned successfully. Failed
    // attempts may never publish a file and must not leave unbounded entries
    // that directory-based eviction can never discover.
    void published(const std::string& path,const uint32_t* tokens,uint32_t count) {
        std::lock_guard<std::mutex> lk(m_);
        prune_missing_cached_paths_locked();
        tokens_[path]=std::vector<uint32_t>(tokens,tokens+count);
        meta_.erase(path);
    }

    // save_state can throw before rename or after rename while syncing the
    // directory. In either case discard memory-only bookkeeping; if a file did
    // land, the next lookup/eviction can reconstruct its tokens by peeking it.
    void save_failed(const std::string& path) {
        std::lock_guard<std::mutex> lk(m_);
        meta_.erase(path);
        tokens_.erase(path);
    }

    std::pair<size_t,size_t> cache_entry_counts() {
        std::lock_guard<std::mutex> lk(m_);
        return {meta_.size(),tokens_.size()};
    }

    // Budget enforcement until the directory fits. The just-written file is
    // deletable too — the budget is a hard cap, and the gate asserts the
    // total never exceeds it. Returns {files, bytes} removed so the --trace
    // stream can record the eviction decision.
    //
    // T1 (2026-07-18-t1-snapshot-eviction-classes.md): with spine_pin_, a
    // snapshot that is a strict token-prefix of another stored snapshot (a
    // growing conversation's spine) is evicted only AFTER every non-spine
    // (leaf) snapshot. Recency is already persisted by best_match()'s
    // touch-on-hit mtime, so within each class the order stays mtime-oldest
    // first — the pin only reorders ACROSS the spine/leaf classes. Eviction
    // only ever deletes files: a wrong victim costs a re-prefill, never a
    // wrong result (the restore path is untouched).
    std::pair<size_t,uint64_t> evict_past_budget() {
        if(!enabled() || !max_bytes_) return {0,0};
        std::pair<size_t,uint64_t> result{0,0};
        with_eviction_lock([&] {
            result=evict_past_budget_locked();
            return true;
        });
        return result;
    }

    std::pair<size_t,uint64_t> evict_past_budget_locked() {
        if(!enabled() || !max_bytes_) return {0,0};
        std::unique_lock<std::mutex> lk(m_);
        reclaim_stale_temporaries_locked();
        prune_missing_cached_paths_locked();
        // mtime folds to an opaque ordering key for the shared ordering
        // function (snapshot_evict.h — header-only so the T1 ordering is
        // unit-testable offline without a resident model).
        std::vector<q27::EvictCandidate> files; uint64_t total=0;
        std::map<std::string,std::pair<uint64_t,uint64_t>> identities;
        std::error_code ec;
        for(const auto& e:std::filesystem::directory_iterator(dir_,ec)) {
            const std::string path=e.path().string();
            const std::string name=e.path().filename().string();
            struct stat st{};
            if(e.path().extension()!=".q27snap" || name.rfind(tag_,0)!=0 ||
               ::lstat(path.c_str(),&st)!=0 || !S_ISREG(st.st_mode) || st.st_size<0)
                continue;
            std::error_code metadata_ec;
            const auto write_time=e.last_write_time(metadata_ec);
            if(metadata_ec) continue;
            const auto mt=std::chrono::duration_cast<std::chrono::nanoseconds>(
                              write_time.time_since_epoch()).count();
            const uint64_t size=(uint64_t)st.st_size;
            files.push_back({path,size,(uint64_t)mt,false});
            identities[path]={(uint64_t)st.st_dev,(uint64_t)st.st_ino};
            total+=size;
        }
        if(spine_pin_) {
            // Lazily read each file's stored token ids once (small next to
            // the state blobs), then mark spine = strict prefix of another.
            for(const auto& f:files)
                if(tokens_.find(f.path)==tokens_.end()) {
                    const int fd=::open(f.path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
                    if(fd>=0) {
                        try { tokens_[f.path]=peek_(fd,f.path).tokens; }
                        catch(...) { /* unreadable: stays a leaf, evictable */ }
                        ::close(fd);
                    }
                }
            q27::mark_spine(files,tokens_);
        }
        q27::eviction_order(files,spine_pin_);
        lk.unlock();
        size_t n=0; uint64_t freed=0; bool rescan=false;
        for(const auto& f:files) {
            if(total<=max_bytes_) break;
            bool missing=false,replaced=false,removed=false;
            uint64_t current_size=0;
            with_publication_lock(f.path,[&] {
                struct stat current{};
                if(::lstat(f.path.c_str(),&current)!=0) {
                    missing=errno==ENOENT;
                    return false;
                }
                current_size=current.st_size<0?0:(uint64_t)current.st_size;
                const auto identity=identities.find(f.path);
                replaced=identity==identities.end() || !S_ISREG(current.st_mode) ||
                    identity->second.first!=(uint64_t)current.st_dev ||
                    identity->second.second!=(uint64_t)current.st_ino;
                if(replaced) return false;
                removed=(::unlink(f.path.c_str())==0 || errno==ENOENT);
                return removed;
            });
            if(missing) {
                total=total>f.size?total-f.size:0;
                { std::lock_guard<std::mutex> cache_lk(m_);
                  tokens_.erase(f.path); meta_.erase(f.path); }
                continue;
            }
            if(replaced) {
                total=total>f.size?total-f.size:0;
                total+=current_size;
                { std::lock_guard<std::mutex> cache_lk(m_);
                  tokens_.erase(f.path); meta_.erase(f.path); }
                rescan=true;
                continue;
            }
            if(removed) {
                total=total>f.size?total-f.size:0;
                n++; freed+=current_size;
                { std::lock_guard<std::mutex> cache_lk(m_);
                  tokens_.erase(f.path);
                  meta_.erase(f.path); }
                if(f.spine) evicted_spine++; else evicted_leaf++;
            }
        }
        if(total>max_bytes_ && (rescan || n>0)) {
            const auto retry=evict_past_budget_locked();
            n+=retry.first;
            freed+=retry.second;
        }
        return {n,freed};
    }

    std::atomic<uint64_t> hits{0}, saves{0}, evicted_spine{0}, evicted_leaf{0};
  private:
    template<class EvictFn>
    bool with_eviction_lock(EvictFn&& evict) {
        std::lock_guard<std::mutex> local_lk(eviction_mutex_);
        uint64_t hash=1469598103934665603ull;
        for(const unsigned char c:tag_) {
            hash^=c;
            hash*=1099511628211ull;
        }
        const std::string lock_path=dir_+"/.q27-evict-"+std::to_string(hash)+".lock";
        const int fd=::open(lock_path.c_str(),O_RDWR|O_CREAT|O_CLOEXEC,0666);
        if(fd<0) throw std::runtime_error("cannot open snapshot eviction lock: "+lock_path);
        struct stat st{};
        if(::fstat(fd,&st)!=0 || !S_ISREG(st.st_mode)) {
            ::close(fd);
            throw std::runtime_error("invalid snapshot eviction lock: "+lock_path);
        }
        if(::flock(fd,LOCK_EX)!=0) {
            ::close(fd);
            throw std::runtime_error("cannot acquire snapshot eviction lock: "+lock_path);
        }
        try {
            const bool result=evict();
            (void)::flock(fd,LOCK_UN);
            ::close(fd);
            return result;
        } catch(...) {
            (void)::flock(fd,LOCK_UN);
            ::close(fd);
            throw;
        }
    }

    void prune_missing_cached_paths_locked() {
        auto prune=[](auto& cache) {
            for(auto it=cache.begin();it!=cache.end();) {
                struct stat st{};
                if(::lstat(it->first.c_str(),&st)!=0 || !S_ISREG(st.st_mode))
                    it=cache.erase(it);
                else ++it;
            }
        };
        prune(meta_);
        prune(tokens_);
    }

    static bool is_hex_key(const std::string& value,size_t begin,size_t count) {
        if(begin+count>value.size()) return false;
        for(size_t i=begin;i<begin+count;i++) {
            const unsigned char c=(unsigned char)value[i];
            if(!((c>='0' && c<='9') || (c>='a' && c<='f') || (c>='A' && c<='F')))
                return false;
        }
        return true;
    }

    bool owns_temporary_name(const std::string& name) const {
        static constexpr char marker[]=".q27snap.tmp.";
        static constexpr size_t marker_len=sizeof(marker)-1;
        static constexpr size_t suffix_len=6;
        if(name.rfind(tag_,0)!=0 || name.size()<tag_.size()+40+marker_len+suffix_len)
            return false;
        const size_t marker_pos=name.size()-marker_len-suffix_len;
        if(name.compare(marker_pos,marker_len,marker)!=0) return false;
        for(size_t i=name.size()-suffix_len;i<name.size();i++) {
            const unsigned char c=(unsigned char)name[i];
            if(!((c>='0' && c<='9') || (c>='a' && c<='z') || (c>='A' && c<='Z')))
                return false;
        }
        const size_t key_begin=tag_.size();
        const size_t key_len=marker_pos-key_begin;
        return key_len==40 && is_hex_key(name,key_begin,40);
    }

    void reclaim_stale_temporaries_locked() {
        if(dir_.empty()) return;
        std::error_code ec;
        const auto stale_before=std::filesystem::file_time_type::clock::now()-
                                std::chrono::seconds(60);
        for(const auto& e:std::filesystem::directory_iterator(dir_,ec)) {
            const std::string name=e.path().filename().string();
            if(!owns_temporary_name(name)) continue;
            std::error_code time_ec;
            const auto write_time=e.last_write_time(time_ec);
            if(time_ec || write_time>stale_before) continue;
            const std::string path=e.path().string();
            const int fd=::open(path.c_str(),O_RDWR|O_CLOEXEC);
            if(fd<0) continue;
            struct stat opened{},current{};
            const bool removable=::fstat(fd,&opened)==0 && S_ISREG(opened.st_mode) &&
                ::flock(fd,LOCK_EX|LOCK_NB)==0 && ::lstat(path.c_str(),&current)==0 &&
                opened.st_dev==current.st_dev && opened.st_ino==current.st_ino;
            if(removable) (void)::unlink(path.c_str());
            ::close(fd);
        }
    }


    struct CachedMeta {
        SnapPeekInfo info;
        uint64_t device=0;
        uint64_t inode=0;
    };
    bool peek_fd_current(const std::string& path,int fd,const struct stat& st,
                         SnapPeekInfo& info) {
        auto cached=meta_.find(path);
        if(cached!=meta_.end() && cached->second.device==(uint64_t)st.st_dev &&
           cached->second.inode==(uint64_t)st.st_ino) {
            info=cached->second.info;
            return true;
        }
        try { info=peek_(fd,path); }
        catch(...) { return false; }
        meta_[path]={info,(uint64_t)st.st_dev,(uint64_t)st.st_ino};
        return true;
    }
    PeekFn peek_;
    HashFn hash_;
    std::string dir_; uint64_t max_bytes_=0; std::string tag_; std::mutex m_,eviction_mutex_;
    bool spine_pin_=false;
    std::map<std::string,std::vector<uint32_t>> tokens_;   // path -> token ids (eviction spine check)
    // Atomic replacement changes inode, forcing a fresh peek even when a
    // different server process publishes the same token-key pathname.
    std::map<std::string,CachedMeta> meta_;
    std::map<std::string,std::shared_ptr<std::mutex>> publication_locks_;
};
