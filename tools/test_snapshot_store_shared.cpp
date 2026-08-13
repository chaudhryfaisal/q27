#include "disk_snapshot_store.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <poll.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;

static SnapPeekInfo fixture_peek(int fd,const std::string&) {
    uint8_t resident=0;
    uint32_t count=0;
    if(::pread(fd,&resident,sizeof resident,0)!=(ssize_t)sizeof resident ||
       ::pread(fd,&count,sizeof count,sizeof resident)!=(ssize_t)sizeof count)
        throw std::runtime_error("invalid snapshot fixture");
    SnapPeekInfo info;
    info.logits_resident=resident!=0;
    info.position=count;
    info.tokens.resize(count);
    if(count && ::pread(fd,info.tokens.data(),(size_t)count*sizeof(uint32_t),
                        sizeof resident+sizeof count)!=(ssize_t)((size_t)count*sizeof(uint32_t)))
        throw std::runtime_error("invalid snapshot fixture");
    return info;
}

static void fixture_hash(const uint32_t* tokens,uint32_t count,char out[41]) {
    uint64_t hash=1469598103934665603ull;
    for(uint32_t i=0;i<count;i++) {
        hash^=tokens[i];
        hash*=1099511628211ull;
    }
    std::snprintf(out,41,"%040llx",(unsigned long long)hash);
}

static uint64_t publication_stripe(const std::string& path) {
    uint64_t hash=1469598103934665603ull;
    for(const unsigned char c:path) {
        hash^=c;
        hash*=1099511628211ull;
    }
    return hash%64;
}

static void write_file(const std::string& path,size_t bytes,int age_seconds) {
    std::ofstream out(path,std::ios::binary);
    std::string payload(bytes,'x');
    out.write(payload.data(),(std::streamsize)payload.size());
    out.close();
    std::error_code ec;
    fs::last_write_time(path,fs::file_time_type::clock::now()-
        std::chrono::seconds(age_seconds),ec);
}

static void publish_fixture(const std::string& path,
                            const std::vector<uint32_t>& tokens,bool resident) {
    const std::string tmp=path+".next";
    std::ofstream out(tmp,std::ios::binary);
    const uint8_t flag=resident?1:0;
    const uint32_t count=(uint32_t)tokens.size();
    out.write(reinterpret_cast<const char*>(&flag),sizeof flag);
    out.write(reinterpret_cast<const char*>(&count),sizeof count);
    if(count) out.write(reinterpret_cast<const char*>(tokens.data()),
                        (std::streamsize)count*sizeof(uint32_t));
    out.close();
    fs::rename(tmp,path);
}

static bool publication_lock_same_process(DiskSnapshotStore& store,
                                          const std::string& path) {
    std::atomic<bool> entered{false};
    bool blocked=false;
    std::thread contender;
    const bool outer=store.with_publication_lock(path,[&] {
        contender=std::thread([&] {
            store.with_publication_lock(path,[&] {
                entered.store(true,std::memory_order_release);
                return true;
            });
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        blocked=!entered.load(std::memory_order_acquire);
        return true;
    });
    contender.join();
    return outer && blocked && entered.load(std::memory_order_acquire);
}

static bool publication_lock_cross_process(DiskSnapshotStore& store,
                                           const std::string& path) {
    int start_pipe[2],entered_pipe[2];
    if(::pipe(start_pipe)!=0 || ::pipe(entered_pipe)!=0) return false;
    const pid_t child=::fork();
    if(child<0) return false;
    if(child==0) {
        ::close(start_pipe[1]);
        ::close(entered_pipe[0]);
        char byte=0;
        if(::read(start_pipe[0],&byte,1)!=1) _exit(2);
        const bool ok=store.with_publication_lock(path,[&] {
            byte=1;
            return ::write(entered_pipe[1],&byte,1)==1;
        });
        _exit(ok?0:3);
    }
    ::close(start_pipe[0]);
    ::close(entered_pipe[1]);
    bool blocked=false;
    const bool outer=store.with_publication_lock(path,[&] {
        char byte=1;
        if(::write(start_pipe[1],&byte,1)!=1) return false;
        pollfd wait_for_child{entered_pipe[0],POLLIN,0};
        blocked=::poll(&wait_for_child,1,150)==0;
        return true;
    });
    pollfd wait_for_child{entered_pipe[0],POLLIN,0};
    const bool entered=::poll(&wait_for_child,1,2000)==1;
    char byte=0;
    if(entered) (void)::read(entered_pipe[0],&byte,1);
    int status=0;
    const bool reaped=::waitpid(child,&status,0)==child;
    ::close(start_pipe[1]);
    ::close(entered_pipe[0]);
    return outer && blocked && entered && byte==1 && reaped &&
           WIFEXITED(status) && WEXITSTATUS(status)==0;
}

int main() {
    const std::string dir=(::getenv("TMPDIR")?::getenv("TMPDIR"):"/tmp")+
        std::string("/q27-snapshot-shared-")+std::to_string((long long)getpid());
    std::error_code ec;
    fs::remove_all(dir,ec);
    fs::create_directories(dir,ec);
    if(ec) {
        std::fprintf(stderr,"cannot create fixture directory: %s\n",ec.message().c_str());
        return 1;
    }

    const std::string owned_temp=dir+"/own-"+std::string(40,'a')+
        ".q27snap.tmp.A1b2C3";
    const std::string unrelated_temp=dir+"/notes.q27snap.tmp.backup";
    const std::string malformed_owned=dir+"/own-"+std::string(40,'a')+
        ".q27snap.tmp.backups";
    const std::string foreign_temp=dir+"/other-"+std::string(40,'a')+
        ".q27snap.tmp.A1b2C3";
    write_file(owned_temp,1,100);
    write_file(unrelated_temp,1,100);
    write_file(malformed_owned,1,100);
    write_file(foreign_temp,1,100);

    const std::string own=dir+"/own-current.q27snap";
    const std::string foreign=dir+"/other-older.q27snap";
    write_file(own,100,10);
    write_file(foreign,200,100);

    DiskSnapshotStore store(&fixture_peek,&fixture_hash);
    store.init(dir,100,"own-",false);
    const auto evicted=store.evict_past_budget();
    const bool owned_removed=!fs::exists(owned_temp,ec);
    const bool unrelated_kept=fs::exists(unrelated_temp,ec);
    const bool malformed_kept=fs::exists(malformed_owned,ec);
    const bool foreign_kept=fs::exists(foreign_temp,ec);
    const bool stale_cleanup_ok=owned_removed && unrelated_kept && malformed_kept &&
        foreign_kept;
    if(!stale_cleanup_ok) {
        std::fprintf(stderr,"snapshot temporary cleanup crossed ownership boundary "
            "(owned=%d unrelated=%d malformed=%d foreign=%d)\n",
            owned_removed,unrelated_kept,malformed_kept,foreign_kept);
        fs::remove_all(dir,ec);
        return 1;
    }
    const bool eviction_ok=evicted.first==0 && fs::exists(own,ec) && fs::exists(foreign,ec);
    if(!eviction_ok) {
        std::fprintf(stderr,"snapshot eviction crossed artifact tag boundary\n");
        fs::remove_all(dir,ec);
        return 1;
    }

    fs::remove(own,ec);
    fs::remove(foreign,ec);
    const uint64_t eviction_stripe=publication_stripe(dir+"/.race-eviction");
    std::string colliding_path;
    for(int i=0;colliding_path.empty();i++) {
        const std::string candidate=dir+"/race-collision-"+
            std::to_string(i)+".q27snap";
        if(publication_stripe(candidate)==eviction_stripe) colliding_path=candidate;
    }
    write_file(colliding_path,100,30);
    write_file(dir+"/race-newer-a.q27snap",100,20);
    write_file(dir+"/race-newer-b.q27snap",100,10);
    DiskSnapshotStore evictor_a(&fixture_peek,&fixture_hash);
    DiskSnapshotStore evictor_b(&fixture_peek,&fixture_hash);
    evictor_a.init(dir,150,"race-",false);
    evictor_b.init(dir,150,"race-",false);
    std::thread eviction_a([&] { (void)evictor_a.evict_past_budget(); });
    std::thread eviction_b([&] { (void)evictor_b.evict_past_budget(); });
    eviction_a.join();
    eviction_b.join();
    size_t remaining_files=0;
    uint64_t remaining_bytes=0;
    for(const auto& entry:fs::directory_iterator(dir,ec)) {
        if(entry.path().extension()==".q27snap" &&
           entry.path().filename().string().rfind("race-",0)==0) {
            remaining_files++;
            remaining_bytes+=(uint64_t)entry.file_size();
        }
    }
    if(remaining_files!=1 || remaining_bytes!=100) {
        std::fprintf(stderr,"concurrent snapshot eviction over-deleted files\n");
        fs::remove_all(dir,ec);
        return 1;
    }

    fs::remove_all(dir,ec);
    fs::create_directories(dir,ec);
    DiskSnapshotStore shared(&fixture_peek,&fixture_hash);
    shared.init(dir,0,"own-",false);
    const std::vector<uint32_t> tokens={1,2,3};
    const std::string path=shared.path_for(tokens.data(),(uint32_t)tokens.size());
    publish_fixture(path,tokens,false);
    const bool saw_stale=!shared.exact_resident(tokens.data(),(uint32_t)tokens.size());
    publish_fixture(path,tokens,true);
    const bool saw_external_resident=shared.exact_resident(tokens.data(),(uint32_t)tokens.size());
    publish_fixture(path,tokens,false);
    auto stale_match=shared.best_match(tokens);
    const bool rejected_external_stale=!stale_match;
    const bool peer_removed_meta=fs::remove(path,ec);
    (void)shared.best_match(tokens);
    const auto cached_after_meta_remove=shared.cache_entry_counts();
    publish_fixture(path,tokens,false);
    shared.published(path,tokens.data(),(uint32_t)tokens.size());
    const bool peer_removed_tokens=fs::remove(path,ec);
    (void)shared.best_match(tokens);
    const auto cached_after_token_remove=shared.cache_entry_counts();
    const bool pruned_peer_metadata=peer_removed_meta && peer_removed_tokens &&
        cached_after_meta_remove.first==0 && cached_after_meta_remove.second==0 &&
        cached_after_token_remove.first==0 && cached_after_token_remove.second==0;
    const bool same_process_lock=publication_lock_same_process(shared,path);
    const bool cross_process_lock=publication_lock_cross_process(shared,path);
    publish_fixture(path,tokens,false);
    const int old_fd=::open(path.c_str(),O_RDONLY|O_CLOEXEC);
    publish_fixture(path,tokens,true);
    shared.reject(path,old_fd);
    if(old_fd>=0) ::close(old_fd);
    const bool replacement_survived=fs::exists(path,ec) &&
        shared.exact_resident(tokens.data(),(uint32_t)tokens.size());
    const int current_fd=::open(path.c_str(),O_RDONLY|O_CLOEXEC);
    shared.reject(path,current_fd);
    if(current_fd>=0) ::close(current_fd);
    const bool failed_current_removed=!fs::exists(path,ec);
    publish_fixture(path,tokens,true);
    const std::string real_path=path+".real";
    fs::rename(path,real_path,ec);
    const bool rename_ok=!ec;
    const bool symlink_created=rename_ok && ::symlink(real_path.c_str(),path.c_str())==0;
    auto symlink_match=shared.best_match(tokens);
    const bool symlink_rejected=symlink_created && !symlink_match;
    (void)::unlink(path.c_str());
    if(rename_ok) fs::rename(real_path,path,ec);
    DiskSnapshotStore peer(&fixture_peek,&fixture_hash);
    peer.init(dir,0,"own-",false);
    auto peer_match=peer.best_match(tokens);
    const bool shared_across_instances=peer_match &&
        peer_match.path==path && peer_match.info.tokens.size()==tokens.size() &&
        peer.exact_resident(tokens.data(),(uint32_t)tokens.size());
    peer_match=DiskSnapshotStore::Match{};
    fs::remove_all(dir,ec);
    if(!saw_stale || !saw_external_resident || !rejected_external_stale ||
       !pruned_peer_metadata || !same_process_lock || !cross_process_lock ||
       !replacement_survived || !failed_current_removed || !symlink_rejected ||
       !shared_across_instances) {
        std::fprintf(stderr,"snapshot metadata, shared reuse, or publication lock failed\n");
        return 1;
    }
    std::puts("snapshot store metadata, shared reuse, and publication locking: PASS");
    return 0;
}
