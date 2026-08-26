// Does llama.cpp's model load corrupt the same way q27's does?
// Loads the GGUF exactly as llama-server would (mmap, n_gpu_layers=999), then
// reads every device tensor BACK and compares it byte-for-byte against the
// file bytes it was uploaded from. Same source type as q27 (page cache),
// same pageable per-tensor cudaMemcpy, but ONE big device buffer per backend
// instead of q27's per-tensor cudaMalloc. A flip here exonerates q27.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <utility>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
// internal API (src/llama-model.h), exported from libllama for tests/benchmarks
const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model);

static uint64_t sum64(const void* p, size_t n){ const uint64_t* w=(const uint64_t*)p; uint64_t s=0; size_t n64=n/8; for(size_t i=0;i<n64;i++) s+=w[i]; return s; }

int main(int argc, char** argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf\n",argv[0]); return 2; }
    llama_log_set([](ggml_log_level,const char*,void*){},nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if(!model){ fprintf(stderr,"load failed\n"); return 1; }
    // source bytes: gguf metadata for offsets + our own mmap of the file
    gguf_init_params gp{ /*no_alloc*/ true, /*ctx*/ nullptr };
    gguf_context* g = gguf_init_from_file(argv[1], gp);
    if(!g){ fprintf(stderr,"gguf parse failed\n"); return 1; }
    const size_t data_off = gguf_get_data_offset(g);
    int fd=open(argv[1],O_RDONLY); struct stat st; fstat(fd,&st);
    const unsigned char* file=(const unsigned char*)mmap(nullptr,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    if(file==MAP_FAILED){ perror("mmap"); return 1; }
    size_t dev_tensors=0, dev_bytes=0; int bad=0; std::vector<unsigned char> back;
    for(const auto& [name, t] : llama_internal_get_tensor_map(model)){
        if(!t->buffer || ggml_backend_buffer_is_host(t->buffer)) continue;
        const int64_t id = gguf_find_tensor(g, name.c_str()); if(id<0) continue;
        const size_t nb = ggml_nbytes(t);
        const unsigned char* src = file + data_off + gguf_get_tensor_offset(g, id);
        back.resize(nb); ggml_backend_tensor_get(t, back.data(), 0, nb);
        dev_tensors++; dev_bytes += nb;
        if(memcmp(back.data(), src, nb)==0) continue;
        bad++;
        const long long delta=(long long)(sum64(back.data(),nb)-sum64(src,nb));
        printf("FLIP tensor=%s bytes=%zu buffer=%s delta=%+lld\n", name.c_str(), nb, ggml_backend_buffer_name(t->buffer), delta);
        int shown=0;
        for(size_t i=0;i<nb;i++){ if(back[i]==src[i]) continue; unsigned x=back[i]^src[i];
            for(int b=0;b<8;b++) if((x>>b)&1) printf("   offset %zu/%zu word %zu dword-bit %zu %s %s-edge\n", i, nb, i/8, (size_t)(b+8*(i%4)), ((back[i]>>b)&1)?"0->1":"1->0", (i<64||i+64>=nb)?"AT":"not-at");
            if(++shown>=8){ printf("   ...\n"); break; } }
        // Which DIRECTION was corrupt? Read back twice more. If every readback
        // agrees with each other but not the file, VRAM holds the wrong bits:
        // the H2D upload corrupted. If a later readback matches the FILE, VRAM
        // was fine and the first D2H readback was what corrupted. This is the
        // evidence that the fault is in the DMA path rather than in storage.
        std::vector<unsigned char> r2(nb), r3(nb);
        ggml_backend_tensor_get(t, r2.data(), 0, nb); ggml_backend_tensor_get(t, r3.data(), 0, nb);
        const bool r2f = memcmp(r2.data(),src,nb)==0, r3f = memcmp(r3.data(),src,nb)==0;
        const bool r2b = memcmp(r2.data(),back.data(),nb)==0, r3b = memcmp(r3.data(),back.data(),nb)==0;
        const char* verdict = (r2f && r3f) ? "D2H_READBACK_FLIP (VRAM clean: later reads match the file)"
                            : (r2b && r3b) ? "H2D_UPLOAD_FLIP (VRAM wrong: every read agrees, none match the file)"
                            : "MIXED";
        printf("   re-read2 %s file / %s first, re-read3 %s file / %s first -> %s\n",
               r2f?"==":"!=", r2b?"==":"!=", r3f?"==":"!=", r3b?"==":"!=", verdict);
    }
    printf("%s: %zu device tensors, %.2f GB compared, %d mismatched\n", bad?"LLAMA_FLIP":"LLAMA_CLEAN", dev_tensors, dev_bytes/1e9, bad);
    llama_model_free(model); llama_backend_free();
    return bad?5:0;
}
