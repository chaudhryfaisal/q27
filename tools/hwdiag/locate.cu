// Corruption LOCATOR. Every prior instrument reported only a checksum delta.
// This one finds the flipped words: upload, device-checksum, and on mismatch
// read back and diff against the source. Each event yields (offset, dword
// bit, direction, DMA mode), which is the evidence that separates
//   storage fault  -> same offset recurs
//   data-path fault-> random offsets, fixed bit position
//   per-transaction-> offsets aligned to a DMA chunk boundary
// Alternates PINNED (direct DMA from the source pages) and PAGEABLE (driver
// stages through its own pinned buffer, CPU memcpy first) on the SAME
// physical host pages, so the only variable is the DMA path.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(3);} }while(0)

__global__ void k_sum(const unsigned long long* p, size_t n, unsigned long long* out){
    unsigned long long s=0;
    for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x) s+=p[i];
    for(int o=16;o>0;o>>=1) s+=__shfl_down_sync(0xffffffff,s,o);
    if((threadIdx.x&31)==0) atomicAdd(out,s);
}
static unsigned long long dsum(const void* d, size_t n64, unsigned long long* dout){
    CK(cudaMemset(dout,0,8));
    k_sum<<<4096,256>>>((const unsigned long long*)d,n64,dout);
    CK(cudaGetLastError());
    unsigned long long h=0; CK(cudaMemcpy(&h,dout,8,cudaMemcpyDeviceToHost)); return h;
}
static unsigned long long hsum(const uint64_t* p, size_t n64){ unsigned long long s=0; for(size_t i=0;i<n64;i++) s+=p[i]; return s; }

int main(int argc,char** argv){
    if(argc<3){fprintf(stderr,"usage: %s file iters\n",argv[0]);return 2;}
    bool use_mmap=false, thp=false; size_t chunk=0;
    for(int a=3;a<argc;a++){ if(!strcmp(argv[a],"--mmap")) use_mmap=true; else if(!strcmp(argv[a],"--thp")) thp=true; else if(!strcmp(argv[a],"--chunk")&&a+1<argc) chunk=(size_t)atoll(argv[++a])<<20; }
    int fd=open(argv[1],O_RDONLY); if(fd<0){perror("open");return 1;}
    struct stat st; fstat(fd,&st); size_t n=(size_t)st.st_size & ~7ull; size_t n64=n/8;
    uint64_t* src=nullptr;
    if(use_mmap){
        // exactly q27's loader: MAP_PRIVATE read-only over the page cache
        void* m=mmap(nullptr,n,PROT_READ,MAP_PRIVATE,fd,0); if(m==MAP_FAILED){perror("mmap");return 1;}
        src=(uint64_t*)m;
    } else {
        src=(uint64_t*)aligned_alloc(2u<<20,n); if(!src){fprintf(stderr,"alloc\n");return 1;}
        // --thp: ask for 2 MB pages (THP is in madvise mode on this box), so the
        // anonymous source gets the same large-page IOMMU mappings a page-cache
        // large folio would. Isolates "large IOMMU mapping" from "file-backed".
        if(thp){ if(madvise(src,n,MADV_HUGEPAGE)) perror("madvise"); }
        size_t got=0; while(got<n){ ssize_t r=pread(fd,(char*)src+got,n-got,got); if(r<=0){perror("read");return 1;} got+=r; }
        close(fd);
    }
    fprintf(stderr,"source=%s%s chunk=%zu MB\n",use_mmap?"mmap(page cache)":"anonymous",thp?"+THP":"",chunk>>20);
    uint64_t* back=(uint64_t*)aligned_alloc(4096,n);
    const unsigned long long ref=hsum(src,n64);
    fprintf(stderr,"%s: %.2f GB, host ref sum %016llx\n",argv[1],n/1e9,ref);
    void* d; CK(cudaMalloc(&d,n)); unsigned long long* dout; CK(cudaMalloc(&dout,8));
    int iters=atoi(argv[2]); int events=0;
    for(int it=0;it<iters;it++){
        const bool pinned=(it&1);
        if(pinned) CK(cudaHostRegister(src,n,cudaHostRegisterDefault));
        if(chunk){ for(size_t off=0;off<n;off+=chunk){ size_t len=(n-off<chunk)?(n-off):chunk;
                       CK(cudaMemcpy((char*)d+off,(const char*)src+off,len,cudaMemcpyHostToDevice)); } }
        else CK(cudaMemcpy(d,src,n,cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());
        unsigned long long ws=dsum(d,n64,dout);
        if(ws==ref){ printf("%4d %-8s CLEAN\n",it,pinned?"pinned":"pageable"); fflush(stdout); }
        else{
            events++;
            unsigned long long ws2=dsum(d,n64,dout);
            long long delta=(long long)(ws-ref);
            printf("%4d %-8s FLIP wsum=%016llx delta=%+lld recompute_stable=%d\n",it,pinned?"pinned":"pageable",ws,delta,(int)(ws2==ws));
            // is the host source still intact? (re-sum before touching anything)
            unsigned long long h2=hsum(src,n64);
            printf("      host source re-sum %s\n",h2==ref?"UNCHANGED":"CHANGED");
            CK(cudaMemcpy(back,d,n,cudaMemcpyDeviceToHost)); CK(cudaDeviceSynchronize());
            unsigned long long bs=hsum(back,n64);
            printf("      readback sum %016llx (%s device sum)\n",bs,bs==ws?"MATCHES":"DIFFERS FROM");
            int shown=0;
            for(size_t i=0;i<n64;i++) if(back[i]!=src[i]){
                uint64_t x=back[i]^src[i];
                for(int b=0;b<64;b++) if((x>>b)&1ull){
                    const int dword=(int)(b/32), dbit=b%32;
                    const bool set=(back[i]>>b)&1ull;
                    printf("      offset 0x%012zx word %zu  bit %2d (dword %d bit %2d, byte %d)  %s   off%%4096=%4zu off%%2M=%7zu\n",
                        i*8,i,b,dword,dbit,b/8,set?"0->1":"1->0",(i*8)%4096,(i*8)%(2u<<20));
                }
                if(++shown>=16){ printf("      ... more\n"); break; }
            }
            fflush(stdout);
        }
        if(pinned) CK(cudaHostUnregister(src));
    }
    printf("done: %d iters, %d events\n",iters,events);
    return 0;
}
