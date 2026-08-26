// Is it DMA reading lines that are still DIRTY IN CPU CACHE?
// The driver's pageable path memcpy()s each chunk into a pinned ring and DMAs
// it at once, so the DMA read is served from the CCD's cache via a fabric
// probe, not from DRAM. Pinned user buffers are cold: served from DRAM. Three
// arms with MY OWN 64 MB pinned ring, so the driver's pages are out of it:
//   HOT   memcpy chunk -> ring (cached stores)        -> DMA immediately
//   COLD  non-temporal stores -> ring (bypass cache) -> sfence -> DMA
//   DIRECT pinned user buffer, no staging (control, expected clean)
// HOT flips and COLD does not  -> cache-served DMA reads, i.e. the CCD<->IOD
//                                 fabric probe path
// both flip                     -> the ring's pages / the memcpy-then-DMA
//                                 ordering, not cache residency
// neither flips (DIRECT too)    -> the driver's OWN ring pages are the weak ones
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <immintrin.h>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(3);} }while(0)
__global__ void k_sum(const unsigned long long* p,size_t n,unsigned long long* out){ unsigned long long s=0; for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x) s+=p[i]; for(int o=16;o>0;o>>=1) s+=__shfl_down_sync(0xffffffff,s,o); if((threadIdx.x&31)==0) atomicAdd(out,s); }
static unsigned long long dsum(const void* d,size_t n64,unsigned long long* o){ CK(cudaMemset(o,0,8)); k_sum<<<4096,256>>>((const unsigned long long*)d,n64,o); CK(cudaGetLastError()); unsigned long long h; CK(cudaMemcpy(&h,o,8,cudaMemcpyDeviceToHost)); return h; }
static unsigned long long hsum(const uint64_t* p,size_t n64){ unsigned long long s=0; for(size_t i=0;i<n64;i++) s+=p[i]; return s; }
static void nt_copy(void* dst,const void* src,size_t n){ // 32B non-temporal stores, bypass cache
    const char* s=(const char*)src; char* d=(char*)dst; size_t i=0;
    for(;i+32<=n;i+=32){ __m256i v=_mm256_loadu_si256((const __m256i*)(s+i)); _mm256_stream_si256((__m256i*)(d+i),v); }
    for(;i<n;i++) d[i]=s[i]; _mm_sfence(); }
int main(int argc,char** argv){
    int fd=open(argv[1],O_RDONLY); struct stat st; fstat(fd,&st); size_t n=(size_t)st.st_size&~7ull, n64=n/8;
    uint64_t* src=(uint64_t*)aligned_alloc(4096,n); size_t got=0; while(got<n){ssize_t r=pread(fd,(char*)src+got,n-got,got); if(r<=0)exit(1); got+=r;} close(fd);
    uint64_t* rb=(uint64_t*)aligned_alloc(4096,n);
    const unsigned long long ref=hsum(src,n64);
    const size_t RING=64u<<20; void* ring; CK(cudaHostAlloc(&ring,RING,cudaHostAllocDefault));
    void* d; CK(cudaMalloc(&d,n)); unsigned long long* o; CK(cudaMalloc(&o,8));
    int rounds=atoi(argv[2]); int ev[3]={0,0,0}; const char* nm[3]={"HOT-staged","COLD-staged","DIRECT-pinned"};
    for(int r=0;r<rounds;r++){
        for(int arm=0;arm<3;arm++){
            if(arm<2){
                for(size_t off=0;off<n;off+=RING){ size_t len=(n-off<RING)?(n-off):RING;
                    if(arm==0) memcpy(ring,(const char*)src+off,len); else nt_copy(ring,(const char*)src+off,len);
                    CK(cudaMemcpy((char*)d+off,ring,len,cudaMemcpyHostToDevice)); }
            } else {
                CK(cudaHostRegister(src,n,cudaHostRegisterDefault));
                CK(cudaMemcpy(d,src,n,cudaMemcpyHostToDevice)); CK(cudaDeviceSynchronize());
                CK(cudaHostUnregister(src));
            }
            CK(cudaDeviceSynchronize());
            unsigned long long ws=dsum(d,n64,o);
            if(ws!=ref){ ev[arm]++; printf("%4d %-14s FLIP delta=%+lld recompute=%s\n",r,nm[arm],(long long)(ws-ref),dsum(d,n64,o)==ws?"stable":"CHANGED");
                CK(cudaMemcpy(rb,d,n,cudaMemcpyDeviceToHost)); CK(cudaDeviceSynchronize());
                if(hsum(rb,n64)==ws){ int shown=0; for(size_t i=0;i<n64&&shown<4;i++) if(rb[i]!=src[i]){ uint64_t x=rb[i]^src[i]; for(int b=0;b<64;b++) if((x>>b)&1ull) printf("   offset 0x%012zx dword %d bit %2d %s\n",i*8,b/32,b%32,((rb[i]>>b)&1ull)?"0->1":"1->0"); shown++; } }
                else printf("   (readback disagreed with device sum; offsets withheld)\n");
            } else printf("%4d %-14s CLEAN\n",r,nm[arm]);
            fflush(stdout);
        }
        if((r+1)%10==0){ printf("--- after %d rounds: ",r+1); for(int a=0;a<3;a++) printf("%s=%d ",nm[a],ev[a]); printf("\n"); fflush(stdout); }
    }
    printf("done %d rounds: ",rounds); for(int a=0;a<3;a++) printf("%s=%d/%d ",nm[a],ev[a],rounds); printf("\nHOT_DONE\n");
}
