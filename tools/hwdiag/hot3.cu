// Which CCD's probe path? Same staging ring, same HOT memcpy->DMA path, but
// the copying thread pinned to CCD0 or CCD1 (cores given on the command line).
// Plus FLUSHED (memcpy, then clflushopt+sfence so the lines are CLEAN in
// DRAM before DMA) to separate "written recently" from "dirty in cache", and
// the DRIVER pageable path as the same-window control. Each arm streams the
// file PASSES times per round so a round carries enough bytes to see events.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <sys/stat.h>
#include <immintrin.h>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(3);} }while(0)
__global__ void k_sum(const unsigned long long* p,size_t n,unsigned long long* out){ unsigned long long s=0; for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x) s+=p[i]; for(int o=16;o>0;o>>=1) s+=__shfl_down_sync(0xffffffff,s,o); if((threadIdx.x&31)==0) atomicAdd(out,s); }
static unsigned long long dsum(const void* d,size_t n64,unsigned long long* o){ CK(cudaMemset(o,0,8)); k_sum<<<4096,256>>>((const unsigned long long*)d,n64,o); CK(cudaGetLastError()); unsigned long long h; CK(cudaMemcpy(&h,o,8,cudaMemcpyDeviceToHost)); return h; }
static unsigned long long hsum(const uint64_t* p,size_t n64){ unsigned long long s=0; for(size_t i=0;i<n64;i++) s+=p[i]; return s; }
static void pin_cores(const char* list){ cpu_set_t cs; CPU_ZERO(&cs); char buf[256]; strncpy(buf,list,255); for(char* t=strtok(buf,",");t;t=strtok(nullptr,",")){ int a,b; if(sscanf(t,"%d-%d",&a,&b)==2){ for(int c=a;c<=b;c++) CPU_SET(c,&cs);} else CPU_SET(atoi(t),&cs);} if(sched_setaffinity(0,sizeof cs,&cs)) perror("affinity"); }
static void flush_range(const void* p,size_t n){ const char* c=(const char*)p; for(size_t i=0;i<n;i+=64) _mm_clflushopt((void*)(c+i)); _mm_sfence(); }
int main(int argc,char** argv){
    if(argc<6){ fprintf(stderr,"usage: %s file rounds passes ccd0cores ccd1cores\n",argv[0]); return 2; }
    const char *ccd0=argv[4], *ccd1=argv[5]; const int passes=atoi(argv[3]);
    int fd=open(argv[1],O_RDONLY); struct stat st; fstat(fd,&st); size_t n=(size_t)st.st_size&~7ull, n64=n/8;
    uint64_t* src=(uint64_t*)aligned_alloc(4096,n); size_t got=0; while(got<n){ssize_t r=pread(fd,(char*)src+got,n-got,got); if(r<=0)exit(1); got+=r;} close(fd);
    uint64_t* rb=(uint64_t*)aligned_alloc(4096,n);
    const unsigned long long ref=hsum(src,n64);
    const size_t RING=64u<<20; void* ring; CK(cudaHostAlloc(&ring,RING,cudaHostAllocDefault));
    void* d; CK(cudaMalloc(&d,n)); unsigned long long* o; CK(cudaMalloc(&o,8));
    int rounds=atoi(argv[2]); int ev[4]={0,0,0,0}; long xfers[4]={0,0,0,0};
    const char* nm[4]={"HOT-ccd0","HOT-ccd1","FLUSHED-ccd0","DRIVER-pageable"};
    for(int r=0;r<rounds;r++){
        for(int arm=0;arm<4;arm++){
            for(int p=0;p<passes;p++){
                if(arm==3){ CK(cudaMemcpy(d,src,n,cudaMemcpyHostToDevice)); }
                else {
                    pin_cores(arm==1?ccd1:ccd0);
                    for(size_t off=0;off<n;off+=RING){ size_t len=(n-off<RING)?(n-off):RING;
                        memcpy(ring,(const char*)src+off,len);
                        if(arm==2) flush_range(ring,len);
                        CK(cudaMemcpy((char*)d+off,ring,len,cudaMemcpyHostToDevice)); }
                }
                CK(cudaDeviceSynchronize()); xfers[arm]++;
                unsigned long long ws=dsum(d,n64,o);
                if(ws!=ref){ ev[arm]++; printf("%4d.%d %-16s FLIP delta=%+lld recompute=%s\n",r,p,nm[arm],(long long)(ws-ref),dsum(d,n64,o)==ws?"stable":"CHANGED");
                    CK(cudaMemcpy(rb,d,n,cudaMemcpyDeviceToHost)); CK(cudaDeviceSynchronize());
                    if(hsum(rb,n64)==ws){ int shown=0; for(size_t i=0;i<n64&&shown<4;i++) if(rb[i]!=src[i]){ uint64_t x=rb[i]^src[i]; for(int b=0;b<64;b++) if((x>>b)&1ull) printf("   offset 0x%012zx dword %d bit %2d %s\n",i*8,b/32,b%32,((rb[i]>>b)&1ull)?"0->1":"1->0"); shown++; } }
                    else printf("   (readback disagreed; offsets withheld)\n");
                    fflush(stdout); }
            }
        }
        if((r+1)%5==0){ printf("--- after %d rounds: ",r+1); for(int a=0;a<4;a++) printf("%s=%d/%ld ",nm[a],ev[a],xfers[a]); printf("\n"); fflush(stdout); }
    }
    printf("done: "); for(int a=0;a<4;a++) printf("%s=%d/%ld ",nm[a],ev[a],xfers[a]); printf("\nHOT3_DONE\n");
}
