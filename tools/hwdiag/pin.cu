// Pinned vs pageable, BOTH directions, same buffer, same window, 5090.
// Pageable transfers are CPU-staged through the driver's small pinned ring;
// pinned transfers DMA the user's pages directly. A weak DRAM cell in the
// staging ring flips pageable transfers only. A fault in the IO-die fabric
// flips both. Every event reports offset, dword bit and direction.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(3);} }while(0)
__global__ void k_sum(const unsigned long long* p, size_t n, unsigned long long* out){
    unsigned long long s=0;
    for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x) s+=p[i];
    for(int o=16;o>0;o>>=1) s+=__shfl_down_sync(0xffffffff,s,o);
    if((threadIdx.x&31)==0) atomicAdd(out,s);
}
static unsigned long long dsum(const void* d,size_t n64,unsigned long long* o){ CK(cudaMemset(o,0,8)); k_sum<<<4096,256>>>((const unsigned long long*)d,n64,o); CK(cudaGetLastError()); unsigned long long h; CK(cudaMemcpy(&h,o,8,cudaMemcpyDeviceToHost)); return h; }
static unsigned long long hsum(const uint64_t* p,size_t n64){ unsigned long long s=0; for(size_t i=0;i<n64;i++) s+=p[i]; return s; }
static void diff(const char* tag,const uint64_t* a,const uint64_t* b,size_t n64){
    int shown=0; for(size_t i=0;i<n64;i++) if(a[i]!=b[i]){ uint64_t x=a[i]^b[i];
        for(int bt=0;bt<64;bt++) if((x>>bt)&1ull) printf("   %s offset 0x%012zx dword %d bit %2d %s\n",tag,i*8,bt/32,bt%32,((b[i]>>bt)&1ull)?"0->1":"1->0");
        if(++shown>=6){ printf("   ...\n"); break; } }
}
int main(int argc,char** argv){
    int fd=open(argv[1],O_RDONLY); struct stat st; fstat(fd,&st); size_t n=(size_t)st.st_size&~7ull, n64=n/8;
    uint64_t* src=(uint64_t*)aligned_alloc(4096,n); size_t got=0; while(got<n){ssize_t r=pread(fd,(char*)src+got,n-got,got); if(r<=0)exit(1); got+=r;} close(fd);
    uint64_t* rb=(uint64_t*)aligned_alloc(4096,n); memset(rb,0,n);
    const unsigned long long ref=hsum(src,n64);
    void* d; CK(cudaMalloc(&d,n)); unsigned long long* o; CK(cudaMalloc(&o,8));
    int rounds=atoi(argv[2]); int ev[4]={0,0,0,0}; const char* nm[4]={"H2D-pageable","D2H-pageable","H2D-pinned","D2H-pinned"};
    for(int r=0;r<rounds;r++){
        for(int arm=0;arm<4;arm++){
            const bool pinned=arm>=2, d2h=(arm&1);
            if(!d2h){
                if(pinned) CK(cudaHostRegister(src,n,cudaHostRegisterDefault));
                CK(cudaMemcpy(d,src,n,cudaMemcpyHostToDevice)); CK(cudaDeviceSynchronize());
                if(pinned) CK(cudaHostUnregister(src));
                unsigned long long ws=dsum(d,n64,o);
                if(ws!=ref){ ev[arm]++; long long dl=(long long)(ws-ref);
                    printf("%4d %-13s FLIP delta=%+lld recompute=%s\n",r,nm[arm],dl,dsum(d,n64,o)==ws?"stable":"CHANGED");
                    CK(cudaMemcpy(rb,d,n,cudaMemcpyDeviceToHost)); CK(cudaDeviceSynchronize());
                    if(hsum(rb,n64)==ws) diff("vram",src,rb,n64); else printf("   (readback itself disagreed with device sum; offsets withheld)\n");
                    // restore a clean device image so later arms compare against ref
                    CK(cudaMemcpy(d,src,n,cudaMemcpyHostToDevice)); CK(cudaDeviceSynchronize());
                    if(dsum(d,n64,o)!=ref) printf("   (re-upload also off; skipping D2H arms this round)\n");
                } else printf("%4d %-13s CLEAN\n",r,nm[arm]);
            } else {
                if(dsum(d,n64,o)!=ref){ printf("%4d %-13s SKIP (device not clean)\n",r,nm[arm]); continue; }
                if(pinned) CK(cudaHostRegister(rb,n,cudaHostRegisterDefault));
                CK(cudaMemcpy(rb,d,n,cudaMemcpyDeviceToHost)); CK(cudaDeviceSynchronize());
                if(pinned) CK(cudaHostUnregister(rb));
                unsigned long long hs=hsum(rb,n64);
                if(hs!=ref){ ev[arm]++; printf("%4d %-13s FLIP delta=%+lld\n",r,nm[arm],(long long)(hs-ref)); diff("host",src,rb,n64); }
                else printf("%4d %-13s CLEAN\n",r,nm[arm]);
            }
            fflush(stdout);
        }
        if((r+1)%10==0){ printf("--- after %d rounds: ",r+1); for(int a=0;a<4;a++) printf("%s=%d ",nm[a],ev[a]); printf("\n"); fflush(stdout); }
    }
    printf("done %d rounds: ",rounds); for(int a=0;a<4;a++) printf("%s=%d/%d ",nm[a],ev[a],rounds); printf("\nPIN_DONE\n");
}
