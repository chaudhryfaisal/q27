// ADDRESS or DATA? hot4 showed the SAME byte offset corrupting twice, in
// different arms and on different CCDs:
//   0x000276f3f570 dword0 bit27 0->1  (DRIVER-pageable r55, HOT-ccd1 r92)
//   0x0004ddddf970 dword0 bit27 1->0  (HOT-ccd0 r213,      HOT-ccd1 r250)
// A random DMA fault in 22.5 GB does not hit the same byte twice in 450 runs.
// Two explanations remain, and they point at different hardware:
//   (a) a weak VRAM cell at that DEVICE address   -> the GPU's memory
//   (b) a data-pattern-dependent path fault        -> the bytes themselves
// Upload the SAME slice to TWO different device regions each round and diff
// both against the source. Flips that follow the data land at the same
// SLICE offset in both copies; flips that follow the address land at one
// device region only.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(3);} }while(0)
__global__ void k_sum(const unsigned long long* p,size_t n,unsigned long long* out){ unsigned long long s=0; for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x) s+=p[i]; for(int o=16;o>0;o>>=1) s+=__shfl_down_sync(0xffffffff,s,o); if((threadIdx.x&31)==0) atomicAdd(out,s); }
static unsigned long long dsum(const void* d,size_t n64,unsigned long long* o){ CK(cudaMemset(o,0,8)); k_sum<<<4096,256>>>((const unsigned long long*)d,n64,o); CK(cudaGetLastError()); unsigned long long h; CK(cudaMemcpy(&h,o,8,cudaMemcpyDeviceToHost)); return h; }
static unsigned long long hsum(const uint64_t* p,size_t n64){ unsigned long long s=0; for(size_t i=0;i<n64;i++) s+=p[i]; return s; }
int main(int argc,char** argv){
    const size_t SLICE=(size_t)8<<30;
    int fd=open(argv[1],O_RDONLY); struct stat st; fstat(fd,&st);
    size_t n=SLICE, n64=n/8;
    uint64_t* src=(uint64_t*)aligned_alloc(4096,n); size_t got=0; while(got<n){ssize_t r=pread(fd,(char*)src+got,n-got,got); if(r<=0)exit(1); got+=r;} close(fd);
    uint64_t* rb=(uint64_t*)aligned_alloc(4096,n);
    const unsigned long long ref=hsum(src,n64);
    void *A,*B; CK(cudaMalloc(&A,n)); CK(cudaMalloc(&B,n)); unsigned long long* o; CK(cudaMalloc(&o,8));
    printf("slice %.2f GB  region A=%p  B=%p  (delta %.2f GB)\n",n/1e9,A,B,((char*)B-(char*)A)/1e9); fflush(stdout);
    const int target=atoi(argv[2]); int evA=0,evB=0; long x=0;
    for(int r=0;r<20000 && evA+evB<target;r++){
        for(int which=0;which<2;which++){
            void* d = which? B : A; const char* nm = which? "B" : "A";
            CK(cudaMemcpy(d,src,n,cudaMemcpyHostToDevice)); CK(cudaDeviceSynchronize()); x++;
            unsigned long long ws=dsum(d,n64,o);
            if(ws==ref) continue;
            (which?evB:evA)++;
            printf("%5d region %s dev=%p FLIP delta=%+lld recompute=%s\n",r,nm,d,(long long)(ws-ref),dsum(d,n64,o)==ws?"stable":"CHANGED");
            CK(cudaMemcpy(rb,d,n,cudaMemcpyDeviceToHost)); CK(cudaDeviceSynchronize());
            if(hsum(rb,n64)==ws){ int shown=0;
                for(size_t i=0;i<n64&&shown<4;i++) if(rb[i]!=src[i]){ uint64_t v=rb[i]^src[i];
                    for(int b=0;b<64;b++) if((v>>b)&1ull)
                        printf("   slice-offset 0x%012zx  DEVICE addr %p  dword %d bit %2d %s\n",
                               i*8,(void*)((char*)d+i*8),b/32,b%32,((rb[i]>>b)&1ull)?"0->1":"1->0");
                    shown++; } }
            else printf("   (readback disagreed; offsets withheld)\n");
            fflush(stdout);
        }
        if((r+1)%25==0){ printf("--- round %d: A=%d B=%d over %ld transfers\n",r+1,evA,evB,x); fflush(stdout); }
    }
    printf("done: A=%d B=%d over %ld transfers (%.1f TB)\nADDR_DONE\n",evA,evB,x,x*8.0/1000);
}
