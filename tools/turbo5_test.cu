// turbo5 (5-bit K) format validation: prove src/turbo5.cuh's device format,
// packing, and every read path agree with an independent CPU oracle BEFORE any
// engine wiring. Plan: docs/plans/2026-08-01-5bit-k.md, phase P0.
//
// This is deliberately a tighter gate than tools/turbo3_test.cu was:
//   * it exercises the REAL cooperative kernel (turbo5_quant_group, 128
//     threads/block) that the decode and prefill stores call, not a
//     single-thread reimplementation of it -- turbo3_test never covered
//     turbo3_quant_group at all, so the turbo3 leg is folded in here too;
//   * the primary oracle reproduces the device's TREE-order reductions, so it
//     demands BITWISE equality (0 mismatches) rather than a tie tolerance. A
//     second serial-order oracle is kept as the independent-implementation
//     check, where near-boundary ties are expected and tolerated;
//   * every read path (deq_elem, the fd2 lane load, the prefill stage8) is
//     gated against the same dequant, because at 5 bits the packing crosses
//     byte boundaries differently than turbo3's and that is exactly where a
//     silent layout bug would live.
// Build: nvcc -std=c++17 -arch=sm_120 tools/turbo5_test.cu -o build/turbo5_test
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "../src/turbo5.cuh"
using namespace q27turbo;

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} }while(0)

static const float C5[32] = Q27_TURBO5_CENTROIDS_LIST;
static const float T5[31] = Q27_TURBO5_THRESH_LIST;
static const float C3[8]  = Q27_TURBO_CENTROIDS_LIST;
static const float S1[128] = Q27_TURBO_S1_LIST;
static const float S2[128] = Q27_TURBO_S2_LIST;

// ---- CPU oracle ------------------------------------------------------------
// The shared rotation, in the fork's operand order. This IS the device's
// butterfly order: turbo3_butterfly128 computes xs[j] = (j&h) ? (b-a) : (a+b)
// with a=xs[j], b=xs[j^h], which for the upper index equals x[lo]-x[hi] and
// for the lower x[lo]+x[hi] -- the same two expressions this loop writes.
static int cpu5_nearest(float v){ int i=0; for(int k=0;k<31;k++) i += (v>=T5[k]); return i; }
static void cpu_fwht(float* x){
    const float inv=0.08838834764831845f;
    for(int i=0;i<128;i++) x[i]*=S1[i];
    for(int h=1;h<128;h*=2) for(int i=0;i<128;i+=h*2) for(int j=i;j<i+h;j++){
        float a=x[j],b=x[j+h]; x[j]=a+b; x[j+h]=a-b; }
    for(int i=0;i<128;i++) x[i]*=inv*S2[i];
}
static void cpu_fwht_inv(float* x){
    const float inv=0.08838834764831845f;
    for(int i=0;i<128;i++) x[i]*=S2[i];
    for(int h=1;h<128;h*=2) for(int i=0;i<128;i+=h*2) for(int j=i;j<i+h;j++){
        float a=x[j],b=x[j+h]; x[j]=a+b; x[j+h]=a-b; }
    for(int i=0;i<128;i++) x[i]*=inv*S1[i];
}
// TREE == the device's `for (s=64;s>0;s>>=1) if (j<s) red[j]+=red[j+s]`.
static float red_tree(const float* v){
    float r[128]; for(int j=0;j<128;j++) r[j]=v[j];
    for(int s=64;s>0;s>>=1) for(int j=0;j<s;j++) r[j]+=r[j+s];
    return r[0];
}
static float red_serial(const float* v){ float s=0.f; for(int j=0;j<128;j++) s+=v[j]; return s; }

// TREE=true reproduces the device bit-for-bit; TREE=false is the independent
// serial-summation implementation (near-boundary ties expected).
static void cpu_quant5(const float* x, block_turbo5* y, bool tree){
    float buf[128], sq[128];
    for(int j=0;j<128;j++){ buf[j]=x[j]; sq[j]=x[j]*x[j]; }
    float gn=sqrtf(tree?red_tree(sq):red_serial(sq));
    float inv=gn>1e-10f?1.f/gn:0.f;
    for(int j=0;j<128;j++) buf[j]*=inv;
    cpu_fwht(buf);
    memset(y->qs,0,64); memset(y->hi,0,16);
    float rq[128];
    int idx[128];
    for(int j=0;j<128;j++){ idx[j]=cpu5_nearest(buf[j]); rq[j]=C5[idx[j]]*C5[idx[j]]; }
    for(int j=0;j<128;j++){
        y->qs[j/2] |= (uint8_t)((idx[j]&15) << ((j%2)*4));
        if(idx[j]&16) y->hi[j/8] |= (uint8_t)(1<<(j%8)); }
    float rn=sqrtf(tree?red_tree(rq):red_serial(rq));
    float corr=rn>1e-10f?gn/rn:gn;
    y->norm=__float2half(corr);
}
static void cpu_dequant5(const block_turbo5* x, float* y){
    float norm=__half2float(x->norm);
    for(int j=0;j<128;j++){ uint8_t l=(x->qs[j/2]>>((j%2)*4))&0xF;
        uint8_t h=(x->hi[j/8]>>(j%8))&1; y[j]=C5[l|(h<<4)]*norm; }
}
// turbo3 CPU dequant, for the bit-width comparison at the end.
static void cpu_dequant3(const block_turbo3* x, float* y){
    float norm=__half2float(x->norm);
    for(int j=0;j<128;j++){ uint8_t l=(x->qs[j/4]>>((j%4)*2))&3;
        uint8_t h=(x->signs[j/8]>>(j%8))&1; y[j]=C3[l|(h<<2)]*norm; }
}

// ---- device: the REAL cooperative kernels the engine stores through --------
__global__ void k_quant5_coop(const float* X, block_turbo5* Y){
    __shared__ float xs[128], red[128];
    turbo5_quant_group(X[blockIdx.x*128+threadIdx.x], Y+blockIdx.x, threadIdx.x, xs, red);
}
__global__ void k_quant3_coop(const float* X, block_turbo3* Y){
    __shared__ float xs[128], red[128];
    turbo3_quant_group(X[blockIdx.x*128+threadIdx.x], Y+blockIdx.x, threadIdx.x, xs, red);
}
// deq_elem over a 2-block row: block b covers dims (b&1)*128 .. +127 of row b>>1
__global__ void k_deq5(const block_turbo5* X, float* Y, int n){
    int b=blockIdx.x*blockDim.x+threadIdx.x; if(b>=n) return;
    for(int j=0;j<128;j++) Y[b*128+j]=turbo5_dequant(&X[b],j,__half2float(X[b].norm));
}
// the two staged read paths, over rows of 2 blocks (head_dim 256)
__global__ void k_lane5(const block_turbo5* X, float* Y){   // 32 threads = 32 lanes
    const block_turbo5* row = X + blockIdx.x*2;
    float o[8]; turbo5_ld8_lane(row, threadIdx.x, o);
    for(int i=0;i<4;i++){
        Y[blockIdx.x*256 + 4*threadIdx.x + i]       = o[i];
        Y[blockIdx.x*256 + 128 + 4*threadIdx.x + i] = o[4+i]; }
}
__global__ void k_stage5(const block_turbo5* X, __half* Y){ // 32 threads = 32 d8 chunks
    const block_turbo5* row = X + blockIdx.x*2;
    __half2 h2[4]; turbo5_stage8_h2(row, threadIdx.x*8, h2);
    for(int j=0;j<4;j++) *(__half2*)(Y + blockIdx.x*256 + threadIdx.x*8 + 2*j) = h2[j];
}
__global__ void k_deqrow5_h(const block_turbo5* X, __half* Y){ // fp16 reference row
    const block_turbo5* row = X + blockIdx.x*2;
    for(int d=threadIdx.x; d<256; d+=blockDim.x)
        Y[blockIdx.x*256+d] = __float2half_rn(turbo5_deq_elem(row, d));
}

int fails=0;
static void gate(const char* what, long bad, long limit){
    printf("%-58s %8ld (want <=%ld) %s\n", what, bad, limit, bad<=limit?"PASS":"FAIL");
    if(bad>limit) fails++;
}

int main(){
    if(sizeof(block_turbo5)!=82){ printf("FAIL sizeof(block_turbo5)=%zu, want 82\n",
                                         sizeof(block_turbo5)); return 1; }
    // table sanity: strictly increasing, symmetric, thresholds interleaved
    for(int i=0;i+1<32;i++) if(!(C5[i]<C5[i+1])){ printf("FAIL centroids not increasing @%d\n",i); fails++; }
    for(int i=0;i<32;i++) if(fabsf(C5[i]+C5[31-i])>1e-7f){ printf("FAIL centroids not symmetric @%d\n",i); fails++; }
    for(int i=0;i<31;i++) if(!(C5[i]<T5[i] && T5[i]<C5[i+1])){ printf("FAIL threshold %d not between its centroids\n",i); fails++; }
    for(int i=0;i<31;i++){ // threshold rule == true nearest centroid
        float mid=T5[i], lo=nextafterf(mid,-1.f), hi=nextafterf(mid,1.f);
        if(cpu5_nearest(lo)!=i || cpu5_nearest(hi)!=i+1){ printf("FAIL nearest across threshold %d\n",i); fails++; } }

    const int N=8192;               // 4096 rows of 2 blocks (head_dim 256)
    std::vector<float> hx(N*128);
    unsigned s=1234567;
    for(auto& v:hx){ s=s*1664525u+1013904223u; v=((s>>8)&0xFFFF)/65536.f-0.5f; }
    float *dX,*dY; block_turbo5* dB5; block_turbo3* dB3; __half *dH,*dHR;
    CK(cudaMalloc(&dX,(size_t)N*128*4));   CK(cudaMalloc(&dY,(size_t)N*128*4));
    CK(cudaMalloc(&dB5,(size_t)N*sizeof(block_turbo5)));
    CK(cudaMalloc(&dB3,(size_t)N*sizeof(block_turbo3)));
    CK(cudaMalloc(&dH,(size_t)N*128*2));   CK(cudaMalloc(&dHR,(size_t)N*128*2));
    CK(cudaMemcpy(dX,hx.data(),(size_t)N*128*4,cudaMemcpyHostToDevice));

    k_quant5_coop<<<N,128>>>(dX,dB5);            CK(cudaDeviceSynchronize());
    k_quant3_coop<<<N,128>>>(dX,dB3);            CK(cudaDeviceSynchronize());
    k_deq5<<<(N+63)/64,64>>>(dB5,dY,N);          CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());

    std::vector<block_turbo5> hb5(N); std::vector<block_turbo3> hb3(N);
    std::vector<float> hy(N*128);
    CK(cudaMemcpy(hb5.data(),dB5,(size_t)N*sizeof(block_turbo5),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hb3.data(),dB3,(size_t)N*sizeof(block_turbo3),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hy.data(),dY,(size_t)N*128*4,cudaMemcpyDeviceToHost));

    // 1. device cooperative quant == CPU tree-order oracle, BITWISE
    long qtree=0,ntree=0,qser=0,nser=0,deqbad=0;
    double cos_num=0,cos_a=0,cos_b=0,mse5=0,mse3=0;
    for(int b=0;b<N;b++){
        block_turbo5 ct,cs;
        cpu_quant5(hx.data()+b*128,&ct,true);
        cpu_quant5(hx.data()+b*128,&cs,false);
        if(memcmp(ct.qs,hb5[b].qs,64)||memcmp(ct.hi,hb5[b].hi,16)) qtree++;
        if(memcmp(&ct.norm,&hb5[b].norm,2)) ntree++;
        bool tie = memcmp(cs.qs,hb5[b].qs,64)||memcmp(cs.hi,hb5[b].hi,16);
        if(tie) qser++;
        // Norm is only independently meaningful where the INDICES agree: the
        // corrected norm is gn/rn and rn is built from the chosen centroids,
        // so a midpoint tie moves the norm by a real amount, not a rounding
        // step. Counting those here would just double-count the tie.
        if(!tie && memcmp(&cs.norm,&hb5[b].norm,2)){ // allow 1 ULP on fp16
            int d=abs((int)*(uint16_t*)&cs.norm - (int)*(uint16_t*)&hb5[b].norm);
            if(d>1) nser++; }
        // 2. device dequant == CPU dequant
        float cy[128]; cpu_dequant5(&hb5[b],cy);
        for(int j=0;j<128;j++) if(cy[j]!=hy[b*128+j]) deqbad++;
        // 3. round-trip vs input, and the same for turbo3 on identical data
        float r5[128],r3[128]; cpu_dequant3(&hb3[b],r3);
        for(int j=0;j<128;j++) r5[j]=hy[b*128+j];
        cpu_fwht_inv(r5); cpu_fwht_inv(r3);
        const float* ox=hx.data()+b*128;
        for(int j=0;j<128;j++){
            cos_num+=r5[j]*ox[j]; cos_a+=r5[j]*r5[j]; cos_b+=ox[j]*ox[j];
            mse5+=(r5[j]-ox[j])*(r5[j]-ox[j]); mse3+=(r3[j]-ox[j])*(r3[j]-ox[j]); }
    }
    double cosine=cos_num/(sqrt(cos_a)*sqrt(cos_b)+1e-12);
    mse5/=(N*128.0); mse3/=(N*128.0);

    printf("\n-- format gates ------------------------------------------------------------\n");
    gate("device coop quant vs CPU TREE oracle: qs/hi blocks differing", qtree, 0);
    gate("device coop quant vs CPU TREE oracle: norm blocks differing", ntree, 0);
    gate("device dequant vs CPU dequant: elements differing", deqbad, 0);
    gate("device coop quant vs CPU SERIAL oracle: qs/hi (midpoint ties)", qser, N/200);
    gate("device coop quant vs CPU SERIAL oracle: norm >1ULP (untied)", nser, 0);

    printf("\n-- fidelity ----------------------------------------------------------------\n");
    printf("round-trip q->deq->invWHT vs input: cosine=%.6f  MSE=%.8f\n", cosine, mse5);
    printf("same data through turbo3 (3-bit)  :                 MSE=%.8f  (%.2fx worse)\n",
           mse3, mse3/mse5);
    if(cosine<0.998){ printf("FAIL round-trip cosine (5-bit Lloyd-Max predicts 0.998747)\n"); fails++; }
    if(!(mse5<mse3)){ printf("FAIL turbo5 is not better than turbo3 on identical input\n"); fails++; }

    // 4. dot invariance (the read contract): WHT(q).deq(K) must track q.K
    double dn=0,da=0,db=0;
    for(int b=0;b<512;b++){
        const float* K=hx.data()+b*128;
        float q[128],qw[128]; unsigned tt=99+b;
        for(int j=0;j<128;j++){ tt=tt*1664525u+1013904223u; q[j]=((tt>>8)&0xFFFF)/65536.f-0.5f; qw[j]=q[j]; }
        cpu_fwht(qw);
        float dqK[128]; cpu_dequant5(&hb5[b],dqK);
        double lhs=0,rhs=0;
        for(int j=0;j<128;j++){ lhs+=qw[j]*dqK[j]; rhs+=q[j]*K[j]; }
        dn+=lhs*rhs; da+=lhs*lhs; db+=rhs*rhs;
    }
    double dotcos=dn/(sqrt(da)*sqrt(db)+1e-12);
    // The ceiling here is set by the format's own distortion, not by the read
    // path: deq(K) = WHT(K) + e with |e|^2/|K|^2 = D, so the score cosine
    // maxes at 1/sqrt(1+D) = 0.998750 for 5-bit (and 0.983161 for 3-bit --
    // turbo3 would FAIL this bar, which is what makes it a bit-width gate and
    // not just a "did the dot survive" gate).
    printf("dot invariance WHT(q).deq(K) vs q.K: score cosine=%.6f over 512 pairs"
           " (5-bit ceiling 0.998750)\n",dotcos);
    if(dotcos<0.998){ printf("FAIL dot invariance\n"); fails++; }

    // 5. read-path gates: the fd2 lane load and the prefill stage8 must both
    //    reproduce deq_elem exactly over full 256-dim rows.
    const int R=N/2;
    float* dLane; CK(cudaMalloc(&dLane,(size_t)R*256*4));
    k_lane5<<<R,32>>>(dB5,dLane);                CK(cudaDeviceSynchronize());
    k_stage5<<<R,32>>>(dB5,dH);                  CK(cudaDeviceSynchronize());
    k_deqrow5_h<<<R,64>>>(dB5,dHR);              CK(cudaDeviceSynchronize());
    std::vector<float> hl((size_t)R*256);
    std::vector<uint16_t> hh((size_t)R*256), hr((size_t)R*256);
    CK(cudaMemcpy(hl.data(),dLane,(size_t)R*256*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hh.data(),dH,(size_t)R*256*2,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hr.data(),dHR,(size_t)R*256*2,cudaMemcpyDeviceToHost));
    long lanebad=0, stagebad=0;
    for(int r=0;r<R;r++){
        float ref[256];
        cpu_dequant5(&hb5[2*r],ref); cpu_dequant5(&hb5[2*r+1],ref+128);
        for(int d=0;d<256;d++){
            if(hl[(size_t)r*256+d]!=ref[d]) lanebad++;
            if(hh[(size_t)r*256+d]!=hr[(size_t)r*256+d]) stagebad++; }
    }
    printf("\n-- read paths --------------------------------------------------------------\n");
    gate("fd2 turbo5_ld8_lane vs CPU dequant: elements differing", lanebad, 0);
    gate("prefill turbo5_stage8_h2 vs deq_elem: halves differing", stagebad, 0);

    // 6. size ledger
    const double t3=(double)sizeof(block_turbo3), t5=(double)sizeof(block_turbo5);
    printf("\nblock bytes/128 dims: turbo3 %.0f, turbo5 %.0f -> turbo5k K+V = %.2fx turbo3\n",
           t3, t5, (t5+t3)/(2*t3));
    printf("             at turbo3's measured 14.1 KB/tok -> turbo5k ~%.1f KB/tok\n",
           14.1*(t5+t3)/(2*t3));

    printf(fails? "\nturbo5_test: %d FAILURES\n":"\nturbo5_test: ALL PASS\n",fails);
    return fails?1:0;
}
