#include <stdio.h>
#include <assert.h>

// Optimized WKV CUDA kernel with [B, C, T] layout for coalesced memory access.
//
// Original layout [B, T, C]: stride between consecutive time steps is C.
//   => thread (b,c) reads k[i * C], k[(i+1) * C], ... — stride C is non-coalesced.
//
// Optimized layout [B, C, T]: stride between consecutive time steps is 1.
//   => thread (b,c) reads k[i], k[i+1], ... — stride 1 is fully coalesced,
//      and adjacent threads (same b, different c) read k[0], k[T], k[2T]...
//      which are also coalesced within a cache line when C is the minor dim.
//
// The thread grid is identical to the original: one thread per (batch, channel) pair.
// What changes is only the offset calculation and the inner-loop stride.

#define MIN_VALUE (-1e38)

template <typename F>
__global__ void kernel_forward_opt(const int B, const int T, const int C,
                                   const F *__restrict__ const _w,
                                   const F *__restrict__ const _u,
                                   const F *__restrict__ const _k,  // [B, C, T]
                                   const F *__restrict__ const _v,  // [B, C, T]
                                   F *__restrict__ const _y) {      // [B, C, T]
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;

    // In [B, C, T] layout the offset to the start of the (b, c) time-series is:
    //   batch_offset  = _b * C * T
    //   channel_offset = _c * T
    const int _offset = _b * C * T + _c * T;

    F u = _u[_c];
    F w = _w[_c];
    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    F *__restrict__ const y = _y + _offset;

    F p = 0, q = 0, o = MIN_VALUE;
    for (int i = 0; i < T; i++) {
        // Stride-1 access — fully coalesced
        F no = max(o, u + k[i]);
        F A = exp(o - no);
        F B = exp(u + k[i] - no);
        y[i] = (A * p + B * v[i]) / (A * q + B);

        no = max(w + o, k[i]);
        A = exp(w + o - no);
        B = exp(k[i] - no);
        p = A * p + B * v[i];
        q = A * q + B;
        o = no;
    }
}

template <typename F>
__global__ void kernel_backward_opt(const int B, const int T, const int C,
                                    const F *__restrict__ const _w,
                                    const F *__restrict__ const _u,
                                    const F *__restrict__ const _k,   // [B, C, T]
                                    const F *__restrict__ const _v,   // [B, C, T]
                                    const F *__restrict__ const _gy,  // [B, C, T]
                                    F *__restrict__ const _gw,
                                    F *__restrict__ const _gu,
                                    F *__restrict__ const _gk,        // [B, C, T]
                                    F *__restrict__ const _gv) {      // [B, C, T]
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * C * T + _c * T;

    F u = _u[_c];
    F w = _w[_c];
    const F *__restrict__ const k  = _k  + _offset;
    const F *__restrict__ const v  = _v  + _offset;
    const F *__restrict__ const gy = _gy + _offset;

    F *__restrict__ const gk = _gk + _offset;
    F *__restrict__ const gv = _gv + _offset;

    F y[Tmax], z[Tmax], zexp[Tmax];

    F gw = 0, gu = 0;
    F p = 0, q = 0;
    F dpdw = 0, dqdw = 0;
    F o = MIN_VALUE;
    for (int i = 0; i < T; i++) {
        F no = max(o, k[i] + u);
        F A = exp(o - no);
        F B = exp(k[i] + u - no);

        F num = A * p + B * v[i];
        F iden = 1 / (A * q + B);

        y[i] = num * iden;
        z[i] = iden;
        zexp[i] = k[i] + u - no;

        gw += gy[i] * (dpdw - dqdw * y[i]) * iden * A;
        gu += gy[i] * (v[i] - y[i]) * B * iden;

        no = max(w + o, k[i]);
        A = exp(w + o - no);
        B = exp(k[i] - no);
        dpdw = A * (p + dpdw);
        dqdw = A * (q + dqdw);
        p = A * p + B * v[i];
        q = A * q + B;
        o = no;
    }

    F gp = 0, gq = 0;
    o = MIN_VALUE;
    for (int i = T - 1; i >= 0; i--) {
        F A = gy[i] * z[i] * exp(zexp[i]);
        F B = exp(k[i] + o);
        gk[i] = A * (v[i] - y[i]) + B * (gp * v[i] + gq);
        gv[i] = A + B * gp;

        F no = max(w + o, zexp[i] - k[i] - u);
        A = exp(w + o - no);
        B = gy[i] * z[i] * exp(zexp[i] - k[i] - u - no);
        gp = A * gp + B;
        gq = A * gq - B * y[i];
        o = no;
    }

    const int _offsetBC = _b * C + _c;
    _gw[_offsetBC] += gw * _w[_c];
    _gu[_offsetBC] += gu;
}

void cuda_forward_opt(int B, int T, int C, float *w, float *u, float *k, float *v, float *y) {
    dim3 threadsPerBlock(min(C, 32));
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward_opt<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, y);
}

void cuda_backward_opt(int B, int T, int C, float *w, float *u, float *k, float *v, float *gy, float *gw, float *gu, float *gk, float *gv) {
    dim3 threadsPerBlock(min(C, 32));
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_backward_opt<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, gy, gw, gu, gk, gv);
}
