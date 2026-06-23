#ifndef STAGEB_PERHASH_MSORT_CUH
#define STAGEB_PERHASH_MSORT_CUH
// =====================================================================
// Per-HASH cooperative merge-path mergesort (astronv CUZSORT replica).
// One block per hash sorts the hash's whole coarse-sorted SA range by suffix.
//   PHMSORT  (v0): naive comparator, GLOBAL ping-pong (3.82 KH/s, 2x slow).
//   PHMLCP   (v1): + LCP-carrying merge (known-broken: co-rank loses LCP ctx).
//   PHMTILE  (v2): SHARED-TILED — within-tile sort fully on-chip (SA indices in
//                  shared, register-blocked), only the tile-merge touches global.
//                  This is astronv's actual structure -> the path to ~parity.
// Comparator = parity-safe cmp_suffix (depth=0). Buffers: d_vals_out<->d_vals_in.
// =====================================================================
#include "gpu/stageb_refine.cuh"   // cmp_suffix, cmp_suffix_lcp

// merge-path co-rank: largest i in [iLow,iHigh] s.t. A[lo+i-1] <= B[mid+(d-i)]
__device__ __forceinline__ int sb_pm_corank(
        const uint8_t* Tl, int n, const uint32_t* sv,
        int lo, int mid, int d, int aLen, int bLen) {
    int iLow = d - bLen; if (iLow < 0) iLow = 0;
    int iHigh = (d < aLen) ? d : aLen;
    while (iLow < iHigh) {
        const int i = (iLow + iHigh + 1) >> 1;
        if (cmp_suffix(Tl, n, sv[lo + i - 1], sv[mid + (d - i)], 0) > 0) iHigh = i - 1;
        else                                                              iLow  = i;
    }
    return iLow;
}

// Block-cooperative bottom-up merge-path sort of buf0[0..L) (buf1 = scratch),
// starting from sorted runs of size startW. Per-thread contiguous output ranges;
// one co-rank + sequential merge per (thread, group, round). Returns the buffer
// holding the sorted result. buf0/buf1 may be shared OR global. ALL block threads
// must call (internal __syncthreads). Comparator reads Tl (global, L2-hot/hash).
template<int BLOCK>
__device__ __forceinline__ uint32_t* sb_coop_msort_range(
        const uint8_t* Tl, int n, uint32_t* buf0, uint32_t* buf1, int L, int startW) {
    uint32_t* src = buf0; uint32_t* dst = buf1;
    const int t = threadIdx.x;
    const int per = (L + BLOCK - 1) / BLOCK;
    const int os = t * per; int oe = os + per; if (oe > L) oe = L;
    for (int w = startW; w < L; w <<= 1) {
        const int two = w << 1; int o = os;
        while (o < oe) {
            const int group = o / two; const int lo = group * two;
            int mid = lo + w;   if (mid > L) mid = L;
            int hi  = lo + two; if (hi  > L) hi  = L;
            const int segEnd = (oe < hi) ? oe : hi;
            if (mid >= hi) { for (; o < segEnd; ++o) dst[o] = src[o]; continue; }
            const int aLen = mid - lo, bLen = hi - mid; const int d = o - lo;
            int i = sb_pm_corank(Tl, n, src, lo, mid, d, aLen, bLen); int j = d - i;
            for (; o < segEnd; ++o) {
                bool takeA;
                if (i >= aLen)      takeA = false;
                else if (j >= bLen) takeA = true;
                else                takeA = (cmp_suffix(Tl, n, src[lo + i], src[mid + j], 0) <= 0);
                if (takeA) { dst[o] = src[lo + i]; ++i; }
                else       { dst[o] = src[mid + j]; ++j; }
            }
        }
        __syncthreads();
        uint32_t* tmp = src; src = dst; dst = tmp;
    }
    return src;
}

// ---- v0: naive cooperative mergesort, global ping-pong ----
template<int BLOCK>
__global__ void sb_perhash_msort_kernel(
        const uint8_t* __restrict__ T, const int32_t* __restrict__ d_data_len,
        const int* __restrict__ d_offsets, uint32_t* __restrict__ V0,
        uint32_t* __restrict__ Vtmp, int active_batch) {
    const int h = blockIdx.x; if (h >= active_batch) return;
    const int base = d_offsets[h]; const int n = (int)d_data_len[h]; if (n < 2) return;
    const uint8_t* Tl = T + (size_t)h * (72 * 1024);
    uint32_t* A = V0 + base; uint32_t* B = Vtmp + base;
    uint32_t* res = sb_coop_msort_range<BLOCK>(Tl, n, A, B, n, 1);
    if (res != A) for (int i = threadIdx.x; i < n; i += BLOCK) A[i] = res[i];
}

// ---- v2: SHARED-TILED cooperative mergesort (astronv structure) ----
// TILE = elements sorted on-chip per shared pass (double-buffered: 2*TILE*4 bytes).
template<int BLOCK, int TILE>
__global__ void sb_perhash_msort_tiled_kernel(
        const uint8_t* __restrict__ T, const int32_t* __restrict__ d_data_len,
        const int* __restrict__ d_offsets, uint32_t* __restrict__ V0,
        uint32_t* __restrict__ Vtmp, int active_batch) {
    const int h = blockIdx.x; if (h >= active_batch) return;
    const int base = d_offsets[h]; const int n = (int)d_data_len[h]; if (n < 2) return;
    const uint8_t* Tl = T + (size_t)h * (72 * 1024);
    uint32_t* A = V0 + base; uint32_t* B = Vtmp + base;
    __shared__ uint32_t s[2 * TILE];

    // Phase A: sort each TILE-run fully on-chip (SA indices in shared), write to A.
    for (int t0 = 0; t0 < n; t0 += TILE) {
        int tlen = n - t0; if (tlen > TILE) tlen = TILE;
        for (int i = threadIdx.x; i < tlen; i += BLOCK) s[i] = A[t0 + i];
        __syncthreads();
        uint32_t* res = sb_coop_msort_range<BLOCK>(Tl, n, s, s + TILE, tlen, 1);
        for (int i = threadIdx.x; i < tlen; i += BLOCK) A[t0 + i] = res[i];
        __syncthreads();
    }

    // Phase B: merge the sorted TILE-runs in global (ping-pong A<->B), from w=TILE.
    if (n > TILE) {
        uint32_t* res = sb_coop_msort_range<BLOCK>(Tl, n, A, B, n, TILE);
        if (res != A) for (int i = threadIdx.x; i < n; i += BLOCK) A[i] = res[i];
    }
}

// ---- v1: LCP-carrying cooperative mergesort (KNOWN-BROKEN, kept for record) ----
template<int BLOCK>
__global__ void sb_perhash_msort_lcp_kernel(
        const uint8_t* __restrict__ T, const int32_t* __restrict__ d_data_len,
        const int* __restrict__ d_offsets, uint32_t* __restrict__ V0,
        uint32_t* __restrict__ Vtmp, int* __restrict__ L0, int* __restrict__ Ltmp,
        int active_batch) {
    const int h = blockIdx.x; if (h >= active_batch) return;
    const int base = d_offsets[h]; const int n = (int)d_data_len[h]; if (n < 2) return;
    const uint8_t* Tl = T + (size_t)h * (72 * 1024);
    uint32_t* Av = V0 + base; uint32_t* Bv = Vtmp + base;
    int* Al = L0 + base; int* Bl = Ltmp + base;
    uint32_t* sv = Av; uint32_t* dv = Bv; int* sl = Al; int* dl = Bl;
    const int t = threadIdx.x; const int per = (n + BLOCK - 1) / BLOCK;
    const int os = t * per; int oe = os + per; if (oe > n) oe = n;
    for (int w = 1; w < n; w <<= 1) {
        const int two = w << 1; int o = os;
        while (o < oe) {
            const int group = o / two; const int lo = group * two;
            int mid = lo + w; if (mid > n) mid = n;
            int hi = lo + two; if (hi > n) hi = n;
            const int segEnd = (oe < hi) ? oe : hi;
            if (mid >= hi) { for (; o < segEnd; ++o) { dv[o] = sv[o]; dl[o] = sl[o]; } continue; }
            const int aLen = mid - lo, bLen = hi - mid; const int d = o - lo;
            const int i0 = sb_pm_corank(Tl, n, sv, lo, mid, d, aLen, bLen);
            int ia = lo + i0, ib = mid + (d - i0); int la = 0, lb = 0; int k = o;
            while (k < segEnd && ia < mid && ib < hi) {
                if (la > lb)      { dl[k] = la; dv[k] = sv[ia]; ++k; ++ia; la = (ia < mid) ? sl[ia] : -1; }
                else if (lb > la) { dl[k] = lb; dv[k] = sv[ib]; ++k; ++ib; lb = (ib < hi) ? sl[ib] : -1; }
                else { int split = 0; int cc = cmp_suffix_lcp(Tl, n, sv[ia], sv[ib], la, &split);
                    if (cc < 0) { dl[k] = la; dv[k] = sv[ia]; ++k; ++ia; la = (ia < mid) ? sl[ia] : -1; lb = split; }
                    else        { dl[k] = lb; dv[k] = sv[ib]; ++k; ++ib; lb = (ib < hi) ? sl[ib] : -1; la = split; } }
            }
            while (k < segEnd && ia < mid) { dl[k] = la; dv[k] = sv[ia]; ++k; ++ia; la = (ia < mid) ? sl[ia] : -1; }
            while (k < segEnd && ib < hi)  { dl[k] = lb; dv[k] = sv[ib]; ++k; ++ib; lb = (ib < hi) ? sl[ib] : -1; }
            o = segEnd;
        }
        __syncthreads();
        uint32_t* tv = sv; sv = dv; dv = tv; int* tl = sl; sl = dl; dl = tl;
    }
    if (sv != Av) for (int o = os; o < oe; ++o) Av[o] = sv[o];
}

#endif // STAGEB_PERHASH_MSORT_CUH
