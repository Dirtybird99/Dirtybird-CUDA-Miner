#ifndef STAGEB_PERHASH_COOP_CUH
#define STAGEB_PERHASH_COOP_CUH
// =====================================================================
// Per-HASH fused refine (astronv CUFES+CUFTT mapping).
// ONE BLOCK PER HASH detects tied runs directly from the sorted coarse keys and
// sorts each run in place — fusing tie-detection + sort and ELIMINATING the
// default Stage-B's separate unpack/bin-count/scatter machinery (~18us/hash).
//   - small runs (<= PH_SMALL): proven LCP-insertion (sb_lcp_insort_run), regs.
//   - large runs (> PH_SMALL):  block-cooperative shared-mem bitonic in s_buf.
// Comparator = cmp_suffix (8-byte loads). Parity-exact vs the byte oracle.
// Requires NARROWKEY (default): reads uint32 keys via reinterpret of d_keys_in.
// Env-gated PERHASH=1.
// =====================================================================
#include "gpu/stageb_refine.cuh"   // cmp_suffix, sb_less, sb_lcp_insort_run, sb_bitonic_array, sb_next_pow2, STAGEB_SENTINEL

#ifndef PH_SMALL
#define PH_SMALL 32          // runs up to this length: per-thread register LCP-insertion
#endif
#ifndef PH_SHARED_MAX
#define PH_SHARED_MAX 2048   // largest run handled in shared (next-pow2); bigger -> global fallback
#endif
#ifndef PH_LARGE_CAP
#define PH_LARGE_CAP 1024    // max buffered large-run descriptors per hash (>32-len runs are rare)
#endif

template<int BLOCK>
__global__ void sb_perhash_coop_kernel(
        const uint8_t*  __restrict__ T,            // d_sdata, 72KB stride
        const int32_t*  __restrict__ d_data_len,
        const int*      __restrict__ d_offsets,
        const uint32_t* __restrict__ d_keys,       // sorted coarse keys (uint32 narrow), packed by d_offsets
        uint32_t*       __restrict__ SA,           // d_vals_out, sorted in place per run
        int active_batch, int depth) {
    int hash_id = blockIdx.x;
    if (hash_id >= active_batch) return;
    int base = d_offsets[hash_id];
    int n = (int)d_data_len[hash_id];
    if (n < 2 || n > 70911) return;
    static constexpr int HOST_SDATA_STRIDE = 72 * 1024;
    const uint8_t*  Tl = T + (size_t)hash_id * HOST_SDATA_STRIDE;
    const uint32_t* K  = d_keys + base;
    uint32_t*       V  = SA + base;

    __shared__ int      s_nlarge;
    __shared__ int      s_large_i[PH_LARGE_CAP];
    __shared__ int      s_large_len[PH_LARGE_CAP];
    __shared__ uint32_t s_buf[PH_SHARED_MAX];
    if (threadIdx.x == 0) s_nlarge = 0;
    __syncthreads();

    // ---- Phase 1: detect runs; sort small runs in registers, queue large ----
    for (int i = threadIdx.x; i < n; i += BLOCK) {
        bool is_start = (i == 0) || (K[i] != K[i-1]);
        if (!is_start) continue;
        int j = i + 1;
        while (j < n && K[j] == K[i]) ++j;
        int len = j - i;
        if (len <= 1) continue;
        if (len <= PH_SMALL) {
            uint32_t r[PH_SMALL];
            int      L[PH_SMALL];
            #pragma unroll
            for (int k = 0; k < PH_SMALL; ++k) if (k < len) r[k] = V[i + k];
            sb_lcp_insort_run<PH_SMALL>(Tl, n, depth, r, L, 0, len);
            #pragma unroll
            for (int k = 0; k < PH_SMALL; ++k) if (k < len) V[i + k] = r[k];
        } else {
            int idx = atomicAdd(&s_nlarge, 1);
            if (idx < PH_LARGE_CAP) { s_large_i[idx] = i; s_large_len[idx] = len; }
        }
    }
    __syncthreads();

    // ---- Phase 2: block-cooperative sort of large (>PH_SMALL) runs ----
    int nl = s_nlarge; if (nl > PH_LARGE_CAP) nl = PH_LARGE_CAP;
    for (int q = 0; q < nl; ++q) {
        int i   = s_large_i[q];
        int len = s_large_len[q];
        int m   = (int)sb_next_pow2((uint32_t)len);
        if (m > PH_SHARED_MAX) {
            // rare: run too big for shared -> single-thread insertion in global (correctness).
            if (threadIdx.x == 0) {
                for (int a = i + 1; a < i + len; ++a) {
                    uint32_t key = V[a]; int b = a - 1;
                    while (b >= i && cmp_suffix(Tl, n, V[b], key, depth) > 0) { V[b+1] = V[b]; --b; }
                    V[b+1] = key;
                }
            }
            __syncthreads();
            continue;
        }
        for (int k = threadIdx.x; k < m; k += BLOCK) s_buf[k] = (k < len) ? V[i + k] : STAGEB_SENTINEL;
        __syncthreads();
        sb_bitonic_array(s_buf, m, Tl, n, depth, threadIdx.x, BLOCK);   // has internal __syncthreads
        for (int k = threadIdx.x; k < len; k += BLOCK) V[i + k] = s_buf[k];
        __syncthreads();
    }
}

#endif // STAGEB_PERHASH_COOP_CUH
