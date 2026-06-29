#ifndef STAGEB_REFINE_CUH
#define STAGEB_REFINE_CUH
// =====================================================================
// Stage-B cooperative active-segment sorter for DERO AstroBWTv3 GPU miner.
//
// Sorts each "active segment" (a set of suffix indices tied on their first
// `depth` bytes) into correct suffix order.  Ordering is lexicographic on
// bytes with END-OF-STRING SMALLEST (implicit sentinel $ < every byte at
// position n).  This is a strict TOTAL order: two distinct suffixes never
// compare equal.
//
// Dispatch (length class), cooperative, NO one-thread-per-segment:
//   - len <= 1            : nothing to do.
//   - len <= 32           : single warp, bitonic sort via __shfl (registers).
//   - len <= BLK_MAX      : single block, bitonic sort in shared memory.
//   - len  > BLK_MAX      : single block, bitonic sort in GLOBAL scratch
//                           (cooperative across the block, multi-pass).
// Padding to next power of two uses SENTINEL = 0xFFFFFFFF treated as +inf.
// =====================================================================
#include <cstdint>
#include <cuda_runtime.h>

#ifndef STAGEB_SENTINEL
#define STAGEB_SENTINEL 0xFFFFFFFFu
#endif

// ---- unaligned vector loads -----------------------------------------
__device__ __forceinline__ uint64_t sb_load_u64(const uint8_t* p) {
    const uintptr_t addr = (uintptr_t)p;
    const uintptr_t base_addr = addr & ~(uintptr_t)7;
    const uint64_t* base = (const uint64_t*)base_addr;
    const uint64_t lo = __ldg(base);
    const uint32_t shift = (uint32_t)((addr & 7u) * 8u);
    if (shift == 0u) return lo;
    const uint64_t hi = __ldg(base + 1);
    return (lo >> shift) | (hi << (64u - shift));
}
__device__ __forceinline__ uint32_t sb_load_u32(const uint8_t* p) {
    const uintptr_t addr = (uintptr_t)p;
    const uintptr_t base_addr = addr & ~(uintptr_t)3;
    const uint32_t* base = (const uint32_t*)base_addr;
    const uint32_t lo = __ldg(base);
    const uint32_t shift = (uint32_t)((addr & 3u) * 8u);
    if (shift == 0u) return lo;
    const uint32_t hi = __ldg(base + 1);
    return (lo >> shift) | (hi << (32u - shift));
}

// ---------------------------------------------------------------------
// Reusable device compare.  Returns <0 if suffix a < b, 0 iff a==b, >0 else.
// End-of-string is SMALLEST.  `depth` bytes are assumed equal (skip them);
// depth==0 is a correct full compare.
// ---------------------------------------------------------------------
__device__ int g_sb_fakecmp = 0;
__device__ int g_sb_widecmp = 0;
__device__ int g_sb_fastdiff = 0;
__device__ __forceinline__ int cmp_suffix(const uint8_t* T, int n,
                                          uint32_t a, uint32_t b, int depth) {
    if (a == b) return 0;
    if (g_sb_fakecmp) return (a < b) ? -1 : 1;
    uint32_t aa = a + (uint32_t)depth;
    uint32_t bb = b + (uint32_t)depth;
    const uint32_t un = (uint32_t)n;
    // 8-byte strides while both have >=8 bytes left
    while (aa + 8u <= un && bb + 8u <= un) {
        uint64_t av = sb_load_u64(T + aa);
        uint64_t bv = sb_load_u64(T + bb);
        if (av != bv) {
            if (g_sb_fastdiff) {
                const uint64_t diff = av ^ bv;
                const int byte = (__ffsll((long long)diff) - 1) >> 3;
                const uint8_t ca = (uint8_t)(av >> (8 * byte));
                const uint8_t cb = (uint8_t)(bv >> (8 * byte));
                return ca < cb ? -1 : 1;
            }
            // little-endian load: first differing byte is the lowest address;
            // compare byte-by-byte from the low byte.
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                uint8_t ca = (uint8_t)(av >> (8*k));
                uint8_t cb = (uint8_t)(bv >> (8*k));
                if (ca != cb) return ca < cb ? -1 : 1;
            }
        }
        aa += 8; bb += 8;
    }
    while (aa + 4u <= un && bb + 4u <= un) {
        uint32_t av = sb_load_u32(T + aa);
        uint32_t bv = sb_load_u32(T + bb);
        if (av != bv) {
            if (g_sb_fastdiff) {
                const uint32_t diff = av ^ bv;
                const int byte = (__ffs((int)diff) - 1) >> 3;
                const uint8_t ca = (uint8_t)(av >> (8 * byte));
                const uint8_t cb = (uint8_t)(bv >> (8 * byte));
                return ca < cb ? -1 : 1;
            }
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                uint8_t ca = (uint8_t)(av >> (8*k));
                uint8_t cb = (uint8_t)(bv >> (8*k));
                if (ca != cb) return ca < cb ? -1 : 1;
            }
        }
        aa += 4; bb += 4;
    }
    while (aa < un && bb < un) {
        uint8_t ca = T[aa], cb = T[bb];
        if (ca != cb) return ca < cb ? -1 : 1;
        ++aa; ++bb;
    }
    // one (or both) ran out: shorter remaining suffix is SMALLER.
    if (aa >= un && bb >= un) {
        uint32_t ra = un - a, rb = un - b;       // distinct a,b => distinct lengths
        return ra < rb ? -1 : (ra > rb ? 1 : 0);
    }
    return (aa >= un) ? -1 : 1;                    // a ended first => a smaller
}

// less-than with SENTINEL = +inf
__device__ __forceinline__ bool sb_less(const uint8_t* T, int n,
                                        uint32_t a, uint32_t b, int depth) {
    if (a == STAGEB_SENTINEL) return false;   // +inf is never < anything
    if (b == STAGEB_SENTINEL) return true;    // anything (non-inf) < +inf
    return cmp_suffix(T, n, a, b, depth) < 0;
}

__device__ __forceinline__ uint32_t sb_next_pow2(uint32_t x) {
    if (x <= 1) return 1;
    return 1u << (32 - __clz(x - 1));
}

// =====================================================================
// WARP bitonic sort (len <= 32). One warp, value held in a register per lane.
// Lanes >= len hold SENTINEL. Produces ascending order in lanes [0,len).
// =====================================================================
// =====================================================================
// Generic bitonic compare-exchange over an index array `a` of logical
// length `m` (power of two), executed cooperatively by `nthreads`.
// Works on shared OR global memory.
// =====================================================================
__device__ __forceinline__ void sb_bitonic_array(uint32_t* a, int m,
                                                  const uint8_t* T, int n, int depth,
                                                  int tid, int nthreads) {
    for (int k = 2; k <= m; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int idx = tid; idx < m; idx += nthreads) {
                int ixj = idx ^ j;
                if (ixj > idx) {
                    bool ascending = ((idx & k) == 0);
                    uint32_t x = a[idx], y = a[ixj];
                    bool swap;
                    if (x == y) swap = false;
                    else {
                        bool x_less = sb_less(T, n, x, y, depth);
                        // ascending: want smaller at idx. swap if x>y i.e. !x_less
                        swap = ascending ? !x_less : x_less;
                    }
                    if (swap) { a[idx] = y; a[ixj] = x; }
                }
            }
            __syncthreads();
        }
    }
}


// =====================================================================
// KERNELS / DISPATCH
// =====================================================================
// Per-segment metadata arrays (length num_segments):
//   seg_off[s]   : start offset into SA[] for segment s
//   seg_len[s]   : number of suffix indices in segment s
//   seg_depth[s] : bytes known equal (compare may start here)
//   seg_n[s]     : n (length of the lane string) for segment s
//   seg_Tbase[s] : byte offset of the lane's T within the global T buffer
//
// SA[] holds the suffix indices; segment s occupies SA[seg_off[s] ..
// seg_off[s]+seg_len[s]) in arbitrary order on input, sorted in place.

#ifndef STAGEB_BLK_MAX
#define STAGEB_BLK_MAX 1024     // largest len handled fully in shared memory
#endif

// One block handles one segment (warp class or shared-mem block class).
// Large-segment kernel: bitonic in GLOBAL scratch (one block per segment).
// scratch is a per-block region of size >= max padded length.
template<int BLOCK>
__global__ void sb_sort_large_kernel(
        const uint8_t* __restrict__ T,
        const uint32_t* __restrict__ seg_off,
        const uint32_t* __restrict__ seg_len,
        const uint32_t* __restrict__ seg_depth,
        const uint32_t* __restrict__ seg_n,
        const uint64_t* __restrict__ seg_Tbase,
        const uint32_t* __restrict__ big_seg_ids,   // which segments are "large"
        uint32_t num_big,
        uint32_t scratch_stride,
        uint32_t* __restrict__ scratch,
        uint32_t* __restrict__ SA) {
    uint32_t bi = blockIdx.x;
    if (bi >= num_big) return;
    uint32_t s = big_seg_ids[bi];
    int len = (int)seg_len[s];
    if (len <= 1) return;
    const uint8_t* Tl = T + seg_Tbase[s];
    int n = (int)seg_n[s];
    int depth = (int)seg_depth[s];
    uint32_t* sa = SA + seg_off[s];
    uint32_t* a = scratch + (size_t)bi * scratch_stride;
    int tid = threadIdx.x;
    int m = (int)sb_next_pow2((uint32_t)len);
    for (int i = tid; i < m; i += BLOCK)
        a[i] = (i < len) ? sa[i] : STAGEB_SENTINEL;
    __syncthreads();
    sb_bitonic_array(a, m, Tl, n, depth, tid, BLOCK);
    for (int i = tid; i < len; i += BLOCK)
        sa[i] = a[i];
}

// trivial baseline: one thread per segment, insertion sort (for benchmark).
// =====================================================================
// BIN-PACKED DISPATCH (bb_segsort style: bin by size, pack many tiny
// segments — one thread per segment — into each block).
//
// The tiny/small bins use ONE THREAD PER SEGMENT with a bounded register
// insertion sort.  Because each thread owns a full, contiguous segment,
// there is NO cross-segment contamination possible (the failure mode of
// warp-packed shuffles).  This is the dominant case (avg len ~5) and gives
// ~O(1) threads per element instead of a whole warp/block per tiny segment.
//
// `bin_ids[]` lists the segment indices assigned to this bin; `bin_count`
// is how many.  MAXLEN is the compile-time bound on segment length for the
// bin (registers sized accordingly).
// =====================================================================
template<int MAXLEN>
__global__ void sb_sort_tiny_packed_kernel(
        const uint8_t* __restrict__ T,
        const uint32_t* __restrict__ seg_off,
        const uint32_t* __restrict__ seg_len,
        const uint32_t* __restrict__ seg_depth,
        const uint32_t* __restrict__ seg_n,
        const uint64_t* __restrict__ seg_Tbase,
        const uint32_t* __restrict__ bin_ids,
        uint32_t bin_count,
        uint32_t* __restrict__ SA) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= bin_count) return;
    uint32_t s = bin_ids[t];
    int len = (int)seg_len[s];
    if (len <= 1) return;
    const uint8_t* Tl = T + seg_Tbase[s];
    int n = (int)seg_n[s];
    int depth = (int)seg_depth[s];
    uint32_t* sa = SA + seg_off[s];

    // load into registers, insertion sort (len <= MAXLEN), store back.
    uint32_t r[MAXLEN];
    #pragma unroll
    for (int i = 0; i < MAXLEN; ++i) if (i < len) r[i] = sa[i];
    // insertion sort over [0,len)
    for (int i = 1; i < len; ++i) {
        uint32_t key = r[i];
        int j = i - 1;
        while (j >= 0 && cmp_suffix(Tl, n, r[j], key, depth) > 0) {
            r[j+1] = r[j]; --j;
        }
        r[j+1] = key;
    }
    #pragma unroll
    for (int i = 0; i < MAXLEN; ++i) if (i < len) sa[i] = r[i];
}

// Medium bin: one segment per block, shared-memory bitonic (reuses the
// verified sb_bitonic_array). bin_ids selects which segments. BLOCK threads.
template<int BLOCK, int SH_MAX>
__global__ void sb_sort_medium_binned_kernel(
        const uint8_t* __restrict__ T,
        const uint32_t* __restrict__ seg_off,
        const uint32_t* __restrict__ seg_len,
        const uint32_t* __restrict__ seg_depth,
        const uint32_t* __restrict__ seg_n,
        const uint64_t* __restrict__ seg_Tbase,
        const uint32_t* __restrict__ bin_ids,
        uint32_t bin_count,
        uint32_t* __restrict__ SA) {
    __shared__ uint32_t buf[SH_MAX];
    uint32_t bi = blockIdx.x;
    if (bi >= bin_count) return;
    uint32_t s = bin_ids[bi];
    int len = (int)seg_len[s];
    if (len <= 1) return;
    const uint8_t* Tl = T + seg_Tbase[s];
    int n = (int)seg_n[s];
    int depth = (int)seg_depth[s];
    uint32_t* sa = SA + seg_off[s];
    int tid = threadIdx.x;
    int m = (int)sb_next_pow2((uint32_t)len);
    for (int i = tid; i < m; i += BLOCK) buf[i] = (i < len) ? sa[i] : STAGEB_SENTINEL;
    __syncthreads();
    sb_bitonic_array(buf, m, Tl, n, depth, tid, BLOCK);
    for (int i = tid; i < len; i += BLOCK) sa[i] = buf[i];
}

// =====================================================================
// SINGLE SOURCE OF TRUTH for bin thresholds. The kernel template params
// and the binner MUST agree; everything derives from these.
//   bin 0: len in [2, 8]      -> tiny  packed, MAXLEN=8,  1 thread/seg
//   bin 1: len in [9, 32]     -> tiny2 packed, MAXLEN=32, 1 thread/seg
//   bin 2: len in [33, 512]   -> medium, block bitonic, SH_MAX=512
//   bin 3: len in [513, 1024] -> medium-large, block bitonic, SH_MAX=1024
//   bin 4: len > 1024         -> huge, global-scratch bitonic
//   (len <= 1 : skipped, sorted trivially)
// =====================================================================
#define SB_BIN0_MAX 8
#define SB_BIN1_MAX 32
#define SB_BIN2_MAX 512
#define SB_BIN3_MAX 1024
#define SB_NBINS 5

__device__ __forceinline__ int sb_bin_of(int len) {
    if (len <= 1)            return -1;          // trivial
    if (len <= SB_BIN0_MAX)  return 0;
    if (len <= SB_BIN1_MAX)  return 1;
    if (len <= SB_BIN2_MAX)  return 2;
    if (len <= SB_BIN3_MAX)  return 3;
    return 4;
}
__host__ __forceinline__ int sb_bin_of_host(int len) {
    if (len <= 1)            return -1;
    if (len <= SB_BIN0_MAX)  return 0;
    if (len <= SB_BIN1_MAX)  return 1;
    if (len <= SB_BIN2_MAX)  return 2;
    if (len <= SB_BIN3_MAX)  return 3;
    return 4;
}

// Device binning pass 1: count segments per bin (atomic histogram).
__global__ void sb_bin_count_kernel(const uint32_t* __restrict__ seg_len,
                                    uint32_t num_segments,
                                    uint32_t* __restrict__ bin_counts /*[SB_NBINS]*/) {
    uint32_t s = blockIdx.x*blockDim.x+threadIdx.x;
    if (s >= num_segments) return;
    int b = sb_bin_of((int)seg_len[s]);
    if (b >= 0) atomicAdd(&bin_counts[b], 1u);
}

// Device binning pass 2: scatter segment ids into per-bin id lists.
// bin_base[b] is the start offset of bin b's region within a single packed
// id array; bin_cursor[b] is an atomic write cursor (init to bin_base[b]).
__global__ void sb_bin_scatter_kernel(const uint32_t* __restrict__ seg_len,
                                      uint32_t num_segments,
                                      uint32_t* __restrict__ bin_cursor /*[SB_NBINS]*/,
                                      uint32_t* __restrict__ ids_packed) {
    uint32_t s = blockIdx.x*blockDim.x+threadIdx.x;
    if (s >= num_segments) return;
    int b = sb_bin_of((int)seg_len[s]);
    if (b < 0) return;
    uint32_t pos = atomicAdd(&bin_cursor[b], 1u);
    ids_packed[pos] = s;
}

// =====================================================================
// OCTET cooperative sorter (replicates astronv CUWSORTST mechanism).
//
// 8 THREADS PER SEGMENT ("octet"). A BLOCK of OB_BLOCK threads sorts
// (OB_BLOCK/8) segments simultaneously. Each octet:
//   - segment_index = global_tid >> 3 ; lane_in_octet = tid & 7.
//   - cooperatively loads its segment's <=MAXLEN suffix indices into a
//     per-octet SHARED-MEMORY slot (padded to next-pow2 with SENTINEL =
//     +inf) -> no register-array / local-memory spill.
//   - the 8 lanes run a cooperative bitonic compare-exchange network over
//     the shared slot, synchronizing only within the octet via
//     __syncwarp(octet_mask). Comparator is the SAME byte-walk cmp_suffix
//     used by every other refine path (end-of-string smallest), so the
//     result is byte-identical to the insertion path.
//
// MAXLEN must be a power of two (8 for bin0 len<=8, 32 for bin1 len<=32).
// Signature mirrors sb_sort_tiny_packed_kernel for drop-in wiring.
// =====================================================================
template<int MAXLEN, int OB_BLOCK, bool FLOOR=false>
__global__ void sb_sort_octet_kernel(
        const uint8_t* __restrict__ T,
        const uint32_t* __restrict__ seg_off,
        const uint32_t* __restrict__ seg_len,
        const uint32_t* __restrict__ seg_depth,
        const uint32_t* __restrict__ seg_n,
        const uint64_t* __restrict__ seg_Tbase,
        const uint32_t* __restrict__ bin_ids,
        uint32_t bin_count,
        uint32_t* __restrict__ SA) {
    static constexpr int OCTETS_PER_BLOCK = OB_BLOCK / 8;
    __shared__ uint32_t s_slot[OCTETS_PER_BLOCK * MAXLEN];

    const int tid          = (int)threadIdx.x;
    const int lane         = tid & 7;
    const int octet_local  = tid >> 3;
    const uint32_t gtid    = blockIdx.x * (uint32_t)OB_BLOCK + (uint32_t)tid;
    const uint32_t seg_idx = gtid >> 3;

    if (seg_idx >= bin_count) return;

    const uint32_t s   = bin_ids[seg_idx];
    const int len      = (int)seg_len[s];
    uint32_t* slot     = s_slot + (size_t)octet_local * MAXLEN;

    const unsigned octet_mask = 0xFFu << (unsigned)((threadIdx.x & 31) & ~7);

    if (len <= 1) return;

    const uint8_t* Tl  = T + seg_Tbase[s];
    const int n        = (int)seg_n[s];
    const int depth    = (int)seg_depth[s];
    uint32_t* sa       = SA + seg_off[s];

    // length-sized network: pad only to next-pow2(len), not MAXLEN.
    const int m = (int)sb_next_pow2((uint32_t)len);   // 2..MAXLEN
    for (int i = lane; i < m; i += 8) {
        slot[i] = (i < len) ? sa[i] : STAGEB_SENTINEL;
    }
    __syncwarp(octet_mask);

    for (int k = 2; k <= m; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int idx = lane; idx < m; idx += 8) {
                int ixj = idx ^ j;
                if (ixj > idx) {
                    bool ascending = ((idx & k) == 0);
                    uint32_t x = slot[idx], y = slot[ixj];
                    bool swap;
                    if (x == y) swap = false;
                    else {
                        bool x_less = FLOOR ? (x < y) : sb_less(Tl, n, x, y, depth);
                        swap = ascending ? !x_less : x_less;
                    }
                    if (swap) { slot[idx] = y; slot[ixj] = x; }
                }
            }
            __syncwarp(octet_mask);
        }
    }

    for (int i = lane; i < len; i += 8) {
        sa[i] = slot[i];
    }
}






// =====================================================================
// ZLCP: LCP-aware insertion sort (comparison sort, immune to the
// level-count wall that kills counting sort on AstroBWTv3's slow shrink
// curve). One thread per segment, bounded len, registers r[MAXLEN] +
// parallel L[MAXLEN] holding LCP(r[i-1], r[i]) for the sorted prefix.
// LCP lets the inner loop SKIP re-walking the shared prefix on most
// comparisons (the byte cost the naive byte-walk insertion pays O(len^2)
// times). Same ordering semantics as cmp_suffix (end-of-string smallest),
// so the existing oracle catches any bug. mismatch MUST be 0.
//

// cmp_suffix_lcp_wide: parity-identical to cmp_suffix_lcp but issues 4 independent
// 8-byte loads per side in parallel (32B/iteration) to add memory-level parallelism
// to deep-LCP byte-walks (the refine bottleneck). Compares in byte order; returns at
// first mismatch with identical sign + lcp, so the sort order is unchanged.
__device__ __forceinline__ int cmp_suffix_lcp_wide(const uint8_t* T, int n,
                                                   uint32_t a, uint32_t b,
                                                   int from, int* out_lcp) {
    const uint32_t un = (uint32_t)n;
    uint32_t aa = a + (uint32_t)from;
    uint32_t bb = b + (uint32_t)from;
    while (aa + 32u <= un && bb + 32u <= un) {
        uint64_t a0=sb_load_u64(T+aa),     b0=sb_load_u64(T+bb);
        uint64_t a1=sb_load_u64(T+aa+8),   b1=sb_load_u64(T+bb+8);
        uint64_t a2=sb_load_u64(T+aa+16),  b2=sb_load_u64(T+bb+16);
        uint64_t a3=sb_load_u64(T+aa+24),  b3=sb_load_u64(T+bb+24);
        uint64_t av, bv; uint32_t off;
        if      (a0!=b0){ av=a0; bv=b0; off=0;  }
        else if (a1!=b1){ av=a1; bv=b1; off=8;  }
        else if (a2!=b2){ av=a2; bv=b2; off=16; }
        else if (a3!=b3){ av=a3; bv=b3; off=24; }
        else { aa+=32; bb+=32; continue; }
        #pragma unroll
        for (int k=0;k<8;++k){ uint8_t ca=(uint8_t)(av>>(8*k)), cb=(uint8_t)(bv>>(8*k));
            if (ca!=cb){ *out_lcp=(int)(aa+off+(uint32_t)k - a); return ca<cb?-1:1; } }
    }
    while (aa + 8u <= un && bb + 8u <= un) {
        uint64_t av=sb_load_u64(T+aa), bv=sb_load_u64(T+bb);
        if (av!=bv){
            #pragma unroll
            for (int k=0;k<8;++k){ uint8_t ca=(uint8_t)(av>>(8*k)),cb=(uint8_t)(bv>>(8*k));
                if(ca!=cb){*out_lcp=(int)(aa+(uint32_t)k-a);return ca<cb?-1:1;} } }
        aa+=8; bb+=8;
    }
    while (aa + 4u <= un && bb + 4u <= un) {
        uint32_t av=sb_load_u32(T+aa), bv=sb_load_u32(T+bb);
        if (av!=bv){
            #pragma unroll
            for(int k=0;k<4;++k){uint8_t ca=(uint8_t)(av>>(8*k)),cb=(uint8_t)(bv>>(8*k));
                if(ca!=cb){*out_lcp=(int)(aa+(uint32_t)k-a);return ca<cb?-1:1;}} }
        aa+=4; bb+=4;
    }
    while (aa<un && bb<un){ uint8_t ca=T[aa],cb=T[bb]; if(ca!=cb){*out_lcp=(int)(aa-a);return ca<cb?-1:1;} ++aa;++bb; }
    *out_lcp=(int)(aa-a);
    if (aa>=un && bb>=un){ uint32_t ra=un-a,rb=un-b; return ra<rb?-1:(ra>rb?1:0); }
    return (aa>=un)?-1:1;
}

// cmp_suffix_lcp: compares suffixes a,b from byte position `from` (an
// absolute offset from each suffix's start, i.e. both are known equal on
// [0,from)). Returns <0,0,>0 like cmp_suffix and writes the resulting LCP
// (total matched bytes from suffix start, >= from) to *out_lcp.
// =====================================================================
__device__ __forceinline__ int cmp_suffix_lcp(const uint8_t* T, int n,
                                              uint32_t a, uint32_t b,
                                              int from, int* out_lcp) {
    if (g_sb_fakecmp) { *out_lcp = 0; return (a < b) ? -1 : 1; }
    if (g_sb_widecmp) return cmp_suffix_lcp_wide(T, n, a, b, from, out_lcp);
    const uint32_t un = (uint32_t)n;
    uint32_t aa = a + (uint32_t)from;
    uint32_t bb = b + (uint32_t)from;
    // 8-byte strides while both have >=8 bytes left.
    while (aa + 8u <= un && bb + 8u <= un) {
        uint64_t av = sb_load_u64(T + aa);
        uint64_t bv = sb_load_u64(T + bb);
        if (av != bv) {
            if (g_sb_fastdiff) {
                const uint64_t diff = av ^ bv;
                const int byte = (__ffsll((long long)diff) - 1) >> 3;
                const uint8_t ca = (uint8_t)(av >> (8 * byte));
                const uint8_t cb = (uint8_t)(bv >> (8 * byte));
                *out_lcp = (int)(aa + (uint32_t)byte - a);
                return ca < cb ? -1 : 1;
            }
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                uint8_t ca = (uint8_t)(av >> (8*k));
                uint8_t cb = (uint8_t)(bv >> (8*k));
                if (ca != cb) { *out_lcp = (int)(aa + (uint32_t)k - a); return ca < cb ? -1 : 1; }
            }
        }
        aa += 8; bb += 8;
    }
    while (aa + 4u <= un && bb + 4u <= un) {
        uint32_t av = sb_load_u32(T + aa);
        uint32_t bv = sb_load_u32(T + bb);
        if (av != bv) {
            if (g_sb_fastdiff) {
                const uint32_t diff = av ^ bv;
                const int byte = (__ffs((int)diff) - 1) >> 3;
                const uint8_t ca = (uint8_t)(av >> (8 * byte));
                const uint8_t cb = (uint8_t)(bv >> (8 * byte));
                *out_lcp = (int)(aa + (uint32_t)byte - a);
                return ca < cb ? -1 : 1;
            }
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                uint8_t ca = (uint8_t)(av >> (8*k));
                uint8_t cb = (uint8_t)(bv >> (8*k));
                if (ca != cb) { *out_lcp = (int)(aa + (uint32_t)k - a); return ca < cb ? -1 : 1; }
            }
        }
        aa += 4; bb += 4;
    }
    while (aa < un && bb < un) {
        uint8_t ca = T[aa], cb = T[bb];
        if (ca != cb) { *out_lcp = (int)(aa - a); return ca < cb ? -1 : 1; }
        ++aa; ++bb;
    }
    // one or both ran out: shorter remaining suffix is SMALLER.
    *out_lcp = (int)(aa - a);   // == matched length (both advanced equally)
    if (aa >= un && bb >= un) {
        uint32_t ra = un - a, rb = un - b;
        return ra < rb ? -1 : (ra > rb ? 1 : 0);
    }
    return (aa >= un) ? -1 : 1;
}

template<int MAXLEN, bool HALF=false>
__global__ void sb_sort_lcp_packed_kernel(
        const uint8_t* __restrict__ T,
        const uint32_t* __restrict__ seg_off,
        const uint32_t* __restrict__ seg_len,
        const uint32_t* __restrict__ seg_depth,
        const uint32_t* __restrict__ seg_n,
        const uint64_t* __restrict__ seg_Tbase,
        const uint32_t* __restrict__ bin_ids,
        uint32_t bin_count,
        uint32_t* __restrict__ SA) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= bin_count) return;
    uint32_t s = bin_ids[t];
    int len = (int)seg_len[s];
    if (len <= 1) return;
    const uint8_t* Tl = T + seg_Tbase[s];
    int n = (int)seg_n[s];
    int depth = (int)seg_depth[s];
    uint32_t* sa = SA + seg_off[s];

    uint32_t r[MAXLEN];   // sorted suffixes
    int      L[MAXLEN];   // L[k] = LCP(r[k-1], r[k]) for k in [1,size); L[0]=0
    if (HALF) {
        // PERF-ONLY lower bound for a 2-way split: sort [0,len/2) then
        // [len/2,len) as two INDEPENDENT LCP insertions, NO merge. Result is
        // intentionally NOT globally sorted (expect mismatch>0); cost is a
        // true lower bound on any 2-way scheme (merge only adds).
        const int mid = len >> 1;
        #pragma unroll
        for (int i = 0; i < MAXLEN; ++i) if (i < len) r[i] = sa[i];
        for (int pass = 0; pass < 2; ++pass) {
            const int lo = pass ? mid : 0;
            const int hi = pass ? len : mid;
            L[lo] = 0;
            for (int i = lo + 1; i < hi; ++i) {
                const uint32_t key = r[i];
                int h = 0;
                int c = cmp_suffix_lcp(Tl, n, key, r[i-1], depth, &h);
                if (c > 0) { L[i] = h; continue; }
                int right_lcp = h;
                int j = i - 1;
                r[j+1] = r[j];
                for (;;) {
                    const int Lj = L[j];
                    if (j == lo) { break; }
                    if (Lj > h) { right_lcp = h; r[j] = r[j-1]; L[j+1] = L[j]; --j; continue; }
                    else if (Lj < h) { right_lcp = Lj; break; }
                    else { int nl=0; int cc=cmp_suffix_lcp(Tl,n,key,r[j-1],h,&nl);
                           if (cc>0){ right_lcp=nl; break; } else { h=nl; right_lcp=nl; r[j]=r[j-1]; L[j+1]=L[j]; --j; } }
                }
                r[j] = key;
                if (j > lo) L[j] = right_lcp;
                L[j+1] = h;
            }
        }
        // --- LCP-aware merge of the two sorted halves [0,mid) and [mid,len) ---
        // Track la=LCP(z,headA), lb=LCP(z,headB) vs last-emitted z. la>lb means
        // headA<headB (A matches z past where B diverged). Only la==lb needs a
        // byte-walk. When we emit from a run, the new z is the just-emitted head;
        // the new head LCP with z is that run stored L[]; the other run l is
        // unchanged by the LCP triangle property (loser first-differs from z at
        // min(la,lb), which equals its own l, so its LCP with the new z is the same).
        {
            uint32_t mr[MAXLEN];
            int      mL[MAXLEN];
            int ia = 0, ib = mid, k = 0;
            int la, lb;
            int seed = 0;
            int c0 = cmp_suffix_lcp(Tl, n, r[ia], r[ib], depth, &seed);
            if (c0 < 0) {
                mr[k] = r[ia]; mL[k] = 0; ++k; ++ia;
                la = (ia < mid) ? L[ia] : -1;
                lb = seed;
            } else {
                mr[k] = r[ib]; mL[k] = 0; ++k; ++ib;
                lb = (ib < len) ? L[ib] : -1;
                la = seed;
            }
            while (ia < mid && ib < len) {
                if (la > lb) {
                    mL[k] = la; mr[k] = r[ia]; ++k; ++ia;
                    la = (ia < mid) ? L[ia] : -1;
                } else if (lb > la) {
                    mL[k] = lb; mr[k] = r[ib]; ++k; ++ib;
                    lb = (ib < len) ? L[ib] : -1;
                } else {
                    int split = 0;
                    int cc = cmp_suffix_lcp(Tl, n, r[ia], r[ib], la, &split);
                    if (cc < 0) { mL[k]=la; mr[k]=r[ia]; ++k; ++ia; la=(ia<mid)?L[ia]:-1; lb=split; }
                    else        { mL[k]=lb; mr[k]=r[ib]; ++k; ++ib; lb=(ib<len)?L[ib]:-1; la=split; }
                }
            }
            while (ia < mid) { mL[k]=la; mr[k]=r[ia]; ++k; ++ia; la=(ia<mid)?L[ia]:-1; }
            while (ib < len) { mL[k]=lb; mr[k]=r[ib]; ++k; ++ib; lb=(ib<len)?L[ib]:-1; }
            #pragma unroll
            for (int i = 0; i < MAXLEN; ++i) if (i < len) { r[i]=mr[i]; L[i]=mL[i]; }
        }
        #pragma unroll
        for (int i = 0; i < MAXLEN; ++i) if (i < len) sa[i] = r[i];
        return;
    }
    #pragma unroll
    for (int i = 0; i < MAXLEN; ++i) if (i < len) r[i] = sa[i];
    L[0] = 0;

    // LCP-aware insertion sort. Invariant before iteration i: r[0..i-1] sorted
    // ascending and L[1..i-1] hold adjacent LCPs. Insert key=r[i].
    for (int i = 1; i < len; ++i) {
        const uint32_t key = r[i];
        // Compare key vs current right neighbor r[i-1] from `depth`.
        int h = 0;   // running LCP(key, r[j]) for the element last compared
        int c = cmp_suffix_lcp(Tl, n, key, r[i-1], depth, &h);
        if (c > 0) {            // key > r[i-1]: stays at i (c==0 impossible).
            L[i] = h;           // LCP(r[i-1], key)
            continue;
        }
        // key < r[i-1]: it must move left. h = LCP(key, r[i-1]).
        // right_lcp = LCP(key, the element currently to key's right) — starts
        // as LCP(key, r[i-1]) = h, updated each time we pass an element left.
        int right_lcp = h;      // LCP(key, r[j+1]) once key lands at j+1
        int j = i - 1;
        // shift r[i-1] up to slot i (its right-LCP becomes LCP(key,r[i-1])=h).
        r[j+1] = r[j];
        // walk left.
        for (;;) {
            const int Lj = L[j];        // LCP(r[j-1], r[j]) (unchanged by shifts left of j)
            if (j == 0) {               // no further left neighbor.
                break;
            }
            if (Lj > h) {
                // r[j-1] agrees with r[j] beyond key's divergence => key < r[j-1];
                // LCP(key, r[j-1]) == h. Skip compare. r[j-1]'s right-LCP (with
                // the element now at j) stays Lj. Shift and continue.
                right_lcp = h;          // LCP(key, r[j]) stays h as key passes r[j]
                r[j] = r[j-1];
                L[j+1] = L[j];          // adjacency (r[j], r[j+1]) carries old L[j]
                --j;
                continue;
            } else if (Lj < h) {
                // r[j-1] diverges from r[j] before key did => key > r[j-1]. Stop.
                // key lands at slot j. LCP(key, r[j]) == Lj.
                right_lcp = Lj;
                break;
            } else {
                // Lj == h: tie at known prefix; compare key vs r[j-1] from h.
                int nl = 0;
                int cc = cmp_suffix_lcp(Tl, n, key, r[j-1], h, &nl);
                if (cc > 0) {           // key > r[j-1]: stop. key lands at slot j.
                    right_lcp = nl;     // LCP(key, r[j])  (== nl since key vs r[j-1] split at nl >= h)
                    break;
                } else {                // key < r[j-1]: shift, continue.
                    h = nl;
                    right_lcp = nl;     // LCP(key, r[j]) becomes nl
                    r[j] = r[j-1];
                    L[j+1] = L[j];
                    --j;
                }
            }
        }
        // key lands at slot j. Set adjacency LCPs:
        //   L[j]   = LCP(r[j-1], key)  (left neighbor, if j>0) = h
        //   L[j+1] = LCP(key, r[j+1])  (right neighbor)        = right_lcp
        //   L[j]   = LCP(r[j-1], key) = split value (right_lcp)
        //   L[j+1] = LCP(key, r[j+1]) = h (the element key stepped over)
        r[j] = key;
        if (j > 0)   L[j]   = right_lcp;
        L[j+1] = h;
    }

    #pragma unroll
    for (int i = 0; i < MAXLEN; ++i) if (i < len) sa[i] = r[i];
}

// =====================================================================
// BOTTOM-UP LCP MERGE-SORT (generalizes the 2-way HALF merge to split
// DEPTH d). 2^d balanced leaf runs are LCP-insertion-sorted, then merged
// bottom-up with the frontier-LCP merge (ping-pong buffers r<->m). d is a
// runtime arg: d=1 reproduces HALF exactly (regression anchor). Same
// ordering as cmp_suffix (end-of-string smallest); ZLCP oracle => mismatch 0.
// =====================================================================
// Merge adjacent sorted runs src[a0,a1) and src[a1,a2) into dst[a0,a2).
// Each run starts at L=0; Ldst gets the merged adjacency LCPs (Ldst[a0]=0).
__device__ __forceinline__ void sb_lcp_merge_runs(
        const uint8_t* Tl, int n, int depth,
        const uint32_t* src, const int* Lsrc,
        uint32_t* dst, int* Ldst,
        int a0, int a1, int a2) {
    int ia = a0, ib = a1, k = a0;
    int la, lb;
    int seed = 0;
    int c0 = cmp_suffix_lcp(Tl, n, src[ia], src[ib], depth, &seed);
    if (c0 < 0) {
        dst[k] = src[ia]; Ldst[k] = 0; ++k; ++ia;
        la = (ia < a1) ? Lsrc[ia] : -1;
        lb = seed;
    } else {
        dst[k] = src[ib]; Ldst[k] = 0; ++k; ++ib;
        lb = (ib < a2) ? Lsrc[ib] : -1;
        la = seed;
    }
    while (ia < a1 && ib < a2) {
        if (la > lb) {
            Ldst[k] = la; dst[k] = src[ia]; ++k; ++ia;
            la = (ia < a1) ? Lsrc[ia] : -1;
        } else if (lb > la) {
            Ldst[k] = lb; dst[k] = src[ib]; ++k; ++ib;
            lb = (ib < a2) ? Lsrc[ib] : -1;
        } else {
            int split = 0;
            int cc = cmp_suffix_lcp(Tl, n, src[ia], src[ib], la, &split);
            if (cc < 0) { Ldst[k]=la; dst[k]=src[ia]; ++k; ++ia; la=(ia<a1)?Lsrc[ia]:-1; lb=split; }
            else        { Ldst[k]=lb; dst[k]=src[ib]; ++k; ++ib; lb=(ib<a2)?Lsrc[ib]:-1; la=split; }
        }
    }
    while (ia < a1) { Ldst[k]=la; dst[k]=src[ia]; ++k; ++ia; la=(ia<a1)?Lsrc[ia]:-1; }
    while (ib < a2) { Ldst[k]=lb; dst[k]=src[ib]; ++k; ++ib; lb=(ib<a2)?Lsrc[ib]:-1; }
}

// LCP-insertion-sort the run [lo,hi) in place in r[], filling L[lo..hi)
// (L[lo]=0). Same algorithm as the non-HALF kernel, restricted to a run.
template<int MAXLEN>
__device__ __forceinline__ void sb_lcp_insort_run(
        const uint8_t* Tl, int n, int depth,
        uint32_t* r, int* L, int lo, int hi) {
    L[lo] = 0;
    for (int i = lo + 1; i < hi; ++i) {
        const uint32_t key = r[i];
        int h = 0;
        int c = cmp_suffix_lcp(Tl, n, key, r[i-1], depth, &h);
        if (c > 0) { L[i] = h; continue; }
        int right_lcp = h;
        int j = i - 1;
        r[j+1] = r[j];
        for (;;) {
            const int Lj = L[j];
            if (j == lo) { break; }
            if (Lj > h) { right_lcp = h; r[j] = r[j-1]; L[j+1] = L[j]; --j; continue; }
            else if (Lj < h) { right_lcp = Lj; break; }
            else { int nl=0; int cc=cmp_suffix_lcp(Tl,n,key,r[j-1],h,&nl);
                   if (cc>0){ right_lcp=nl; break; } else { h=nl; right_lcp=nl; r[j]=r[j-1]; L[j+1]=L[j]; --j; } }
        }
        r[j] = key;
        if (j > lo) L[j] = right_lcp;
        L[j+1] = h;
    }
}

template<int MAXLEN>
__global__ void sb_sort_lcp_msort_kernel(
        const uint8_t* __restrict__ T,
        const uint32_t* __restrict__ seg_off,
        const uint32_t* __restrict__ seg_len,
        const uint32_t* __restrict__ seg_depth,
        const uint32_t* __restrict__ seg_n,
        const uint64_t* __restrict__ seg_Tbase,
        const uint32_t* __restrict__ bin_ids,
        uint32_t bin_count,
        int d,
        uint32_t* __restrict__ SA) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= bin_count) return;
    uint32_t s = bin_ids[t];
    int len = (int)seg_len[s];
    if (len <= 1) return;
    const uint8_t* Tl = T + seg_Tbase[s];
    int n = (int)seg_n[s];
    int depth = (int)seg_depth[s];
    uint32_t* sa = SA + seg_off[s];

    uint32_t r[MAXLEN];
    int      L[MAXLEN];
    uint32_t m[MAXLEN];
    int      Lm[MAXLEN];
    #pragma unroll
    for (int i = 0; i < MAXLEN; ++i) if (i < len) r[i] = sa[i];

    // cap leaf count so each leaf has >=1 element; nleaves = min(2^d, len).
    int nleaves = 1 << d;
    if (nleaves > len) nleaves = len;
    // balanced boundaries bnd[i] = (i*len)/nleaves; insertion-sort each leaf.
    for (int li = 0; li < nleaves; ++li) {
        int lo = (int)(((long)li * len) / nleaves);
        int hi = (int)(((long)(li+1) * len) / nleaves);
        sb_lcp_insort_run<MAXLEN>(Tl, n, depth, r, L, lo, hi);
    }

    // bottom-up merge: runs double each level. Ping-pong r<->m.
    // 'src' points at the buffer holding current runs; flip per level.
    uint32_t* src = r;  int* Ls = L;
    uint32_t* dst = m;  int* Ld = Lm;
    int width = 1;  // current run count is ceil(nleaves/width-ish); iterate by run size in leaf units
    // run size in ELEMENTS doubles via leaf pairing; track leaf-index step.
    int runs = nleaves;
    int leafstep = 1;
    while (runs > 1) {
        int li = 0;
        while (li < nleaves) {
            int a0 = (int)(((long)li * len) / nleaves);
            int lmid = li + leafstep;
            if (lmid >= nleaves) {
                // lone trailing run: copy through preserving L.
                int a2 = len;
                for (int x = a0; x < a2; ++x) { dst[x] = src[x]; Ld[x] = Ls[x]; }
                li = nleaves;
            } else {
                int a1 = (int)(((long)lmid * len) / nleaves);
                int lend = li + 2*leafstep;
                if (lend > nleaves) lend = nleaves;
                int a2 = (int)(((long)lend * len) / nleaves);
                sb_lcp_merge_runs(Tl, n, depth, src, Ls, dst, Ld, a0, a1, a2);
                li = lend;
            }
        }
        // swap buffers
        uint32_t* ts=src; src=dst; dst=ts;
        int* tls=Ls; Ls=Ld; Ld=tls;
        leafstep <<= 1;
        runs = (runs + 1) >> 1;
        (void)width;
    }
    // src now holds the fully merged run.
    #pragma unroll
    for (int i = 0; i < MAXLEN; ++i) if (i < len) sa[i] = src[i];
}

#endif // STAGEB_REFINE_CUH
