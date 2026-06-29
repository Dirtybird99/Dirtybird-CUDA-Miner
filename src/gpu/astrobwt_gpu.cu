// OpenAstronV — GPU-accelerated AstroBWT v3 mining
// Exact mode follows the current official DERO v3 source semantics.
// Fast mode is an optimized experimental path and must be verified separately.

#include "astrobwt_gpu.cuh"
#include "crypto/sha256.h"
#include "crypto/salsa20.h"
#include "crypto/rc4.h"
#include "crypto/hash_utils.h"
#include "crypto/sais.h"

extern "C" {
#include "libsais.h"
}
#include "crypto/astrobwt.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "gpu/stageb_refine.cuh"   // Stage-B bin-packed cooperative segment sorter
#include "gpu/stageb_perhash_coop.cuh"  // PERHASH: per-hash fused refine (astronv CUFES+CUFTT mapping)
#include "gpu/stageb_perhash_msort.cuh"  // PHMSORT: per-hash whole-array cooperative merge-path mergesort (CUZSORT replica)
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <array>
#include <mutex>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)

struct GPUContext {
    int device_id;
    int batch_size;
    int block_size;
    int total_sa_elements;
    GPUEngineMode engine_mode;
    int staged_subbatch;
    bool perf_logging;

    uint8_t*  d_sdata;
    int32_t*  d_data_len;
    int32_t*  d_sa;
    uint64_t* d_solutions;

    uint64_t* d_keys_in;
    uint64_t* d_keys_out;
    uint32_t* d_vals_in;
    uint32_t* d_vals_out;
    uint32_t* d_bucket_counts;
    uint32_t* d_ambiguous_counts;
    uint32_t* d_max_bucket_sizes;
    uint32_t* d_bucket_total_count;
    uint64_t* d_bucket_descs;
    int*      d_offsets;
    void*     d_sort_temp;
    size_t    sort_temp_size;

    // ---- Stage-B cooperative refine (bin-packed segment sorter) ----
    // Per-segment metadata arrays (cap = sb_seg_cap segments).
    uint32_t* d_sb_seg_off;
    uint32_t* d_sb_seg_len;
    uint32_t* d_sb_seg_depth;
    uint32_t* d_sb_seg_n;
    uint64_t* d_sb_seg_Tbase;
    uint32_t* d_sb_ids_packed;     // packed per-bin segment-id lists
    uint32_t* d_sb_bin_counts;     // [SB_NBINS]
    uint32_t* d_sb_bin_base;       // [SB_NBINS] (exclusive prefix of counts)
    uint32_t* d_sb_bin_cursor;     // [SB_NBINS] (atomic scatter cursor)
    uint32_t* d_sb_split_counts;   // [SB_SPLIT_NBINS] for env-gated bin1 split tests
    uint32_t* d_sb_split_cursor;   // [SB_SPLIT_NBINS] for env-gated bin1 split tests
    uint32_t* d_sb_max_len;        // [1] atomicMax of seg_len (for bin4 stride)
    uint32_t* d_sb_scratch;        // bin4 global-scratch bitonic workspace
    uint32_t* h_sb_meta;           // pinned: SB_NBINS counts + 1 max_len
    uint32_t* h_sb_split_meta;     // pinned: SB_SPLIT_NBINS counts + 1 max_len
    size_t    sb_seg_cap;          // max segments handled by Stage-B
    size_t    sb_scratch_cap_elems;// total uint32 elements in d_sb_scratch

    uint64_t* h_solutions;
    int32_t*  h_data_len;
    uint32_t* h_bucket_counts;
    uint32_t* h_ambiguous_counts;
    uint32_t* h_max_bucket_sizes;

    uint8_t work[112];
    uint64_t difficulty;
    cudaStream_t stream;
};

static constexpr int kInitialPrefixSymbols = 2;  // optimal: symbols=3 re-tested after exact-tiny/BIN2LCP = net loss (8.43 vs 8.78 KH/s; radix sort +53ms, refine -33ms).
static constexpr int kSortEndBit = kInitialPrefixSymbols * 9;
static constexpr int kFastPrefixBytes = 4;
[[maybe_unused]] static constexpr int kDefaultStagedSubbatch = 32;
static constexpr uint32_t kBucketLenBits = 17;
static constexpr uint32_t kBucketStartBits = 17;
static constexpr uint32_t kBucketDepthBits = 17;

struct GPUSABuildStats {
    float offsets_ms = 0.0f;
    float keys_ms = 0.0f;
    float sort_ms = 0.0f;
    float refine_ms = 0.0f;
    uint32_t bucket_count = 0;
    uint32_t ambiguous_count = 0;
    uint32_t max_bucket_size = 0;
};

static inline bool gpu_engine_is_exact(GPUEngineMode mode) {
    return mode == GPUEngineMode::Exact;
}

static inline bool gpu_engine_is_recovered(GPUEngineMode mode) {
    return mode == GPUEngineMode::Recovered;
}

static inline bool gpu_engine_is_staged(GPUEngineMode mode) {
    return mode == GPUEngineMode::Staged;
}

static inline bool gpu_engine_is_cleanroom(GPUEngineMode mode) {
    return mode == GPUEngineMode::Cleanroom;
}

static const char* gpu_engine_mode_name(GPUEngineMode mode) {
    switch (mode) {
        case GPUEngineMode::Exact: return "exact";
        case GPUEngineMode::Recovered: return "recovered";
        case GPUEngineMode::Staged: return "staged";
        case GPUEngineMode::Cleanroom: return "cleanroom";
    }
    return "unknown";
}

__constant__ uint32_t d_CodeLUT[257];
__constant__ uint8_t  d_work_template[48]; 
__constant__ uint8_t  d_difficulty_target[32];
__device__   uint32_t d_near_miss_count; 

__device__ uint32_t d_rotl(uint32_t v, int n) { return (v << n) | (v >> (32 - n)); }
__device__ uint8_t d_rl8(uint8_t x, int y) { int s = y & 7; return (uint8_t)((x << s) | (x >> (8 - s))); }
__device__ uint8_t d_reverse8(uint8_t b) { b = (uint8_t)(((b & 0xAA) >> 1) | ((b & 0x55) << 1)); b = (uint8_t)(((b & 0xCC) >> 2) | ((b & 0x33) << 2)); b = (uint8_t)(((b & 0xF0) >> 4) | ((b & 0x0F) << 4)); return b; }
__device__ uint32_t d_rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ uint32_t d_bswap32(uint32_t x) {
    return (x >> 24) |
           ((x >> 8) & 0x0000FF00U) |
           ((x << 8) & 0x00FF0000U) |
           (x << 24);
}

__device__ __forceinline__ uint32_t d_load_u32_unaligned(const uint8_t* p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

__device__ __forceinline__ uint64_t d_load_u64_unaligned(const uint8_t* p) {
    return (uint64_t)d_load_u32_unaligned(p) |
           ((uint64_t)d_load_u32_unaligned(p + 4) << 32);
}

__device__ __forceinline__ uint64_t d_pack_bucket_desc(uint32_t hash_id,
                                                       uint32_t start,
                                                       uint32_t len,
                                                       uint32_t depth) {
    return ((uint64_t)hash_id << (kBucketDepthBits + kBucketStartBits + kBucketLenBits)) |
           ((uint64_t)depth << (kBucketStartBits + kBucketLenBits)) |
           ((uint64_t)start << kBucketLenBits) |
           (uint64_t)len;
}

__device__ __forceinline__ void d_unpack_bucket_desc(uint64_t desc,
                                                     uint32_t& hash_id,
                                                     uint32_t& start,
                                                     uint32_t& len,
                                                     uint32_t& depth) {
    const uint64_t len_mask = (1ULL << kBucketLenBits) - 1ULL;
    const uint64_t start_mask = (1ULL << kBucketStartBits) - 1ULL;
    const uint64_t depth_mask = (1ULL << kBucketDepthBits) - 1ULL;
    len = (uint32_t)(desc & len_mask);
    start = (uint32_t)((desc >> kBucketLenBits) & start_mask);
    depth = (uint32_t)((desc >> (kBucketLenBits + kBucketStartBits)) & depth_mask);
    hash_id = (uint32_t)(desc >> (kBucketLenBits + kBucketStartBits + kBucketDepthBits));
}

static __device__ const uint32_t d_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ uint8_t d_wolfBranch(uint8_t val, uint8_t pos2_val, uint32_t opcode) {
    #pragma unroll 4
    for (int i = 3; i >= 0; --i) {
        uint8_t insn = (opcode >> (i << 3)) & 0xFF;
        switch (insn) {
            case 0:  val = val + val; break;
            case 1:  val = val - (val ^ 97); break;
            case 2:  val = val * val; break;
            case 3:  val = val ^ pos2_val; break;
            case 4:  val = ~val; break;
            case 5:  val = val & pos2_val; break;
            case 6:  val = (uint8_t)(val << (val & 3)); break;
            case 7:  val = (uint8_t)(val >> (val & 3)); break;
            case 8:  val = d_reverse8(val); break;
            case 9:  val = val ^ (uint8_t)__popc((unsigned int)val); break;
            case 10: val = d_rl8(val, val); break;
            case 11: val = d_rl8(val, 1); break;
            case 12: val = val ^ d_rl8(val, 2); break;
            case 13: val = d_rl8(val, 3); break;
            case 14: val = val ^ d_rl8(val, 4); break;
            case 15: val = d_rl8(val, 5); break;
        }
    }
    return val;
}

__device__ void d_sha256(const uint8_t* data, int len, uint8_t hash[32]) {
    uint32_t state[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint8_t buf[128]; int off = 0;
    while (off + 64 <= len) {
        uint32_t W[16]; uint32_t a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
        for (int i = 0; i < 64; i++) {
            uint32_t w_i;
            if (i < 16) { w_i = ((uint32_t)data[off+i*4]<<24)|((uint32_t)data[off+i*4+1]<<16)|((uint32_t)data[off+i*4+2]<<8)|data[off+i*4+3]; W[i] = w_i; }
            else { uint32_t w15 = W[(i-15)&15], w2 = W[(i-2)&15]; uint32_t s0 = d_rotr(w15,7)^d_rotr(w15,18)^(w15>>3); uint32_t s1 = d_rotr(w2,17)^d_rotr(w2,19)^(w2>>10); w_i = W[(i-16)&15] + s0 + W[(i-7)&15] + s1; W[i&15] = w_i; }
            uint32_t S1 = d_rotr(e,6)^d_rotr(e,11)^d_rotr(e,25); uint32_t ch = (e&f)^(~e&g); uint32_t t1 = h+S1+ch+d_K[i]+w_i; uint32_t S0 = d_rotr(a,2)^d_rotr(a,13)^d_rotr(a,22); uint32_t mj = (a&b)^(a&c)^(b&c); uint32_t t2 = S0+mj; h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
        off += 64;
    }
    int rem = len - off; for (int i = 0; i < rem; i++) buf[i] = data[off + i];
    buf[rem] = 0x80; int padlen = (rem < 56) ? (56 - rem - 1) : (120 - rem - 1);
    for (int i = 0; i < padlen; i++) buf[rem + 1 + i] = 0;
    uint64_t total = (uint64_t)len * 8; int finoff = rem + 1 + padlen;
    buf[finoff+0]=(uint8_t)(total>>56); buf[finoff+1]=(uint8_t)(total>>48); buf[finoff+2]=(uint8_t)(total>>40); buf[finoff+3]=(uint8_t)(total>>32);
    buf[finoff+4]=(uint8_t)(total>>24); buf[finoff+5]=(uint8_t)(total>>16); buf[finoff+6]=(uint8_t)(total>>8);  buf[finoff+7]=(uint8_t)total;
    int blocks = (finoff + 8) / 64;
    for (int blk = 0; blk < blocks; blk++) {
        uint32_t W[16]; uint32_t a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
        for (int i = 0; i < 64; i++) {
            uint32_t w_i;
            if (i < 16) { int p = blk*64+i*4; w_i = ((uint32_t)buf[p]<<24)|((uint32_t)buf[p+1]<<16)|((uint32_t)buf[p+2]<<8)|buf[p+3]; W[i] = w_i; }
            else { uint32_t w15 = W[(i-15)&15], w2 = W[(i-2)&15]; uint32_t s0 = d_rotr(w15,7)^d_rotr(w15,18)^(w15>>3); uint32_t s1 = d_rotr(w2,17)^d_rotr(w2,19)^(w2>>10); w_i = W[(i-16)&15] + s0 + W[(i-7)&15] + s1; W[i&15] = w_i; }
            uint32_t S1 = d_rotr(e,6)^d_rotr(e,11)^d_rotr(e,25); uint32_t ch = (e&f)^(~e&g); uint32_t t1 = h+S1+ch+d_K[i]+w_i; uint32_t S0 = d_rotr(a,2)^d_rotr(a,13)^d_rotr(a,22); uint32_t mj = (a&b)^(a&c)^(b&c); uint32_t t2 = S0+mj; h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
    }
    for (int i = 0; i < 8; i++) { hash[i*4]=(uint8_t)(state[i]>>24); hash[i*4+1]=(uint8_t)(state[i]>>16); hash[i*4+2]=(uint8_t)(state[i]>>8); hash[i*4+3]=(uint8_t)state[i]; }
}

__device__ void d_sha256_words_le(const uint32_t* data_words, int word_count, uint8_t hash[32]) {
    uint32_t state[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    const int full_blocks = word_count >> 4;
    for (int blk = 0; blk < full_blocks; ++blk) {
        uint32_t W[16];
        uint32_t a = state[0], b = state[1], c = state[2], d = state[3], e = state[4], f = state[5], g = state[6], h = state[7];
        const uint32_t* src = data_words + (blk << 4);
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            uint32_t w_i;
            if (i < 16) {
                w_i = d_bswap32(src[i]);
                W[i] = w_i;
            } else {
                const uint32_t w15 = W[(i - 15) & 15];
                const uint32_t w2 = W[(i - 2) & 15];
                const uint32_t s0 = d_rotr(w15, 7) ^ d_rotr(w15, 18) ^ (w15 >> 3);
                const uint32_t s1 = d_rotr(w2, 17) ^ d_rotr(w2, 19) ^ (w2 >> 10);
                w_i = W[(i - 16) & 15] + s0 + W[(i - 7) & 15] + s1;
                W[i & 15] = w_i;
            }
            const uint32_t S1 = d_rotr(e, 6) ^ d_rotr(e, 11) ^ d_rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + S1 + ch + d_K[i] + w_i;
            const uint32_t S0 = d_rotr(a, 2) ^ d_rotr(a, 13) ^ d_rotr(a, 22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }

    uint8_t buf[128];
    const int rem_words = word_count & 15;
    const uint32_t* rem_src = data_words + (full_blocks << 4);
    int rem = rem_words * 4;
    for (int i = 0; i < rem_words; ++i) {
        const uint32_t w = rem_src[i];
        buf[i * 4 + 0] = (uint8_t)w;
        buf[i * 4 + 1] = (uint8_t)(w >> 8);
        buf[i * 4 + 2] = (uint8_t)(w >> 16);
        buf[i * 4 + 3] = (uint8_t)(w >> 24);
    }
    buf[rem] = 0x80;
    const int padlen = (rem < 56) ? (56 - rem - 1) : (120 - rem - 1);
    for (int i = 0; i < padlen; ++i) buf[rem + 1 + i] = 0;
    const uint64_t total = (uint64_t)word_count * 32ULL;
    const int finoff = rem + 1 + padlen;
    buf[finoff+0]=(uint8_t)(total>>56); buf[finoff+1]=(uint8_t)(total>>48); buf[finoff+2]=(uint8_t)(total>>40); buf[finoff+3]=(uint8_t)(total>>32);
    buf[finoff+4]=(uint8_t)(total>>24); buf[finoff+5]=(uint8_t)(total>>16); buf[finoff+6]=(uint8_t)(total>>8);  buf[finoff+7]=(uint8_t)total;
    const int blocks = (finoff + 8) / 64;
    for (int blk = 0; blk < blocks; ++blk) {
        uint32_t W[16];
        uint32_t a = state[0], b = state[1], c = state[2], d = state[3], e = state[4], f = state[5], g = state[6], h = state[7];
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            uint32_t w_i;
            if (i < 16) {
                const int p = blk * 64 + i * 4;
                w_i = ((uint32_t)buf[p] << 24) | ((uint32_t)buf[p + 1] << 16) | ((uint32_t)buf[p + 2] << 8) | buf[p + 3];
                W[i] = w_i;
            } else {
                const uint32_t w15 = W[(i - 15) & 15];
                const uint32_t w2 = W[(i - 2) & 15];
                const uint32_t s0 = d_rotr(w15, 7) ^ d_rotr(w15, 18) ^ (w15 >> 3);
                const uint32_t s1 = d_rotr(w2, 17) ^ d_rotr(w2, 19) ^ (w2 >> 10);
                w_i = W[(i - 16) & 15] + s0 + W[(i - 7) & 15] + s1;
                W[i & 15] = w_i;
            }
            const uint32_t S1 = d_rotr(e, 6) ^ d_rotr(e, 11) ^ d_rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + S1 + ch + d_K[i] + w_i;
            const uint32_t S0 = d_rotr(a, 2) ^ d_rotr(a, 13) ^ d_rotr(a, 22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }
    for (int i = 0; i < 8; ++i) {
        hash[i * 4 + 0] = (uint8_t)(state[i] >> 24);
        hash[i * 4 + 1] = (uint8_t)(state[i] >> 16);
        hash[i * 4 + 2] = (uint8_t)(state[i] >> 8);
        hash[i * 4 + 3] = (uint8_t)state[i];
    }
}

__device__ void d_salsa20_block(const uint32_t input[16], uint8_t output[64]) {
    uint32_t x[16]; for (int i = 0; i < 16; i++) x[i] = input[i];
    for (int i = 0; i < 10; i++) {
        x[4]^=d_rotl(x[0]+x[12], 7); x[8]^=d_rotl(x[4]+x[0], 9); x[12]^=d_rotl(x[8]+x[4],13); x[0]^=d_rotl(x[12]+x[8],18);
        x[9]^=d_rotl(x[5]+x[1], 7); x[13]^=d_rotl(x[9]+x[5], 9); x[1]^=d_rotl(x[13]+x[9],13); x[5]^=d_rotl(x[1]+x[13],18);
        x[14]^=d_rotl(x[10]+x[6], 7); x[2]^=d_rotl(x[14]+x[10], 9); x[6]^=d_rotl(x[2]+x[14],13); x[10]^=d_rotl(x[6]+x[2],18);
        x[3]^=d_rotl(x[15]+x[11], 7); x[7]^=d_rotl(x[3]+x[15], 9); x[11]^=d_rotl(x[7]+x[3],13); x[15]^=d_rotl(x[11]+x[7],18);
        x[1]^=d_rotl(x[0]+x[3], 7); x[2]^=d_rotl(x[1]+x[0], 9); x[3]^=d_rotl(x[2]+x[1],13); x[0]^=d_rotl(x[3]+x[2],18);
        x[6]^=d_rotl(x[5]+x[4], 7); x[7]^=d_rotl(x[6]+x[5], 9); x[4]^=d_rotl(x[7]+x[6],13); x[5]^=d_rotl(x[4]+x[7],18);
        x[11]^=d_rotl(x[10]+x[9], 7); x[8]^=d_rotl(x[11]+x[10], 9); x[9]^=d_rotl(x[8]+x[11],13); x[10]^=d_rotl(x[9]+x[8],18);
        x[12]^=d_rotl(x[15]+x[14], 7); x[13]^=d_rotl(x[12]+x[15], 9); x[14]^=d_rotl(x[13]+x[12],13); x[15]^=d_rotl(x[14]+x[13],18);
    }
    for (int i = 0; i < 16; i++) { uint32_t v = x[i] + input[i]; output[i*4]=(uint8_t)v; output[i*4+1]=(uint8_t)(v>>8); output[i*4+2]=(uint8_t)(v>>16); output[i*4+3]=(uint8_t)(v>>24); }
}

__device__ void d_rc4_init(uint8_t S[256], const uint8_t* key, int keylen) {
    for (int i = 0; i < 256; i++) S[i] = (uint8_t)i;
    uint8_t j = 0; for (int i = 0; i < 256; i++) { j = j + S[i] + key[i % keylen]; uint8_t t = S[i]; S[i] = S[j]; S[j] = t; }
}

__device__ void d_rc4_process(uint8_t S[256], uint8_t* si, uint8_t* sj, const uint8_t* in, uint8_t* out, int len) {
    uint8_t i = *si, j = *sj;
    for (int k = 0; k < len; k++) { i++; j += S[i]; uint8_t t = S[i]; S[i] = S[j]; S[j] = t; out[k] = in[k] ^ S[(uint8_t)(S[i] + S[j])]; }
    *si = i; *sj = j;
}

__device__ uint64_t d_fnv1a_256(const uint8_t* data) { uint64_t h = 0xcbf29ce484222325ULL; for (int i = 0; i < 256; i++) { h ^= data[i]; h *= 0x100000001b3ULL; } return h; }
__device__ uint64_t d_fnv1a(const uint8_t* data, int len) { uint64_t h = 0xcbf29ce484222325ULL; for (int i = 0; i < len; i++) { h ^= data[i]; h *= 0x100000001b3ULL; } return h; }
__device__ uint64_t d_rotl64(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
__device__ uint64_t d_xxh64_round(uint64_t acc, uint64_t input) { const uint64_t P1 = 0x9E3779B185EBCA87ULL; const uint64_t P2 = 0xC2B2AE3D27D4EB4FULL; return d_rotl64(acc + input * P2, 31) * P1; }

__device__ uint64_t d_xxhash64(const uint8_t* data, int len, uint64_t seed) {
    const uint64_t P1 = 0x9E3779B185EBCA87ULL; const uint64_t P2 = 0xC2B2AE3D27D4EB4FULL; const uint64_t P3 = 0x165667B19E3779F9ULL; const uint64_t P4 = 0x85EBCA77C2B2AE63ULL; const uint64_t P5 = 0x27D4EB2F165667C5ULL;
    const uint8_t* p = data; const uint8_t* end = data + len; uint64_t h64;
    if (len >= 32) {
        uint64_t v1 = seed + P1 + P2, v2 = seed + P2, v3 = seed, v4 = seed - P1;
        do { uint64_t m1; memcpy(&m1, p, 8); v1 = d_xxh64_round(v1, m1); p += 8; uint64_t m2; memcpy(&m2, p, 8); v2 = d_xxh64_round(v2, m2); p += 8; uint64_t m3; memcpy(&m3, p, 8); v3 = d_xxh64_round(v3, m3); p += 8; uint64_t m4; memcpy(&m4, p, 8); v4 = d_xxh64_round(v4, m4); p += 8; } while (p <= end - 32);
        h64 = d_rotl64(v1,1) + d_rotl64(v2,7) + d_rotl64(v3,12) + d_rotl64(v4,18);
        h64 = (h64 ^ d_xxh64_round(0, v1)) * P1 + P4; h64 = (h64 ^ d_xxh64_round(0, v2)) * P1 + P4; h64 = (h64 ^ d_xxh64_round(0, v3)) * P1 + P4; h64 = (h64 ^ d_xxh64_round(0, v4)) * P1 + P4;
    } else h64 = seed + P5;
    h64 += (uint64_t)len;
    while (p + 8 <= end) { uint64_t m; memcpy(&m, p, 8); h64 = d_rotl64(h64 ^ d_xxh64_round(0, m), 27) * P1 + P4; p += 8; }
    while (p + 4 <= end) { uint32_t m; memcpy(&m, p, 4); h64 = d_rotl64(h64 ^ (m * P1), 23) * P2 + P3; p += 4; }
    while (p < end) { h64 = d_rotl64(h64 ^ (*p++ * P5), 11) * P1; }
    h64 ^= h64 >> 33; h64 *= P2; h64 ^= h64 >> 29; h64 *= P3; h64 ^= h64 >> 32; return h64;
}

__device__ uint64_t d_siphash(const uint64_t key[2], const uint8_t* data, int len) {
    uint64_t v0 = key[0] ^ 0x736f6d6570736575ULL, v1 = key[1] ^ 0x646f72616e646f6dULL, v2 = key[0] ^ 0x6c7967656e657261ULL, v3 = key[1] ^ 0x7465646279746573ULL;
    int blocks = len / 8;
    for (int i = 0; i < blocks; i++) {
        uint64_t m; memcpy(&m, data + i * 8, 8); v3 ^= m;
        for (int r = 0; r < 2; r++) { v0 += v1; v1 = d_rotl64(v1, 13); v1 ^= v0; v0 = d_rotl64(v0, 32); v2 += v3; v3 = d_rotl64(v3, 16); v3 ^= v2; v0 += v3; v3 = d_rotl64(v3, 21); v3 ^= v0; v2 += v1; v1 = d_rotl64(v1, 17); v1 ^= v2; v2 = d_rotl64(v2, 32); }
        v0 ^= m;
    }
    uint64_t last = (uint64_t)len << 56; const uint8_t* tail = data + blocks * 8;
    switch (len & 7) { case 7: last |= (uint64_t)tail[6] << 48; case 6: last |= (uint64_t)tail[5] << 40; case 5: last |= (uint64_t)tail[4] << 32; case 4: last |= (uint64_t)tail[3] << 24; case 3: last |= (uint64_t)tail[2] << 16; case 2: last |= (uint64_t)tail[1] << 8; case 1: last |= (uint64_t)tail[0]; }
    v3 ^= last;
    for (int r = 0; r < 2; r++) { v0 += v1; v1 = d_rotl64(v1, 13); v1 ^= v0; v0 = d_rotl64(v0, 32); v2 += v3; v3 = d_rotl64(v3, 16); v3 ^= v2; v0 += v3; v3 = d_rotl64(v3, 21); v3 ^= v0; v2 += v1; v1 = d_rotl64(v1, 17); v1 ^= v2; v2 = d_rotl64(v2, 32); }
    v0 ^= last; v2 ^= 0xff;
    for (int r = 0; r < 4; r++) { v0 += v1; v1 = d_rotl64(v1, 13); v1 ^= v0; v0 = d_rotl64(v0, 32); v2 += v3; v3 = d_rotl64(v3, 16); v3 ^= v2; v0 += v3; v3 = d_rotl64(v3, 21); v3 ^= v0; v2 += v1; v1 = d_rotl64(v1, 17); v1 ^= v2; v2 = d_rotl64(v2, 32); }
    return v0 ^ v1 ^ v2 ^ v3;
}

__device__ bool d_check_hash_target(const uint8_t hash[32], const uint8_t target[32]) {
    for (int i = 0; i < 32; i++) { if (hash[i] != target[i]) return hash[i] < target[i]; } return true;
}

__global__ void astrobwt_branch_compute_kernel(uint8_t* __restrict__ d_sdata_out, int32_t* __restrict__ d_data_len_out, uint64_t* __restrict__ d_solutions, uint32_t nonce_start, int batch_size) {
    int hash_id = blockIdx.x; if (hash_id >= batch_size) return;
    int lane = threadIdx.x; uint32_t nonce = nonce_start + hash_id;
    constexpr int HOST_SDATA_STRIDE = 72 * 1024;
    uint8_t* sData = d_sdata_out + (size_t)hash_id * HOST_SDATA_STRIDE;
    __shared__ uint8_t sha_key[32], scratch[256], rc4_S[256], rc4_i, rc4_j; __shared__ uint64_t lhash, prev_lhash;
    if (lane == 0) { uint8_t work[48]; for (int i = 0; i < 48; i++) work[i] = d_work_template[i]; work[43] = (uint8_t)(nonce >> 24); work[44] = (uint8_t)(nonce >> 16); work[45] = (uint8_t)(nonce >> 8);  work[46] = (uint8_t)(nonce); d_sha256(work, 48, sha_key); }
    __syncthreads();
    if (lane == 0) { static const uint8_t sigma[16] = {'e','x','p','a','n','d',' ','3','2','-','b','y','t','e',' ','k'}; uint32_t st[16]; auto le32 = [](const uint8_t* p) -> uint32_t { return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24); }; st[0]=le32(&sigma[0]); st[1]=le32(&sha_key[0]); st[2]=le32(&sha_key[4]); st[3]=le32(&sha_key[8]); st[4]=le32(&sha_key[12]); st[5]=le32(&sigma[4]); st[6]=0; st[7]=0; st[8]=0; st[9]=0; st[10]=le32(&sigma[8]); st[11]=le32(&sha_key[16]); st[12]=le32(&sha_key[20]); st[13]=le32(&sha_key[24]); st[14]=le32(&sha_key[28]); st[15]=le32(&sigma[12]); for (int blk = 0; blk < 4; blk++) { uint8_t block[64]; d_salsa20_block(st, block); for (int i = 0; i < 64; i++) scratch[blk * 64 + i] = block[i]; st[8]++; } }
    __syncthreads();
    if (lane == 0) { d_rc4_init(rc4_S, scratch, 256); rc4_i = 0; rc4_j = 0; d_rc4_process(rc4_S, &rc4_i, &rc4_j, scratch, scratch, 256); lhash = d_fnv1a_256(scratch); prev_lhash = lhash; }
    __syncthreads();
    for (int i = lane; i < 256; i += 32) sData[i] = scratch[i];
    __syncthreads();
    int tries = 0;
    for (int iteration = 1; iteration <= 278; ++iteration) {
        tries++; uint64_t sw; if (lane == 0) sw = prev_lhash ^ lhash ^ (uint64_t)tries;
        sw = __shfl_sync(0xFFFFFFFF, sw, 0);
        uint8_t op = (uint8_t)(sw), p1 = (uint8_t)(sw >> 8), p2 = (uint8_t)(sw >> 16);
        if (p1 > p2) { uint8_t t = p1; p1 = p2; p2 = t; } if (p2 - p1 > 32) p2 = p1 + ((p2 - p1) & 0x1f);
        uint8_t* chunk = &sData[(tries - 1) * 256]; uint8_t* prev_chunk = (tries == 1) ? chunk : &sData[(tries - 2) * 256];
        if (tries > 1) { for (int i = lane; i < 256; i += 32) chunk[i] = prev_chunk[i]; __syncthreads(); }
        if (op == 253) {
            for (int i = p1; i < p2; i++) {
                if (i % 32 == lane) {
                    chunk[i] = d_rl8(chunk[i], 3);
                    chunk[i] ^= d_rl8(chunk[i], 2);
                    chunk[i] ^= prev_chunk[p2];
                    chunk[i] = d_rl8(chunk[i], 3);
                }
                __syncthreads();
                if (lane == 0) {
                    prev_lhash = lhash + prev_lhash;
                    lhash = d_xxhash64(chunk, p2, 0);
                }
                __syncthreads();
            }
        }
        else { if (op >= 254) { if (lane == 0) { d_rc4_init(rc4_S, prev_chunk, 256); rc4_i = 0; rc4_j = 0; } __syncthreads(); } uint32_t opcode = d_CodeLUT[op]; uint8_t pos2_val = prev_chunk[p2]; for (int i = p1 + lane; i < p2; i += 32) chunk[i] = d_wolfBranch(prev_chunk[i], pos2_val, opcode); __syncthreads(); if (op == 0 && ((p2 - p1) & 1) == 1) { if (lane == 0) { uint8_t t1 = chunk[p1], t2 = chunk[p2]; chunk[p1] = d_reverse8(t2); chunk[p2] = d_reverse8(t1); } __syncthreads(); } }
        if (lane == 0) {
            uint8_t A = (uint8_t)((256 + ((chunk[p1] - chunk[p2]) % 256)) % 256);
            if (A < 0x30) {
                if (A < 0x10) { prev_lhash = lhash + prev_lhash; lhash = d_xxhash64(chunk, p2, 0); }
                if (A < 0x20) { prev_lhash = lhash + prev_lhash; lhash = d_fnv1a(chunk, p2); }
                prev_lhash = lhash + prev_lhash;
                uint64_t sip_key[2] = { (uint64_t)tries, prev_lhash };
                lhash = d_siphash(sip_key, chunk, p2);
            }
            if (A <= 0x40) d_rc4_process(rc4_S, &rc4_i, &rc4_j, chunk, chunk, 256);
        }
        __syncthreads();
        if (lane == 0) chunk[255] ^= chunk[p1] ^ chunk[p2]; __syncthreads(); if (tries > 260 + 16) break; if (chunk[255] >= 0xf0 && tries > 260) break;
    }
    if (lane == 0) { uint8_t* last_chunk = &sData[(tries - 1) * 256]; int dlen = (tries - 4) * 256 + (((last_chunk[253] << 8) | last_chunk[254]) & 0x3ff); if (dlen < 256) dlen = 256; if (dlen > 70911) dlen = 70911; d_data_len_out[hash_id] = dlen; for (int i = dlen; i < dlen + 16 && i < (72 * 1024); i++) sData[i] = 0; }
}

__global__ void astrobwt_final_hash_kernel(const int32_t* __restrict__ d_sa, const int32_t* __restrict__ d_data_len, uint64_t* __restrict__ d_solutions, uint32_t nonce_start, int batch_size) {
    static constexpr int MAX_SA_N = 71429; int tid = blockIdx.x * blockDim.x + threadIdx.x; if (tid >= batch_size) return;
    int n = d_data_len[tid]; if (n <= 0) return;
    const uint32_t* sa_words = reinterpret_cast<const uint32_t*>(d_sa + (size_t)tid * MAX_SA_N); uint8_t final_hash[32]; d_sha256_words_le(sa_words, n, final_hash);
    if (final_hash[0] == 0) atomicAdd(&d_near_miss_count, 1);
    if (d_check_hash_target(final_hash, d_difficulty_target)) {
        uint64_t idx = atomicAdd((unsigned long long*)&d_solutions[0], 1ULL);
        if (idx < 1024) { uint32_t nonce = nonce_start + tid; uint64_t* slot = &d_solutions[1 + idx * 5]; slot[0] = (uint64_t)nonce; for (int i = 0; i < 4; i++) { uint64_t val = 0; for (int j = 0; j < 8; j++) val |= ((uint64_t)final_hash[i * 8 + j]) << (j * 8); slot[1 + i] = val; } }
    }
}

__global__ void build_offsets_kernel(const int32_t* d_data_len, int* d_offsets, int batch_size) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int offset = 0;
    for (int i = 0; i < batch_size; i++) {
        d_offsets[i] = offset;
        int n = d_data_len[i];
        int clamp_n = (n < 0) ? 0 : (n > 70911 ? 70911 : n);
        offset += clamp_n;
    }
    d_offsets[batch_size] = offset;
}

__global__ void init_sort_keys_kernel(const uint8_t* __restrict__ d_sdata,
                                      const int32_t* __restrict__ d_data_len,
                                      const int* __restrict__ d_offsets,
                                      uint64_t* __restrict__ d_keys,
                                      uint32_t* __restrict__ d_vals,
                                      int batch_size,
                                      int shift) {
    int hash_id = blockIdx.y; if (hash_id >= batch_size) return;
    int n = d_data_len[hash_id]; if (n < 2 || n > 70911) return;
    static constexpr int HOST_SDATA_STRIDE = 72 * 1024; int base_offset = d_offsets[hash_id]; const uint8_t* T = d_sdata + (size_t)hash_id * HOST_SDATA_STRIDE;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        uint64_t key = 0;
        #pragma unroll
        for (int b = 0; b < kInitialPrefixSymbols; ++b) {
            const int pos = i + shift + b;
            const uint64_t sym = (pos < n) ? ((uint64_t)T[pos] + 1ULL) : 0ULL;
            key = (key << 9) | sym;
        }
        d_keys[base_offset + i] = key;
        d_vals[base_offset + i] = (uint32_t)i;
    }
}

__global__ void init_sort_bytes64_kernel(const uint8_t* __restrict__ d_sdata,
                                         const int32_t* __restrict__ d_data_len,
                                         const int* __restrict__ d_offsets,
                                         uint64_t* __restrict__ d_keys,
                                         uint32_t* __restrict__ d_vals,
                                         int batch_size,
                                         int shift) {
    int hash_id = blockIdx.y; if (hash_id >= batch_size) return;
    int n = d_data_len[hash_id]; if (n < 2 || n > 70911) return;
    static constexpr int HOST_SDATA_STRIDE = 72 * 1024;
    const int base_offset = d_offsets[hash_id];
    const uint8_t* T = d_sdata + (size_t)hash_id * HOST_SDATA_STRIDE;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        uint64_t key = 0;
        #pragma unroll
        for (int b = 0; b < kFastPrefixBytes; ++b) {
            key <<= 8;
            const int pos = i + shift + b;
            if (pos < n) key |= (uint64_t)T[pos];
        }
        d_keys[base_offset + i] = key;
        d_vals[base_offset + i] = (uint32_t)i;
    }
}

__global__ void build_ordered_bytes64_kernel(const uint8_t* __restrict__ d_sdata,
                                             const int32_t* __restrict__ d_data_len,
                                             const int* __restrict__ d_offsets,
                                             const uint32_t* __restrict__ d_src_vals,
                                             uint64_t* __restrict__ d_keys,
                                             uint32_t* __restrict__ d_vals,
                                             int batch_size,
                                             int shift) {
    int hash_id = blockIdx.y; if (hash_id >= batch_size) return;
    int n = d_data_len[hash_id]; if (n < 2 || n > 70911) return;
    static constexpr int HOST_SDATA_STRIDE = 72 * 1024;
    const int base_offset = d_offsets[hash_id];
    const uint8_t* T = d_sdata + (size_t)hash_id * HOST_SDATA_STRIDE;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const uint32_t suffix = d_src_vals[base_offset + i];
        uint64_t key = 0;
        #pragma unroll
        for (int b = 0; b < kFastPrefixBytes; ++b) {
            key <<= 8;
            const uint32_t pos = suffix + (uint32_t)shift + (uint32_t)b;
            if (pos < (uint32_t)n) key |= (uint64_t)T[pos];
        }
        d_keys[base_offset + i] = key;
        d_vals[base_offset + i] = suffix;
    }
}

__global__ void astrobwt_final_hash_segments_kernel(const uint32_t* __restrict__ d_sorted_sa,
                                                    const int32_t* __restrict__ d_data_len,
                                                    const int* __restrict__ d_offsets,
                                                    uint64_t* __restrict__ d_solutions,
                                                    uint32_t nonce_start,
                                                    int batch_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;
    const int n = d_data_len[tid];
    if (n <= 0) return;

    const int base_offset = d_offsets[tid];
    uint8_t final_hash[32];
    d_sha256_words_le(d_sorted_sa + base_offset, n, final_hash);
    if (final_hash[0] == 0) atomicAdd(&d_near_miss_count, 1);
    if (d_check_hash_target(final_hash, d_difficulty_target)) {
        uint64_t idx = atomicAdd((unsigned long long*)&d_solutions[0], 1ULL);
        if (idx < 1024) {
            const uint32_t nonce = nonce_start + tid;
            uint64_t* slot = &d_solutions[1 + idx * 5];
            slot[0] = (uint64_t)nonce;
            for (int i = 0; i < 4; i++) {
                uint64_t val = 0;
                for (int j = 0; j < 8; j++) val |= ((uint64_t)final_hash[i * 8 + j]) << (j * 8);
                slot[1 + i] = val;
            }
        }
    }
}

__global__ void build_rank_keys_kernel(const uint32_t* __restrict__ d_ranks,
                                       const int32_t* __restrict__ d_data_len,
                                       const int* __restrict__ d_offsets,
                                       uint64_t* __restrict__ d_keys,
                                       uint32_t* __restrict__ d_vals,
                                       int batch_size, int step) {
    int hash_id = blockIdx.y; if (hash_id >= batch_size) return;
    int n = d_data_len[hash_id]; if (n < 2 || n > 70911) return;
    const int base_offset = d_offsets[hash_id];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const uint32_t r1 = d_ranks[base_offset + i];
        const uint32_t r2 = (i + step < n) ? d_ranks[base_offset + i + step] : 0U;
        d_keys[base_offset + i] = ((uint64_t)r1 << 32) | (uint64_t)r2;
        d_vals[base_offset + i] = (uint32_t)i;
    }
}

__device__ int d_compare_suffixes_from(const uint8_t* T, int n, uint32_t a, uint32_t b, uint32_t depth) {
    if (a == b) return 0;
    uint32_t aa = a + depth;
    uint32_t bb = b + depth;
    while (aa + 8 <= (uint32_t)n && bb + 8 <= (uint32_t)n) {
        const uint64_t av = d_load_u64_unaligned(T + aa);
        const uint64_t bv = d_load_u64_unaligned(T + bb);
        if (av != bv) break;
        aa += 8;
        bb += 8;
    }
    while (aa + 4 <= (uint32_t)n && bb + 4 <= (uint32_t)n) {
        const uint32_t av = d_load_u32_unaligned(T + aa);
        const uint32_t bv = d_load_u32_unaligned(T + bb);
        if (av != bv) break;
        aa += 4;
        bb += 4;
    }
    while (aa < (uint32_t)n && bb < (uint32_t)n) {
        const uint8_t av = T[aa];
        const uint8_t bv = T[bb];
        if (av != bv) return (av < bv) ? -1 : 1;
        ++aa;
        ++bb;
    }
    if (aa >= (uint32_t)n && bb >= (uint32_t)n) {
        const uint32_t rem_a = (uint32_t)n - a;
        const uint32_t rem_b = (uint32_t)n - b;
        if (rem_a == rem_b) return 0;
        return (rem_a < rem_b) ? -1 : 1;
    }
    return (aa >= (uint32_t)n) ? -1 : 1;
}

// Block-per-hash, grid-stride over elements. Each thread tests a LOCAL
// run-start (idx==0 || key[idx]!=key[idx-1]); a run-start forward-scans to
// the run end and (if size>1) emits one descriptor via a WARP-AGGREGATED
// atomic on d_bucket_total_count (one atomicAdd per warp; lanes claim slots
// from the warp base). Descriptor emission order is irrelevant (each desc is
// self-contained, per-bin sorts independent). Stats (bucket/ambiguous/max)
// are diagnostic-only and accumulated via a block reduction when requested.
// [NARROWKEY] uint32 variant of init_sort_keys_kernel: identical 18-bit key
// VALUE (9-bit symbols, byte+1 so 0=end-of-string sentinel) but stored in 32
// bits to halve per-pass key bandwidth in the coarse radix sort. Parity-exact
// vs the uint64 path (same key bits, stable radix). Env-gated by NARROWKEY.
__global__ void init_sort_keys18_u32_kernel(const uint8_t* __restrict__ d_sdata,
                                            const int32_t* __restrict__ d_data_len,
                                            const int* __restrict__ d_offsets,
                                            uint32_t* __restrict__ d_keys,
                                            uint32_t* __restrict__ d_vals,
                                            int batch_size,
                                            int shift) {
    int hash_id = blockIdx.y; if (hash_id >= batch_size) return;
    int n = d_data_len[hash_id]; if (n < 2 || n > 70911) return;
    static constexpr int HOST_SDATA_STRIDE = 72 * 1024; int base_offset = d_offsets[hash_id]; const uint8_t* T = d_sdata + (size_t)hash_id * HOST_SDATA_STRIDE;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        uint32_t key = 0;
        #pragma unroll
        for (int b = 0; b < kInitialPrefixSymbols; ++b) {
            const int pos = i + shift + b;
            const uint32_t sym = (pos < n) ? ((uint32_t)T[pos] + 1u) : 0u;
            key = (key << 9) | sym;
        }
        d_keys[base_offset + i] = key;
        d_vals[base_offset + i] = (uint32_t)i;
    }
}

template<typename K>
__global__ void build_bucket_descriptors_kernel(const K* __restrict__ d_primary_keys,
                                                const K* __restrict__ d_secondary_keys,
                                                const int32_t* __restrict__ d_data_len,
                                                const int* __restrict__ d_offsets,
                                                uint64_t* __restrict__ d_bucket_descs,
                                                uint32_t* __restrict__ d_bucket_total_count,
                                                uint32_t* __restrict__ d_bucket_counts,
                                                uint32_t* __restrict__ d_ambiguous_counts,
                                                uint32_t* __restrict__ d_max_bucket_sizes,
                                                uint32_t bucket_depth,
                                                int batch_size) {
    const int hash_id = blockIdx.x;
    if (hash_id >= batch_size) return;
    const int tid = (int)threadIdx.x;
    const bool want_stats = (d_bucket_counts != nullptr);

    const int n = d_data_len[hash_id];
    if (n <= 1 || n > 70911) {
        if (tid == 0) {
            if (d_bucket_counts) d_bucket_counts[hash_id] = 0;
            if (d_ambiguous_counts) d_ambiguous_counts[hash_id] = 0;
            if (d_max_bucket_sizes) d_max_bucket_sizes[hash_id] = 0;
        }
        return;
    }

    const int base_offset = d_offsets[hash_id];
    const K* P = d_primary_keys + base_offset;
    const K* S = d_secondary_keys ? (d_secondary_keys + base_offset) : nullptr;
    uint32_t local_bucket_count = 0;
    uint32_t local_ambiguous_count = 0;
    uint32_t local_max_bucket_size = 1;
    const unsigned lane = (unsigned)(threadIdx.x & 31);

    for (int i = tid; i < n; i += (int)blockDim.x) {
        // local run-start test: i is the start of a maximal equal-key run iff
        // i==0 or key[i] differs from key[i-1].
        const K pi = P[i];
        const K si = S ? S[i] : (K)0;
        bool is_start = (i == 0);
        if (!is_start) {
            is_start = (P[i-1] != pi) || (S && S[i-1] != si);
        }
        uint32_t bucket_size = 0;
        if (is_start) {
            int j = i + 1;
            while (j < n && P[j] == pi && (!S || S[j] == si)) ++j;
            bucket_size = (uint32_t)(j - i);
        }
        const bool emit = (bucket_size > 1);
        // warp-aggregated atomic: one atomicAdd per warp for all emitters.
        const unsigned active = __activemask();
        const unsigned emask = __ballot_sync(active, emit);
        if (emit) {
            const unsigned leader_lane = (unsigned)__ffs((int)emask) - 1u;
            const unsigned rank = (unsigned)__popc(emask & ((1u << lane) - 1u));
            uint32_t warp_base = 0;
            if (lane == leader_lane) {
                warp_base = atomicAdd(d_bucket_total_count, (uint32_t)__popc(emask));
            }
            warp_base = __shfl_sync(emask, warp_base, (int)leader_lane);
            const uint32_t slot = warp_base + rank;
            d_bucket_descs[slot] = d_pack_bucket_desc((uint32_t)hash_id, (uint32_t)i, bucket_size, bucket_depth);
            if (want_stats) {
                local_bucket_count++;
                local_ambiguous_count += bucket_size;
                if (bucket_size > local_max_bucket_size) local_max_bucket_size = bucket_size;
            }
        }
    }

    if (want_stats) {
        // block reduction of the three diagnostic accumulators.
        __shared__ uint32_t s_cnt[1024/32], s_amb[1024/32], s_mx[1024/32];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            local_bucket_count    += __shfl_down_sync(0xFFFFFFFFu, local_bucket_count, off);
            local_ambiguous_count += __shfl_down_sync(0xFFFFFFFFu, local_ambiguous_count, off);
            uint32_t o = __shfl_down_sync(0xFFFFFFFFu, local_max_bucket_size, off);
            if (o > local_max_bucket_size) local_max_bucket_size = o;
        }
        const int warp_id = tid >> 5;
        const int nwarps = ((int)blockDim.x + 31) >> 5;
        if (lane == 0) { s_cnt[warp_id] = local_bucket_count; s_amb[warp_id] = local_ambiguous_count; s_mx[warp_id] = local_max_bucket_size; }
        __syncthreads();
        if (warp_id == 0) {
            uint32_t c = (int)lane < nwarps ? s_cnt[lane] : 0u;
            uint32_t a = (int)lane < nwarps ? s_amb[lane] : 0u;
            uint32_t m = (int)lane < nwarps ? s_mx[lane]  : 1u;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                c += __shfl_down_sync(0xFFFFFFFFu, c, off);
                a += __shfl_down_sync(0xFFFFFFFFu, a, off);
                uint32_t o = __shfl_down_sync(0xFFFFFFFFu, m, off);
                if (o > m) m = o;
            }
            if (lane == 0) {
                if (d_bucket_counts) d_bucket_counts[hash_id] = c;
                if (d_ambiguous_counts) d_ambiguous_counts[hash_id] = a;
                if (d_max_bucket_sizes) d_max_bucket_sizes[hash_id] = (c ? m : 0u);
            }
        }
    }
}

__global__ void refine_bucket_descs_kernel(const uint8_t* __restrict__ d_sdata,
                                           const int32_t* __restrict__ d_data_len,
                                           const int* __restrict__ d_offsets,
                                           const uint64_t* __restrict__ d_bucket_descs,
                                           uint32_t bucket_total_count,
                                           uint32_t* __restrict__ d_sorted_vals) {
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t bucket_id = blockIdx.x * blockDim.x + threadIdx.x;
         bucket_id < bucket_total_count;
         bucket_id += stride) {
        uint32_t hash_id = 0;
        uint32_t start = 0;
        uint32_t len = 0;
        uint32_t depth = 0;
        d_unpack_bucket_desc(d_bucket_descs[bucket_id], hash_id, start, len, depth);
        if (len <= 1) continue;

        static constexpr int HOST_SDATA_STRIDE = 72 * 1024;
        const int n = d_data_len[hash_id];
        const int base_offset = d_offsets[hash_id] + (int)start;
        const uint8_t* T = d_sdata + (size_t)hash_id * HOST_SDATA_STRIDE;
        uint32_t* values = d_sorted_vals + base_offset;

        for (uint32_t i = 1; i < len; ++i) {
            const uint32_t value = values[i];
            int pos = (int)i - 1;
            while (pos >= 0 && d_compare_suffixes_from(T, n, value, values[pos], depth) < 0) {
                values[pos + 1] = values[pos];
                --pos;
            }
            values[pos + 1] = value;
        }
    }
}

// ---------------------------------------------------------------------
// Stage-B: unpack the packed bucket descriptors into the 5 per-segment
// metadata arrays consumed by stageb_refine.cuh, and track the max segment
// length (atomicMax) so the host can size the bin4 global-scratch stride.
//   seg_Tbase[s] = hash_id * HOST_SDATA_STRIDE   (byte offset of lane's T)
//   seg_off[s]   = d_offsets[hash_id] + start    (offset into d_vals_out / SA)
//   seg_len[s]   = len
//   seg_depth[s] = depth (14)
//   seg_n[s]     = d_data_len[hash_id]
// ---------------------------------------------------------------------
__global__ void unpack_bucket_descs_kernel(const int32_t* __restrict__ d_data_len,
                                           const int* __restrict__ d_offsets,
                                           const uint64_t* __restrict__ d_bucket_descs,
                                           uint32_t bucket_total_count,
                                           uint32_t* __restrict__ seg_off,
                                           uint32_t* __restrict__ seg_len,
                                           uint32_t* __restrict__ seg_depth,
                                           uint32_t* __restrict__ seg_n,
                                           uint64_t* __restrict__ seg_Tbase,
                                           uint32_t* __restrict__ d_max_len) {
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
         s < bucket_total_count; s += stride) {
        uint32_t hash_id = 0, start = 0, len = 0, depth = 0;
        d_unpack_bucket_desc(d_bucket_descs[s], hash_id, start, len, depth);
        static constexpr int HOST_SDATA_STRIDE = 72 * 1024;
        seg_Tbase[s] = (uint64_t)hash_id * (uint64_t)HOST_SDATA_STRIDE;
        seg_off[s]   = (uint32_t)((int)d_offsets[hash_id] + (int)start);
        seg_len[s]   = len;
        seg_depth[s] = depth;
        seg_n[s]     = (uint32_t)d_data_len[hash_id];
        if (len > 1) atomicMax(d_max_len, len);
    }
}

__global__ void sb_bin_count_desc_kernel(const uint64_t* __restrict__ d_bucket_descs,
                                         uint32_t bucket_total_count,
                                         uint32_t* __restrict__ bin_counts,
                                         uint32_t* __restrict__ d_max_len) {
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
         s < bucket_total_count; s += stride) {
        uint32_t hash_id = 0, start = 0, len = 0, depth = 0;
        d_unpack_bucket_desc(d_bucket_descs[s], hash_id, start, len, depth);
        int b = sb_bin_of((int)len);
        if (b >= 0) atomicAdd(&bin_counts[b], 1u);
        if (len > 1) atomicMax(d_max_len, len);
    }
}

__global__ void sb_bin_scatter_desc_kernel(const uint64_t* __restrict__ d_bucket_descs,
                                           uint32_t bucket_total_count,
                                           uint32_t* __restrict__ bin_cursor,
                                           uint32_t* __restrict__ ids_packed) {
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
         s < bucket_total_count; s += stride) {
        uint32_t hash_id = 0, start = 0, len = 0, depth = 0;
        d_unpack_bucket_desc(d_bucket_descs[s], hash_id, start, len, depth);
        int b = sb_bin_of((int)len);
        if (b < 0) continue;
        uint32_t pos = atomicAdd(&bin_cursor[b], 1u);
        ids_packed[pos] = s;
    }
}

static constexpr int SB_SPLIT_NBINS = 11;
static constexpr int SB_EXACT_NBINS = 34;     // len2..32 exact, 33..512, 513..1024, >1024
static constexpr int SB_EXT_NBINS = 36;       // exact tiny plus 33..64, 65..128, 129..512
static constexpr int SB_META_NBINS = SB_EXT_NBINS;
static constexpr int SB_LEN_HIST_SLOTS = 34;  // exact len 0..32 plus slot 33 for >32

__device__ __forceinline__ int sb_split_bin_of_len(int len) {
    if (len <= 1)            return -1;
    if (len == 2)            return 0;
    if (len <= 4)            return 1;  // 3..4
    if (len <= 6)            return 2;  // 5..6
    if (len <= SB_BIN0_MAX)  return 3;  // 7..8
    if (len <= 12)           return 4;  // 9..12
    if (len <= 16)           return 5;  // 13..16
    if (len <= 24)           return 6;  // 17..24
    if (len <= SB_BIN1_MAX)  return 7;  // 25..32
    if (len <= SB_BIN2_MAX)  return 8;
    if (len <= SB_BIN3_MAX)  return 9;
    return 10;
}

__device__ __forceinline__ int sb_exact_bin_of_len(int len) {
    if (len <= 1)            return -1;
    if (len <= SB_BIN1_MAX)  return len - 2;  // bins 0..30 for len 2..32
    if (len <= SB_BIN2_MAX)  return 31;
    if (len <= SB_BIN3_MAX)  return 32;
    return 33;
}

__device__ __forceinline__ int sb_ext_bin_of_len(int len) {
    if (len <= 1)            return -1;
    if (len <= SB_BIN1_MAX)  return len - 2;  // bins 0..30 for len 2..32
    if (len <= 64)           return 31;
    if (len <= 128)          return 32;
    if (len <= SB_BIN2_MAX)  return 33;
    if (len <= SB_BIN3_MAX)  return 34;
    return 35;
}

__global__ void sb_bin_count_desc_split_kernel(const uint64_t* __restrict__ d_bucket_descs,
                                               uint32_t bucket_total_count,
                                               uint32_t* __restrict__ split_counts,
                                               uint32_t* __restrict__ d_max_len) {
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
         s < bucket_total_count; s += stride) {
        uint32_t hash_id = 0, start = 0, len = 0, depth = 0;
        d_unpack_bucket_desc(d_bucket_descs[s], hash_id, start, len, depth);
        int b = sb_split_bin_of_len((int)len);
        if (b >= 0) atomicAdd(&split_counts[b], 1u);
        if (len > 1) atomicMax(d_max_len, len);
    }
}

__global__ void sb_bin_scatter_desc_split_kernel(const uint64_t* __restrict__ d_bucket_descs,
                                                 uint32_t bucket_total_count,
                                                 uint32_t* __restrict__ split_cursor,
                                                 uint32_t* __restrict__ ids_packed) {
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
         s < bucket_total_count; s += stride) {
        uint32_t hash_id = 0, start = 0, len = 0, depth = 0;
        d_unpack_bucket_desc(d_bucket_descs[s], hash_id, start, len, depth);
        int b = sb_split_bin_of_len((int)len);
        if (b < 0) continue;
        uint32_t pos = atomicAdd(&split_cursor[b], 1u);
        ids_packed[pos] = s;
    }
}

__global__ void sb_bin_count_desc_exact_kernel(const uint64_t* __restrict__ d_bucket_descs,
                                               uint32_t bucket_total_count,
                                               uint32_t* __restrict__ exact_counts,
                                               uint32_t* __restrict__ d_max_len) {
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
         s < bucket_total_count; s += stride) {
        uint32_t hash_id = 0, start = 0, len = 0, depth = 0;
        d_unpack_bucket_desc(d_bucket_descs[s], hash_id, start, len, depth);
        int b = sb_exact_bin_of_len((int)len);
        if (b >= 0) atomicAdd(&exact_counts[b], 1u);
        if (len > 1) atomicMax(d_max_len, len);
    }
}

__global__ void sb_bin_scatter_desc_exact_kernel(const uint64_t* __restrict__ d_bucket_descs,
                                                 uint32_t bucket_total_count,
                                                 uint32_t* __restrict__ exact_cursor,
                                                 uint32_t* __restrict__ ids_packed) {
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
         s < bucket_total_count; s += stride) {
        uint32_t hash_id = 0, start = 0, len = 0, depth = 0;
        d_unpack_bucket_desc(d_bucket_descs[s], hash_id, start, len, depth);
        int b = sb_exact_bin_of_len((int)len);
        if (b < 0) continue;
        uint32_t pos = atomicAdd(&exact_cursor[b], 1u);
        ids_packed[pos] = s;
    }
}

__global__ void sb_bin_count_desc_ext_kernel(const uint64_t* __restrict__ d_bucket_descs,
                                             uint32_t bucket_total_count,
                                             uint32_t* __restrict__ ext_counts,
                                             uint32_t* __restrict__ d_max_len) {
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
         s < bucket_total_count; s += stride) {
        uint32_t hash_id = 0, start = 0, len = 0, depth = 0;
        d_unpack_bucket_desc(d_bucket_descs[s], hash_id, start, len, depth);
        int b = sb_ext_bin_of_len((int)len);
        if (b >= 0) atomicAdd(&ext_counts[b], 1u);
        if (len > 1) atomicMax(d_max_len, len);
    }
}

__global__ void sb_bin_scatter_desc_ext_kernel(const uint64_t* __restrict__ d_bucket_descs,
                                               uint32_t bucket_total_count,
                                               uint32_t* __restrict__ ext_cursor,
                                               uint32_t* __restrict__ ids_packed) {
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
         s < bucket_total_count; s += stride) {
        uint32_t hash_id = 0, start = 0, len = 0, depth = 0;
        d_unpack_bucket_desc(d_bucket_descs[s], hash_id, start, len, depth);
        int b = sb_ext_bin_of_len((int)len);
        if (b < 0) continue;
        uint32_t pos = atomicAdd(&ext_cursor[b], 1u);
        ids_packed[pos] = s;
    }
}

__global__ void sb_len_hist_desc_kernel(const uint64_t* __restrict__ d_bucket_descs,
                                        uint32_t bucket_total_count,
                                        uint32_t* __restrict__ len_hist) {
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
         s < bucket_total_count; s += stride) {
        uint32_t hash_id = 0, start = 0, len = 0, depth = 0;
        d_unpack_bucket_desc(d_bucket_descs[s], hash_id, start, len, depth);
        const uint32_t slot = (len <= 32u) ? len : 33u;
        atomicAdd(&len_hist[slot], 1u);
    }
}

__device__ __forceinline__ void sb_desc_segment(const uint64_t* __restrict__ descs,
                                                uint32_t desc_id,
                                                const int32_t* __restrict__ d_data_len,
                                                const int* __restrict__ d_offsets,
                                                const uint8_t* __restrict__ T,
                                                const uint32_t* __restrict__ SA_base,
                                                const uint8_t*& Tl,
                                                uint32_t*& sa,
                                                int& n,
                                                int& len,
                                                int& depth) {
    uint32_t hash_id = 0, start = 0, ulen = 0, udepth = 0;
    d_unpack_bucket_desc(descs[desc_id], hash_id, start, ulen, udepth);
    static constexpr int HOST_SDATA_STRIDE = 72 * 1024;
    Tl = T + (size_t)hash_id * HOST_SDATA_STRIDE;
    sa = const_cast<uint32_t*>(SA_base) + (uint32_t)d_offsets[hash_id] + start;
    n = (int)d_data_len[hash_id];
    len = (int)ulen;
    depth = (int)udepth;
}

__global__ void sb_sort_len2_desc_kernel(
        const uint8_t* __restrict__ T,
        const int32_t* __restrict__ d_data_len,
        const int* __restrict__ d_offsets,
        const uint64_t* __restrict__ descs,
        const uint32_t* __restrict__ bin_ids,
        uint32_t bin_count,
        uint32_t* __restrict__ SA) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= bin_count) return;
    const uint8_t* Tl = nullptr;
    uint32_t* sa = nullptr;
    int n = 0, len = 0, depth = 0;
    sb_desc_segment(descs, bin_ids[t], d_data_len, d_offsets, T, SA, Tl, sa, n, len, depth);
    if (len != 2) return;
    const uint32_t a = sa[0];
    const uint32_t b = sa[1];
    if (cmp_suffix(Tl, n, b, a, depth) < 0) {
        sa[0] = b;
        sa[1] = a;
    }
}

// [PACKDESC] Descriptor-based packed LCP-insertion sort for tiny segments. Same fast
// single-pass LCP-aware insertion as sb_sort_lcp_packed_kernel (stageb_refine.cuh:680, HALF=false),
// but reads the segment from the packed bucket descriptor (one thread per segment). Opt-in faster
// alternative to the merge-sort sb_sort_lcp_msort_desc_kernel for the tiny bins (len 2..32).
template<int MAXLEN>
__global__ void sb_sort_lcp_packed_desc_kernel(
        const uint8_t* __restrict__ T,
        const int32_t* __restrict__ d_data_len,
        const int* __restrict__ d_offsets,
        const uint64_t* __restrict__ descs,
        const uint32_t* __restrict__ bin_ids,
        uint32_t bin_count,
        uint32_t* __restrict__ SA) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= bin_count) return;
    const uint8_t* Tl = nullptr;
    uint32_t* sa = nullptr;
    int n = 0, len = 0, depth = 0;
    sb_desc_segment(descs, bin_ids[t], d_data_len, d_offsets, T, SA, Tl, sa, n, len, depth);
    if (len <= 1) return;

    uint32_t r[MAXLEN];   // sorted suffixes
    int      L[MAXLEN];   // L[k] = LCP(r[k-1], r[k]); L[0]=0
    #pragma unroll
    for (int i = 0; i < MAXLEN; ++i) if (i < len) r[i] = sa[i];
    L[0] = 0;
    for (int i = 1; i < len; ++i) {
        const uint32_t key = r[i];
        int h = 0;
        int c = cmp_suffix_lcp(Tl, n, key, r[i-1], depth, &h);
        if (c > 0) { L[i] = h; continue; }
        int right_lcp = h;
        int j = i - 1;
        r[j+1] = r[j];
        for (;;) {
            const int Lj = L[j];
            if (j == 0) break;
            if (Lj > h) { right_lcp = h; r[j] = r[j-1]; L[j+1] = L[j]; --j; continue; }
            else if (Lj < h) { right_lcp = Lj; break; }
            else { int nl = 0; int cc = cmp_suffix_lcp(Tl, n, key, r[j-1], h, &nl);
                   if (cc > 0) { right_lcp = nl; break; }
                   else { h = nl; right_lcp = nl; r[j] = r[j-1]; L[j+1] = L[j]; --j; } }
        }
        r[j] = key;
        if (j > 0) L[j] = right_lcp;
        L[j+1] = h;
    }
    #pragma unroll
    for (int i = 0; i < MAXLEN; ++i) if (i < len) sa[i] = r[i];
}

template<int MAXLEN>
__global__ void sb_sort_lcp_msort_desc_kernel(
        const uint8_t* __restrict__ T,
        const int32_t* __restrict__ d_data_len,
        const int* __restrict__ d_offsets,
        const uint64_t* __restrict__ descs,
        const uint32_t* __restrict__ bin_ids,
        uint32_t bin_count,
        int d,
        uint32_t* __restrict__ SA) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= bin_count) return;
    const uint8_t* Tl = nullptr;
    uint32_t* sa = nullptr;
    int n = 0, len = 0, depth = 0;
    sb_desc_segment(descs, bin_ids[t], d_data_len, d_offsets, T, SA, Tl, sa, n, len, depth);
    if (len <= 1) return;

    uint32_t r[MAXLEN];
    int      L[MAXLEN];
    uint32_t m[MAXLEN];
    int      Lm[MAXLEN];
    #pragma unroll
    for (int i = 0; i < MAXLEN; ++i) if (i < len) r[i] = sa[i];

    int nleaves = 1 << d;
    if (nleaves > len) nleaves = len;
    for (int li = 0; li < nleaves; ++li) {
        int lo = (int)(((long)li * len) / nleaves);
        int hi = (int)(((long)(li + 1) * len) / nleaves);
        sb_lcp_insort_run<MAXLEN>(Tl, n, depth, r, L, lo, hi);
    }

    uint32_t* src = r;  int* Ls = L;
    uint32_t* dst = m;  int* Ld = Lm;
    int runs = nleaves;
    int leafstep = 1;
    while (runs > 1) {
        int li = 0;
        while (li < nleaves) {
            int a0 = (int)(((long)li * len) / nleaves);
            int lmid = li + leafstep;
            if (lmid >= nleaves) {
                for (int x = a0; x < len; ++x) { dst[x] = src[x]; Ld[x] = Ls[x]; }
                li = nleaves;
            } else {
                int a1 = (int)(((long)lmid * len) / nleaves);
                int lend = li + 2 * leafstep;
                if (lend > nleaves) lend = nleaves;
                int a2 = (int)(((long)lend * len) / nleaves);
                sb_lcp_merge_runs(Tl, n, depth, src, Ls, dst, Ld, a0, a1, a2);
                li = lend;
            }
        }
        uint32_t* ts = src; src = dst; dst = ts;
        int* tls = Ls; Ls = Ld; Ld = tls;
        leafstep <<= 1;
        runs = (runs + 1) >> 1;
    }

    #pragma unroll
    for (int i = 0; i < MAXLEN; ++i) if (i < len) sa[i] = src[i];
}

template<int MAXLEN, int MINLEN, int MAXRUN>
__global__ void sb_sort_lcp_msort_desc_scan_kernel(
        const uint8_t* __restrict__ T,
        const int32_t* __restrict__ d_data_len,
        const int* __restrict__ d_offsets,
        const uint64_t* __restrict__ descs,
        uint32_t bucket_total_count,
        int d,
        uint32_t* __restrict__ SA) {
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
         s < bucket_total_count; s += stride) {
        uint32_t hash_id = 0, start = 0, ulen = 0, udepth = 0;
        d_unpack_bucket_desc(descs[s], hash_id, start, ulen, udepth);
        if (ulen < (uint32_t)MINLEN || ulen > (uint32_t)MAXRUN) continue;
        static constexpr int HOST_SDATA_STRIDE = 72 * 1024;
        const uint8_t* Tl = T + (size_t)hash_id * HOST_SDATA_STRIDE;
        uint32_t* sa = SA + (uint32_t)d_offsets[hash_id] + start;
        const int n = (int)d_data_len[hash_id];
        const int len = (int)ulen;
        const int depth = (int)udepth;

        uint32_t r[MAXLEN];
        int      L[MAXLEN];
        uint32_t m[MAXLEN];
        int      Lm[MAXLEN];
        #pragma unroll
        for (int i = 0; i < MAXLEN; ++i) if (i < len) r[i] = sa[i];

        int nleaves = 1 << d;
        if (nleaves > len) nleaves = len;
        for (int li = 0; li < nleaves; ++li) {
            int lo = (int)(((long)li * len) / nleaves);
            int hi = (int)(((long)(li + 1) * len) / nleaves);
            sb_lcp_insort_run<MAXLEN>(Tl, n, depth, r, L, lo, hi);
        }

        uint32_t* src = r;  int* Ls = L;
        uint32_t* dst = m;  int* Ld = Lm;
        int runs = nleaves;
        int leafstep = 1;
        while (runs > 1) {
            int li = 0;
            while (li < nleaves) {
                int a0 = (int)(((long)li * len) / nleaves);
                int lmid = li + leafstep;
                if (lmid >= nleaves) {
                    for (int x = a0; x < len; ++x) { dst[x] = src[x]; Ld[x] = Ls[x]; }
                    li = nleaves;
                } else {
                    int a1 = (int)(((long)lmid * len) / nleaves);
                    int lend = li + 2 * leafstep;
                    if (lend > nleaves) lend = nleaves;
                    int a2 = (int)(((long)lend * len) / nleaves);
                    sb_lcp_merge_runs(Tl, n, depth, src, Ls, dst, Ld, a0, a1, a2);
                    li = lend;
                }
            }
            uint32_t* ts = src; src = dst; dst = ts;
            int* tls = Ls; Ls = Ld; Ld = tls;
            leafstep <<= 1;
            runs = (runs + 1) >> 1;
        }

        #pragma unroll
        for (int i = 0; i < MAXLEN; ++i) if (i < len) sa[i] = src[i];
    }
}

__global__ void sb_bin_count_desc_tail_kernel(const uint64_t* __restrict__ d_bucket_descs,
                                              uint32_t bucket_total_count,
                                              uint32_t* __restrict__ tail_counts,
                                              uint32_t* __restrict__ d_max_len) {
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
         s < bucket_total_count; s += stride) {
        uint32_t hash_id = 0, start = 0, len = 0, depth = 0;
        d_unpack_bucket_desc(d_bucket_descs[s], hash_id, start, len, depth);
        int b = sb_bin_of((int)len);
        if (b >= 2) atomicAdd(&tail_counts[b - 2], 1u);
        if (len > 1) atomicMax(d_max_len, len);
    }
}

__global__ void sb_bin_scatter_desc_tail_kernel(const uint64_t* __restrict__ d_bucket_descs,
                                                uint32_t bucket_total_count,
                                                uint32_t* __restrict__ tail_cursor,
                                                uint32_t* __restrict__ ids_packed) {
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
         s < bucket_total_count; s += stride) {
        uint32_t hash_id = 0, start = 0, len = 0, depth = 0;
        d_unpack_bucket_desc(d_bucket_descs[s], hash_id, start, len, depth);
        int b = sb_bin_of((int)len);
        if (b < 2) continue;
        uint32_t pos = atomicAdd(&tail_cursor[b - 2], 1u);
        ids_packed[pos] = s;
    }
}

template<int BLOCK, int SH_MAX>
__global__ void sb_sort_medium_binned_desc_kernel(
        const uint8_t* __restrict__ T,
        const int32_t* __restrict__ d_data_len,
        const int* __restrict__ d_offsets,
        const uint64_t* __restrict__ descs,
        const uint32_t* __restrict__ bin_ids,
        uint32_t bin_count,
        uint32_t* __restrict__ SA) {
    __shared__ uint32_t buf[SH_MAX];
    uint32_t bi = blockIdx.x;
    if (bi >= bin_count) return;
    const uint8_t* Tl = nullptr;
    uint32_t* sa = nullptr;
    int n = 0, len = 0, depth = 0;
    sb_desc_segment(descs, bin_ids[bi], d_data_len, d_offsets, T, SA, Tl, sa, n, len, depth);
    if (len <= 1) return;
    int tid = threadIdx.x;
    int m = (int)sb_next_pow2((uint32_t)len);
    for (int i = tid; i < m; i += BLOCK) buf[i] = (i < len) ? sa[i] : STAGEB_SENTINEL;
    __syncthreads();
    sb_bitonic_array(buf, m, Tl, n, depth, tid, BLOCK);
    for (int i = tid; i < len; i += BLOCK) sa[i] = buf[i];
}

template<int BLOCK>
__global__ void sb_sort_large_desc_kernel(
        const uint8_t* __restrict__ T,
        const int32_t* __restrict__ d_data_len,
        const int* __restrict__ d_offsets,
        const uint64_t* __restrict__ descs,
        const uint32_t* __restrict__ big_seg_ids,
        uint32_t num_big,
        uint32_t scratch_stride,
        uint32_t* __restrict__ scratch,
        uint32_t* __restrict__ SA) {
    uint32_t bi = blockIdx.x;
    if (bi >= num_big) return;
    const uint8_t* Tl = nullptr;
    uint32_t* sa = nullptr;
    int n = 0, len = 0, depth = 0;
    sb_desc_segment(descs, big_seg_ids[bi], d_data_len, d_offsets, T, SA, Tl, sa, n, len, depth);
    if (len <= 1) return;
    uint32_t* a = scratch + (size_t)bi * scratch_stride;
    int tid = threadIdx.x;
    int m = (int)sb_next_pow2((uint32_t)len);
    for (int i = tid; i < m; i += BLOCK) a[i] = (i < len) ? sa[i] : STAGEB_SENTINEL;
    __syncthreads();
    sb_bitonic_array(a, m, Tl, n, depth, tid, BLOCK);
    for (int i = tid; i < len; i += BLOCK) sa[i] = a[i];
}

// Build the bin4 (large-segment) id list: scan ids_packed[bin_base[4]..]
// already gives bin4 ids contiguously, so the large kernel can index it
// directly via (ids_packed + bin_base[4]).  No separate kernel needed.

__global__ void extract_sa_from_sorted_kernel(const uint32_t* __restrict__ d_sorted_vals, const int32_t* d_data_len, const int* d_offsets, int32_t* d_sa, int batch_size) {
    int hash_id = blockIdx.y; if (hash_id >= batch_size) return;
    int n = d_data_len[hash_id]; if (n < 2 || n > 70911) return;
    int base_offset = d_offsets[hash_id]; static constexpr int MAX_SA_N = 71429;
    for (int i = threadIdx.x; i < n; i += blockDim.x) d_sa[(size_t)hash_id * MAX_SA_N + i] = (int32_t)d_sorted_vals[base_offset + i];
}

GPUContext* gpu_create_context(const GPUMinerConfig& config) {
    auto* ctx = new GPUContext{};
    ctx->device_id = config.device_id;
    ctx->block_size = 256;
    ctx->engine_mode = config.engine_mode;
    ctx->staged_subbatch = std::max(1, config.staged_subbatch);
    ctx->perf_logging = config.perf_logging;
    CUDA_CHECK(cudaSetDevice(config.device_id));
    if (config.batch_size <= 0) {
        if (gpu_engine_is_exact(ctx->engine_mode)) ctx->batch_size = 96;
        else if (gpu_engine_is_cleanroom(ctx->engine_mode)) ctx->batch_size = 96;
        else if (gpu_engine_is_staged(ctx->engine_mode)) ctx->batch_size = 2954;
    else if (gpu_engine_is_recovered(ctx->engine_mode)) ctx->batch_size = 4096;  // EXACTTINY+BIN2LCP @4096 = ~12.8 KH/s bench / ~13 live on RTX 4070 Laptop (beats astronv 12.53). exact-tiny resource cliff now sits in (4096,4224] (was 4256); 4096 is the safe peak below it. Split-small (EXACTTINY_OFF) is cliff-free to ~4608 but slower (~12.2).
        else ctx->batch_size = 1536;
    } else {
        ctx->batch_size = config.batch_size;
    }
    ctx->total_sa_elements = ctx->batch_size * 71429;

    const size_t total = (size_t)ctx->total_sa_elements;
    CUDA_CHECK(cudaMalloc(&ctx->d_sdata, (size_t)ctx->batch_size * (72 * 1024)));
    CUDA_CHECK(cudaMalloc(&ctx->d_data_len, (size_t)ctx->batch_size * sizeof(int32_t)));
    if (gpu_engine_is_exact(ctx->engine_mode) || gpu_engine_is_cleanroom(ctx->engine_mode))
        CUDA_CHECK(cudaMalloc(&ctx->d_sa, total * sizeof(int32_t)));  // exact/cleanroom only; recovered SA lives in d_vals_out
    CUDA_CHECK(cudaMalloc(&ctx->d_keys_in, total * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&ctx->d_keys_out, total * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&ctx->d_vals_in, total * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx->d_vals_out, total * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx->d_bucket_counts, (size_t)ctx->batch_size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx->d_ambiguous_counts, (size_t)ctx->batch_size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx->d_max_bucket_sizes, (size_t)ctx->batch_size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx->d_bucket_total_count, sizeof(uint32_t)));
    if (gpu_engine_is_exact(ctx->engine_mode) || gpu_engine_is_cleanroom(ctx->engine_mode)) {
        CUDA_CHECK(cudaMalloc(&ctx->d_bucket_descs, total * sizeof(uint64_t)));
    } else {
        // [Stage 2: drop bucket_descs] recovered/staged: alias onto d_keys_out (same total*8 bytes).
        // d_keys_out holds the dead shifted-keys byproduct after the final ordered-keys pass;
        // bucket-detect reads d_keys_in and writes descs here, then unpack consumes them into the
        // seg arrays -> no live d_keys_out reader. Saves 0.545 MB/hash. (No fallback in this path.)
        ctx->d_bucket_descs = ctx->d_keys_out;
    }
    CUDA_CHECK(cudaMalloc(&ctx->d_offsets, (size_t)(ctx->batch_size + 1) * sizeof(int)));

    // ---- Stage-B cooperative refine buffers ----
    // Cap segments well above the observed real-scratch bucket count
    // (~15.6M @ batch 1536); clamp to total_sa_elements (the absolute max).
    ctx->sb_seg_cap = std::min<size_t>(total, (size_t)ctx->batch_size * 13800);  // 1.29x measured max segs/hash (10,674); over-cap falls back to descs refine (still correct)
    const bool use_descdirect_stageb =
        gpu_engine_is_recovered(ctx->engine_mode) && getenv("DESCDIRECT_OFF") == nullptr;
    // bin4 global-scratch budget: ~512 MB of uint32.
    ctx->sb_scratch_cap_elems = (size_t)64 * 1024;  // floor: bin4 unused (max_len~674); guard trips fallback if a real bin4 seg ever appears (was 512MB)
    if (!use_descdirect_stageb) {
        CUDA_CHECK(cudaMalloc(&ctx->d_sb_seg_off,   ctx->sb_seg_cap * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&ctx->d_sb_seg_len,   ctx->sb_seg_cap * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&ctx->d_sb_seg_depth, ctx->sb_seg_cap * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&ctx->d_sb_seg_n,     ctx->sb_seg_cap * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&ctx->d_sb_seg_Tbase, ctx->sb_seg_cap * sizeof(uint64_t)));
    }
    CUDA_CHECK(cudaMalloc(&ctx->d_sb_ids_packed, ctx->sb_seg_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx->d_sb_bin_counts, SB_NBINS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx->d_sb_bin_base,   SB_NBINS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx->d_sb_bin_cursor, SB_NBINS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx->d_sb_split_counts, SB_META_NBINS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx->d_sb_split_cursor, SB_META_NBINS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx->d_sb_max_len,    sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx->d_sb_scratch,    ctx->sb_scratch_cap_elems * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_sb_meta,   (SB_NBINS + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_sb_split_meta, (SB_META_NBINS + 1) * sizeof(uint32_t)));

    ctx->sort_temp_size = 0;
    if (gpu_engine_is_exact(ctx->engine_mode) || gpu_engine_is_cleanroom(ctx->engine_mode)) {
        // Exact/cleanroom (gpu_build_exact_sa) uses the separate-pointer SortPairs
        // API -> CUB allocates an O(total) alternate-buffer temp (~0.82 MB/hash).
        cub::DeviceSegmentedRadixSort::SortPairs(nullptr, ctx->sort_temp_size,
            ctx->d_keys_in, ctx->d_keys_out,
            ctx->d_vals_in, ctx->d_vals_out,
            (int)total, ctx->batch_size, ctx->d_offsets, ctx->d_offsets + 1, 0, 64);
        size_t sort_temp_size32 = 0;
        cub::DeviceSegmentedRadixSort::SortPairs(nullptr, sort_temp_size32,
            reinterpret_cast<uint32_t*>(ctx->d_keys_in), reinterpret_cast<uint32_t*>(ctx->d_keys_out),
            ctx->d_vals_in, ctx->d_vals_out,
            (int)total, ctx->batch_size, ctx->d_offsets, ctx->d_offsets + 1, 0, 32);
        if (sort_temp_size32 > ctx->sort_temp_size) ctx->sort_temp_size = sort_temp_size32;
    } else {
        // Recovered/staged (gpu_build_fast_sa_window) uses cub::DoubleBuffer, ping-ponging
        // in the owned d_keys_in/out + d_vals_in/out buffers -> only a tiny histogram temp
        // instead of the ~0.82 MB/hash alternate-buffer blob. [Stage 1: drop CUB temp]
        cub::DoubleBuffer<uint64_t> dk(ctx->d_keys_in, ctx->d_keys_out);
        cub::DoubleBuffer<uint32_t> dv(ctx->d_vals_in, ctx->d_vals_out);
        cub::DeviceSegmentedRadixSort::SortPairs(nullptr, ctx->sort_temp_size, dk, dv,
            (int)total, ctx->batch_size, ctx->d_offsets, ctx->d_offsets + 1, 0, 64);
        // [NARROWKEY] also size the uint32-key sort temp and take the max so the
        // shared d_sort_temp is large enough whichever key width runs at mine-time.
        size_t st32 = 0;
        cub::DoubleBuffer<uint32_t> dk32(reinterpret_cast<uint32_t*>(ctx->d_keys_in), reinterpret_cast<uint32_t*>(ctx->d_keys_out));
        cub::DoubleBuffer<uint32_t> dv32(ctx->d_vals_in, ctx->d_vals_out);
        cub::DeviceSegmentedRadixSort::SortPairs(nullptr, st32, dk32, dv32,
            (int)total, ctx->batch_size, ctx->d_offsets, ctx->d_offsets + 1, 0, 32);
        if (st32 > ctx->sort_temp_size) ctx->sort_temp_size = st32;
    }
    CUDA_CHECK(cudaMalloc(&ctx->d_sort_temp, ctx->sort_temp_size));
    CUDA_CHECK(cudaMalloc(&ctx->d_solutions, (1 + 1024 * 5) * sizeof(uint64_t)));

    CUDA_CHECK(cudaMallocHost(&ctx->h_solutions, (1 + 1024 * 5) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_data_len, (size_t)ctx->batch_size * sizeof(int32_t)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_bucket_counts, (size_t)ctx->batch_size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_ambiguous_counts, (size_t)ctx->batch_size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_max_bucket_sizes, (size_t)ctx->batch_size * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpyToSymbol(d_CodeLUT, astrobwt::CodeLUT, sizeof(astrobwt::CodeLUT)));
    CUDA_CHECK(cudaStreamCreate(&ctx->stream));
    return ctx;
}

void gpu_destroy_context(GPUContext* ctx) {
    if (!ctx) return;
    cudaSetDevice(ctx->device_id);
    cudaFree(ctx->d_sdata);
    cudaFree(ctx->d_data_len);
    cudaFree(ctx->d_sa);
    cudaFree(ctx->d_keys_in);
    cudaFree(ctx->d_keys_out);
    cudaFree(ctx->d_vals_in);
    cudaFree(ctx->d_vals_out);
    cudaFree(ctx->d_bucket_counts);
    cudaFree(ctx->d_ambiguous_counts);
    cudaFree(ctx->d_max_bucket_sizes);
    cudaFree(ctx->d_bucket_total_count);
    if (ctx->d_bucket_descs != (uint64_t*)ctx->d_keys_out) cudaFree(ctx->d_bucket_descs);  // alias in recovered/staged
    cudaFree(ctx->d_offsets);
    cudaFree(ctx->d_sb_seg_off);
    cudaFree(ctx->d_sb_seg_len);
    cudaFree(ctx->d_sb_seg_depth);
    cudaFree(ctx->d_sb_seg_n);
    cudaFree(ctx->d_sb_seg_Tbase);
    cudaFree(ctx->d_sb_ids_packed);
    cudaFree(ctx->d_sb_bin_counts);
    cudaFree(ctx->d_sb_bin_base);
    cudaFree(ctx->d_sb_bin_cursor);
    cudaFree(ctx->d_sb_split_counts);
    cudaFree(ctx->d_sb_split_cursor);
    cudaFree(ctx->d_sb_max_len);
    cudaFree(ctx->d_sb_scratch);
    cudaFreeHost(ctx->h_sb_meta);
    cudaFreeHost(ctx->h_sb_split_meta);
    cudaFree(ctx->d_sort_temp);
    cudaFree(ctx->d_solutions);
    cudaFreeHost(ctx->h_solutions);
    cudaFreeHost(ctx->h_data_len);
    cudaFreeHost(ctx->h_bucket_counts);
    cudaFreeHost(ctx->h_ambiguous_counts);
    cudaFreeHost(ctx->h_max_bucket_sizes);
    cudaStreamDestroy(ctx->stream);
    delete ctx;
}

void gpu_set_work(GPUContext* ctx, const uint8_t work[112], uint64_t difficulty) { std::memcpy(ctx->work, work, 112); ctx->difficulty = difficulty; cudaSetDevice(ctx->device_id); CUDA_CHECK(cudaMemcpyToSymbol(d_work_template, work, 48)); uint8_t target[32]; astrobwt::compute_target(difficulty, target); CUDA_CHECK(cudaMemcpyToSymbol(d_difficulty_target, target, 32)); }

int gpu_get_batch_size(GPUContext* ctx) { return ctx->batch_size; }

static bool gpu_collect_bucket_stats(GPUContext* ctx, int active_batch, GPUSABuildStats* stats) {
    if (!stats || active_batch <= 0) return true;
    CUDA_CHECK(cudaMemcpyAsync(ctx->h_bucket_counts, ctx->d_bucket_counts, (size_t)active_batch * sizeof(uint32_t), cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->h_ambiguous_counts, ctx->d_ambiguous_counts, (size_t)active_batch * sizeof(uint32_t), cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->h_max_bucket_sizes, ctx->d_max_bucket_sizes, (size_t)active_batch * sizeof(uint32_t), cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    stats->bucket_count = 0;
    stats->ambiguous_count = 0;
    stats->max_bucket_size = 0;
    for (int i = 0; i < active_batch; ++i) {
        stats->bucket_count += ctx->h_bucket_counts[i];
        stats->ambiguous_count += ctx->h_ambiguous_counts[i];
        stats->max_bucket_size = std::max(stats->max_bucket_size, ctx->h_max_bucket_sizes[i]);
    }
    return true;
}

static bool gpu_build_exact_sa(GPUContext* ctx, int active_batch, int& total_elements, GPUSABuildStats* stats = nullptr) {
    cudaSetDevice(ctx->device_id);
    total_elements = 0;
    uint32_t bucket_total_count = 0;
    if (active_batch <= 0) return true;
    cudaEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr, e3 = nullptr, e4 = nullptr;
    if (stats) {
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventCreate(&e2);
        cudaEventCreate(&e3);
        cudaEventCreate(&e4);
        cudaEventRecord(e0, ctx->stream);
    }

    build_offsets_kernel<<<1, 1, 0, ctx->stream>>>(ctx->d_data_len, ctx->d_offsets, active_batch);
    if (stats) cudaEventRecord(e1, ctx->stream);
    CUDA_CHECK(cudaMemcpyAsync(&total_elements, ctx->d_offsets + active_batch, sizeof(int), cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    if (stats) cudaEventSynchronize(e1);
    if (total_elements <= 0) {
        if (stats) {
            cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3); cudaEventDestroy(e4);
        }
        return true;
    }

    if (stats) {
        CUDA_CHECK(cudaMemsetAsync(ctx->d_bucket_counts, 0, (size_t)active_batch * sizeof(uint32_t), ctx->stream));
        CUDA_CHECK(cudaMemsetAsync(ctx->d_ambiguous_counts, 0, (size_t)active_batch * sizeof(uint32_t), ctx->stream));
        CUDA_CHECK(cudaMemsetAsync(ctx->d_max_bucket_sizes, 0, (size_t)active_batch * sizeof(uint32_t), ctx->stream));
    }
    CUDA_CHECK(cudaMemsetAsync(ctx->d_bucket_total_count, 0, sizeof(uint32_t), ctx->stream));

    init_sort_bytes64_kernel<<<dim3(1, active_batch), 256, 0, ctx->stream>>>(
        ctx->d_sdata, ctx->d_data_len, ctx->d_offsets, ctx->d_keys_in, ctx->d_vals_in, active_batch, kFastPrefixBytes);
    if (stats) cudaEventRecord(e2, ctx->stream);
    cub::DeviceSegmentedRadixSort::SortPairs(
        ctx->d_sort_temp, ctx->sort_temp_size,
        ctx->d_keys_in, ctx->d_keys_out,
        ctx->d_vals_in, ctx->d_vals_out,
        total_elements, active_batch, ctx->d_offsets, ctx->d_offsets + 1, 0, 64, ctx->stream);
    build_ordered_bytes64_kernel<<<dim3(1, active_batch), 256, 0, ctx->stream>>>(
        ctx->d_sdata, ctx->d_data_len, ctx->d_offsets, ctx->d_vals_out, ctx->d_keys_in, ctx->d_vals_in, active_batch, 0);
    cub::DeviceSegmentedRadixSort::SortPairs(
        ctx->d_sort_temp, ctx->sort_temp_size,
        ctx->d_keys_in, ctx->d_keys_out,
        ctx->d_vals_in, ctx->d_vals_out,
        total_elements, active_batch, ctx->d_offsets, ctx->d_offsets + 1, 0, 64, ctx->stream);
    if (stats) cudaEventRecord(e3, ctx->stream);

    build_ordered_bytes64_kernel<<<dim3(1, active_batch), 256, 0, ctx->stream>>>(
        ctx->d_sdata, ctx->d_data_len, ctx->d_offsets, ctx->d_vals_out, ctx->d_keys_in, ctx->d_vals_in, active_batch, kFastPrefixBytes);
    build_bucket_descriptors_kernel<<<active_batch, 256, 0, ctx->stream>>>(
        ctx->d_keys_out, ctx->d_keys_in, ctx->d_data_len, ctx->d_offsets,
        ctx->d_bucket_descs, ctx->d_bucket_total_count,
        stats ? ctx->d_bucket_counts : nullptr,
        stats ? ctx->d_ambiguous_counts : nullptr,
        stats ? ctx->d_max_bucket_sizes : nullptr,
        (uint32_t)(kFastPrefixBytes * 2),
        active_batch);
    CUDA_CHECK(cudaMemcpyAsync(&bucket_total_count, ctx->d_bucket_total_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    if (bucket_total_count > 0) {
        const uint32_t refine_blocks = ((bucket_total_count + 255U) / 256U)  /* gridDim.x max is 2^31-1, NOT 65535; capping silently dropped segments >16.78M (batch>~1536) -> corrupt SA */;
        refine_bucket_descs_kernel<<<refine_blocks, 256, 0, ctx->stream>>>(
            ctx->d_sdata, ctx->d_data_len, ctx->d_offsets, ctx->d_bucket_descs, bucket_total_count, ctx->d_vals_out);
    }
    extract_sa_from_sorted_kernel<<<dim3(1, active_batch), 256, 0, ctx->stream>>>(
        ctx->d_vals_out, ctx->d_data_len, ctx->d_offsets, ctx->d_sa, active_batch);
    if (stats) {
        cudaEventRecord(e4, ctx->stream);
        CUDA_CHECK(cudaEventSynchronize(e4));
        cudaEventElapsedTime(&stats->offsets_ms, e0, e1);
        cudaEventElapsedTime(&stats->keys_ms, e1, e2);
        cudaEventElapsedTime(&stats->sort_ms, e2, e3);
        cudaEventElapsedTime(&stats->refine_ms, e3, e4);
        gpu_collect_bucket_stats(ctx, active_batch, stats);
        cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3); cudaEventDestroy(e4);
    }
    return true;
}

// ---------------------------------------------------------------------
// Stage-B cooperative refine. Replaces the single-thread-per-bucket
// insertion-sort refine. Returns true if it ran; false if the segment count
// or bin4 scratch budget is exceeded (caller must fall back to the old
// refine_bucket_descs_kernel for this batch).
// ---------------------------------------------------------------------
__global__ void rd_compare_sa_kernel(const uint32_t* __restrict__ d_vals_a, const uint32_t* __restrict__ d_vals_b, const int32_t* __restrict__ d_data_len, const int* __restrict__ d_offsets, int batch_size, unsigned long long* __restrict__ d_mismatch);
static bool gpu_stageb_refine(GPUContext* ctx,
                              const uint8_t* d_sdata_base,
                              const int32_t* d_data_len_base,
                              uint32_t bucket_total_count,
                              int active_batch,
                              int total_elements) {
    if (bucket_total_count == 0) return true;             // nothing to do
    if (bucket_total_count > ctx->sb_seg_cap) return false; // too many segs -> fallback

    cudaStream_t st = ctx->stream;
    { static bool _fcinit=false; if(!_fcinit){ int _fc=getenv("FAKECMP")?1:0; cudaMemcpyToSymbol(g_sb_fakecmp,&_fc,sizeof(int)); _fcinit=true; } }
    { static bool _wcinit=false; if(!_wcinit){ int _wc=getenv("WIDECMP_OFF")?0:1; cudaMemcpyToSymbol(g_sb_widecmp,&_wc,sizeof(int)); _wcinit=true; } }
    { static bool _fdinit=false; if(!_fdinit){ int _fd=getenv("FASTDIFF_OFF")?0:1; cudaMemcpyToSymbol(g_sb_fastdiff,&_fd,sizeof(int)); _fdinit=true; } }

    // [PERHASH] per-hash fused refine: one block/hash re-detects tied runs from the
    // sorted keys and sorts in place — skips the entire unpack/bin-count/scatter
    // machinery below. Requires NARROWKEY (uint32 keys in d_keys_in). Env PERHASH=1.
    static const bool perhash = (getenv("PERHASH") != nullptr);
    if (perhash) {
        sb_perhash_coop_kernel<128><<<active_batch, 128, 0, st>>>(
            d_sdata_base, d_data_len_base, ctx->d_offsets,
            reinterpret_cast<const uint32_t*>(ctx->d_keys_in), ctx->d_vals_out,
            active_batch, (int)kInitialPrefixSymbols);
        return true;
    }

    // [PHMSORT] per-hash whole-array cooperative merge-path mergesort (astronv CUZSORT replica).
    // One block/hash sorts the hash's coarse-sorted SA range fully by suffix, cooperatively,
    // ping-ponging d_vals_out <-> d_vals_in. Skips the bin machinery. Env PHMSORT=1.
    static const bool phmsort = (getenv("PHMSORT") != nullptr);
    if (phmsort) {
        sb_perhash_msort_kernel<256><<<active_batch, 256, 0, st>>>(
            d_sdata_base, d_data_len_base, ctx->d_offsets,
            ctx->d_vals_out, ctx->d_vals_in, active_batch);
        return true;
    }

    // [PHMLCP] same cooperative per-hash mergesort but with LCP-carrying merges
    // (derocuda's edge astronv lacks). LCP arrays ping-pong in d_keys_in/d_keys_out
    // (free scratch in this path). Env PHMLCP=1.
    static const bool phmlcp = (getenv("PHMLCP") != nullptr);
    if (phmlcp) {
        sb_perhash_msort_lcp_kernel<256><<<active_batch, 256, 0, st>>>(
            d_sdata_base, d_data_len_base, ctx->d_offsets,
            ctx->d_vals_out, ctx->d_vals_in,
            reinterpret_cast<int*>(ctx->d_keys_in), reinterpret_cast<int*>(ctx->d_keys_out),
            active_batch);
        return true;
    }

    // [PHMTILE] shared-tiled cooperative mergesort (astronv structure): within-tile
    // sort fully on-chip (SA indices in shared), only the tile-merge in global. Env PHMTILE=1.
    static const bool phmtile = (getenv("PHMTILE") != nullptr);
    if (phmtile) {
        sb_perhash_msort_tiled_kernel<256, 1024><<<active_batch, 256, 0, st>>>(
            d_sdata_base, d_data_len_base, ctx->d_offsets,
            ctx->d_vals_out, ctx->d_vals_in, active_batch);
        return true;
    }

    // [DESCDIRECT] Keep packed bucket descriptors as the segment metadata.
    // This skips unpack_bucket_descs_kernel and avoids reloading five scattered
    // per-segment arrays in the hot tiny-bin sorters. Default-on for recovered;
    // DESCDIRECT_OFF restores the unpacked-array path for A/B checks.
    static const bool descdirect = (getenv("DESCDIRECT_OFF") == nullptr);
    if (descdirect) {
        // [TINYDIRECT] Opt-in rollback: descriptor scans for len<=32 are stable
        // at high batch but slower (~9.5 KH/s). Default stays on the exact-tiny
        // launch train, which is parity-green and peaks near 13.0 KH/s at b4224.
        static const bool tinydirect = (getenv("TINYDIRECT") != nullptr);
        if (tinydirect) {
            CUDA_CHECK(cudaMemsetAsync(ctx->d_sb_max_len, 0, sizeof(uint32_t), st));
            CUDA_CHECK(cudaMemsetAsync(ctx->d_sb_bin_counts, 0, 3 * sizeof(uint32_t), st));
            {
                const uint32_t blocks = ((bucket_total_count + 255U) / 256U);
                sb_bin_count_desc_tail_kernel<<<blocks, 256, 0, st>>>(
                    ctx->d_bucket_descs, bucket_total_count,
                    ctx->d_sb_bin_counts, ctx->d_sb_max_len);
            }

            uint32_t* h = ctx->h_sb_meta;
            CUDA_CHECK(cudaMemcpyAsync(h, ctx->d_sb_bin_counts, 3 * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, st));
            CUDA_CHECK(cudaMemcpyAsync(h + SB_NBINS, ctx->d_sb_max_len, sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            const uint32_t bin2 = h[0], bin3 = h[1], bin4 = h[2];
            const uint32_t max_len = h[SB_NBINS];
            uint32_t tail_base[3] = {0, bin2, bin2 + bin3};

            uint32_t scratch_stride = 1u;
            while (scratch_stride < max_len) scratch_stride <<= 1;
            if (scratch_stride < 2u) scratch_stride = 2u;
            if (bin4 > 0 &&
                (size_t)bin4 * (size_t)scratch_stride > ctx->sb_scratch_cap_elems) {
                return false;
            }

            CUDA_CHECK(cudaMemcpyAsync(ctx->d_sb_bin_cursor, tail_base, 3 * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice, st));
            {
                const uint32_t blocks = ((bucket_total_count + 255U) / 256U);
                sb_bin_scatter_desc_tail_kernel<<<blocks, 256, 0, st>>>(
                    ctx->d_bucket_descs, bucket_total_count,
                    ctx->d_sb_bin_cursor, ctx->d_sb_ids_packed);
            }

            const uint8_t* T = d_sdata_base;
            const uint64_t* descs = ctx->d_bucket_descs;
            const uint32_t* ids = ctx->d_sb_ids_packed;
            uint32_t* SA = ctx->d_vals_out;
            static const int msort_d = (getenv("MSORT_D") ? atoi(getenv("MSORT_D")) : 0);
            static const int msort_d0 = (getenv("MSORT_D0") ? atoi(getenv("MSORT_D0")) : 0);
            static const bool sb_debug = (getenv("STAGEB_DEBUG") != nullptr);
            cudaEvent_t _be[6] = {};
            if (sb_debug) {
                fprintf(stderr,
                    "[TINYDIRECT] segs=%u tail=[%u %u %u] max_len=%u stride=%u scratch_used=%zu/%zu\n",
                    bucket_total_count, bin2, bin3, bin4, max_len, scratch_stride,
                    (size_t)bin4 * (size_t)scratch_stride, ctx->sb_scratch_cap_elems);
                for (int _i = 0; _i < 6; ++_i) cudaEventCreate(&_be[_i]);
                cudaEventRecord(_be[0], st);
            }
            {
                const uint32_t blocks = ((bucket_total_count + 127U) / 128U);
                sb_sort_lcp_msort_desc_scan_kernel<SB_BIN0_MAX, 2, SB_BIN0_MAX><<<blocks, 128, 0, st>>>(
                    T, d_data_len_base, ctx->d_offsets, descs, bucket_total_count,
                    (msort_d0 > 0 ? msort_d0 : 2), SA);
            }
            if (sb_debug) cudaEventRecord(_be[1], st);
            {
                const uint32_t blocks = ((bucket_total_count + 127U) / 128U);
                sb_sort_lcp_msort_desc_scan_kernel<SB_BIN1_MAX, SB_BIN0_MAX + 1, SB_BIN1_MAX><<<blocks, 128, 0, st>>>(
                    T, d_data_len_base, ctx->d_offsets, descs, bucket_total_count,
                    (msort_d > 0 ? msort_d : 3), SA);
            }
            if (sb_debug) cudaEventRecord(_be[2], st);
            if (bin2 > 0) {
                sb_sort_medium_binned_desc_kernel<128, SB_BIN2_MAX><<<bin2, 128, 0, st>>>(
                    T, d_data_len_base, ctx->d_offsets, descs, ids + tail_base[0], bin2, SA);
            }
            if (sb_debug) cudaEventRecord(_be[3], st);
            if (bin3 > 0) {
                sb_sort_medium_binned_desc_kernel<256, SB_BIN3_MAX><<<bin3, 256, 0, st>>>(
                    T, d_data_len_base, ctx->d_offsets, descs, ids + tail_base[1], bin3, SA);
            }
            if (sb_debug) cudaEventRecord(_be[4], st);
            if (bin4 > 0) {
                sb_sort_large_desc_kernel<256><<<bin4, 256, 0, st>>>(
                    T, d_data_len_base, ctx->d_offsets, descs,
                    ids + tail_base[2], bin4, scratch_stride, ctx->d_sb_scratch, SA);
            }
            if (sb_debug) {
                cudaEventRecord(_be[5], st);
                cudaEventSynchronize(_be[5]);
                float _t[5];
                for (int _i = 0; _i < 5; ++_i) cudaEventElapsedTime(&_t[_i], _be[_i], _be[_i + 1]);
                fprintf(stderr,
                        "[TINYDIRECT-T] bin0scan=%.1f bin1scan=%.1f bin2=%.1f bin3=%.1f bin4=%.1f ms  (tail %u/%u/%u)\n",
                        _t[0], _t[1], _t[2], _t[3], _t[4], bin2, bin3, bin4);
                for (int _i = 0; _i < 6; ++_i) cudaEventDestroy(_be[_i]);
            }
            return true;
        }

        // [EXACTTINY] Default-ON: fastest refine on the RTX 4070 Laptop -- ~12.7-12.8 KH/s bench /
        // ~13 live (BEATS astronv 12.53), vs ~12.2/12.5 for split-small. NOT broken -- it has a
        // deterministic RESOURCE cliff: the per-length tiny sorters use more registers/scratch, so
        // it is catastrophic (~1.1 KH/s) above batch ~4096 and fast at/below it. The recovered
        // default batch is 4096 (below the cliff); auto-batch is capped to land there too.
        // EXACTTINY_OFF rolls back to the cliff-free split-small path for A/B or higher-batch use.
        static const bool exacttiny = (getenv("EXACTTINY_OFF") == nullptr);
        if (exacttiny) {
            // [BIN2LCP] Default-on: most 33..512 segments are 33..64, where
            // LCP merge-sort beats the shared bitonic medium sorter by a wide
            // margin. BIN2LCP_OFF restores the single medium-sort bin2 path.
            static const bool bin2lcp = (getenv("BIN2LCP_OFF") == nullptr);
            const int exact_nbins = bin2lcp ? SB_EXT_NBINS : SB_EXACT_NBINS;
            CUDA_CHECK(cudaMemsetAsync(ctx->d_sb_max_len, 0, sizeof(uint32_t), st));
            CUDA_CHECK(cudaMemsetAsync(ctx->d_sb_split_counts, 0, exact_nbins * sizeof(uint32_t), st));
            {
                const uint32_t blocks = ((bucket_total_count + 255U) / 256U);
                if (bin2lcp) {
                    sb_bin_count_desc_ext_kernel<<<blocks, 256, 0, st>>>(
                        ctx->d_bucket_descs, bucket_total_count,
                        ctx->d_sb_split_counts, ctx->d_sb_max_len);
                } else {
                    sb_bin_count_desc_exact_kernel<<<blocks, 256, 0, st>>>(
                        ctx->d_bucket_descs, bucket_total_count,
                        ctx->d_sb_split_counts, ctx->d_sb_max_len);
                }
            }

            uint32_t* h = ctx->h_sb_split_meta;
            CUDA_CHECK(cudaMemcpyAsync(h, ctx->d_sb_split_counts, exact_nbins * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, st));
            CUDA_CHECK(cudaMemcpyAsync(h + exact_nbins, ctx->d_sb_max_len, sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            const uint32_t max_len = h[exact_nbins];

            uint32_t exact_base[SB_META_NBINS];
            uint32_t acc = 0;
            for (int b = 0; b < exact_nbins; ++b) { exact_base[b] = acc; acc += h[b]; }

            const uint32_t bin33_512 = bin2lcp ? 0 : h[31];
            const uint32_t bin33_64 = bin2lcp ? h[31] : 0;
            const uint32_t bin65_128 = bin2lcp ? h[32] : 0;
            const uint32_t bin129_512 = bin2lcp ? h[33] : 0;
            const uint32_t bin513_1024 = bin2lcp ? h[34] : h[32];
            const uint32_t bin_big = bin2lcp ? h[35] : h[33];
            uint32_t scratch_stride = 1u;
            while (scratch_stride < max_len) scratch_stride <<= 1;
            if (scratch_stride < 2u) scratch_stride = 2u;
            if (bin_big > 0 &&
                (size_t)bin_big * (size_t)scratch_stride > ctx->sb_scratch_cap_elems) {
                return false;
            }

            CUDA_CHECK(cudaMemcpyAsync(ctx->d_sb_split_cursor, exact_base, exact_nbins * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice, st));
            {
                const uint32_t blocks = ((bucket_total_count + 255U) / 256U);
                if (bin2lcp) {
                    sb_bin_scatter_desc_ext_kernel<<<blocks, 256, 0, st>>>(
                        ctx->d_bucket_descs, bucket_total_count,
                        ctx->d_sb_split_cursor, ctx->d_sb_ids_packed);
                } else {
                    sb_bin_scatter_desc_exact_kernel<<<blocks, 256, 0, st>>>(
                        ctx->d_bucket_descs, bucket_total_count,
                        ctx->d_sb_split_cursor, ctx->d_sb_ids_packed);
                }
            }

            const uint8_t* T = d_sdata_base;
            const uint64_t* descs = ctx->d_bucket_descs;
            const uint32_t* ids = ctx->d_sb_ids_packed;
            uint32_t* SA = ctx->d_vals_out;
            static const int msort_d = (getenv("MSORT_D") ? atoi(getenv("MSORT_D")) : 0);
            static const int msort_d0 = (getenv("MSORT_D0") ? atoi(getenv("MSORT_D0")) : 0);
            static const int msort_d0lo = (getenv("MSORT_D0LO") ? atoi(getenv("MSORT_D0LO")) : 0);
            static const int msort_d0hi = (getenv("MSORT_D0HI") ? atoi(getenv("MSORT_D0HI")) : 0);
            static const int msort_d1lo = (getenv("MSORT_D1LO") ? atoi(getenv("MSORT_D1LO")) : 0);
            static const int msort_d1hi = (getenv("MSORT_D1HI") ? atoi(getenv("MSORT_D1HI")) : 0);
            static const int msort_d9_12 = (getenv("MSORT_D9_12") ? atoi(getenv("MSORT_D9_12")) : 0);
            static const int msort_d13_16 = (getenv("MSORT_D13_16") ? atoi(getenv("MSORT_D13_16")) : 0);
            static const int msort_d17_24 = (getenv("MSORT_D17_24") ? atoi(getenv("MSORT_D17_24")) : 0);
            static const int msort_d25_32 = (getenv("MSORT_D25_32") ? atoi(getenv("MSORT_D25_32")) : 0);
            const int d3_4 = (msort_d0lo > 0 ? msort_d0lo : (msort_d0 > 0 ? msort_d0 : 2));
            const int d5_8 = (msort_d0hi > 0 ? msort_d0hi : (msort_d0 > 0 ? msort_d0 : 2));
            const int d9_16 = (msort_d1lo > 0 ? msort_d1lo : (msort_d > 0 ? msort_d : 3));
            const int d17_32 = (msort_d1hi > 0 ? msort_d1hi : (msort_d > 0 ? msort_d : 3));
            const int d9_12 = (msort_d9_12 > 0 ? msort_d9_12 : d9_16);
            const int d13_16 = (msort_d13_16 > 0 ? msort_d13_16 : d9_16);
            const int d17_24 = (msort_d17_24 > 0 ? msort_d17_24 : d17_32);
            const int d25_32 = (msort_d25_32 > 0 ? msort_d25_32 : d17_32);
            static const bool sb_debug = (getenv("STAGEB_DEBUG") != nullptr);
            cudaEvent_t _be[7] = {};
            if (sb_debug) {
                uint32_t tiny = 0;
                for (int i = 0; i <= 30; ++i) tiny += h[i];
                fprintf(stderr,
                    "[EXACTTINY%s] segs=%u tiny=%u bin33_512=%u bin33_64=%u bin65_128=%u bin129_512=%u bin513_1024=%u bin4=%u max_len=%u stride=%u scratch_used=%zu/%zu\n",
                    bin2lcp ? "+BIN2LCP" : "", bucket_total_count, tiny, bin33_512, bin33_64,
                    bin65_128, bin129_512, bin513_1024, bin_big, max_len, scratch_stride,
                    (size_t)bin_big * (size_t)scratch_stride, ctx->sb_scratch_cap_elems);
                for (int _i = 0; _i < 7; ++_i) cudaEventCreate(&_be[_i]);
                cudaEventRecord(_be[0], st);
            }

            if (h[0] > 0) {
                sb_sort_len2_desc_kernel<<<(h[0] + 127) / 128, 128, 0, st>>>(
                    T, d_data_len_base, ctx->d_offsets, descs, ids + exact_base[0], h[0], SA);
            }
#define SB_LAUNCH_EXACT_LEN(L, DVAL) do { \
                const uint32_t _cnt = h[(L) - 2]; \
                if (_cnt > 0) { \
                    sb_sort_lcp_msort_desc_kernel<(L)><<<(_cnt + 127) / 128, 128, 0, st>>>( \
                        T, d_data_len_base, ctx->d_offsets, descs, ids + exact_base[(L) - 2], _cnt, (DVAL), SA); \
                } \
            } while (0)
            SB_LAUNCH_EXACT_LEN(3, d3_4);
            SB_LAUNCH_EXACT_LEN(4, d3_4);
            SB_LAUNCH_EXACT_LEN(5, d5_8);
            SB_LAUNCH_EXACT_LEN(6, d5_8);
            SB_LAUNCH_EXACT_LEN(7, d5_8);
            SB_LAUNCH_EXACT_LEN(8, d5_8);
            if (sb_debug) cudaEventRecord(_be[1], st);
            SB_LAUNCH_EXACT_LEN(9, d9_12);
            SB_LAUNCH_EXACT_LEN(10, d9_12);
            SB_LAUNCH_EXACT_LEN(11, d9_12);
            SB_LAUNCH_EXACT_LEN(12, d9_12);
            SB_LAUNCH_EXACT_LEN(13, d13_16);
            SB_LAUNCH_EXACT_LEN(14, d13_16);
            SB_LAUNCH_EXACT_LEN(15, d13_16);
            SB_LAUNCH_EXACT_LEN(16, d13_16);
            if (sb_debug) cudaEventRecord(_be[2], st);
            SB_LAUNCH_EXACT_LEN(17, d17_24);
            SB_LAUNCH_EXACT_LEN(18, d17_24);
            SB_LAUNCH_EXACT_LEN(19, d17_24);
            SB_LAUNCH_EXACT_LEN(20, d17_24);
            SB_LAUNCH_EXACT_LEN(21, d17_24);
            SB_LAUNCH_EXACT_LEN(22, d17_24);
            SB_LAUNCH_EXACT_LEN(23, d17_24);
            SB_LAUNCH_EXACT_LEN(24, d17_24);
            SB_LAUNCH_EXACT_LEN(25, d25_32);
            SB_LAUNCH_EXACT_LEN(26, d25_32);
            SB_LAUNCH_EXACT_LEN(27, d25_32);
            SB_LAUNCH_EXACT_LEN(28, d25_32);
            SB_LAUNCH_EXACT_LEN(29, d25_32);
            SB_LAUNCH_EXACT_LEN(30, d25_32);
            SB_LAUNCH_EXACT_LEN(31, d25_32);
            SB_LAUNCH_EXACT_LEN(32, d25_32);
#undef SB_LAUNCH_EXACT_LEN
            if (sb_debug) cudaEventRecord(_be[3], st);
            if (bin2lcp) {
                if (bin33_64 > 0) {
                    sb_sort_lcp_msort_desc_kernel<64><<<(bin33_64 + 127) / 128, 128, 0, st>>>(
                        T, d_data_len_base, ctx->d_offsets, descs, ids + exact_base[31], bin33_64,
                        d17_32, SA);
                }
                if (bin65_128 > 0) {
                    sb_sort_lcp_msort_desc_kernel<128><<<(bin65_128 + 127) / 128, 128, 0, st>>>(
                        T, d_data_len_base, ctx->d_offsets, descs, ids + exact_base[32], bin65_128,
                        d17_32, SA);
                }
                if (bin129_512 > 0) {
                    sb_sort_medium_binned_desc_kernel<128, SB_BIN2_MAX><<<bin129_512, 128, 0, st>>>(
                        T, d_data_len_base, ctx->d_offsets, descs, ids + exact_base[33], bin129_512, SA);
                }
            } else if (bin33_512 > 0) {
                sb_sort_medium_binned_desc_kernel<128, SB_BIN2_MAX><<<bin33_512, 128, 0, st>>>(
                    T, d_data_len_base, ctx->d_offsets, descs, ids + exact_base[31], bin33_512, SA);
            }
            if (sb_debug) cudaEventRecord(_be[4], st);
            if (bin513_1024 > 0) {
                sb_sort_medium_binned_desc_kernel<256, SB_BIN3_MAX><<<bin513_1024, 256, 0, st>>>(
                    T, d_data_len_base, ctx->d_offsets, descs,
                    ids + exact_base[bin2lcp ? 34 : 32], bin513_1024, SA);
            }
            if (sb_debug) cudaEventRecord(_be[5], st);
            if (bin_big > 0) {
                sb_sort_large_desc_kernel<256><<<bin_big, 256, 0, st>>>(
                    T, d_data_len_base, ctx->d_offsets, descs,
                    ids + exact_base[bin2lcp ? 35 : 33], bin_big, scratch_stride, ctx->d_sb_scratch, SA);
            }
            if (sb_debug) {
                cudaEventRecord(_be[6], st);
                cudaEventSynchronize(_be[6]);
                float _t[6];
                for (int _i = 0; _i < 6; ++_i) cudaEventElapsedTime(&_t[_i], _be[_i], _be[_i + 1]);
                fprintf(stderr,
                        "[EXACTTINY-T] len2_8=%.1f len9_16=%.1f len17_32=%.1f bin2=%.1f bin3=%.1f bin4=%.1f ms\n",
                        _t[0], _t[1], _t[2], _t[3], _t[4], _t[5]);
                for (int _i = 0; _i < 7; ++_i) cudaEventDestroy(_be[_i]);
            }
            return true;
        }

        // [SPLITSMALL] Coarser fallback when EXACTTINY_OFF is set. LENHIST
        // shows most bin0/bin1 work sits in shorter halves, so this still
        // avoids paying the largest local-array footprint for every tiny
        // segment. SPLITSMALL_OFF or SPLITBIN1_OFF restores the single-bin path.
        static const bool splitbin1 =
            (getenv("SPLITSMALL_OFF") == nullptr && getenv("SPLITBIN1_OFF") == nullptr);
        if (splitbin1) {
            CUDA_CHECK(cudaMemsetAsync(ctx->d_sb_max_len, 0, sizeof(uint32_t), st));
            CUDA_CHECK(cudaMemsetAsync(ctx->d_sb_split_counts, 0, SB_SPLIT_NBINS * sizeof(uint32_t), st));
            {
                const uint32_t blocks = ((bucket_total_count + 255U) / 256U);
                sb_bin_count_desc_split_kernel<<<blocks, 256, 0, st>>>(
                    ctx->d_bucket_descs, bucket_total_count,
                    ctx->d_sb_split_counts, ctx->d_sb_max_len);
            }

            uint32_t* h = ctx->h_sb_split_meta;
            CUDA_CHECK(cudaMemcpyAsync(h, ctx->d_sb_split_counts, SB_SPLIT_NBINS * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, st));
            CUDA_CHECK(cudaMemcpyAsync(h + SB_SPLIT_NBINS, ctx->d_sb_max_len, sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            const uint32_t bin_len2 = h[0], bin3_4 = h[1], bin5_6 = h[2], bin7_8 = h[3];
            const uint32_t bin9_12 = h[4], bin13_16 = h[5], bin17_24 = h[6], bin25_32 = h[7];
            const uint32_t bin33_512 = h[8], bin513_1024 = h[9], bin_big = h[10];
            const uint32_t max_len = h[SB_SPLIT_NBINS];

            uint32_t split_base[SB_SPLIT_NBINS];
            uint32_t acc = 0;
            for (int b = 0; b < SB_SPLIT_NBINS; ++b) { split_base[b] = acc; acc += h[b]; }

            uint32_t scratch_stride = 1u;
            while (scratch_stride < max_len) scratch_stride <<= 1;
            if (scratch_stride < 2u) scratch_stride = 2u;
            if (bin_big > 0 &&
                (size_t)bin_big * (size_t)scratch_stride > ctx->sb_scratch_cap_elems) {
                return false;
            }

            static const bool sb_debug = (getenv("STAGEB_DEBUG") != nullptr);
            if (sb_debug) {
                fprintf(stderr,
                    "[SPLITSMALL] segs=%u bins=[%u %u %u %u %u %u %u %u %u %u %u] max_len=%u stride=%u scratch_used=%zu/%zu\n",
                    bucket_total_count, bin_len2, bin3_4, bin5_6, bin7_8, bin9_12, bin13_16, bin17_24, bin25_32,
                    bin33_512, bin513_1024, bin_big, max_len, scratch_stride,
                    (size_t)bin_big * (size_t)scratch_stride, ctx->sb_scratch_cap_elems);
            }

            CUDA_CHECK(cudaMemcpyAsync(ctx->d_sb_split_cursor, split_base, SB_SPLIT_NBINS * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice, st));
            {
                const uint32_t blocks = ((bucket_total_count + 255U) / 256U);
                sb_bin_scatter_desc_split_kernel<<<blocks, 256, 0, st>>>(
                    ctx->d_bucket_descs, bucket_total_count,
                    ctx->d_sb_split_cursor, ctx->d_sb_ids_packed);
            }

            const uint8_t* T = d_sdata_base;
            const uint64_t* descs = ctx->d_bucket_descs;
            const uint32_t* ids = ctx->d_sb_ids_packed;
            uint32_t* SA = ctx->d_vals_out;
            static const int msort_d = (getenv("MSORT_D") ? atoi(getenv("MSORT_D")) : 0);
            static const int msort_d0 = (getenv("MSORT_D0") ? atoi(getenv("MSORT_D0")) : 0);
            static const int msort_d0lo = (getenv("MSORT_D0LO") ? atoi(getenv("MSORT_D0LO")) : 0);
            static const int msort_d0hi = (getenv("MSORT_D0HI") ? atoi(getenv("MSORT_D0HI")) : 0);
            static const int msort_d1lo = (getenv("MSORT_D1LO") ? atoi(getenv("MSORT_D1LO")) : 0);
            static const int msort_d1hi = (getenv("MSORT_D1HI") ? atoi(getenv("MSORT_D1HI")) : 0);
            cudaEvent_t _be[12] = {};
            if (sb_debug) { for (int _i = 0; _i < 12; ++_i) cudaEventCreate(&_be[_i]); cudaEventRecord(_be[0], st); }
            // [PACKDESC] route tiny bins (len 3..32) through the packed LCP-insertion kernel
            // instead of the merge-sort msort kernel. A/B-safe: default (no env) unchanged.
            static const bool packdesc = (getenv("PACKDESC") != nullptr);
#define SB_TINY(ML, CNT, BIDX, DVAL) do { \
    if ((CNT) > 0) { \
        if (packdesc) sb_sort_lcp_packed_desc_kernel<ML><<<((CNT) + 127) / 128, 128, 0, st>>>( \
            T, d_data_len_base, ctx->d_offsets, descs, ids + split_base[BIDX], (CNT), SA); \
        else sb_sort_lcp_msort_desc_kernel<ML><<<((CNT) + 127) / 128, 128, 0, st>>>( \
            T, d_data_len_base, ctx->d_offsets, descs, ids + split_base[BIDX], (CNT), (DVAL), SA); \
    } } while (0)
            if (bin_len2 > 0) {
                sb_sort_len2_desc_kernel<<<(bin_len2 + 127) / 128, 128, 0, st>>>(
                    T, d_data_len_base, ctx->d_offsets, descs, ids + split_base[0], bin_len2, SA);
            }
            if (sb_debug) cudaEventRecord(_be[1], st);
            SB_TINY(4, bin3_4, 1, (msort_d0lo > 0 ? msort_d0lo : (msort_d0 > 0 ? msort_d0 : 2)));
            if (sb_debug) cudaEventRecord(_be[2], st);
            SB_TINY(6, bin5_6, 2, (msort_d0hi > 0 ? msort_d0hi : (msort_d0 > 0 ? msort_d0 : 2)));
            if (sb_debug) cudaEventRecord(_be[3], st);
            SB_TINY(SB_BIN0_MAX, bin7_8, 3, (msort_d0hi > 0 ? msort_d0hi : (msort_d0 > 0 ? msort_d0 : 2)));
            if (sb_debug) cudaEventRecord(_be[4], st);
            SB_TINY(12, bin9_12, 4, (msort_d1lo > 0 ? msort_d1lo : (msort_d > 0 ? msort_d : 3)));
            if (sb_debug) cudaEventRecord(_be[5], st);
            SB_TINY(16, bin13_16, 5, (msort_d1lo > 0 ? msort_d1lo : (msort_d > 0 ? msort_d : 3)));
            if (sb_debug) cudaEventRecord(_be[6], st);
            SB_TINY(24, bin17_24, 6, (msort_d1hi > 0 ? msort_d1hi : (msort_d > 0 ? msort_d : 3)));
            if (sb_debug) cudaEventRecord(_be[7], st);
            SB_TINY(SB_BIN1_MAX, bin25_32, 7, (msort_d1hi > 0 ? msort_d1hi : (msort_d > 0 ? msort_d : 3)));
#undef SB_TINY
            if (sb_debug) cudaEventRecord(_be[8], st);
            if (bin33_512 > 0) {
                sb_sort_medium_binned_desc_kernel<128, SB_BIN2_MAX><<<bin33_512, 128, 0, st>>>(
                    T, d_data_len_base, ctx->d_offsets, descs, ids + split_base[8], bin33_512, SA);
            }
            if (sb_debug) cudaEventRecord(_be[9], st);
            if (bin513_1024 > 0) {
                sb_sort_medium_binned_desc_kernel<256, SB_BIN3_MAX><<<bin513_1024, 256, 0, st>>>(
                    T, d_data_len_base, ctx->d_offsets, descs, ids + split_base[9], bin513_1024, SA);
            }
            if (sb_debug) cudaEventRecord(_be[10], st);
            if (bin_big > 0) {
                sb_sort_large_desc_kernel<256><<<bin_big, 256, 0, st>>>(
                    T, d_data_len_base, ctx->d_offsets, descs,
                    ids + split_base[10], bin_big, scratch_stride, ctx->d_sb_scratch, SA);
            }
            if (sb_debug) {
                cudaEventRecord(_be[11], st);
                cudaEventSynchronize(_be[11]);
                float _t[11];
                for (int _i = 0; _i < 11; ++_i) cudaEventElapsedTime(&_t[_i], _be[_i], _be[_i + 1]);
                fprintf(stderr,
                        "[SPLITSMALL-T] len2=%.1f len3_4=%.1f len5_6=%.1f len7_8=%.1f len9_12=%.1f len13_16=%.1f len17_24=%.1f len25_32=%.1f bin2=%.1f bin3=%.1f bin4=%.1f ms  (segs %u/%u/%u/%u/%u/%u/%u/%u/%u/%u/%u)\n",
                        _t[0], _t[1], _t[2], _t[3], _t[4], _t[5], _t[6], _t[7], _t[8], _t[9], _t[10],
                        bin_len2, bin3_4, bin5_6, bin7_8, bin9_12, bin13_16, bin17_24, bin25_32,
                        bin33_512, bin513_1024, bin_big);
                for (int _i = 0; _i < 12; ++_i) cudaEventDestroy(_be[_i]);
            }
            return true;
        }

        CUDA_CHECK(cudaMemsetAsync(ctx->d_sb_max_len, 0, sizeof(uint32_t), st));
        CUDA_CHECK(cudaMemsetAsync(ctx->d_sb_bin_counts, 0, SB_NBINS * sizeof(uint32_t), st));
        {
            const uint32_t blocks = ((bucket_total_count + 255U) / 256U);
            sb_bin_count_desc_kernel<<<blocks, 256, 0, st>>>(
                ctx->d_bucket_descs, bucket_total_count,
                ctx->d_sb_bin_counts, ctx->d_sb_max_len);
        }

        uint32_t* h = ctx->h_sb_meta;
        CUDA_CHECK(cudaMemcpyAsync(h, ctx->d_sb_bin_counts, SB_NBINS * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(h + SB_NBINS, ctx->d_sb_max_len, sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
        const uint32_t bin0 = h[0], bin1 = h[1], bin2 = h[2], bin3 = h[3], bin4 = h[4];
        const uint32_t max_len = h[SB_NBINS];

        static const bool lenhist = (getenv("LENHIST") != nullptr);
        if (lenhist) {
            uint32_t* d_len_hist = nullptr;
            uint32_t h_len_hist[SB_LEN_HIST_SLOTS] = {};
            CUDA_CHECK(cudaMalloc(&d_len_hist, SB_LEN_HIST_SLOTS * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemsetAsync(d_len_hist, 0, SB_LEN_HIST_SLOTS * sizeof(uint32_t), st));
            const uint32_t blocks = ((bucket_total_count + 255U) / 256U);
            sb_len_hist_desc_kernel<<<blocks, 256, 0, st>>>(
                ctx->d_bucket_descs, bucket_total_count, d_len_hist);
            CUDA_CHECK(cudaMemcpyAsync(h_len_hist, d_len_hist,
                                       SB_LEN_HIST_SLOTS * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            CUDA_CHECK(cudaFree(d_len_hist));
            uint32_t sum_2_8 = 0, sum_9_16 = 0, sum_17_32 = 0;
            for (int i = 2; i <= 8; ++i) sum_2_8 += h_len_hist[i];
            for (int i = 9; i <= 16; ++i) sum_9_16 += h_len_hist[i];
            for (int i = 17; i <= 32; ++i) sum_17_32 += h_len_hist[i];
            fprintf(stderr,
                    "[LENHIST] len2_8=%u exact2_8=[%u %u %u %u %u %u %u] len9_16=%u len17_32=%u gt32=%u exact9_24=[%u %u %u %u %u %u %u %u | %u %u %u %u %u %u %u %u]\n",
                    sum_2_8,
                    h_len_hist[2], h_len_hist[3], h_len_hist[4], h_len_hist[5],
                    h_len_hist[6], h_len_hist[7], h_len_hist[8],
                    sum_9_16, sum_17_32, h_len_hist[33],
                    h_len_hist[9], h_len_hist[10], h_len_hist[11], h_len_hist[12],
                    h_len_hist[13], h_len_hist[14], h_len_hist[15], h_len_hist[16],
                    h_len_hist[17], h_len_hist[18], h_len_hist[19], h_len_hist[20],
                    h_len_hist[21], h_len_hist[22], h_len_hist[23], h_len_hist[24]);
        }

        uint32_t bin_base[SB_NBINS];
        uint32_t acc = 0;
        for (int b = 0; b < SB_NBINS; ++b) { bin_base[b] = acc; acc += h[b]; }

        uint32_t scratch_stride = 1u;
        while (scratch_stride < max_len) scratch_stride <<= 1;
        if (scratch_stride < 2u) scratch_stride = 2u;
        if (bin4 > 0 &&
            (size_t)bin4 * (size_t)scratch_stride > ctx->sb_scratch_cap_elems) {
            return false;
        }

        static const bool sb_debug = (getenv("STAGEB_DEBUG") != nullptr);
        if (sb_debug) {
            fprintf(stderr,
                "[DESCDIRECT] segs=%u bins=[%u %u %u %u %u] max_len=%u stride=%u scratch_used=%zu/%zu\n",
                bucket_total_count, bin0, bin1, bin2, bin3, bin4, max_len, scratch_stride,
                (size_t)bin4 * (size_t)scratch_stride, ctx->sb_scratch_cap_elems);
        }

        CUDA_CHECK(cudaMemcpyAsync(ctx->d_sb_bin_cursor, bin_base, SB_NBINS * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, st));
        {
            const uint32_t blocks = ((bucket_total_count + 255U) / 256U);
            sb_bin_scatter_desc_kernel<<<blocks, 256, 0, st>>>(
                ctx->d_bucket_descs, bucket_total_count, ctx->d_sb_bin_cursor, ctx->d_sb_ids_packed);
        }

        const uint8_t* T = d_sdata_base;
        const uint64_t* descs = ctx->d_bucket_descs;
        const uint32_t* ids = ctx->d_sb_ids_packed;
        uint32_t* SA = ctx->d_vals_out;
        static const int msort_d = (getenv("MSORT_D") ? atoi(getenv("MSORT_D")) : 0);
        static const int msort_d0 = (getenv("MSORT_D0") ? atoi(getenv("MSORT_D0")) : 0);
        cudaEvent_t _be[6] = {};
        if (sb_debug) { for (int _i = 0; _i < 6; ++_i) cudaEventCreate(&_be[_i]); cudaEventRecord(_be[0], st); }
        if (bin0 > 0) {
            sb_sort_lcp_msort_desc_kernel<SB_BIN0_MAX><<<(bin0 + 127) / 128, 128, 0, st>>>(
                T, d_data_len_base, ctx->d_offsets, descs, ids + bin_base[0], bin0,
                (msort_d0 > 0 ? msort_d0 : 2), SA);
        }
        if (sb_debug) cudaEventRecord(_be[1], st);
        if (bin1 > 0) {
            sb_sort_lcp_msort_desc_kernel<SB_BIN1_MAX><<<(bin1 + 127) / 128, 128, 0, st>>>(
                T, d_data_len_base, ctx->d_offsets, descs, ids + bin_base[1], bin1,
                (msort_d > 0 ? msort_d : 3), SA);
        }
        if (sb_debug) cudaEventRecord(_be[2], st);
        if (bin2 > 0) {
            sb_sort_medium_binned_desc_kernel<128, SB_BIN2_MAX><<<bin2, 128, 0, st>>>(
                T, d_data_len_base, ctx->d_offsets, descs, ids + bin_base[2], bin2, SA);
        }
        if (sb_debug) cudaEventRecord(_be[3], st);
        if (bin3 > 0) {
            sb_sort_medium_binned_desc_kernel<256, SB_BIN3_MAX><<<bin3, 256, 0, st>>>(
                T, d_data_len_base, ctx->d_offsets, descs, ids + bin_base[3], bin3, SA);
        }
        if (sb_debug) cudaEventRecord(_be[4], st);
        if (bin4 > 0) {
            sb_sort_large_desc_kernel<256><<<bin4, 256, 0, st>>>(
                T, d_data_len_base, ctx->d_offsets, descs,
                ids + bin_base[4], bin4, scratch_stride, ctx->d_sb_scratch, SA);
        }
        if (sb_debug) {
            cudaEventRecord(_be[5], st);
            cudaEventSynchronize(_be[5]);
            float _t[5];
            for (int _i = 0; _i < 5; ++_i) cudaEventElapsedTime(&_t[_i], _be[_i], _be[_i + 1]);
            fprintf(stderr,
                    "[DESCDIRECT-T] bin0=%.1f bin1=%.1f bin2=%.1f bin3=%.1f bin4=%.1f ms  (segs %u/%u/%u/%u/%u)\n",
                    _t[0], _t[1], _t[2], _t[3], _t[4], bin0, bin1, bin2, bin3, bin4);
            for (int _i = 0; _i < 6; ++_i) cudaEventDestroy(_be[_i]);
        }
        return true;
    }

    // 1) Unpack descriptors into the 5 seg arrays; track max seg length.
    CUDA_CHECK(cudaMemsetAsync(ctx->d_sb_max_len, 0, sizeof(uint32_t), st));
    {
        const uint32_t blocks = ((bucket_total_count + 255U) / 256U)  /* gridDim.x max is 2^31-1, NOT 65535; capping silently dropped segments >16.78M (batch>~1536) -> corrupt SA */;
        unpack_bucket_descs_kernel<<<blocks, 256, 0, st>>>(
            d_data_len_base, ctx->d_offsets, ctx->d_bucket_descs, bucket_total_count,
            ctx->d_sb_seg_off, ctx->d_sb_seg_len, ctx->d_sb_seg_depth,
            ctx->d_sb_seg_n, ctx->d_sb_seg_Tbase, ctx->d_sb_max_len);
    }

    // 2) Bin-count histogram.
    CUDA_CHECK(cudaMemsetAsync(ctx->d_sb_bin_counts, 0, SB_NBINS * sizeof(uint32_t), st));
    {
        const uint32_t blocks = ((bucket_total_count + 255U) / 256U)  /* gridDim.x max is 2^31-1, NOT 65535; capping silently dropped segments >16.78M (batch>~1536) -> corrupt SA */;
        sb_bin_count_kernel<<<blocks, 256, 0, st>>>(
            ctx->d_sb_seg_len, bucket_total_count, ctx->d_sb_bin_counts);
    }

    // 3) Pull counts + max_len to host (one sync), compute exclusive prefix.
    uint32_t* h = ctx->h_sb_meta;
    CUDA_CHECK(cudaMemcpyAsync(h, ctx->d_sb_bin_counts, SB_NBINS * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaMemcpyAsync(h + SB_NBINS, ctx->d_sb_max_len, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    const uint32_t bin0 = h[0], bin1 = h[1], bin2 = h[2], bin3 = h[3], bin4 = h[4];
    const uint32_t max_len = h[SB_NBINS];

    uint32_t bin_base[SB_NBINS];
    uint32_t acc = 0;
    for (int b = 0; b < SB_NBINS; ++b) { bin_base[b] = acc; acc += h[b]; }

    // bin4 scratch sizing: stride = next pow2 of max segment length.
    uint32_t scratch_stride = 1u;
    while (scratch_stride < max_len) scratch_stride <<= 1;
    if (scratch_stride < 2u) scratch_stride = 2u;
    if (bin4 > 0 &&
        (size_t)bin4 * (size_t)scratch_stride > ctx->sb_scratch_cap_elems) {
        return false;  // bin4 scratch would overflow -> fallback
    }

    static const bool sb_debug = (getenv("STAGEB_DEBUG") != nullptr);
    if (sb_debug) {
        fprintf(stderr,
            "[STAGEB] segs=%u bins=[%u %u %u %u %u] max_len=%u stride=%u scratch_used=%zu/%zu\n",
            bucket_total_count, bin0, bin1, bin2, bin3, bin4, max_len, scratch_stride,
            (size_t)bin4 * (size_t)scratch_stride, ctx->sb_scratch_cap_elems);
    }

    // 4) Scatter segment ids into per-bin packed lists.
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_sb_bin_cursor, bin_base, SB_NBINS * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, st));
    {
        const uint32_t blocks = ((bucket_total_count + 255U) / 256U)  /* gridDim.x max is 2^31-1, NOT 65535; capping silently dropped segments >16.78M (batch>~1536) -> corrupt SA */;
        sb_bin_scatter_kernel<<<blocks, 256, 0, st>>>(
            ctx->d_sb_seg_len, bucket_total_count, ctx->d_sb_bin_cursor, ctx->d_sb_ids_packed);
    }

    // 5) Per-bin sort kernels (each over its packed sub-array of ids).
    const uint8_t* T = d_sdata_base;
    uint32_t* SA = ctx->d_vals_out;
    auto seg_off = ctx->d_sb_seg_off; auto seg_len = ctx->d_sb_seg_len;
    auto seg_depth = ctx->d_sb_seg_depth; auto seg_n = ctx->d_sb_seg_n;
    auto seg_Tbase = ctx->d_sb_seg_Tbase;
    uint32_t* ids = ctx->d_sb_ids_packed;

    // [HASHORDER] sort each bin's ids ascending (== hash order) so each block's
    // segments share one hash -> per-block L1-resident text -> faster comparator reads.
    static const bool hashorder = (getenv("HASHORDER") != nullptr);
    if (hashorder) {
        static void* ho_temp = nullptr; static size_t ho_temp_sz = 0;
        uint32_t* ho_out = ctx->d_vals_in;   // scratch (>= sb_seg_cap), unused in binned path
        if (ho_temp == nullptr) {
            size_t need = 0;
            cub::DeviceRadixSort::SortKeys(nullptr, need, ids, ho_out, (int)ctx->sb_seg_cap, 0, 32, st);
            cudaMalloc(&ho_temp, need); ho_temp_sz = need;
        }
        for (int b = 0; b < SB_NBINS; ++b) {
            uint32_t cnt = h[b]; if (cnt < 2) continue;
            size_t tsz = ho_temp_sz;
            cub::DeviceRadixSort::SortKeys(ho_temp, tsz, ids + bin_base[b], ho_out + bin_base[b], (int)cnt, 0, 32, st);
            cudaMemcpyAsync(ids + bin_base[b], ho_out + bin_base[b], (size_t)cnt * sizeof(uint32_t), cudaMemcpyDeviceToDevice, st);
        }
    }

    // ---------------------------------------------------------------------
    // OCTET parity+bench harness (env OCTET). Replaces bin0/bin1 thread-per-
    // segment insertion with the 8-thread cooperative shared-mem octet sort.
    // Oracle = FULL proven byte path (all bins). Test = octet bin0/bin1 +
    // identical bin2/3/4 kernels. mismatch MUST be 0. Default path (env
    // unset) falls through and is byte-identical.
    // ---------------------------------------------------------------------
    {
        static const bool octet_on = (getenv("OCTET") != nullptr);
        static const bool octet_floor = (getenv("OCTET_FLOOR") != nullptr);
        if (octet_on) {
            uint32_t* snap   = reinterpret_cast<uint32_t*>(ctx->d_keys_out);
            uint32_t* oracle = reinterpret_cast<uint32_t*>(ctx->d_keys_out) + (size_t)ctx->total_sa_elements;
            const size_t Noc = (size_t)total_elements;
            // 1) snapshot pristine refine input
            cudaMemcpyAsync(snap, SA, Noc * sizeof(uint32_t), cudaMemcpyDeviceToDevice, st);
            // 2) FULL proven byte path -> oracle (all bins)
            if (bin0 > 0)
                sb_sort_tiny_packed_kernel<SB_BIN0_MAX><<<(bin0+127)/128,128,0,st>>>(
                    T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[0], bin0, SA);
            if (bin1 > 0)
                sb_sort_tiny_packed_kernel<SB_BIN1_MAX><<<(bin1+127)/128,128,0,st>>>(
                    T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[1], bin1, SA);
            if (bin2 > 0)
                sb_sort_medium_binned_kernel<128, SB_BIN2_MAX><<<bin2,128,0,st>>>(
                    T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[2], bin2, SA);
            if (bin3 > 0)
                sb_sort_medium_binned_kernel<256, SB_BIN3_MAX><<<bin3,256,0,st>>>(
                    T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[3], bin3, SA);
            if (bin4 > 0)
                sb_sort_large_kernel<256><<<bin4,256,0,st>>>(
                    T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[4], bin4, scratch_stride, ctx->d_sb_scratch, SA);
            cudaMemcpyAsync(oracle, SA, Noc * sizeof(uint32_t), cudaMemcpyDeviceToDevice, st);
            // 3) restore pristine input
            cudaMemcpyAsync(SA, snap, Noc * sizeof(uint32_t), cudaMemcpyDeviceToDevice, st);
            unsigned long long *d_mm=nullptr, *d_pc=nullptr;
            cudaMalloc(&d_mm,sizeof(unsigned long long));
            cudaMalloc(&d_pc,sizeof(unsigned long long));
            cudaMemsetAsync(d_mm,0,sizeof(unsigned long long),st);
            cudaMemsetAsync(d_pc,0,sizeof(unsigned long long),st);
            rd_compare_sa_kernel<<<active_batch,256,0,st>>>(snap, oracle, d_data_len_base, ctx->d_offsets, active_batch, d_pc);
            // 4) octet path for bin0/bin1, timed SEPARATELY
            cudaEvent_t e_a=nullptr,e_b=nullptr,e_c=nullptr;
            cudaEventCreate(&e_a); cudaEventCreate(&e_b); cudaEventCreate(&e_c);
            cudaEventRecord(e_a, st);
            if (bin0 > 0) {
                if (octet_floor)
                    sb_sort_octet_kernel<SB_BIN0_MAX,256,true><<<(bin0+31)/32,256,0,st>>>(
                        T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[0], bin0, SA);
                else
                    sb_sort_octet_kernel<SB_BIN0_MAX,256,false><<<(bin0+31)/32,256,0,st>>>(
                        T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[0], bin0, SA);
            }
            cudaEventRecord(e_b, st);
            if (bin1 > 0) {
                if (octet_floor)
                    sb_sort_octet_kernel<SB_BIN1_MAX,256,true><<<(bin1+31)/32,256,0,st>>>(
                        T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[1], bin1, SA);
                else
                    sb_sort_octet_kernel<SB_BIN1_MAX,256,false><<<(bin1+31)/32,256,0,st>>>(
                        T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[1], bin1, SA);
            }
            cudaEventRecord(e_c, st);
            cudaError_t oc_sync = cudaEventSynchronize(e_c);
            float ms_bin0=0.f, ms_bin1=0.f;
            cudaEventElapsedTime(&ms_bin0, e_a, e_b);
            cudaEventElapsedTime(&ms_bin1, e_b, e_c);
            // remaining bins (2/3/4) identical to oracle path
            if (bin2 > 0)
                sb_sort_medium_binned_kernel<128, SB_BIN2_MAX><<<bin2,128,0,st>>>(
                    T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[2], bin2, SA);
            if (bin3 > 0)
                sb_sort_medium_binned_kernel<256, SB_BIN3_MAX><<<bin3,256,0,st>>>(
                    T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[3], bin3, SA);
            if (bin4 > 0)
                sb_sort_large_kernel<256><<<bin4,256,0,st>>>(
                    T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[4], bin4, scratch_stride, ctx->d_sb_scratch, SA);
            // 5) compare octet result vs oracle -> must be 0
            rd_compare_sa_kernel<<<active_batch,256,0,st>>>(SA, oracle, d_data_len_base, ctx->d_offsets, active_batch, d_mm);
            cudaStreamSynchronize(st);
            unsigned long long h_mm=0,h_pc=0;
            cudaMemcpy(&h_mm,d_mm,sizeof(unsigned long long),cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_pc,d_pc,sizeof(unsigned long long),cudaMemcpyDeviceToHost);
            fprintf(stderr,"[OCTET]%s octet_bin0_ms=%.2f octet_bin1_ms=%.2f mismatch=%llu poscontrol=%llu sync=%s (bin0=%u bin1=%u batch=%d)%c", (octet_floor?"[FLOOR/perf-only]":""),
                    ms_bin0, ms_bin1, h_mm, h_pc, cudaGetErrorString(oc_sync), bin0, bin1, active_batch, 10);
            cudaFree(d_mm); cudaFree(d_pc);
            cudaEventDestroy(e_a); cudaEventDestroy(e_b); cudaEventDestroy(e_c);
            return true;
        }
    }



    {
        static const bool zlcp_on = (getenv("ZLCP") != nullptr);
        if (zlcp_on) {
            uint32_t* snap   = reinterpret_cast<uint32_t*>(ctx->d_keys_out);
            uint32_t* oracle = reinterpret_cast<uint32_t*>(ctx->d_keys_out) + (size_t)ctx->total_sa_elements;
            const size_t Nzl = (size_t)total_elements;
            cudaMemcpyAsync(snap, SA, Nzl * sizeof(uint32_t), cudaMemcpyDeviceToDevice, st);
            // FULL proven byte path -> oracle (all bins)
            if (bin0 > 0)
                sb_sort_tiny_packed_kernel<SB_BIN0_MAX><<<(bin0+127)/128,128,0,st>>>(
                    T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[0], bin0, SA);
            if (bin1 > 0)
                sb_sort_tiny_packed_kernel<SB_BIN1_MAX><<<(bin1+127)/128,128,0,st>>>(
                    T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[1], bin1, SA);
            if (bin2 > 0)
                sb_sort_medium_binned_kernel<128, SB_BIN2_MAX><<<bin2,128,0,st>>>(
                    T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[2], bin2, SA);
            if (bin3 > 0)
                sb_sort_medium_binned_kernel<256, SB_BIN3_MAX><<<bin3,256,0,st>>>(
                    T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[3], bin3, SA);
            if (bin4 > 0)
                sb_sort_large_kernel<256><<<bin4,256,0,st>>>(
                    T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[4], bin4, scratch_stride, ctx->d_sb_scratch, SA);
            cudaMemcpyAsync(oracle, SA, Nzl * sizeof(uint32_t), cudaMemcpyDeviceToDevice, st);
            cudaMemcpyAsync(SA, snap, Nzl * sizeof(uint32_t), cudaMemcpyDeviceToDevice, st);
            unsigned long long *d_mm=nullptr, *d_pc=nullptr;
            cudaMalloc(&d_mm,sizeof(unsigned long long));
            cudaMalloc(&d_pc,sizeof(unsigned long long));
            cudaMemsetAsync(d_mm,0,sizeof(unsigned long long),st);
            cudaMemsetAsync(d_pc,0,sizeof(unsigned long long),st);
            rd_compare_sa_kernel<<<active_batch,256,0,st>>>(snap, oracle, d_data_len_base, ctx->d_offsets, active_batch, d_pc);
            const char* z_b1 = getenv("ZLCP_BIN1");  // also do bin1 with LCP when set
            const char* z_half = getenv("ZLCP_HALF"); // PERF-ONLY: bin1 2-way half-sort (no merge), expect mismatch>0
            cudaEvent_t z_a=nullptr,z_b=nullptr,z_c=nullptr;
            cudaEventCreate(&z_a); cudaEventCreate(&z_b); cudaEventCreate(&z_c);
            cudaEventRecord(z_a, st);
            const char* z_msd0 = getenv("MSORT_D0");
            if (bin0 > 0) {
                if (z_msd0)
                    sb_sort_lcp_msort_kernel<SB_BIN0_MAX><<<(bin0+127)/128,128,0,st>>>(
                        T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[0], bin0, atoi(z_msd0), SA);
                else
                    sb_sort_lcp_packed_kernel<SB_BIN0_MAX><<<(bin0+127)/128,128,0,st>>>(
                        T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[0], bin0, SA);
            }
            cudaEventRecord(z_b, st);
            const char* z_msd = getenv("MSORT_D");
            if (bin1 > 0) {
                if (z_msd)
                    sb_sort_lcp_msort_kernel<SB_BIN1_MAX><<<(bin1+127)/128,128,0,st>>>(
                        T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[1], bin1, atoi(z_msd), SA);
                else if (z_half)
                    sb_sort_lcp_packed_kernel<SB_BIN1_MAX,true><<<(bin1+127)/128,128,0,st>>>(
                        T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[1], bin1, SA);
                else if (z_b1)
                    sb_sort_lcp_packed_kernel<SB_BIN1_MAX><<<(bin1+127)/128,128,0,st>>>(
                        T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[1], bin1, SA);
                else
                    sb_sort_tiny_packed_kernel<SB_BIN1_MAX><<<(bin1+127)/128,128,0,st>>>(
                        T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[1], bin1, SA);
            }
            cudaEventRecord(z_c, st);
            cudaError_t zl_sync = cudaEventSynchronize(z_c);
            float zms_bin0=0.f, zms_bin1=0.f;
            cudaEventElapsedTime(&zms_bin0, z_a, z_b);
            cudaEventElapsedTime(&zms_bin1, z_b, z_c);
            if (bin2 > 0)
                sb_sort_medium_binned_kernel<128, SB_BIN2_MAX><<<bin2,128,0,st>>>(
                    T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[2], bin2, SA);
            if (bin3 > 0)
                sb_sort_medium_binned_kernel<256, SB_BIN3_MAX><<<bin3,256,0,st>>>(
                    T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[3], bin3, SA);
            if (bin4 > 0)
                sb_sort_large_kernel<256><<<bin4,256,0,st>>>(
                    T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids+bin_base[4], bin4, scratch_stride, ctx->d_sb_scratch, SA);
            rd_compare_sa_kernel<<<active_batch,256,0,st>>>(SA, oracle, d_data_len_base, ctx->d_offsets, active_batch, d_mm);
            cudaStreamSynchronize(st);
            unsigned long long h_mm=0,h_pc=0;
            cudaMemcpy(&h_mm,d_mm,sizeof(unsigned long long),cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_pc,d_pc,sizeof(unsigned long long),cudaMemcpyDeviceToHost);
            fprintf(stderr,"[ZLCP]%s zlcp_bin0_ms=%.2f bin1_ms=%.2f mismatch=%llu poscontrol=%llu sync=%s (bin0=%u bin1=%u batch=%d)%c", (z_b1?"[bin1=LCP]":"[bin1=insertion]"), zms_bin0, zms_bin1, h_mm, h_pc, cudaGetErrorString(zl_sync), bin0, bin1, active_batch, 10);
            cudaFree(d_mm); cudaFree(d_pc);
            cudaEventDestroy(z_a); cudaEventDestroy(z_b); cudaEventDestroy(z_c);
            return true;
        }
    }

    cudaEvent_t _be[6] = {};
    if (sb_debug) { for (int _i = 0; _i < 6; ++_i) cudaEventCreate(&_be[_i]); cudaEventRecord(_be[0], st); }
    static const bool lcp_default_off = (getenv("LCP_DEFAULT_OFF") != nullptr);
    static const int  msort_d = (getenv("MSORT_D") ? atoi(getenv("MSORT_D")) : 0);
    static const int  msort_d0 = (getenv("MSORT_D0") ? atoi(getenv("MSORT_D0")) : 0);
    if (bin0 > 0) {  // len in [2,8]   LCP-aware insertion (byte-identical to byte-walk)
        if (lcp_default_off)
            sb_sort_tiny_packed_kernel<SB_BIN0_MAX><<<(bin0 + 127) / 128, 128, 0, st>>>(
                T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids + bin_base[0], bin0, SA);
        else
            sb_sort_lcp_msort_kernel<SB_BIN0_MAX><<<(bin0 + 127) / 128, 128, 0, st>>>(
                T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids + bin_base[0], bin0,
                (msort_d0 > 0 ? msort_d0 : 2), SA);
    }
    if (sb_debug) cudaEventRecord(_be[1], st);
    if (bin1 > 0) {  // len in [9,32]  LCP-aware insertion (byte-identical to byte-walk)
        static const bool pack_bin1 = (getenv("PACK_BIN1") != nullptr);  // 2-array insertion vs 4-array msort (local-mem test)
        static const bool medium_bin1 = (getenv("MEDIUM_BIN1") != nullptr);  // route bin1 through SHARED-mem block-cooperative sort (astronv CUFTT style) instead of per-thread local-array msort
        if (lcp_default_off)
            sb_sort_tiny_packed_kernel<SB_BIN1_MAX><<<(bin1 + 127) / 128, 128, 0, st>>>(
                T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids + bin_base[1], bin1, SA);
        else if (medium_bin1)
            sb_sort_medium_binned_kernel<32, SB_BIN1_MAX><<<bin1, 32, 0, st>>>(
                T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids + bin_base[1], bin1, SA);
        else if (pack_bin1)
            sb_sort_lcp_packed_kernel<SB_BIN1_MAX><<<(bin1 + 127) / 128, 128, 0, st>>>(
                T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids + bin_base[1], bin1, SA);
        else
            sb_sort_lcp_msort_kernel<SB_BIN1_MAX><<<(bin1 + 127) / 128, 128, 0, st>>>(
                T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase, ids + bin_base[1], bin1,
                (msort_d > 0 ? msort_d : 3), SA);
    }
    if (sb_debug) cudaEventRecord(_be[2], st);
    if (bin2 > 0) {  // len in [33,512]  block bitonic, SH_MAX=512
        sb_sort_medium_binned_kernel<128, SB_BIN2_MAX><<<bin2, 128, 0, st>>>(
            T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase,
            ids + bin_base[2], bin2, SA);
    }
    if (sb_debug) cudaEventRecord(_be[3], st);
    if (bin3 > 0) {  // len in [513,1024]  block bitonic, SH_MAX=1024
        sb_sort_medium_binned_kernel<256, SB_BIN3_MAX><<<bin3, 256, 0, st>>>(
            T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase,
            ids + bin_base[3], bin3, SA);
    }
    if (sb_debug) cudaEventRecord(_be[4], st);
    if (bin4 > 0) {  // len > 1024  global-scratch bitonic
        sb_sort_large_kernel<256><<<bin4, 256, 0, st>>>(
            T, seg_off, seg_len, seg_depth, seg_n, seg_Tbase,
            ids + bin_base[4], bin4, scratch_stride, ctx->d_sb_scratch, SA);
    }
    if (sb_debug) {
        cudaEventRecord(_be[5], st);
        cudaEventSynchronize(_be[5]);
        float _t[5];
        for (int _i = 0; _i < 5; ++_i) cudaEventElapsedTime(&_t[_i], _be[_i], _be[_i + 1]);
        fprintf(stderr, "[STAGEB-T] bin0=%.1f bin1=%.1f bin2=%.1f bin3=%.1f bin4=%.1f ms  (segs %u/%u/%u/%u/%u)\n",
                _t[0], _t[1], _t[2], _t[3], _t[4], bin0, bin1, bin2, bin3, bin4);
        for (int _i = 0; _i < 6; ++_i) cudaEventDestroy(_be[_i]);
    }
    return true;
}

// ---------------------------------------------------------------------
// SHRINK-CURVE DIAGNOSTIC (gate spike). Given the FINAL correct SA, a suffix
// is "still tied at prefix depth L" iff it shares >= L leading bytes with a
// sorted neighbor, i.e. max(leftLCP, rightLCP) >= L. This is the EXACT residual
// ambiguous fraction prefix-doubling would face at depth L. One-shot, env-gated.
// ---------------------------------------------------------------------
#define SC_NDEPTH 9
__global__ void shrink_curve_kernel(const uint8_t* __restrict__ d_sdata,
                                    const int* __restrict__ d_offsets,
                                    const uint32_t* __restrict__ d_sa,
                                    int active_batch,
                                    unsigned long long* __restrict__ d_ambig,
                                    unsigned long long* __restrict__ d_total) {
    const int depths[SC_NDEPTH] = {1, 2, 3, 4, 7, 14, 28, 56, 96};
    const int CAP = 112;
    int hash_id = blockIdx.x;
    if (hash_id >= active_batch) return;
    static constexpr int HOST_SDATA_STRIDE = 72 * 1024;
    const uint8_t* T = d_sdata + (size_t)hash_id * HOST_SDATA_STRIDE;
    int base = d_offsets[hash_id];
    int n = d_offsets[hash_id + 1] - base;
    if (n <= 0) return;
    for (int p = threadIdx.x; p < n; p += blockDim.x) {
        uint32_t sp = d_sa[base + p];
        int rl = 0, ll = 0;
        if (p + 1 < n) {
            uint32_t sq = d_sa[base + p + 1];
            int k = 0;
            while (k < CAP) {
                int ia = sp + k, ib = sq + k;
                if (ia >= n || ib >= n) break;
                if (T[ia] != T[ib]) break;
                k++;
            }
            rl = k;
        }
        if (p > 0) {
            uint32_t sq = d_sa[base + p - 1];
            int k = 0;
            while (k < CAP) {
                int ia = sp + k, ib = sq + k;
                if (ia >= n || ib >= n) break;
                if (T[ia] != T[ib]) break;
                k++;
            }
            ll = k;
        }
        int m = rl > ll ? rl : ll;
        atomicAdd(d_total, 1ULL);
        #pragma unroll
        for (int di = 0; di < SC_NDEPTH; di++)
            if (m >= depths[di]) atomicAdd(&d_ambig[di], 1ULL);
    }
}

__global__ void rd_assign_ranks_parallel(const uint64_t* __restrict__ keys, const uint32_t* __restrict__ vals, const int32_t* __restrict__ dlen, const int* __restrict__ doff, uint32_t* __restrict__ ranks_out, uint32_t* __restrict__ uniq_out, int batch){
    int h=blockIdx.x; if(h>=batch) return;
    int n=dlen[h]; if(n<=0||n>70911){ if(threadIdx.x==0) uniq_out[h]=(n<=0?0u:1u); return; }
    int base=doff[h]; int T=blockDim.x; int chunk=(n+T-1)/T;
    int lo=threadIdx.x*chunk; if(lo>n) lo=n; int hi=lo+chunk; if(hi>n) hi=n;
    __shared__ uint32_t s_cmax[256]; __shared__ uint32_t s_ex[256]; __shared__ unsigned int s_uniq;
    if(threadIdx.x==0) s_uniq=0u; __syncthreads();
    uint32_t cmax=0u; unsigned int cnt=0u;
    for(int i=lo;i<hi;i++){ uint32_t v; if(i==0){ v=0u; cnt++; } else { v=(keys[base+i]!=keys[base+i-1])?(uint32_t)i:0u; if(v) cnt++; } if(v>cmax) cmax=v; }
    s_cmax[threadIdx.x]=cmax; if(cnt) atomicAdd(&s_uniq,cnt); __syncthreads();
    if(threadIdx.x==0){ uint32_t run=0u; for(int t=0;t<T;t++){ s_ex[t]=run; if(s_cmax[t]>run) run=s_cmax[t]; } uniq_out[h]=s_uniq; }
    __syncthreads();
    uint32_t run=s_ex[threadIdx.x];
    for(int i=lo;i<hi;i++){ uint32_t v; if(i==0) v=0u; else v=(keys[base+i]!=keys[base+i-1])?(uint32_t)i:0u; if(v>run) run=v; ranks_out[base+vals[base+i]]=run+1u; }
}

__global__ void rd_compare_sa_kernel(const uint32_t* __restrict__ d_vals_a, const uint32_t* __restrict__ d_vals_b, const int32_t* __restrict__ d_data_len, const int* __restrict__ d_offsets, int batch_size, unsigned long long* __restrict__ d_mismatch) {
    int hash_id=blockIdx.x; if(hash_id>=batch_size) return;
    int n=d_data_len[hash_id]; if(n<=0||n>70911) return;
    int base=d_offsets[hash_id];
    for(int i=threadIdx.x;i<n;i+=blockDim.x){
        if(d_vals_a[base+i]!=d_vals_b[base+i]) atomicAdd(d_mismatch,1ULL);
    }
}

static bool gpu_build_fast_sa_window(GPUContext* ctx,
                                     const uint8_t* d_sdata_base,
                                     const int32_t* d_data_len_base,
                                     int active_batch,
                                     int& total_elements,
                                     GPUSABuildStats* stats = nullptr) {
    cudaSetDevice(ctx->device_id);
    total_elements = 0;
    uint32_t bucket_total_count = 0;
    if (active_batch <= 0) return true;

    cudaEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr, e3 = nullptr, e4 = nullptr;
    if (stats) {
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventCreate(&e2);
        cudaEventCreate(&e3);
        cudaEventCreate(&e4);
        cudaEventRecord(e0, ctx->stream);
    }

    build_offsets_kernel<<<1, 1, 0, ctx->stream>>>(ctx->d_data_len, ctx->d_offsets, active_batch);
    if (stats) cudaEventRecord(e1, ctx->stream);
    CUDA_CHECK(cudaMemcpyAsync(&total_elements, ctx->d_offsets + active_batch, sizeof(int), cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    if (stats) cudaEventSynchronize(e1);
    if (total_elements <= 0) {
        if (stats) {
            cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3); cudaEventDestroy(e4);
        }
        return true;
    }

    if (stats) {
        CUDA_CHECK(cudaMemsetAsync(ctx->d_bucket_counts, 0, (size_t)active_batch * sizeof(uint32_t), ctx->stream));
        CUDA_CHECK(cudaMemsetAsync(ctx->d_ambiguous_counts, 0, (size_t)active_batch * sizeof(uint32_t), ctx->stream));
        CUDA_CHECK(cudaMemsetAsync(ctx->d_max_bucket_sizes, 0, (size_t)active_batch * sizeof(uint32_t), ctx->stream));
    }
    CUDA_CHECK(cudaMemsetAsync(ctx->d_bucket_total_count, 0, sizeof(uint32_t), ctx->stream));

    // EXP1: single segmented sort by symbols[0..6] (one 63-bit key), then
    // bucket-detect at depth 7 (primary key only, secondary=nullptr) and let the
    // cheap cooperative stage-B refine sort the larger/more-numerous tied groups.
    // [Lever 2: fuse keys-build] init_sort_keys_kernel already emits keys + identity
    // vals; the old build_ordered pass (shift=0, identity src_vals) recomputed the
    // SAME keys from the SAME positions into d_keys_out/d_vals_out. Write the sort
    // input (d_keys_out/d_vals_out) directly in one pass and drop the redundant kernel.
    // [NARROWKEY] default ON: uint32 18-bit keys cut the coarse sort 94.6->50.7ms
    // (1.87x) for +10.5% (6.41->7.09 KH/s), parity-green across all vectors @b3328.
    // Escape hatch NARROWKEY_OFF=1 restores the uint64-key path.
    static const bool narrowkey = (getenv("NARROWKEY_OFF") == nullptr);
    bool sorted_vals_in_dvals_out = false;
    if (narrowkey) {
        // [NARROWKEY] identical 18-bit key VALUE stored in uint32 -> half the
        // per-pass key bandwidth of the uint64 radix sort. Reinterpret the owned
        // uint64 key buffers as uint32 (each total*8 bytes holds >= total u32).
        // Stable radix on identical key bits -> parity-exact vs the uint64 path.
        uint32_t* k_cur = reinterpret_cast<uint32_t*>(ctx->d_keys_out);
        uint32_t* k_alt = reinterpret_cast<uint32_t*>(ctx->d_keys_in);
        init_sort_keys18_u32_kernel<<<dim3(1, active_batch), 256, 0, ctx->stream>>>(
            d_sdata_base, d_data_len_base, ctx->d_offsets, k_cur, ctx->d_vals_in, active_batch, 0);
        if (stats) cudaEventRecord(e2, ctx->stream);
        cub::DoubleBuffer<uint32_t> dk(k_cur, k_alt);
        // Deliberately start values in the opposite buffer from keys: after the
        // same radix pass parity that puts sorted keys in d_keys_in, sorted
        // suffix values land directly in d_vals_out, avoiding a full D2D copy.
        cub::DoubleBuffer<uint32_t> dv(ctx->d_vals_in, ctx->d_vals_out);
        cub::DeviceSegmentedRadixSort::SortPairs(
            ctx->d_sort_temp, ctx->sort_temp_size, dk, dv,
            total_elements, active_batch, ctx->d_offsets, ctx->d_offsets + 1, 0, kSortEndBit, ctx->stream);
        // normalize sorted uint32 keys into reinterpret(d_keys_in) so the
        // descriptor build reads them there; d_keys_out is then free for descs.
        // Also normalize values to d_vals_out only if CUB pass parity changes.
        if (dk.Current() != k_alt) {
            CUDA_CHECK(cudaMemcpyAsync(k_alt, dk.Current(),
                (size_t)total_elements * sizeof(uint32_t), cudaMemcpyDeviceToDevice, ctx->stream));
        }
        if (dv.Current() != ctx->d_vals_out) {
            CUDA_CHECK(cudaMemcpyAsync(ctx->d_vals_out, dv.Current(),
                (size_t)total_elements * sizeof(uint32_t), cudaMemcpyDeviceToDevice, ctx->stream));
        }
        sorted_vals_in_dvals_out = true;
    } else {
    init_sort_keys_kernel<<<dim3(1, active_batch), 256, 0, ctx->stream>>>(
        d_sdata_base, d_data_len_base, ctx->d_offsets, ctx->d_keys_out, ctx->d_vals_out, active_batch, 0);
    if (stats) cudaEventRecord(e2, ctx->stream);
    {
        // [Stage 1: drop CUB temp] DoubleBuffer ping-pongs across the owned in/out
        // buffers -> no ~0.82 MB/hash alternate-buffer temp. Pre-sort input is in
        // d_keys_out/d_vals_out (Current = d_buffers[0]); normalize the sorted result
        // back into d_keys_in/d_vals_in (a DtoD copy only when pass-parity leaves it in
        // the input buffers) so all downstream kernels (build_ordered / bucket_descs /
        // stage-B which reads d_vals_out) are unchanged.
        cub::DoubleBuffer<uint64_t> dk(ctx->d_keys_out, ctx->d_keys_in);
        cub::DoubleBuffer<uint32_t> dv(ctx->d_vals_out, ctx->d_vals_in);
        cub::DeviceSegmentedRadixSort::SortPairs(
            ctx->d_sort_temp, ctx->sort_temp_size, dk, dv,
            total_elements, active_batch, ctx->d_offsets, ctx->d_offsets + 1, 0, kSortEndBit, ctx->stream);
        if (dk.Current() != ctx->d_keys_in) {
            CUDA_CHECK(cudaMemcpyAsync(ctx->d_keys_in, dk.Current(),
                (size_t)total_elements * sizeof(uint64_t), cudaMemcpyDeviceToDevice, ctx->stream));
            CUDA_CHECK(cudaMemcpyAsync(ctx->d_vals_in, dv.Current(),
                (size_t)total_elements * sizeof(uint32_t), cudaMemcpyDeviceToDevice, ctx->stream));
        }
    }
    }
    if (stats) cudaEventRecord(e3, ctx->stream);

    static const bool _ov_dbg = (getenv("STAGEB_DEBUG") != nullptr);
    cudaEvent_t _ov[4] = {};
    if (_ov_dbg) { for (int _i=0;_i<4;++_i) cudaEventCreate(&_ov[_i]); cudaEventRecord(_ov[0], ctx->stream); }
    // Post-sort, d_vals_in already holds the sorted suffix array (SA) and
    // d_keys_in holds the sorted prefix keys consumed by the descriptor build.
    // The old build_ordered_shifted_keys_kernel here re-gathered T to rebuild
    // keys into d_keys_out (DEAD: descriptors read d_keys_in) and copied the
    // suffix into d_vals_out verbatim (d_vals_out[i] = d_vals_in[i]). Replace
    // the ~1.8GB key-rebuild + scattered gather with a plain D2D vals copy.
    if (!sorted_vals_in_dvals_out) {
        CUDA_CHECK(cudaMemcpyAsync(ctx->d_vals_out, ctx->d_vals_in,
            (size_t)total_elements * sizeof(uint32_t), cudaMemcpyDeviceToDevice, ctx->stream));
    }
    if (_ov_dbg) cudaEventRecord(_ov[1], ctx->stream);
    if (narrowkey) {
        build_bucket_descriptors_kernel<uint32_t><<<active_batch, 256, 0, ctx->stream>>>(
            reinterpret_cast<const uint32_t*>(ctx->d_keys_in), (const uint32_t*)nullptr, d_data_len_base, ctx->d_offsets,
            ctx->d_bucket_descs, ctx->d_bucket_total_count,
            stats ? ctx->d_bucket_counts : nullptr,
            stats ? ctx->d_ambiguous_counts : nullptr,
            stats ? ctx->d_max_bucket_sizes : nullptr,
            (uint32_t)(kInitialPrefixSymbols),
            active_batch);
    } else {
        build_bucket_descriptors_kernel<uint64_t><<<active_batch, 256, 0, ctx->stream>>>(
            ctx->d_keys_in, (const uint64_t*)nullptr, d_data_len_base, ctx->d_offsets,
            ctx->d_bucket_descs, ctx->d_bucket_total_count,
            stats ? ctx->d_bucket_counts : nullptr,
            stats ? ctx->d_ambiguous_counts : nullptr,
            stats ? ctx->d_max_bucket_sizes : nullptr,
            (uint32_t)(kInitialPrefixSymbols),
            active_batch);
    }
    CUDA_CHECK(cudaMemcpyAsync(&bucket_total_count, ctx->d_bucket_total_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    if (_ov_dbg) cudaEventRecord(_ov[2], ctx->stream);
    if (bucket_total_count > 0) {
        // Stage-B cooperative bin-packed segment sorter (replaces the slow
        // single-thread-per-bucket insertion refine). Falls back to the old
        // kernel if the segment count or bin4 scratch budget is exceeded.
        if (!gpu_stageb_refine(ctx, d_sdata_base, d_data_len_base, bucket_total_count, active_batch, total_elements)) {
            const uint32_t refine_blocks = ((bucket_total_count + 255U) / 256U)  /* gridDim.x max is 2^31-1, NOT 65535; capping silently dropped segments >16.78M (batch>~1536) -> corrupt SA */;
            refine_bucket_descs_kernel<<<refine_blocks, 256, 0, ctx->stream>>>(
                d_sdata_base, d_data_len_base, ctx->d_offsets, ctx->d_bucket_descs, bucket_total_count, ctx->d_vals_out);
        }
    }
    if (_ov_dbg) {
        cudaEventRecord(_ov[3], ctx->stream);
        cudaEventSynchronize(_ov[3]);
        float _shuf=0.f,_desc=0.f,_sbtot=0.f;
        cudaEventElapsedTime(&_shuf, _ov[0], _ov[1]);
        cudaEventElapsedTime(&_desc, _ov[1], _ov[2]);
        cudaEventElapsedTime(&_sbtot, _ov[2], _ov[3]);
        fprintf(stderr, "[OVHD] shuffle=%.1f descr+sync=%.1f stageb_total=%.1f ms (pre-bin = stageb_total - STAGEB-T_sum)%c", _shuf, _desc, _sbtot, 10);
        for (int _i=0;_i<4;++_i) cudaEventDestroy(_ov[_i]);
    }
    {
        static bool sc_done = false;
        if (!sc_done && getenv("SHRINKCURVE")) {
            sc_done = true;
            unsigned long long *d_amb = nullptr, *d_tot = nullptr;
            cudaMalloc(&d_amb, SC_NDEPTH * sizeof(unsigned long long));
            cudaMalloc(&d_tot, sizeof(unsigned long long));
            cudaMemset(d_amb, 0, SC_NDEPTH * sizeof(unsigned long long));
            cudaMemset(d_tot, 0, sizeof(unsigned long long));
            shrink_curve_kernel<<<active_batch, 256, 0, ctx->stream>>>(
                d_sdata_base, ctx->d_offsets, ctx->d_vals_out, active_batch, d_amb, d_tot);
            cudaStreamSynchronize(ctx->stream);
            unsigned long long h_amb[SC_NDEPTH], h_tot = 0;
            cudaMemcpy(h_amb, d_amb, SC_NDEPTH * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_tot, d_tot, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            const int depths[SC_NDEPTH] = {1, 2, 3, 4, 7, 14, 28, 56, 96};
            fprintf(stderr, "[SHRINK] total_suffixes=%llu  (batch=%d)\n", h_tot, active_batch);
            for (int i = 0; i < SC_NDEPTH; i++)
                fprintf(stderr, "[SHRINK] depth=%3d  tied=%12llu  (%5.1f%%)\n",
                        depths[i], h_amb[i], h_tot ? 100.0 * (double)h_amb[i] / (double)h_tot : 0.0);
            cudaFree(d_amb); cudaFree(d_tot);
        }
    }
    {
        static bool sp_done = false;
        if (!sp_done && getenv("SORTPROBE")) {
            sp_done = true;
            uint64_t *pk_in=nullptr,*pk_out=nullptr; uint32_t *pv_in=nullptr,*pv_out=nullptr;
            cudaMalloc(&pk_in,(size_t)total_elements*sizeof(uint64_t));
            cudaMalloc(&pk_out,(size_t)total_elements*sizeof(uint64_t));
            cudaMalloc(&pv_in,(size_t)total_elements*sizeof(uint32_t));
            cudaMalloc(&pv_out,(size_t)total_elements*sizeof(uint32_t));
            const int Ks[5]={64,56,40,34,24};
            for(int ki=0;ki<5;ki++){
                int K=Ks[ki];
                init_sort_keys_kernel<<<dim3(1,active_batch),256,0,ctx->stream>>>(
                    d_sdata_base,d_data_len_base,ctx->d_offsets,pk_in,pv_in,active_batch,0);
                cudaStreamSynchronize(ctx->stream);
                cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
                cub::DoubleBuffer<uint64_t> dk(pk_in,pk_out);
                cub::DoubleBuffer<uint32_t> dv(pv_in,pv_out);
                cudaEventRecord(a,ctx->stream);
                cub::DeviceSegmentedRadixSort::SortPairs(
                    ctx->d_sort_temp,ctx->sort_temp_size,dk,dv,
                    total_elements,active_batch,ctx->d_offsets,ctx->d_offsets+1,0,K,ctx->stream);
                cudaEventRecord(b,ctx->stream);
                cudaEventSynchronize(b);
                float ms=0.f; cudaEventElapsedTime(&ms,a,b);
                fprintf(stderr,"[SORTPROBE] batch=%d elems=%d bits=%2d pass_ms=%.2f\n",active_batch,total_elements,K,ms);
                cudaEventDestroy(a); cudaEventDestroy(b);
            }
            cudaFree(pk_in);cudaFree(pk_out);cudaFree(pv_in);cudaFree(pv_out);
        }
    }
    {
        static bool rd_done = false;
        if (!rd_done && getenv("RANKDOUBLE")) {
            rd_done = true;
            size_t Nrd = (size_t)total_elements;
            uint32_t *d_oracle=nullptr,*d_ranks=nullptr,*d_ranks2=nullptr,*d_uniq=nullptr;
            unsigned long long* d_mismatch=nullptr;
            cudaMalloc(&d_oracle, Nrd*sizeof(uint32_t));
            cudaMalloc(&d_ranks, Nrd*sizeof(uint32_t));
            cudaMalloc(&d_ranks2, Nrd*sizeof(uint32_t));
            cudaMalloc(&d_uniq, (size_t)active_batch*sizeof(uint32_t));
            cudaMalloc(&d_mismatch, sizeof(unsigned long long));
            cudaMemsetAsync(d_mismatch,0,sizeof(unsigned long long),ctx->stream);
            cudaMemcpyAsync(d_oracle, ctx->d_vals_out, Nrd*sizeof(uint32_t), cudaMemcpyDeviceToDevice, ctx->stream);
            int32_t* h_len=(int32_t*)malloc((size_t)active_batch*sizeof(int32_t));
            uint32_t* h_uniq=(uint32_t*)malloc((size_t)active_batch*sizeof(uint32_t));
            cudaMemcpy(h_len, d_data_len_base, (size_t)active_batch*sizeof(int32_t), cudaMemcpyDeviceToHost);
            long long want=0; for(int i=0;i<active_batch;i++){ if(h_len[i]>0 && h_len[i]<=70911) want+=h_len[i]; }
            cudaEvent_t rt0,rt1; cudaEventCreate(&rt0); cudaEventCreate(&rt1);
            cudaEventRecord(rt0,ctx->stream);
            init_sort_keys_kernel<<<dim3(1,active_batch),256,0,ctx->stream>>>(d_sdata_base,d_data_len_base,ctx->d_offsets,ctx->d_keys_in,ctx->d_vals_in,active_batch,0);
            uint32_t* cur_ranks=d_ranks; uint32_t* nxt_ranks=d_ranks2;
            uint32_t* rd_final_vals=ctx->d_vals_out;
            {
                cub::DoubleBuffer<uint64_t> dk(ctx->d_keys_in,ctx->d_keys_out);
                cub::DoubleBuffer<uint32_t> dv(ctx->d_vals_in,ctx->d_vals_out);
                cub::DeviceSegmentedRadixSort::SortPairs(ctx->d_sort_temp,ctx->sort_temp_size,dk,dv,total_elements,active_batch,ctx->d_offsets,ctx->d_offsets+1,0,64,ctx->stream);
                rd_assign_ranks_parallel<<<active_batch,256,0,ctx->stream>>>(dk.Current(),dv.Current(),d_data_len_base,ctx->d_offsets,cur_ranks,d_uniq,active_batch);
                rd_final_vals=dv.Current();
            }
            cudaStreamSynchronize(ctx->stream);
            long long uq=0; cudaMemcpy(h_uniq,d_uniq,(size_t)active_batch*sizeof(uint32_t),cudaMemcpyDeviceToHost); for(int i=0;i<active_batch;i++) uq+=h_uniq[i];
            fprintf(stderr,"[RANKDOUBLE] round=0 step=7 active=%lld of %lld%c", want-uq, want, 10);
            int step=7; int round=0;
            while(uq<want && round<24){
                round++;
                build_rank_keys_kernel<<<dim3(1,active_batch),256,0,ctx->stream>>>(cur_ranks,d_data_len_base,ctx->d_offsets,ctx->d_keys_in,ctx->d_vals_in,active_batch,step);
                cub::DoubleBuffer<uint64_t> dk(ctx->d_keys_in,ctx->d_keys_out);
                cub::DoubleBuffer<uint32_t> dv(ctx->d_vals_in,ctx->d_vals_out);
                cub::DeviceSegmentedRadixSort::SortPairs(ctx->d_sort_temp,ctx->sort_temp_size,dk,dv,total_elements,active_batch,ctx->d_offsets,ctx->d_offsets+1,0,64,ctx->stream);
                rd_assign_ranks_parallel<<<active_batch,256,0,ctx->stream>>>(dk.Current(),dv.Current(),d_data_len_base,ctx->d_offsets,nxt_ranks,d_uniq,active_batch);
                rd_final_vals=dv.Current();
                cudaStreamSynchronize(ctx->stream);
                uq=0; cudaMemcpy(h_uniq,d_uniq,(size_t)active_batch*sizeof(uint32_t),cudaMemcpyDeviceToHost); for(int i=0;i<active_batch;i++) uq+=h_uniq[i];
                fprintf(stderr,"[RANKDOUBLE] round=%d step=%d active=%lld of %lld%c", round, step*2, want-uq, want, 10);
                uint32_t* tmp=cur_ranks; cur_ranks=nxt_ranks; nxt_ranks=tmp;
                step*=2;
            }
            cudaEventRecord(rt1,ctx->stream); cudaEventSynchronize(rt1);
            float rms=0.f; cudaEventElapsedTime(&rms,rt0,rt1);
            rd_compare_sa_kernel<<<active_batch,256,0,ctx->stream>>>(rd_final_vals,d_oracle,d_data_len_base,ctx->d_offsets,active_batch,d_mismatch);
            cudaStreamSynchronize(ctx->stream);
            unsigned long long h_mismatch=0; cudaMemcpy(&h_mismatch,d_mismatch,sizeof(unsigned long long),cudaMemcpyDeviceToHost);
            fprintf(stderr,"[RANKDOUBLE] DONE rounds=%d total_ms=%.1f mismatch_vs_byterefine=%llu%c", round, rms, h_mismatch, 10);
            free(h_len); free(h_uniq);
            cudaFree(d_oracle); cudaFree(d_ranks); cudaFree(d_ranks2); cudaFree(d_uniq); cudaFree(d_mismatch);
            cudaEventDestroy(rt0); cudaEventDestroy(rt1);
        }
    }

    if (stats) {
        cudaEventRecord(e4, ctx->stream);
        CUDA_CHECK(cudaEventSynchronize(e4));
        cudaEventElapsedTime(&stats->offsets_ms, e0, e1);
        cudaEventElapsedTime(&stats->keys_ms, e1, e2);
        cudaEventElapsedTime(&stats->sort_ms, e2, e3);
        cudaEventElapsedTime(&stats->refine_ms, e3, e4);
        gpu_collect_bucket_stats(ctx, active_batch, stats);
        cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3); cudaEventDestroy(e4);
    }
    return true;
}

static bool gpu_build_fast_sa(GPUContext* ctx, int active_batch, int& total_elements, GPUSABuildStats* stats = nullptr) {
    return gpu_build_fast_sa_window(ctx, ctx->d_sdata, ctx->d_data_len, active_batch, total_elements, stats);
}

static bool gpu_build_staged_sa(GPUContext* ctx, int active_batch, int& total_elements, GPUSABuildStats* stats = nullptr) {
    cudaSetDevice(ctx->device_id);
    total_elements = 0;
    if (active_batch <= 0) return true;

    const int chunk_size = std::max(1, std::min(ctx->staged_subbatch, active_batch));
    GPUSABuildStats accum{};

    for (int batch_start = 0; batch_start < active_batch; batch_start += chunk_size) {
        const int chunk_batch = std::min(chunk_size, active_batch - batch_start);
        int chunk_total = 0;
        GPUSABuildStats chunk_stats{};
        GPUSABuildStats* chunk_stats_ptr = stats ? &chunk_stats : nullptr;
        const uint8_t* chunk_sdata = ctx->d_sdata + (size_t)batch_start * (72 * 1024);
        const int32_t* chunk_data_len = ctx->d_data_len + batch_start;
        if (!gpu_build_fast_sa_window(ctx, chunk_sdata, chunk_data_len, chunk_batch, chunk_total, chunk_stats_ptr)) {
            return false;
        }
        total_elements += chunk_total;
        if (stats) {
            accum.offsets_ms += chunk_stats.offsets_ms;
            accum.keys_ms += chunk_stats.keys_ms;
            accum.sort_ms += chunk_stats.sort_ms;
            accum.refine_ms += chunk_stats.refine_ms;
            accum.bucket_count += chunk_stats.bucket_count;
            accum.ambiguous_count += chunk_stats.ambiguous_count;
            accum.max_bucket_size = std::max(accum.max_bucket_size, chunk_stats.max_bucket_size);
        }
    }

    if (stats) *stats = accum;
    return true;
}

int gpu_mine_batch(GPUContext* ctx, uint32_t nonce_start, std::vector<GPUSolution>& solutions) {
    cudaSetDevice(ctx->device_id);
    const int bs = ctx->batch_size;
    const bool exact_engine = gpu_engine_is_exact(ctx->engine_mode) || gpu_engine_is_cleanroom(ctx->engine_mode);
    const bool recovered_engine = gpu_engine_is_recovered(ctx->engine_mode);
    const bool staged_engine = gpu_engine_is_staged(ctx->engine_mode);
    solutions.clear();

    uint64_t zero = 0;
    uint32_t zero32 = 0;
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_solutions, &zero, sizeof(uint64_t), cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(d_near_miss_count, &zero32, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, ctx->stream));

    cudaEvent_t e_start, e_branch, e_sa_end, e_end;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_branch);
    cudaEventCreate(&e_sa_end);
    cudaEventCreate(&e_end);
    cudaEventRecord(e_start, ctx->stream);

    astrobwt_branch_compute_kernel<<<bs, 32, 0, ctx->stream>>>(ctx->d_sdata, ctx->d_data_len, nullptr, nonce_start, bs);
    cudaEventRecord(e_branch, ctx->stream);

    auto host_wait_start = std::chrono::steady_clock::now();
    GPUSABuildStats sa_stats{};
    GPUSABuildStats* sa_stats_ptr = ctx->perf_logging ? &sa_stats : nullptr;
    int total_elements = 0;
    bool sa_ok = true;
    if (exact_engine) {
        sa_ok = gpu_build_exact_sa(ctx, bs, total_elements, sa_stats_ptr);
        if (sa_ok) cudaEventRecord(e_sa_end, ctx->stream);
    } else if (staged_engine) {
        const int chunk_size = std::max(1, std::min(ctx->staged_subbatch, bs));
        GPUSABuildStats accum{};
        for (int batch_start = 0; batch_start < bs; batch_start += chunk_size) {
            const int chunk_batch = std::min(chunk_size, bs - batch_start);
            int chunk_total = 0;
            GPUSABuildStats chunk_stats{};
            GPUSABuildStats* chunk_stats_ptr = ctx->perf_logging ? &chunk_stats : nullptr;
            const uint8_t* chunk_sdata = ctx->d_sdata + (size_t)batch_start * (72 * 1024);
            const int32_t* chunk_data_len = ctx->d_data_len + batch_start;
            sa_ok = gpu_build_fast_sa_window(ctx, chunk_sdata, chunk_data_len, chunk_batch, chunk_total, chunk_stats_ptr);
            if (!sa_ok) break;
            total_elements += chunk_total;
            if (sa_stats_ptr) {
                accum.offsets_ms += chunk_stats.offsets_ms;
                accum.keys_ms += chunk_stats.keys_ms;
                accum.sort_ms += chunk_stats.sort_ms;
                accum.refine_ms += chunk_stats.refine_ms;
                accum.bucket_count += chunk_stats.bucket_count;
                accum.ambiguous_count += chunk_stats.ambiguous_count;
                accum.max_bucket_size = std::max(accum.max_bucket_size, chunk_stats.max_bucket_size);
            }
            astrobwt_final_hash_segments_kernel<<<(chunk_batch + 63) / 64, 64, 0, ctx->stream>>>(
                ctx->d_vals_out, ctx->d_data_len + batch_start, ctx->d_offsets, ctx->d_solutions, nonce_start + (uint32_t)batch_start, chunk_batch);
        }
        if (sa_stats_ptr) sa_stats = accum;
    } else if (recovered_engine) {
        sa_ok = gpu_build_fast_sa(ctx, bs, total_elements, sa_stats_ptr);
        if (sa_ok) cudaEventRecord(e_sa_end, ctx->stream);
    } else {
        sa_ok = false;
    }
    if (!sa_ok) {
        std::fprintf(stderr, "[GPU %d] %s SA build failed\n", ctx->device_id, gpu_engine_mode_name(ctx->engine_mode));
        cudaEventDestroy(e_start);
        cudaEventDestroy(e_branch);
        cudaEventDestroy(e_sa_end);
        cudaEventDestroy(e_end);
        return 0;
    }

    const int hash_block = exact_engine ? 128 : 64;
    if (exact_engine) {
        astrobwt_final_hash_kernel<<<(bs + hash_block - 1) / hash_block, hash_block, 0, ctx->stream>>>(
            ctx->d_sa, ctx->d_data_len, ctx->d_solutions, nonce_start, bs);
    } else if (recovered_engine) {
        astrobwt_final_hash_segments_kernel<<<(bs + hash_block - 1) / hash_block, hash_block, 0, ctx->stream>>>(
            ctx->d_vals_out, ctx->d_data_len, ctx->d_offsets, ctx->d_solutions, nonce_start, bs);
    }
    cudaEventRecord(e_end, ctx->stream);
    CUDA_CHECK(cudaEventSynchronize(e_end));
    auto host_wait_end = std::chrono::steady_clock::now();

    float t_branch = 0.0f, t_sa = 0.0f, t_hash = 0.0f, t_total_gpu = 0.0f;
    cudaEventElapsedTime(&t_branch, e_start, e_branch);
    cudaEventElapsedTime(&t_total_gpu, e_start, e_end);
    if (staged_engine) {
        t_sa = sa_stats.offsets_ms + sa_stats.keys_ms + sa_stats.sort_ms + sa_stats.refine_ms;
        t_hash = std::max(0.0f, t_total_gpu - t_branch - t_sa);
    } else {
        cudaEventElapsedTime(&t_sa, e_branch, e_sa_end);
        cudaEventElapsedTime(&t_hash, e_sa_end, e_end);
    }
    const float t_host_wait = (float)std::chrono::duration<double, std::milli>(host_wait_end - host_wait_start).count();

    uint32_t near_misses = 0;
    CUDA_CHECK(cudaMemcpyFromSymbol(&near_misses, d_near_miss_count, sizeof(uint32_t)));

    uint64_t sol_count = 0;
    CUDA_CHECK(cudaMemcpy(&sol_count, ctx->d_solutions, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    if (sol_count > 0 && sol_count <= 1024) {
        CUDA_CHECK(cudaMemcpy(ctx->h_solutions, ctx->d_solutions, (1 + sol_count * 5) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        for (uint64_t i = 0; i < sol_count; ++i) {
            GPUSolution sol{};
            uint64_t* slot_ptr = &ctx->h_solutions[1 + i * 5];
            sol.nonce = (uint32_t)slot_ptr[0];
            std::memcpy(sol.hash, &slot_ptr[1], 32);
            solutions.push_back(sol);
        }
    }

    static int log_cnt = 0;
    log_cnt++;
    const bool should_log_perf = ctx->perf_logging
        ? (t_total_gpu > 200.0f || log_cnt % 5 == 0)
        : (t_total_gpu > 500.0f);
    if (should_log_perf) {
        const float avg_len = bs > 0 ? (float)total_elements / (float)bs : 0.0f;
        if (sa_stats_ptr) {
            std::printf("[PERF] Batch: %.1f ms (Branch: %.1f, %s: %.1f [Off: %.1f Keys: %.1f Sort: %.1f Refine: %.1f], Hash: %.1f, HostWait: %.1f) | HR: %.1f KH/s | Near: %u | Sol: %u | Elems: %d | AvgLen: %.1f | RefineSegs: %u | Ambig: %u | MaxBucket: %u\n",
                        t_total_gpu, t_branch, exact_engine ? "ExactSA" : (staged_engine ? "StagedSA" : "Sort"),
                        t_sa, sa_stats.offsets_ms, sa_stats.keys_ms, sa_stats.sort_ms, sa_stats.refine_ms,
                        t_hash, (t_host_wait - t_total_gpu), (bs / t_total_gpu), near_misses,
                        (unsigned)solutions.size(), total_elements, avg_len,
                        sa_stats.bucket_count, sa_stats.ambiguous_count, sa_stats.max_bucket_size);
        } else {
            std::printf("[PERF] Batch: %.1f ms (Branch: %.1f, %s: %.1f, Hash: %.1f, HostWait: %.1f) | HR: %.1f KH/s | Near: %u | Sol: %u | Elems: %d | AvgLen: %.1f\n",
                        t_total_gpu, t_branch, exact_engine ? "ExactSA" : (staged_engine ? "StagedSA" : "Sort"),
                        t_sa, t_hash, (t_host_wait - t_total_gpu), (bs / t_total_gpu), near_misses,
                        (unsigned)solutions.size(), total_elements, avg_len);
        }
        std::fflush(stdout);
    }

    cudaEventDestroy(e_start);
    cudaEventDestroy(e_branch);
    cudaEventDestroy(e_sa_end);
    cudaEventDestroy(e_end);
    return (int)solutions.size();
}

namespace {

struct GPUParityCase {
    const char* name;
    const char* work_hex;
};

static const GPUParityCase kParityCases[] = {
    {"repeat-vector", "419ebb000000001bbdc9bf2200000000635d6e4e24829b4249fe0e67878ad4350000000043f53e5436cf610000086b00"},
    {"fake-submit-1", "4130da000067542bc5e06c2a0000000070d6b9d4cb39116b464668b058f81f4800000000b504f97dbb030129d12a106a"},
    {"fake-submit-2", "4182b70000675447d00670340000000070d6b9d4cb39116b464668b058f81f4800000000fecf77436984066f91da6794"},
    {"real-daemon-1", "41d09c000067ed485d6268420000000070d6b9d4cb39116b464668b058f81f4800000000e071b48a9d04eef3131750ab"},
    {"real-daemon-1+10", "41d09c000067ed485d6268420000000070d6b9d4cb39116b464668b058f81f4800000000e071b48a9d04eef313175aab"},
};

static constexpr const char* kRealDaemonRangeWorkHex =
    "41d09c000067ed485d6268420000000070d6b9d4cb39116b464668b058f81f4800000000e071b48a9d04eef3131750ab";

static bool gpu_decode_hex_48(const char* hex, uint8_t out[48]) {
    if (!hex || std::strlen(hex) != 96) return false;
    auto hex_nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (int i = 0; i < 48; ++i) {
        const int hi = hex_nibble(hex[i * 2]);
        const int lo = hex_nibble(hex[i * 2 + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return true;
}

static void gpu_print_hex32(const uint8_t data[32]) {
    for (int i = 0; i < 32; ++i) std::printf("%02x", data[i]);
}

template <typename T>
static int gpu_first_mismatch_index(const T* lhs, const int32_t* rhs, int n) {
    for (int i = 0; i < n; ++i) {
        if ((int32_t)lhs[i] != rhs[i]) return i;
    }
    return -1;
}

static uint32_t gpu_read_nonce_be(const uint8_t work[48]) {
    return ((uint32_t)work[43] << 24) |
           ((uint32_t)work[44] << 16) |
           ((uint32_t)work[45] << 8) |
           (uint32_t)work[46];
}

static bool gpu_verify_case(GPUContext* ctx, const char* name, const uint8_t work[48]) {
    static constexpr int MAX_SA_N = 71429;
    astrobwt::WorkerState cpu_worker{};
    uint8_t cpu_hash[32] = {};
    uint8_t cpu_sdata_sha[32] = {};
    astrobwt::hash(work, 48, cpu_hash, cpu_worker);
    sha256::hash(cpu_worker.sData, cpu_worker.data_len, cpu_sdata_sha);

    cudaSetDevice(ctx->device_id);
    CUDA_CHECK(cudaMemcpyToSymbol(d_work_template, work, 48));

    const uint32_t nonce = gpu_read_nonce_be(work);
    astrobwt_branch_compute_kernel<<<1, 32, 0, ctx->stream>>>(ctx->d_sdata, ctx->d_data_len, nullptr, nonce, 1);
    int total_elements = 0;
    const bool sa_ok = gpu_build_exact_sa(ctx, 1, total_elements);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    if (!sa_ok) {
        std::printf("[VERIFY][GPU %d] %s | exact SA convergence failed\n", ctx->device_id, name);
        return false;
    }

    int32_t gpu_data_len = 0;
    CUDA_CHECK(cudaMemcpy(&gpu_data_len, ctx->d_data_len, sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (gpu_data_len < 0 || gpu_data_len > MAX_SA_N) {
        std::printf("[VERIFY][GPU %d] %s | invalid data_len=%d\n", ctx->device_id, name, gpu_data_len);
        return false;
    }

    std::vector<uint8_t> gpu_sdata((size_t)std::max(gpu_data_len, 1));
    std::vector<int32_t> gpu_sa((size_t)std::max(gpu_data_len, 1));
    if (gpu_data_len > 0) {
        CUDA_CHECK(cudaMemcpy(gpu_sdata.data(), ctx->d_sdata, (size_t)gpu_data_len, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_sa.data(), ctx->d_sa, (size_t)gpu_data_len * sizeof(int32_t), cudaMemcpyDeviceToHost));
    }

    uint8_t gpu_sdata_sha[32] = {};
    uint8_t gpu_hash[32] = {};
    if (gpu_data_len > 0) {
        sha256::hash(gpu_sdata.data(), gpu_data_len, gpu_sdata_sha);
        sha256::hash(reinterpret_cast<const uint8_t*>(gpu_sa.data()), gpu_data_len * (int)sizeof(int32_t), gpu_hash);
    }

    const bool len_match = (gpu_data_len == cpu_worker.data_len);
    const bool sdata_match = len_match && std::memcmp(gpu_sdata.data(), cpu_worker.sData, (size_t)gpu_data_len) == 0;
    const bool sa_match = len_match && std::memcmp(gpu_sa.data(), cpu_worker.sa, (size_t)gpu_data_len * sizeof(int32_t)) == 0;
    const bool hash_match = len_match && std::memcmp(gpu_hash, cpu_hash, 32) == 0;
    const bool ok = len_match && sdata_match && sa_match && hash_match;

    std::printf("[VERIFY][GPU %d] %s | len:%s sdata:%s sa:%s hash:%s\n",
                ctx->device_id, name,
                len_match ? "OK" : "FAIL",
                sdata_match ? "OK" : "FAIL",
                sa_match ? "OK" : "FAIL",
                hash_match ? "OK" : "FAIL");
    if (!ok) {
        std::printf("  cpu_data_len=%d gpu_data_len=%d\n", cpu_worker.data_len, gpu_data_len);
        if (len_match && !sa_match) {
            const int mismatch = gpu_first_mismatch_index(gpu_sa.data(), cpu_worker.sa, gpu_data_len);
            if (mismatch >= 0) {
                std::printf("  first_sa_mismatch=%d cpu=%d gpu=%d\n",
                            mismatch, cpu_worker.sa[mismatch], gpu_sa[mismatch]);
            }
        }
        std::printf("  cpu_sdata_sha=");
        gpu_print_hex32(cpu_sdata_sha);
        std::printf("\n  gpu_sdata_sha=");
        gpu_print_hex32(gpu_sdata_sha);
        std::printf("\n  cpu_hash=");
        gpu_print_hex32(cpu_hash);
        std::printf("\n  gpu_hash=");
        gpu_print_hex32(gpu_hash);
        std::printf("\n");
    }
    return ok;
}

static bool gpu_verify_range_case(GPUContext* ctx, const char* name, const uint8_t work[48], int active_batch) {
    static constexpr int HOST_SDATA_STRIDE = 72 * 1024;
    static constexpr int MAX_SA_N = 71429;

    if (active_batch <= 0 || active_batch > ctx->batch_size) return false;

    cudaSetDevice(ctx->device_id);
    CUDA_CHECK(cudaMemcpyToSymbol(d_work_template, work, 48));

    const uint32_t nonce_start = gpu_read_nonce_be(work);
    astrobwt_branch_compute_kernel<<<active_batch, 32, 0, ctx->stream>>>(ctx->d_sdata, ctx->d_data_len, nullptr, nonce_start, active_batch);
    int total_elements = 0;
    const bool sa_ok = gpu_build_exact_sa(ctx, active_batch, total_elements);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    if (!sa_ok) {
        std::printf("[VERIFY][GPU %d] %s | exact SA convergence failed for batch=%d nonce_start=%u\n",
                    ctx->device_id, name, active_batch, nonce_start);
        return false;
    }

    std::vector<int32_t> gpu_data_len((size_t)active_batch);
    std::vector<uint8_t> gpu_sdata((size_t)active_batch * HOST_SDATA_STRIDE);
    std::vector<int32_t> gpu_sa((size_t)active_batch * MAX_SA_N);
    CUDA_CHECK(cudaMemcpy(gpu_data_len.data(), ctx->d_data_len, (size_t)active_batch * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_sdata.data(), ctx->d_sdata, (size_t)active_batch * HOST_SDATA_STRIDE, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_sa.data(), ctx->d_sa, (size_t)active_batch * MAX_SA_N * sizeof(int32_t), cudaMemcpyDeviceToHost));

    bool ok = true;
    int first_bad_lane = -1;
    const char* first_bad_reason = "";
    uint8_t cpu_hash[32] = {};
    uint8_t gpu_hash[32] = {};
    uint8_t cpu_sdata_sha[32] = {};
    uint8_t gpu_sdata_sha[32] = {};

    for (int lane = 0; lane < active_batch; ++lane) {
        uint8_t cpu_work[48];
        std::memcpy(cpu_work, work, 48);
        const uint32_t nonce = nonce_start + (uint32_t)lane;
        cpu_work[43] = (uint8_t)(nonce >> 24);
        cpu_work[44] = (uint8_t)(nonce >> 16);
        cpu_work[45] = (uint8_t)(nonce >> 8);
        cpu_work[46] = (uint8_t)nonce;

        astrobwt::WorkerState cpu_worker{};
        astrobwt::hash(cpu_work, 48, cpu_hash, cpu_worker);
        sha256::hash(cpu_worker.sData, cpu_worker.data_len, cpu_sdata_sha);

        const int32_t gpu_n = gpu_data_len[lane];
        if (gpu_n != cpu_worker.data_len) {
            ok = false;
            first_bad_lane = lane;
            first_bad_reason = "len";
            break;
        }

        const uint8_t* lane_sdata = gpu_sdata.data() + (size_t)lane * HOST_SDATA_STRIDE;
        const int32_t* lane_sa = gpu_sa.data() + (size_t)lane * MAX_SA_N;
        if (std::memcmp(lane_sdata, cpu_worker.sData, (size_t)gpu_n) != 0) {
            sha256::hash(lane_sdata, gpu_n, gpu_sdata_sha);
            ok = false;
            first_bad_lane = lane;
            first_bad_reason = "sdata";
            break;
        }
        if (std::memcmp(lane_sa, cpu_worker.sa, (size_t)gpu_n * sizeof(int32_t)) != 0) {
            sha256::hash(reinterpret_cast<const uint8_t*>(lane_sa), gpu_n * (int)sizeof(int32_t), gpu_hash);
            ok = false;
            first_bad_lane = lane;
            first_bad_reason = "sa";
            break;
        }
        sha256::hash(reinterpret_cast<const uint8_t*>(lane_sa), gpu_n * (int)sizeof(int32_t), gpu_hash);
        if (std::memcmp(gpu_hash, cpu_hash, 32) != 0) {
            ok = false;
            first_bad_lane = lane;
            first_bad_reason = "hash";
            break;
        }
    }

    std::printf("[VERIFY][GPU %d] %s | batch:%d %s\n",
                ctx->device_id, name, active_batch, ok ? "OK" : "FAIL");
    if (!ok && first_bad_lane >= 0) {
        const uint32_t bad_nonce = nonce_start + (uint32_t)first_bad_lane;
        std::printf("  lane=%d nonce=%u reason=%s\n", first_bad_lane, bad_nonce, first_bad_reason);
        if (std::strcmp(first_bad_reason, "sdata") == 0) {
            std::printf("  cpu_sdata_sha=");
            gpu_print_hex32(cpu_sdata_sha);
            std::printf("\n  gpu_sdata_sha=");
            gpu_print_hex32(gpu_sdata_sha);
            std::printf("\n");
        } else if (std::strcmp(first_bad_reason, "sa") == 0 || std::strcmp(first_bad_reason, "hash") == 0) {
            std::printf("  cpu_hash=");
            gpu_print_hex32(cpu_hash);
            std::printf("\n  gpu_hash=");
            gpu_print_hex32(gpu_hash);
            std::printf("\n");
        }
    }
    return ok;
}

static bool gpu_verify_recovered_case(GPUContext* ctx, const char* name, const uint8_t work[48]) {
    static constexpr int MAX_SA_N = 71429;

    astrobwt::WorkerState cpu_worker{};
    uint8_t cpu_hash[32] = {};
    astrobwt::hash(work, 48, cpu_hash, cpu_worker);

    cudaSetDevice(ctx->device_id);
    CUDA_CHECK(cudaMemcpyToSymbol(d_work_template, work, 48));

    const uint32_t nonce = gpu_read_nonce_be(work);
    astrobwt_branch_compute_kernel<<<1, 32, 0, ctx->stream>>>(ctx->d_sdata, ctx->d_data_len, nullptr, nonce, 1);
    int total_elements = 0;
    const bool sa_ok = gpu_build_fast_sa(ctx, 1, total_elements);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    if (!sa_ok) {
        std::printf("[VERIFY][GPU %d][RECOVERED] %s | recovered SA build failed\n", ctx->device_id, name);
        return false;
    }

    int32_t gpu_data_len = 0;
    CUDA_CHECK(cudaMemcpy(&gpu_data_len, ctx->d_data_len, sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (gpu_data_len < 0 || gpu_data_len > MAX_SA_N) {
        std::printf("[VERIFY][GPU %d][RECOVERED] %s | invalid data_len=%d\n", ctx->device_id, name, gpu_data_len);
        return false;
    }

    std::vector<uint32_t> gpu_sa((size_t)std::max(gpu_data_len, 1));
    uint8_t gpu_hash[32] = {};
    if (gpu_data_len > 0) {
        CUDA_CHECK(cudaMemcpy(gpu_sa.data(), ctx->d_vals_out, (size_t)gpu_data_len * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        sha256::hash(reinterpret_cast<const uint8_t*>(gpu_sa.data()), gpu_data_len * (int)sizeof(uint32_t), gpu_hash);
    }

    const bool len_match = (gpu_data_len == cpu_worker.data_len);
    const bool sa_match = len_match && std::memcmp(gpu_sa.data(), cpu_worker.sa, (size_t)gpu_data_len * sizeof(int32_t)) == 0;
    const bool hash_match = len_match && std::memcmp(gpu_hash, cpu_hash, 32) == 0;
    const bool ok = len_match && sa_match && hash_match;

    std::printf("[VERIFY][GPU %d][RECOVERED] %s | len:%s sa:%s hash:%s\n",
                ctx->device_id, name,
                len_match ? "OK" : "FAIL",
                sa_match ? "OK" : "FAIL",
                hash_match ? "OK" : "FAIL");
    if (!ok) {
        std::printf("  cpu_data_len=%d gpu_data_len=%d\n", cpu_worker.data_len, gpu_data_len);
        if (len_match && !sa_match) {
            const int mismatch = gpu_first_mismatch_index(gpu_sa.data(), cpu_worker.sa, gpu_data_len);
            if (mismatch >= 0) {
                std::printf("  first_sa_mismatch=%d cpu=%d gpu=%d\n",
                            mismatch, cpu_worker.sa[mismatch], gpu_sa[mismatch]);
            }
        }
        std::printf("  cpu_hash=");
        gpu_print_hex32(cpu_hash);
        std::printf("\n  gpu_hash=");
        gpu_print_hex32(gpu_hash);
        std::printf("\n");
    }
    return ok;
}

static bool gpu_verify_recovered_range_case(GPUContext* ctx, const char* name, const uint8_t work[48], int active_batch) {
    static constexpr int MAX_SA_N = 71429;

    if (active_batch <= 0 || active_batch > ctx->batch_size) return false;

    cudaSetDevice(ctx->device_id);
    CUDA_CHECK(cudaMemcpyToSymbol(d_work_template, work, 48));

    const uint32_t nonce_start = gpu_read_nonce_be(work);
    astrobwt_branch_compute_kernel<<<active_batch, 32, 0, ctx->stream>>>(ctx->d_sdata, ctx->d_data_len, nullptr, nonce_start, active_batch);
    int total_elements = 0;
    const bool sa_ok = gpu_build_fast_sa(ctx, active_batch, total_elements);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    if (!sa_ok) {
        std::printf("[VERIFY][GPU %d][RECOVERED] %s | recovered SA build failed for batch=%d nonce_start=%u\n",
                    ctx->device_id, name, active_batch, nonce_start);
        return false;
    }

    std::vector<int32_t> gpu_data_len((size_t)active_batch);
    std::vector<uint32_t> gpu_sa((size_t)std::max(total_elements, 1));
    CUDA_CHECK(cudaMemcpy(gpu_data_len.data(), ctx->d_data_len, (size_t)active_batch * sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (total_elements > 0) {
        CUDA_CHECK(cudaMemcpy(gpu_sa.data(), ctx->d_vals_out, (size_t)total_elements * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }

    std::vector<int> offsets((size_t)active_batch + 1, 0);
    for (int i = 0; i < active_batch; ++i) {
        offsets[(size_t)i + 1] = offsets[(size_t)i] + gpu_data_len[(size_t)i];
    }

    bool ok = true;
    int first_bad_lane = -1;
    const char* first_bad_reason = "";
    uint8_t cpu_hash[32] = {};
    uint8_t gpu_hash[32] = {};

    for (int lane = 0; lane < active_batch; ++lane) {
        uint8_t cpu_work[48];
        std::memcpy(cpu_work, work, 48);
        const uint32_t nonce = nonce_start + (uint32_t)lane;
        cpu_work[43] = (uint8_t)(nonce >> 24);
        cpu_work[44] = (uint8_t)(nonce >> 16);
        cpu_work[45] = (uint8_t)(nonce >> 8);
        cpu_work[46] = (uint8_t)nonce;

        astrobwt::WorkerState cpu_worker{};
        astrobwt::hash(cpu_work, 48, cpu_hash, cpu_worker);

        const int32_t gpu_n = gpu_data_len[(size_t)lane];
        if (gpu_n != cpu_worker.data_len) {
            ok = false;
            first_bad_lane = lane;
            first_bad_reason = "len";
            break;
        }
        if (gpu_n < 0 || gpu_n > MAX_SA_N) {
            ok = false;
            first_bad_lane = lane;
            first_bad_reason = "len";
            break;
        }

        const uint32_t* lane_sa = gpu_sa.data() + offsets[(size_t)lane];
        if (std::memcmp(lane_sa, cpu_worker.sa, (size_t)gpu_n * sizeof(int32_t)) != 0) {
            sha256::hash(reinterpret_cast<const uint8_t*>(lane_sa), gpu_n * (int)sizeof(uint32_t), gpu_hash);
            ok = false;
            first_bad_lane = lane;
            first_bad_reason = "sa";
            break;
        }
        sha256::hash(reinterpret_cast<const uint8_t*>(lane_sa), gpu_n * (int)sizeof(uint32_t), gpu_hash);
        if (std::memcmp(gpu_hash, cpu_hash, 32) != 0) {
            ok = false;
            first_bad_lane = lane;
            first_bad_reason = "hash";
            break;
        }
    }

    std::printf("[VERIFY][GPU %d][RECOVERED] %s | batch:%d %s\n",
                ctx->device_id, name, active_batch, ok ? "OK" : "FAIL");
    if (!ok && first_bad_lane >= 0) {
        const uint32_t bad_nonce = nonce_start + (uint32_t)first_bad_lane;
        std::printf("  lane=%d nonce=%u reason=%s\n", first_bad_lane, bad_nonce, first_bad_reason);
        std::printf("  cpu_hash=");
        gpu_print_hex32(cpu_hash);
        std::printf("\n  gpu_hash=");
        gpu_print_hex32(gpu_hash);
        std::printf("\n");
    }
    return ok;
}

static bool gpu_verify_staged_case(GPUContext* ctx, const char* name, const uint8_t work[48]) {
    static constexpr int MAX_SA_N = 71429;

    astrobwt::WorkerState cpu_worker{};
    uint8_t cpu_hash[32] = {};
    astrobwt::hash(work, 48, cpu_hash, cpu_worker);

    cudaSetDevice(ctx->device_id);
    CUDA_CHECK(cudaMemcpyToSymbol(d_work_template, work, 48));

    const uint32_t nonce = gpu_read_nonce_be(work);
    astrobwt_branch_compute_kernel<<<1, 32, 0, ctx->stream>>>(ctx->d_sdata, ctx->d_data_len, nullptr, nonce, 1);
    int total_elements = 0;
    const bool sa_ok = gpu_build_staged_sa(ctx, 1, total_elements);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    if (!sa_ok) {
        std::printf("[VERIFY][GPU %d][STAGED] %s | staged SA build failed\n", ctx->device_id, name);
        return false;
    }

    int32_t gpu_data_len = 0;
    CUDA_CHECK(cudaMemcpy(&gpu_data_len, ctx->d_data_len, sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (gpu_data_len < 0 || gpu_data_len > MAX_SA_N) {
        std::printf("[VERIFY][GPU %d][STAGED] %s | invalid data_len=%d\n", ctx->device_id, name, gpu_data_len);
        return false;
    }

    std::vector<uint32_t> gpu_sa((size_t)std::max(gpu_data_len, 1));
    uint8_t gpu_hash[32] = {};
    if (gpu_data_len > 0) {
        CUDA_CHECK(cudaMemcpy(gpu_sa.data(), ctx->d_vals_out, (size_t)gpu_data_len * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        sha256::hash(reinterpret_cast<const uint8_t*>(gpu_sa.data()), gpu_data_len * (int)sizeof(uint32_t), gpu_hash);
    }

    const bool len_match = (gpu_data_len == cpu_worker.data_len);
    const bool sa_match = len_match && std::memcmp(gpu_sa.data(), cpu_worker.sa, (size_t)gpu_data_len * sizeof(int32_t)) == 0;
    const bool hash_match = len_match && std::memcmp(gpu_hash, cpu_hash, 32) == 0;
    const bool ok = len_match && sa_match && hash_match;

    std::printf("[VERIFY][GPU %d][STAGED] %s | len:%s sa:%s hash:%s\n",
                ctx->device_id, name,
                len_match ? "OK" : "FAIL",
                sa_match ? "OK" : "FAIL",
                hash_match ? "OK" : "FAIL");
    if (!ok) {
        std::printf("  cpu_data_len=%d gpu_data_len=%d\n", cpu_worker.data_len, gpu_data_len);
        if (len_match && !sa_match) {
            const int mismatch = gpu_first_mismatch_index(gpu_sa.data(), cpu_worker.sa, gpu_data_len);
            if (mismatch >= 0) {
                std::printf("  first_sa_mismatch=%d cpu=%d gpu=%d\n",
                            mismatch, cpu_worker.sa[mismatch], gpu_sa[mismatch]);
            }
        }
        std::printf("  cpu_hash=");
        gpu_print_hex32(cpu_hash);
        std::printf("\n  gpu_hash=");
        gpu_print_hex32(gpu_hash);
        std::printf("\n");
    }
    return ok;
}

static bool gpu_verify_staged_range_case(GPUContext* ctx, const char* name, const uint8_t work[48], int active_batch) {
    if (active_batch <= 0 || active_batch > ctx->batch_size) return false;
    const uint32_t nonce_start = gpu_read_nonce_be(work);
    for (int lane = 0; lane < active_batch; ++lane) {
        uint8_t lane_work[48];
        std::memcpy(lane_work, work, 48);
        const uint32_t nonce = nonce_start + (uint32_t)lane;
        lane_work[43] = (uint8_t)(nonce >> 24);
        lane_work[44] = (uint8_t)(nonce >> 16);
        lane_work[45] = (uint8_t)(nonce >> 8);
        lane_work[46] = (uint8_t)nonce;
        if (!gpu_verify_staged_case(ctx, lane == 0 ? name : "range-lane", lane_work)) {
            std::printf("[VERIFY][GPU %d][STAGED] %s | batch:%d FAIL at lane=%d nonce=%u\n",
                        ctx->device_id, name, active_batch, lane, nonce);
            return false;
        }
    }
    std::printf("[VERIFY][GPU %d][STAGED] %s | batch:%d OK\n",
                ctx->device_id, name, active_batch);
    return true;
}

} // namespace

bool gpu_verify_parity_suite(GPUContext* ctx) {
    uint8_t original_work[48] = {};
    std::memcpy(original_work, ctx->work, 48);

    uint8_t synthetic[48];
    for (int i = 0; i < 48; ++i) synthetic[i] = (uint8_t)(i * 7 + 13);
    synthetic[0] = 0x71;
    const uint32_t test_nonce = 42;
    synthetic[43] = (uint8_t)(test_nonce >> 24);
    synthetic[44] = (uint8_t)(test_nonce >> 16);
    synthetic[45] = (uint8_t)(test_nonce >> 8);
    synthetic[46] = (uint8_t)test_nonce;
    synthetic[47] = 0;

    bool ok = gpu_verify_case(ctx, "synthetic", synthetic);
    for (const auto& tc : kParityCases) {
        uint8_t work[48] = {};
        if (!gpu_decode_hex_48(tc.work_hex, work)) {
            std::printf("[VERIFY][GPU %d] %s | invalid fixture\n", ctx->device_id, tc.name);
            ok = false;
            continue;
        }
        ok = gpu_verify_case(ctx, tc.name, work) && ok;
    }
    {
        uint8_t work[48] = {};
        if (!gpu_decode_hex_48(kRealDaemonRangeWorkHex, work)) {
            std::printf("[VERIFY][GPU %d] real-daemon-range | invalid fixture\n", ctx->device_id);
            ok = false;
        } else {
            ok = gpu_verify_range_case(ctx, "real-daemon-range", work, ctx->batch_size) && ok;
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_work_template, original_work, 48));
    return ok;
}

static void gpu_make_synthetic_work(uint8_t work[48]) {
    for (int i = 0; i < 48; ++i) work[i] = (uint8_t)(i * 7 + 13);
    work[0] = 0x71;
    const uint32_t test_nonce = 42;
    work[43] = (uint8_t)(test_nonce >> 24);
    work[44] = (uint8_t)(test_nonce >> 16);
    work[45] = (uint8_t)(test_nonce >> 8);
    work[46] = (uint8_t)test_nonce;
    work[47] = 0;
}

bool gpu_verify_recovered_parity_suite(GPUContext* ctx) {
    uint8_t original_work[48] = {};
    std::memcpy(original_work, ctx->work, 48);

    uint8_t synthetic[48];
    gpu_make_synthetic_work(synthetic);

    bool ok = gpu_verify_recovered_case(ctx, "synthetic", synthetic);
    for (const auto& tc : kParityCases) {
        uint8_t work[48] = {};
        if (!gpu_decode_hex_48(tc.work_hex, work)) {
            std::printf("[VERIFY][GPU %d][RECOVERED] %s | invalid fixture\n", ctx->device_id, tc.name);
            ok = false;
            continue;
        }
        ok = gpu_verify_recovered_case(ctx, tc.name, work) && ok;
    }
    {
        uint8_t work[48] = {};
        if (!gpu_decode_hex_48(kRealDaemonRangeWorkHex, work)) {
            std::printf("[VERIFY][GPU %d][RECOVERED] real-daemon-range | invalid fixture\n", ctx->device_id);
            ok = false;
        } else {
            ok = gpu_verify_recovered_range_case(ctx, "real-daemon-range", work, ctx->batch_size) && ok;
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_work_template, original_work, 48));
    return ok;
}

bool gpu_verify_fast_parity_suite(GPUContext* ctx) {
    return gpu_verify_recovered_parity_suite(ctx);
}

bool gpu_verify_staged_parity_suite(GPUContext* ctx) {
    uint8_t original_work[48] = {};
    std::memcpy(original_work, ctx->work, 48);

    uint8_t synthetic[48];
    for (int i = 0; i < 48; ++i) synthetic[i] = (uint8_t)(i * 7 + 13);
    synthetic[0] = 0x71;
    const uint32_t test_nonce = 42;
    synthetic[43] = (uint8_t)(test_nonce >> 24);
    synthetic[44] = (uint8_t)(test_nonce >> 16);
    synthetic[45] = (uint8_t)(test_nonce >> 8);
    synthetic[46] = (uint8_t)test_nonce;
    synthetic[47] = 0;

    bool ok = gpu_verify_staged_case(ctx, "synthetic", synthetic);
    for (const auto& tc : kParityCases) {
        uint8_t work[48] = {};
        if (!gpu_decode_hex_48(tc.work_hex, work)) {
            std::printf("[VERIFY][GPU %d][STAGED] %s | invalid fixture\n", ctx->device_id, tc.name);
            ok = false;
            continue;
        }
        ok = gpu_verify_staged_case(ctx, tc.name, work) && ok;
    }
    {
        uint8_t work[48] = {};
        if (!gpu_decode_hex_48(kRealDaemonRangeWorkHex, work)) {
            std::printf("[VERIFY][GPU %d][STAGED] real-daemon-range | invalid fixture\n", ctx->device_id);
            ok = false;
        } else {
            const int range_batch = std::min(ctx->batch_size, std::max(96, ctx->staged_subbatch * 3));
            ok = gpu_verify_staged_range_case(ctx, "real-daemon-range", work, range_batch) && ok;
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_work_template, original_work, 48));
    return ok;
}

bool gpu_verify_parity_smoke_suite(GPUContext* ctx) {
    uint8_t original_work[48] = {};
    std::memcpy(original_work, ctx->work, 48);

    uint8_t synthetic[48];
    gpu_make_synthetic_work(synthetic);
    bool ok = gpu_verify_case(ctx, "synthetic-smoke", synthetic);

    if (ok) {
        uint8_t work[48] = {};
        if (!gpu_decode_hex_48(kParityCases[0].work_hex, work)) {
            std::printf("[VERIFY][GPU %d] smoke-fixture | invalid fixture\n", ctx->device_id);
            ok = false;
        } else {
            ok = gpu_verify_case(ctx, kParityCases[0].name, work) && ok;
        }
    }
    if (ok) {
        uint8_t work[48] = {};
        if (!gpu_decode_hex_48(kRealDaemonRangeWorkHex, work)) {
            std::printf("[VERIFY][GPU %d] smoke-range | invalid fixture\n", ctx->device_id);
            ok = false;
        } else {
            ok = gpu_verify_range_case(ctx, "startup-range", work, std::min(ctx->batch_size, 8)) && ok;
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_work_template, original_work, 48));
    return ok;
}

bool gpu_verify_recovered_smoke_suite(GPUContext* ctx) {
    uint8_t original_work[48] = {};
    std::memcpy(original_work, ctx->work, 48);

    uint8_t synthetic[48];
    gpu_make_synthetic_work(synthetic);
    bool ok = gpu_verify_recovered_case(ctx, "synthetic-smoke", synthetic);

    if (ok) {
        uint8_t work[48] = {};
        if (!gpu_decode_hex_48(kParityCases[0].work_hex, work)) {
            std::printf("[VERIFY][GPU %d][RECOVERED] smoke-fixture | invalid fixture\n", ctx->device_id);
            ok = false;
        } else {
            ok = gpu_verify_recovered_case(ctx, kParityCases[0].name, work) && ok;
        }
    }
    if (ok) {
        uint8_t work[48] = {};
        if (!gpu_decode_hex_48(kRealDaemonRangeWorkHex, work)) {
            std::printf("[VERIFY][GPU %d][RECOVERED] smoke-range | invalid fixture\n", ctx->device_id);
            ok = false;
        } else {
            ok = gpu_verify_recovered_range_case(ctx, "startup-range", work, std::min(ctx->batch_size, 8)) && ok;
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_work_template, original_work, 48));
    return ok;
}

bool gpu_verify_staged_smoke_suite(GPUContext* ctx) {
    uint8_t original_work[48] = {};
    std::memcpy(original_work, ctx->work, 48);

    uint8_t synthetic[48];
    gpu_make_synthetic_work(synthetic);
    bool ok = gpu_verify_staged_case(ctx, "synthetic-smoke", synthetic);

    if (ok) {
        uint8_t work[48] = {};
        if (!gpu_decode_hex_48(kParityCases[0].work_hex, work)) {
            std::printf("[VERIFY][GPU %d][STAGED] smoke-fixture | invalid fixture\n", ctx->device_id);
            ok = false;
        } else {
            ok = gpu_verify_staged_case(ctx, kParityCases[0].name, work) && ok;
        }
    }
    if (ok) {
        uint8_t work[48] = {};
        if (!gpu_decode_hex_48(kRealDaemonRangeWorkHex, work)) {
            std::printf("[VERIFY][GPU %d][STAGED] smoke-range | invalid fixture\n", ctx->device_id);
            ok = false;
        } else {
            ok = gpu_verify_staged_range_case(ctx, "startup-range", work, std::min(ctx->batch_size, 8)) && ok;
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_work_template, original_work, 48));
    return ok;
}

void gpu_verify_hashes(GPUContext* ctx) {
    std::printf("[VERIFY] Running GPU parity suite...\n");
    const bool ok = gpu_verify_parity_suite(ctx);
    std::printf("[VERIFY] GPU parity %s\n\n", ok ? "PASS" : "FAIL");
}

std::vector<GPUDeviceInfo> gpu_enumerate() {
    int count = 0; cudaGetDeviceCount(&count);
    std::vector<GPUDeviceInfo> gpus;
    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, i);
        GPUDeviceInfo info;
        info.device_id = i;
        info.name = prop.name;
        info.total_mem = prop.totalGlobalMem;
        info.compute_major = prop.major;
        info.compute_minor = prop.minor;
        info.sm_count = prop.multiProcessorCount;
        info.mem_clock_khz = prop.memoryClockRate;
        char bus[32] = {0};
        if (cudaDeviceGetPCIBusId(bus, sizeof(bus), i) == cudaSuccess) info.pci_bus_id = bus;
        size_t fm = 0, tm = 0;
        if (cudaSetDevice(i) == cudaSuccess && cudaMemGetInfo(&fm, &tm) == cudaSuccess) info.free_mem = fm;
        gpus.push_back(info);
    }
    return gpus;
}
