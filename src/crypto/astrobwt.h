#pragma once
// AstroBWT v3 — Complete PoW algorithm for DERO mining.
// Produces byte-identical hashes to the reference DERO implementation.

#include <cstdint>
#include <cstring>
#include <vector>
#include "sha256.h"
#include "salsa20.h"
#include "rc4.h"
#include "hash_utils.h"
#include "sais.h"
extern "C" {
#include "libsais.h"
}

namespace astrobwt {

enum class TraceExitReason : uint8_t {
    none = 0,
    max_iterations = 1,
    chunk_255_threshold = 2,
};

struct TraceIteration {
    int try_index = 0;
    uint64_t random_switcher = 0;
    uint8_t op = 0;
    uint8_t pos1 = 0;
    uint8_t pos2 = 0;
    uint8_t a = 0;
    uint8_t chunk_255 = 0;
    uint64_t lhash = 0;
    uint64_t prev_lhash = 0;
    TraceExitReason exit_reason = TraceExitReason::none;
    uint8_t chunk_digest[32] = {};
};

struct HashTrace {
    uint8_t sha256[32] = {};
    uint8_t salsa[256] = {};
    uint8_t rc4[256] = {};
    uint64_t initial_lhash = 0;
    uint64_t initial_prev_lhash = 0;
    int data_len = 0;
    uint8_t sdata_sha256[32] = {};
    uint8_t sa_sha256[32] = {};
    std::vector<TraceIteration> iterations;
};

// ============================================================================
// Constants
// ============================================================================
static constexpr int MINIBLOCK_SIZE     = 48;             // DERO v3 hashing blob size
static constexpr int MAX_LENGTH         = 256 * 277 - 1;  // 70911
static constexpr int SCRATCH_SIZE       = MAX_LENGTH + 64; // 70975
static constexpr int MAX_ITERATIONS     = 278;
static constexpr int SALSA20_OUTPUT_LEN = 256;
static constexpr int RC4_KEY_LEN        = 256;

// ============================================================================
// CodeLUT — maps op (0-255) to 4 chained micro-operations (32-bit encoding)
// Extracted from DirtyBird wolfbranching.cpp (Wolf9466 contribution)
// ============================================================================
alignas(32) static const uint32_t CodeLUT[257] = {
    0x090F020A, 0x060B0500, 0x09080609, 0x0A0D030B, 0x04070A01, 0x09030607, 0x060D0401, 0x000A0904,
    0x040F0F06, 0x030E070C, 0x04020D02, 0x0B0F050A, 0x0C020C04, 0x0B03070F, 0x07060206, 0x0C060501,
    0x0E020B04, 0x03020F04, 0x0E0D0B0F, 0x010F0600, 0x0503080C, 0x0B030005, 0x0608020B, 0x0D0B0905,
    0x00070E0F, 0x090D0A01, 0x02090008, 0x0F050E0F, 0x0600000F, 0x02030700, 0x050E0F06, 0x040C0602,
    0x0C080D0C, 0x0A0E0802, 0x01060601, 0x00040B03, 0x090B0C0B, 0x0A070702, 0x070D090A, 0x0C030705,
    0x0A030903, 0x0F010D0E, 0x0B0D0C0A, 0x05000501, 0x09090D0A, 0x0F0F0509, 0x09000F0E, 0x0F050F06,
    0x0A04040F, 0x0900080E, 0x080D000B, 0x030E0E0F, 0x0A070409, 0x00090E0E, 0x08030404, 0x080E0E0B,
    0x0C02040B, 0x0A0F0D08, 0x080C0500, 0x0B020A04, 0x0304020D, 0x0F060D0F, 0x05040C00, 0x0F090100,
    0x03080E02, 0x0F0D0C02, 0x0C080E0B, 0x0B090C0F, 0x05040E03, 0x00020807, 0x0302070E, 0x0F040206,
    0x08090306, 0x09080F01, 0x020D0805, 0x0209050E, 0x0A0C0F07, 0x0D000609, 0x0A080201, 0x0E0C0002,
    0x0A060005, 0x0E060A09, 0x03040407, 0x06080D08, 0x010B0600, 0x07030A06, 0x0E0A0E04, 0x000D0E00,
    0x0C0B0204, 0x0002040C, 0x080F0B07, 0x09050E08, 0x09040905, 0x0C020500, 0x0B0A0506, 0x0B040F0F,
    0x0C0C090B, 0x0B060907, 0x0E06070E, 0x0E010807, 0x0A060809, 0x07090704, 0x0D01000D, 0x0B08030A,
    0x08090F00, 0x060D0A0C, 0x080E0B02, 0x070C0F0B, 0x0304050C, 0x020A030C, 0x000C0C07, 0x02080207,
    0x0D040F01, 0x0F0B0904, 0x0B080A04, 0x0A0F050D, 0x05030906, 0x060D0605, 0x0700060F, 0x080C0403,
    0x0C020308, 0x07000902, 0x0E0A0F0C, 0x05040D0D, 0x0C0C0304, 0x080C0007, 0x0D0B0F08, 0x06020503,
    0x0A0C0C0F, 0x04090907, 0x070A0B0E, 0x010B0902, 0x05080F0C, 0x030F0C06, 0x040E0B05, 0x070C0008,
    0x0701030F, 0x0F07080A, 0x03030001, 0x0F0D0C0D, 0x0B0C030F, 0x0B010900, 0x050F080C, 0x050D0706,
    0x0A06040A, 0x080E0C0E, 0x05060509, 0x04060E02, 0x050F0601, 0x03080100, 0x06060605, 0x00060206,
    0x0704060C, 0x0B0D0404, 0x0F040309, 0x01030903, 0x07070D0B, 0x07060A0B, 0x090D000B, 0x01030A03,
    0x07080B0D, 0x03030F0A, 0x02080C01, 0x06010E0B, 0x02090104, 0x0E030600, 0x0D000C04, 0x04040207,
    0x0A050A0B, 0x0B060E05, 0x01080102, 0x0D010908, 0x0E01060B, 0x04060200, 0x040A0909, 0x0D01020F,
    0x0302030F, 0x090C0C05, 0x0500040B, 0x0C000708, 0x070E0301, 0x04060C0F, 0x030B0F0E, 0x00010102,
    0x06020F03, 0x040E0F07, 0x0C0E0107, 0x0304000D, 0x0E090E0E, 0x0F0E0301, 0x0F07050C, 0x000D0A07,
    0x00060002, 0x05060A0B, 0x050A0605, 0x090C030E, 0x0D08060B, 0x0E0A0202, 0x0707080B, 0x04000203,
    0x07090808, 0x0D0C0E04, 0x03040A0F, 0x03050B0A, 0x0F0C0A03, 0x090E0600, 0x0E080809, 0x0F0D0909,
    0x0000070D, 0x0F080901, 0x0C0A0F04, 0x0E00010A, 0x0A0C0303, 0x00060D01, 0x03010704, 0x03050602,
    0x0A040105, 0x0F000B0E, 0x08040201, 0x0E0D0508, 0x0B060806, 0x0F030408, 0x07060302, 0x0D030A01,
    0x0C0B0D06, 0x0407080D, 0x08010203, 0x04060105, 0x00070009, 0x0D0A0C09, 0x02050A0A, 0x0D070308,
    0x02020E0F, 0x0B090D09, 0x05020703, 0x0C020D04, 0x03000501, 0x0F060C0D, 0x00000D01, 0x0F0B0205,
    0x04000506, 0x0E09030B, 0x00000103, 0x0F0C090B, 0x040C080F, 0x010F0C07, 0x000B0700, 0x0F0C0F04,
    0x0401090F, 0x080E0E0A, 0x050A090E, 0x0009080C, 0x080E0C06, 0x0D0C030D, 0x090D0C0D, 0x090D0C0D,
    0x00000000 // sentinel (index 256)
};

// ============================================================================
// Bit manipulation helpers
// ============================================================================
static inline uint8_t rl8(uint8_t x, int y) {
    int s = y & 7;
    return (uint8_t)((x << s) | (x >> (8 - s)));
}

static inline uint8_t reverse8(uint8_t b) {
    b = (uint8_t)(((b & 0xAA) >> 1) | ((b & 0x55) << 1));
    b = (uint8_t)(((b & 0xCC) >> 2) | ((b & 0x33) << 2));
    b = (uint8_t)(((b & 0xF0) >> 4) | ((b & 0x0F) << 4));
    return b;
}

static inline int popcount8(uint8_t v) {
    v = (uint8_t)((v & 0x55) + ((v >> 1) & 0x55));
    v = (uint8_t)((v & 0x33) + ((v >> 2) & 0x33));
    return (v + (v >> 4)) & 0x0F;
}

// ============================================================================
// wolfBranch — apply 4 chained micro-operations to a single byte
// ============================================================================
static inline uint8_t wolfBranch(uint8_t val, uint8_t pos2_val, uint32_t opcode) {
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
            case 8:  val = reverse8(val); break;
            case 9:  val = val ^ (uint8_t)popcount8(val); break;
            case 10: val = rl8(val, val); break;
            case 11: val = rl8(val, 1); break;
            case 12: val = val ^ rl8(val, 2); break;
            case 13: val = rl8(val, 3); break;
            case 14: val = val ^ rl8(val, 4); break;
            case 15: val = rl8(val, 5); break;
        }
    }
    return val;
}

// ============================================================================
// wolfPermute — apply branch operation to byte range [pos1, pos2)
// ============================================================================
static inline void wolfPermute(const uint8_t* in, uint8_t* out,
                                uint8_t op, uint8_t pos1, uint8_t pos2) {
    uint32_t opcode = CodeLUT[op];
    uint8_t pos2_val = in[pos2];
    for (int i = pos1; i < pos2; ++i) {
        out[i] = wolfBranch(in[i], pos2_val, opcode);
    }
}

// ============================================================================
// Worker state for AstroBWT v3
// ============================================================================
struct WorkerState {
    uint8_t  sData[SCRATCH_SIZE + 256]; // Branch compute output (~71KB)
    int32_t  sa[MAX_LENGTH + 2];        // Suffix array output
    uint8_t  bwt[MAX_LENGTH + 2];       // BWT string output
    uint8_t  scratch[384];              // SHA256 + Salsa20 + RC4 working buffer
    rc4::State rc4_state;               // RC4 key state
    int      data_len;                  // Final data length after branch compute
};

// ============================================================================
// AstroBWT v3 — Complete PoW hash computation
// ============================================================================
// Input:  input[0..inputLen-1] = miniblock (48 bytes)
// Output: output[0..31] = 32-byte SHA256 hash
static inline void hash_with_trace(const uint8_t* input, int inputLen, uint8_t output[32],
                                   WorkerState& worker, HashTrace* trace) {
    std::memset(worker.sData, 0, sizeof(worker.sData));

    // Step 1.1: SHA256 of input
    sha256::hash(input, inputLen, &worker.scratch[320]);
    if (trace) {
        std::memcpy(trace->sha256, &worker.scratch[320], 32);
    }

    // Step 1.2: Salsa20 keystream (key = SHA256 output, IV = 8 zero bytes)
    uint8_t zero_iv[8] = {0};
    uint8_t zero_input[256];
    std::memset(zero_input, 0, 256);
    salsa20::xor_stream(&worker.scratch[320], zero_iv, zero_input, worker.scratch, 256);
    if (trace) {
        std::memcpy(trace->salsa, worker.scratch, 256);
    }

    // Step 1.3: RC4 KSA + generate 256 bytes of keystream (in-place)
    rc4::set_key(worker.rc4_state, worker.scratch, 256);
    rc4::process(worker.rc4_state, worker.scratch, worker.scratch, 256);
    if (trace) {
        std::memcpy(trace->rc4, worker.scratch, 256);
    }

    // Step 1.4: Copy to sData and compute FNV-1a hash
    std::memcpy(worker.sData, worker.scratch, 256);
    uint64_t lhash = fnv1a::hash_256(worker.scratch);
    uint64_t prev_lhash = lhash;
    if (trace) {
        trace->initial_lhash = lhash;
        trace->initial_prev_lhash = prev_lhash;
        trace->iterations.clear();
        trace->iterations.reserve(MAX_ITERATIONS);
    }

    int tries = 0;
    for (int iteration = 1; iteration <= MAX_ITERATIONS; ++iteration) {
        tries++;
        uint64_t random_switcher = prev_lhash ^ lhash ^ (uint64_t)tries;
        uint8_t op = (uint8_t)(random_switcher);
        uint8_t p1 = (uint8_t)(random_switcher >> 8);
        uint8_t p2 = (uint8_t)(random_switcher >> 16);
        if (p1 > p2) { uint8_t t = p1; p1 = p2; p2 = t; }
        if (p2 - p1 > 32) p2 = p1 + ((p2 - p1) & 0x1f);
        uint8_t* chunk = &worker.sData[(tries - 1) * 256];
        uint8_t* prev_chunk;
        if (tries == 1) prev_chunk = chunk;
        else { prev_chunk = &worker.sData[(tries - 2) * 256]; std::memcpy(chunk, prev_chunk, 256); }
        uint8_t A = 0;
        if (op == 253) {
            for (int i = p1; i < p2; i++) {
                chunk[i] = rl8(chunk[i], 3); chunk[i] ^= rl8(chunk[i], 2);
                chunk[i] ^= prev_chunk[p2]; chunk[i] = rl8(chunk[i], 3);
                prev_lhash = lhash + prev_lhash;
                lhash = xxhash64::hash(chunk, p2, 0);
            }
        } else {
            if (op >= 254) rc4::set_key(worker.rc4_state, prev_chunk, 256);
            wolfPermute(prev_chunk, chunk, op, p1, p2);
            if (op == 0 && ((p2 - p1) & 1) == 1) {
                uint8_t t1 = chunk[p1]; uint8_t t2 = chunk[p2];
                chunk[p1] = reverse8(t2); chunk[p2] = reverse8(t1);
            }
        }
        A = (uint8_t)((256 + ((chunk[p1] - chunk[p2]) % 256)) % 256);
        if (A < 0x30) {
            if (A < 0x10) { prev_lhash = lhash + prev_lhash; lhash = xxhash64::hash(chunk, p2, 0); }
            if (A < 0x20) { prev_lhash = lhash + prev_lhash; lhash = fnv1a::hash(chunk, p2); }
            prev_lhash = lhash + prev_lhash;
            uint64_t sip_key[2] = { (uint64_t)tries, prev_lhash };
            lhash = siphash::hash(sip_key, chunk, p2);
        }
        if (A <= 0x40) rc4::process(worker.rc4_state, chunk, chunk, 256);
        chunk[255] ^= chunk[p1] ^ chunk[p2];

        TraceExitReason exit_reason = TraceExitReason::none;
        if (tries > 260 + 16) exit_reason = TraceExitReason::max_iterations;
        if (worker.sData[(tries - 1) * 256 + 255] >= 0xf0 && tries > 260) exit_reason = TraceExitReason::chunk_255_threshold;
        if (trace) {
            TraceIteration iter_trace{};
            iter_trace.try_index = tries;
            iter_trace.random_switcher = random_switcher;
            iter_trace.op = op;
            iter_trace.pos1 = p1;
            iter_trace.pos2 = p2;
            iter_trace.a = A;
            iter_trace.chunk_255 = chunk[255];
            iter_trace.lhash = lhash;
            iter_trace.prev_lhash = prev_lhash;
            iter_trace.exit_reason = exit_reason;
            sha256::hash(chunk, 256, iter_trace.chunk_digest);
            trace->iterations.push_back(iter_trace);
        }

        if (exit_reason != TraceExitReason::none) break;
    }

    uint8_t* last_chunk = &worker.sData[(tries - 1) * 256];
    worker.data_len = (tries - 4) * 256 + (((last_chunk[253] << 8) | last_chunk[254]) & 0x3ff);
    if (worker.data_len < 256) worker.data_len = 256;
    if (worker.data_len > MAX_LENGTH) worker.data_len = MAX_LENGTH;
    std::memset(worker.sData + worker.data_len, 0, 16);
    if (trace) {
        trace->data_len = worker.data_len;
        sha256::hash(worker.sData, worker.data_len, trace->sdata_sha256);
    }

    // Current official DERO v3 uses suffix-array construction semantics compatible
    // with libsais/divsufsort for the final ordering stage.
    if (libsais(worker.sData, worker.sa, worker.data_len, 0, nullptr) != 0) {
        sais::build(worker.sData, worker.sa, worker.data_len);
    }

    // Current official DERO v3 finalizes by hashing the raw suffix-array bytes.
    sha256::hash(reinterpret_cast<const uint8_t*>(worker.sa),
                 worker.data_len * static_cast<int>(sizeof(worker.sa[0])),
                 output);
    if (trace) {
        std::memcpy(trace->sa_sha256, output, 32);
    }
}

static inline void hash(const uint8_t* input, int inputLen, uint8_t output[32],
                        WorkerState& worker) {
    hash_with_trace(input, inputLen, output, worker, nullptr);
}

// ============================================================================
// Difficulty check: is hash <= target?
// ============================================================================
// Compute target = 2^256 / difficulty as 32 big-endian bytes
static inline void compute_target(uint64_t difficulty, uint8_t target[32]) {
    std::memset(target, 0xFF, 32);
    if (difficulty <= 1) return;

    uint64_t result[4] = {0};
    uint64_t remainder = 1;
    for (int i = 0; i < 4; i++) {
        uint64_t q = 0;
        for (int bit = 63; bit >= 0; bit--) {
            bool overflow = (remainder >> 63) != 0;
            remainder <<= 1;
            if (overflow || remainder >= difficulty) {
                remainder -= difficulty;
                q |= (1ULL << bit);
            }
        }
        result[i] = q;
    }
    for (int i = 0; i < 4; i++) {
        uint64_t w = result[i];
        for (int b = 0; b < 8; b++)
            target[i * 8 + b] = (uint8_t)(w >> (56 - b * 8));
    }
}

// Check: is hash (big-endian from SHA256) <= target (big-endian)?
static inline bool check_difficulty(const uint8_t hash[32], uint64_t difficulty) {
    if (difficulty == 0) return false;
    uint8_t target[32];
    compute_target(difficulty, target);
    for (int i = 0; i < 32; i++) {
        uint8_t h = hash[i], t = target[i];
        if (h != t) return h < t;
    }
    return true; // equal
}

} // namespace astrobwt
