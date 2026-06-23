#pragma once
// Salsa20 stream cipher — self-contained implementation.
// Matches DirtyBird ucstk::Salsa20 behavior exactly.

#include <cstdint>
#include <cstring>

namespace salsa20 {

inline uint32_t rotl(uint32_t v, int n) { return (v << n) | (v >> (32 - n)); }

inline void quarter_round(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    b ^= rotl(a + d, 7);
    c ^= rotl(b + a, 9);
    d ^= rotl(c + b, 13);
    a ^= rotl(d + c, 18);
}

inline uint32_t le32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}

inline void le32_put(uint8_t* p, uint32_t v) {
    p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); p[2]=(uint8_t)(v>>16); p[3]=(uint8_t)(v>>24);
}

// Salsa20 core: 20-round doubleround on 16 x uint32 state
inline void core(const uint32_t input[16], uint8_t output[64]) {
    uint32_t x[16];
    for (int i = 0; i < 16; i++) x[i] = input[i];

    for (int i = 0; i < 10; i++) { // 20 rounds = 10 doublerounds
        // Column rounds
        quarter_round(x[ 0], x[ 4], x[ 8], x[12]);
        quarter_round(x[ 5], x[ 9], x[13], x[ 1]);
        quarter_round(x[10], x[14], x[ 2], x[ 6]);
        quarter_round(x[15], x[ 3], x[ 7], x[11]);
        // Row rounds
        quarter_round(x[ 0], x[ 1], x[ 2], x[ 3]);
        quarter_round(x[ 5], x[ 6], x[ 7], x[ 4]);
        quarter_round(x[10], x[11], x[ 8], x[ 9]);
        quarter_round(x[15], x[12], x[13], x[14]);
    }

    for (int i = 0; i < 16; i++) le32_put(&output[i*4], x[i] + input[i]);
}

// Encrypt/decrypt len bytes. key=32 bytes, iv=8 bytes (nonce).
// For AstroBWT: key = SHA256 output, iv = first 8 bytes of scratch (zeros)
inline void xor_stream(const uint8_t key[32], const uint8_t iv[8],
                       const uint8_t* in, uint8_t* out, size_t len) {
    // "expand 32-byte k" constants
    static const uint8_t sigma[16] = {
        'e','x','p','a','n','d',' ','3','2','-','b','y','t','e',' ','k'
    };

    uint32_t state[16];
    state[ 0] = le32(&sigma[ 0]);
    state[ 1] = le32(&key[ 0]);
    state[ 2] = le32(&key[ 4]);
    state[ 3] = le32(&key[ 8]);
    state[ 4] = le32(&key[12]);
    state[ 5] = le32(&sigma[ 4]);
    state[ 6] = le32(&iv[0]);
    state[ 7] = le32(&iv[4]);
    state[ 8] = 0; // block counter low
    state[ 9] = 0; // block counter high
    state[10] = le32(&sigma[ 8]);
    state[11] = le32(&key[16]);
    state[12] = le32(&key[20]);
    state[13] = le32(&key[24]);
    state[14] = le32(&key[28]);
    state[15] = le32(&sigma[12]);

    uint8_t block[64];
    size_t off = 0;
    while (off < len) {
        core(state, block);
        size_t n = (len - off > 64) ? 64 : (len - off);
        for (size_t i = 0; i < n; i++) {
            out[off + i] = in[off + i] ^ block[i];
        }
        off += n;
        state[8]++;
        if (state[8] == 0) state[9]++;
    }
}

} // namespace salsa20
