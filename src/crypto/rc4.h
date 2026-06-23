#pragma once
// RC4 stream cipher — matches OpenSSL RC4_set_key + RC4 behavior exactly.

#include <cstdint>

namespace rc4 {

struct State {
    uint8_t S[256];
    uint8_t i, j;
};

inline void set_key(State& state, const uint8_t* key, int keylen) {
    for (int i = 0; i < 256; i++) state.S[i] = (uint8_t)i;
    uint8_t j = 0;
    for (int i = 0; i < 256; i++) {
        j = j + state.S[i] + key[i % keylen];
        uint8_t t = state.S[i]; state.S[i] = state.S[j]; state.S[j] = t;
    }
    state.i = 0;
    state.j = 0;
}

inline void process(State& state, const uint8_t* in, uint8_t* out, int len) {
    uint8_t i = state.i, j = state.j;
    for (int k = 0; k < len; k++) {
        i++; j += state.S[i];
        uint8_t t = state.S[i]; state.S[i] = state.S[j]; state.S[j] = t;
        out[k] = in[k] ^ state.S[(uint8_t)(state.S[i] + state.S[j])];
    }
    state.i = i;
    state.j = j;
}

} // namespace rc4
