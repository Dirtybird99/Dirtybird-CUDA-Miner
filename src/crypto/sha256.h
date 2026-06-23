#pragma once
// SHA-256 implementation (FIPS 180-4) — fully self-contained, no dependencies.
// Produces byte-identical output to OpenSSL EVP_sha256.

#include <cstdint>
#include <cstring>

namespace sha256 {

static constexpr uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
inline uint32_t sigma0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
inline uint32_t sigma1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
inline uint32_t gamma0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
inline uint32_t gamma1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

inline uint32_t be32(const uint8_t* p) {
    return ((uint32_t)p[0]<<24)|((uint32_t)p[1]<<16)|((uint32_t)p[2]<<8)|p[3];
}
inline void be32_put(uint8_t* p, uint32_t v) {
    p[0]=(uint8_t)(v>>24); p[1]=(uint8_t)(v>>16); p[2]=(uint8_t)(v>>8); p[3]=(uint8_t)v;
}

struct Context {
    uint32_t state[8];
    uint64_t count;
    uint8_t  buf[64];
};

inline void transform(Context& ctx) {
    uint32_t W[64], a,b,c,d,e,f,g,h;
    for (int i = 0; i < 16; i++) W[i] = be32(&ctx.buf[i*4]);
    for (int i = 16; i < 64; i++) W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];
    a=ctx.state[0]; b=ctx.state[1]; c=ctx.state[2]; d=ctx.state[3];
    e=ctx.state[4]; f=ctx.state[5]; g=ctx.state[6]; h=ctx.state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sigma1(e) + ch(e,f,g) + K[i] + W[i];
        uint32_t t2 = sigma0(a) + maj(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    ctx.state[0]+=a; ctx.state[1]+=b; ctx.state[2]+=c; ctx.state[3]+=d;
    ctx.state[4]+=e; ctx.state[5]+=f; ctx.state[6]+=g; ctx.state[7]+=h;
}

inline void init(Context& ctx) {
    ctx.state[0]=0x6a09e667; ctx.state[1]=0xbb67ae85;
    ctx.state[2]=0x3c6ef372; ctx.state[3]=0xa54ff53a;
    ctx.state[4]=0x510e527f; ctx.state[5]=0x9b05688c;
    ctx.state[6]=0x1f83d9ab; ctx.state[7]=0x5be0cd19;
    ctx.count = 0;
    std::memset(ctx.buf, 0, 64);
}

inline void update(Context& ctx, const uint8_t* data, size_t len) {
    size_t idx = ctx.count & 63;
    ctx.count += len;
    for (size_t i = 0; i < len; i++) {
        ctx.buf[idx++] = data[i];
        if (idx == 64) { transform(ctx); idx = 0; }
    }
}

inline void final(Context& ctx, uint8_t hash[32]) {
    uint64_t bits = ctx.count * 8;
    size_t idx = ctx.count & 63;
    ctx.buf[idx++] = 0x80;
    if (idx > 56) {
        while (idx < 64) ctx.buf[idx++] = 0;
        transform(ctx); idx = 0;
    }
    while (idx < 56) ctx.buf[idx++] = 0;
    for (int i = 7; i >= 0; i--) ctx.buf[56 + (7-i)] = (uint8_t)(bits >> (i*8));
    transform(ctx);
    for (int i = 0; i < 8; i++) be32_put(&hash[i*4], ctx.state[i]);
}

// One-shot convenience
inline void hash(const uint8_t* data, size_t len, uint8_t out[32]) {
    Context ctx;
    init(ctx);
    update(ctx, data, len);
    final(ctx, out);
}

} // namespace sha256
