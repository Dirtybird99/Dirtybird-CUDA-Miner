#pragma once
// FNV-1a 64-bit, XXHash64, and SipHash — needed for AstroBWT v3 branch compute.

#include <cstdint>
#include <cstring>

namespace fnv1a {
inline uint64_t hash_256(const uint8_t* data) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (int i = 0; i < 256; i++) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}
inline uint64_t hash(const uint8_t* data, int len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (int i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}
} // namespace fnv1a

namespace xxhash64 {
static constexpr uint64_t PRIME1 = 0x9E3779B185EBCA87ULL;
static constexpr uint64_t PRIME2 = 0xC2B2AE3D27D4EB4FULL;
static constexpr uint64_t PRIME3 = 0x165667B19E3779F9ULL;
static constexpr uint64_t PRIME4 = 0x85EBCA77C2B2AE63ULL;
static constexpr uint64_t PRIME5 = 0x27D4EB2F165667C5ULL;

inline uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
inline uint64_t read64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
inline uint32_t read32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint64_t round(uint64_t acc, uint64_t input) {
    return rotl(acc + input * PRIME2, 31) * PRIME1;
}

inline uint64_t hash(const uint8_t* data, int len, uint64_t seed) {
    const uint8_t* p = data;
    const uint8_t* end = data + len;
    uint64_t h64;

    if (len >= 32) {
        uint64_t v1 = seed + PRIME1 + PRIME2;
        uint64_t v2 = seed + PRIME2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - PRIME1;
        do {
            v1 = round(v1, read64(p)); p += 8;
            v2 = round(v2, read64(p)); p += 8;
            v3 = round(v3, read64(p)); p += 8;
            v4 = round(v4, read64(p)); p += 8;
        } while (p <= end - 32);
        h64 = rotl(v1,1) + rotl(v2,7) + rotl(v3,12) + rotl(v4,18);
        auto merge = [](uint64_t h, uint64_t v) {
            return (h ^ round(0, v)) * PRIME1 + PRIME4;
        };
        h64 = merge(h64, v1); h64 = merge(h64, v2);
        h64 = merge(h64, v3); h64 = merge(h64, v4);
    } else {
        h64 = seed + PRIME5;
    }
    h64 += (uint64_t)len;

    while (p + 8 <= end) {
        h64 = rotl(h64 ^ round(0, read64(p)), 27) * PRIME1 + PRIME4;
        p += 8;
    }
    while (p + 4 <= end) {
        h64 = rotl(h64 ^ (read32(p) * PRIME1), 23) * PRIME2 + PRIME3;
        p += 4;
    }
    while (p < end) {
        h64 = rotl(h64 ^ (*p++ * PRIME5), 11) * PRIME1;
    }

    h64 ^= h64 >> 33; h64 *= PRIME2;
    h64 ^= h64 >> 29; h64 *= PRIME3;
    h64 ^= h64 >> 32;
    return h64;
}
} // namespace xxhash64

namespace siphash {
inline uint64_t rotl(uint64_t x, int b) { return (x << b) | (x >> (64 - b)); }
inline uint64_t read64le(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// SipHash-2-4 with 128-bit key (two uint64s)
inline uint64_t hash(const uint64_t key[2], const uint8_t* data, int len) {
    uint64_t v0 = key[0] ^ 0x736f6d6570736575ULL;
    uint64_t v1 = key[1] ^ 0x646f72616e646f6dULL;
    uint64_t v2 = key[0] ^ 0x6c7967656e657261ULL;
    uint64_t v3 = key[1] ^ 0x7465646279746573ULL;

    auto sipround = [&]() {
        v0 += v1; v1 = rotl(v1, 13); v1 ^= v0; v0 = rotl(v0, 32);
        v2 += v3; v3 = rotl(v3, 16); v3 ^= v2;
        v0 += v3; v3 = rotl(v3, 21); v3 ^= v0;
        v2 += v1; v1 = rotl(v1, 17); v1 ^= v2; v2 = rotl(v2, 32);
    };

    int blocks = len / 8;
    for (int i = 0; i < blocks; i++) {
        uint64_t m = read64le(data + i * 8);
        v3 ^= m;
        sipround(); sipround();
        v0 ^= m;
    }

    uint64_t last = (uint64_t)len << 56;
    const uint8_t* tail = data + blocks * 8;
    switch (len & 7) {
        case 7: last |= (uint64_t)tail[6] << 48; [[fallthrough]];
        case 6: last |= (uint64_t)tail[5] << 40; [[fallthrough]];
        case 5: last |= (uint64_t)tail[4] << 32; [[fallthrough]];
        case 4: last |= (uint64_t)tail[3] << 24; [[fallthrough]];
        case 3: last |= (uint64_t)tail[2] << 16; [[fallthrough]];
        case 2: last |= (uint64_t)tail[1] << 8;  [[fallthrough]];
        case 1: last |= (uint64_t)tail[0];        break;
        case 0: break;
    }
    v3 ^= last;
    sipround(); sipround();
    v0 ^= last;

    v2 ^= 0xff;
    sipround(); sipround(); sipround(); sipround();
    return v0 ^ v1 ^ v2 ^ v3;
}
} // namespace siphash
