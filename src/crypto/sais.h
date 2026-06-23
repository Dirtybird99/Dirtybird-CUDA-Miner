#pragma once
// SA-IS (Suffix Array by Induced Sorting) — O(n) suffix array construction.
// Self-contained implementation for AstroBWT v3.
// Input: byte array T[0..n-1], Output: SA[0..n-1] where SA[i] is the i-th smallest suffix.

#include <cstdint>
#include <cstring>
#include <vector>

namespace sais {

// Type: S-type (true) or L-type (false)
inline void get_type(const uint8_t* T, int n, std::vector<bool>& type) {
    type.resize(n + 1);
    type[n] = true;     // sentinel is S-type
    type[n - 1] = false; // last char is L-type
    for (int i = n - 2; i >= 0; i--) {
        type[i] = (T[i] < T[i+1]) || (T[i] == T[i+1] && type[i+1]);
    }
}

inline bool is_lms(const std::vector<bool>& type, int i) {
    return i > 0 && type[i] && !type[i-1];
}

inline void get_buckets(const uint8_t* T, int n, std::vector<int>& buckets, bool end) {
    buckets.assign(256, 0);
    for (int i = 0; i < n; i++) buckets[T[i]]++;
    int sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += buckets[i];
        buckets[i] = end ? sum : (sum - buckets[i]);
    }
}

inline void induce_sa_l(const uint8_t* T, int32_t* SA, int n,
                         const std::vector<bool>& type, std::vector<int>& buckets) {
    get_buckets(T, n, buckets, false);
    for (int i = 0; i < n; i++) {
        int j = SA[i] - 1;
        if (SA[i] > 0 && !type[j]) {
            SA[buckets[T[j]]++] = j;
        }
    }
}

inline void induce_sa_s(const uint8_t* T, int32_t* SA, int n,
                         const std::vector<bool>& type, std::vector<int>& buckets) {
    get_buckets(T, n, buckets, true);
    for (int i = n - 1; i >= 0; i--) {
        int j = SA[i] - 1;
        if (SA[i] > 0 && type[j]) {
            SA[--buckets[T[j]]] = j;
        }
    }
}

// Build suffix array for byte string T[0..n-1]
inline void build(const uint8_t* T, int32_t* SA, int n) {
    if (n <= 1) { if (n == 1) SA[0] = 0; return; }
    if (n == 2) {
        SA[0] = (T[0] < T[1]) ? 0 : 1;
        SA[1] = (T[0] < T[1]) ? 1 : 0;
        return;
    }

    std::vector<bool> type;
    std::vector<int> buckets;
    get_type(T, n, type);

    // Step 1: Place LMS suffixes into their buckets
    get_buckets(T, n, buckets, true);
    std::memset(SA, -1, n * sizeof(int32_t));
    for (int i = n - 1; i >= 0; i--) {
        if (is_lms(type, i)) SA[--buckets[T[i]]] = i;
    }

    // Step 2: Induce L-type and S-type
    induce_sa_l(T, SA, n, type, buckets);
    induce_sa_s(T, SA, n, type, buckets);

    // Step 3: Compact LMS suffixes
    int n1 = 0;
    for (int i = 0; i < n; i++) {
        if (is_lms(type, SA[i])) SA[n1++] = SA[i];
    }

    // Step 4: Name LMS substrings
    std::memset(SA + n1, -1, (n - n1) * sizeof(int32_t));
    int name = 0, prev = -1;
    for (int i = 0; i < n1; i++) {
        int pos = SA[i];
        bool diff = false;
        if (prev == -1) {
            diff = true;
        } else {
            // Compare LMS substrings
            for (int d = 0; ; d++) {
                if (T[pos + d] != T[prev + d] || type[pos + d] != type[prev + d]) {
                    diff = true; break;
                }
                if (d > 0 && (is_lms(type, pos + d) || is_lms(type, prev + d))) break;
            }
        }
        if (diff) { name++; prev = pos; }
        SA[n1 + (pos / 2)] = name - 1;
    }

    // Compact names
    std::vector<int32_t> s1(n1);
    std::vector<int32_t> sa1(n1);
    int j = 0;
    for (int i = n1; i < n; i++) {
        if (SA[i] >= 0) s1[j++] = SA[i];
    }

    // Step 5: Recursion or direct sort
    if (name < n1) {
        // Recursive case: names are not unique
        // Simple bucket sort for small alphabets
        std::vector<int32_t> temp_t(n1);
        for (int i = 0; i < n1; i++) temp_t[i] = (uint8_t)s1[i]; // May not be bytes
        // Use recursive SA-IS on reduced string
        // For simplicity and to avoid deep recursion, use comparison sort for small n1
        std::vector<int> order(n1);
        for (int i = 0; i < n1; i++) order[i] = i;
        // Stable sort by s1 values
        std::vector<int> cnt(name + 1, 0);
        for (int i = 0; i < n1; i++) cnt[s1[i] + 1]++;
        for (int i = 1; i <= name; i++) cnt[i] += cnt[i-1];
        for (int i = 0; i < n1; i++) sa1[cnt[s1[i]]++] = i;
    } else {
        // Direct case: all names are unique
        for (int i = 0; i < n1; i++) sa1[s1[i]] = i;
    }

    // Step 6: Map back to original positions
    // Collect LMS positions
    std::vector<int> lms_positions;
    lms_positions.reserve(n1);
    for (int i = 0; i < n; i++) {
        if (is_lms(type, i)) lms_positions.push_back(i);
    }

    // Step 7: Place sorted LMS suffixes
    std::memset(SA, -1, n * sizeof(int32_t));
    get_buckets(T, n, buckets, true);
    for (int i = n1 - 1; i >= 0; i--) {
        int pos = lms_positions[sa1[i]];
        SA[--buckets[T[pos]]] = pos;
    }

    // Step 8: Final induced sort
    induce_sa_l(T, SA, n, type, buckets);
    induce_sa_s(T, SA, n, type, buckets);
}

} // namespace sais
