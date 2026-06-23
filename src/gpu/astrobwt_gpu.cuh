#pragma once
// GPU miner interface for AstroBWT v3

#include <cstdint>
#include <vector>
#include <string>

struct GPUDeviceInfo {
    int device_id;
    std::string name;
    size_t total_mem;
    int compute_major, compute_minor;
    int sm_count;
    std::string pci_bus_id;   // e.g. "0000:01:00.0"
    size_t free_mem = 0;      // free VRAM at enumeration time
    int mem_clock_khz = 0;    // memory clock (kHz)
};

enum class GPUEngineMode : int {
    Exact = 0,
    Recovered = 1,
    Staged = 2,
    Cleanroom = 3,
};

struct GPUMinerConfig {
    int device_id = 0;
    int batch_size = 0;  // 0 = auto
    int block_size = 128;
    GPUEngineMode engine_mode = GPUEngineMode::Exact;
    int staged_subbatch = 32;
    bool perf_logging = false;
};

struct GPUSolution {
    uint32_t nonce;
    uint8_t  hash[32];
};

// Initialize CUDA, return list of available GPUs
std::vector<GPUDeviceInfo> gpu_enumerate();

// Allocate GPU resources for mining
struct GPUContext;
GPUContext* gpu_create_context(const GPUMinerConfig& config);
void gpu_destroy_context(GPUContext* ctx);

// Set work template (112 bytes) and difficulty
void gpu_set_work(GPUContext* ctx, const uint8_t work[112], uint64_t difficulty);

// Mine a batch of nonces. Returns solutions found.
// nonce_start: first nonce to try
// Returns number of solutions found, fills solutions vector.
int gpu_mine_batch(GPUContext* ctx, uint32_t nonce_start,
                   std::vector<GPUSolution>& solutions);

// Get batch size for this context
int gpu_get_batch_size(GPUContext* ctx);

// Verify GPU hashes match CPU reference (call once at startup)
void gpu_verify_hashes(GPUContext* ctx);

// Run the exact multi-case GPU parity suite against the CPU oracle.
bool gpu_verify_parity_suite(GPUContext* ctx);

// Run the optimized fast-path parity suite against the exact oracle.
bool gpu_verify_recovered_parity_suite(GPUContext* ctx);

// Deprecated alias for the recovered fast-path parity suite.
bool gpu_verify_fast_parity_suite(GPUContext* ctx);

// Run the staged sub-batch GPU path against the exact oracle.
bool gpu_verify_staged_parity_suite(GPUContext* ctx);

// Run cheap startup smoke checks instead of the full corpus.
bool gpu_verify_parity_smoke_suite(GPUContext* ctx);
bool gpu_verify_recovered_smoke_suite(GPUContext* ctx);
bool gpu_verify_staged_smoke_suite(GPUContext* ctx);
