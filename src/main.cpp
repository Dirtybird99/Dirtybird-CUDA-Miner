#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE 1   /* ensure popen/pclose, localtime_r, isatty are declared under -std=c++17 */
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <csignal>
#include <iostream>
#include <fstream>
#include <array>
#include <algorithm>
#include <cctype>
#include <ctime>

#include "gpu/astrobwt_gpu.cuh"
#include "crypto/astrobwt.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET socket_t;
#define SOCKET_INVALID INVALID_SOCKET
#define CLOSE_SOCKET closesocket
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
typedef int socket_t;
#define SOCKET_INVALID -1
#define CLOSE_SOCKET close
#endif

#ifdef HAS_OPENSSL
#include <openssl/ssl.h>
#include <openssl/err.h>
#endif
#include <unistd.h>   // isatty() for --color auto

#ifdef _WIN32
#define COL_GREEN   ""
#define COL_YELLOW  ""
#define COL_RED     ""
#define COL_CYAN    ""
#define COL_RESET   ""
#else
#define COL_GREEN   "\033[32m"
#define COL_YELLOW  "\033[33m"
#define COL_RED     "\033[31m"
#define COL_CYAN    "\033[36m"
#define COL_RESET   "\033[0m"
#endif

// Runtime color gate for the periodic status UI (--color auto|always|never).
// auto = enabled when stdout is a TTY. Wrap codes with CC() so they vanish when
// color is off (e.g. piped to a log file).
static bool g_color_enabled = true;
static inline const char* CC(const char* code) { return g_color_enabled ? code : ""; }
#define UI_RESET  "\033[0m"
#define UI_DIM    "\033[2m"
#define UI_HR     "\033[36;1m"   /* bright cyan: the headline hashrate */
#define UI_GOOD   "\033[32m"
#define UI_BAD    "\033[31m"

// Miner identity shown in the startup banner.
#define MINER_NAME   "derocuda"
#define MINER_VER    "1.0"
#define MINER_AUTHOR "derocuda"

// "DD-MM-YYYY HH:MM:SS" timestamp prefix (astronv-style logs).
static std::string ts() {
    time_t tt = time(nullptr); struct tm tmv; localtime_r(&tt, &tmv);
    char b[24]; strftime(b, sizeof b, "%d-%m-%Y %H:%M:%S", &tmv);
    return std::string(b);
}

// CPU brand string from /proc/cpuinfo (best-effort, display only).
static std::string cpu_brand() {
    std::ifstream f("/proc/cpuinfo"); std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("model name", 0) == 0) {
            size_t c = line.find(':');
            if (c != std::string::npos) {
                size_t s = line.find_first_not_of(" \t", c + 1);
                return s == std::string::npos ? std::string("CPU") : line.substr(s);
            }
        }
    }
    return "CPU";
}

static std::atomic<bool> g_running{true};
static std::atomic<uint64_t> g_total_hashes{0};
static std::atomic<uint64_t> g_accepted{0};
static std::atomic<uint64_t> g_rejected{0};
static std::atomic<uint64_t> g_shares_found{0};  // session-local solutions submitted (pool acc/rej are cumulative)
static constexpr int MAX_GPUS = 16;
static std::atomic<uint64_t> g_gpu_hashes[MAX_GPUS];  // per-GPU hash counters for the GPU Stats block
static std::atomic<uint32_t> g_global_nonce{0};
static std::atomic<int> g_exit_code{0};

static constexpr int MINIBLOCK_SIZE = 48; 

struct MiningJob {
    uint8_t  work[112];
    std::string jobid;
    std::string blockhashing_blob_hex;
    std::string effective_blob_hex;
    uint64_t difficulty;
    int64_t  height;
    uint64_t generation = 0;
    bool     valid = false;
};

static std::mutex g_job_mutex;
static MiningJob g_current_job;
static bool g_has_job = false;
static std::atomic<int64_t> g_job_counter{0};

struct Config {
    std::string wallet;
    std::string pool_host;
    int pool_port = 10100;
    std::string worker_name = "openastronv";
    int log_interval = 10;
    int batch_size = 0;
    GPUEngineMode gpu_engine = GPUEngineMode::Exact;
    int staged_subbatch = 32;
    bool perf_logging = false;
    bool verbose_jobs = false;
    bool quiet = false;
    bool auto_batch = false;
    std::string submit_log_path = "openastronv_submit.jsonl";
};

static Config g_config;
static std::mutex g_submit_log_mutex;

static const char* engine_cli_name(GPUEngineMode mode) {
    switch (mode) {
        case GPUEngineMode::Exact: return "exact";
        case GPUEngineMode::Recovered: return "recovered";
        case GPUEngineMode::Staged: return "staged";
        case GPUEngineMode::Cleanroom: return "cleanroom";
    }
    return "unknown";
}

static const char* engine_banner_desc(GPUEngineMode mode) {
    switch (mode) {
        case GPUEngineMode::Exact: return "official-compatible exact path";
        case GPUEngineMode::Recovered: return "historical fast-path recovery candidate";
        case GPUEngineMode::Staged: return "staged sub-batch GPU path";
        case GPUEngineMode::Cleanroom: return "clean-room research path (exact fallback today)";
    }
    return "unknown";
}

static void print_usage(const char* argv0) {
    std::printf("Usage: %s -d host:port -w wallet [options]\n", argv0 ? argv0 : "openastronv_v3");
    std::printf("  -b, --batch-size N   Override GPU batch size\n");
    std::printf("  --auto-batch         Auto-detect the best batch size per GPU (hunt + per-GPU cache)\n");
    std::printf("  --worker NAME        Worker name appended to the pool path (default: openastronv)\n");
    std::printf("  --status-interval N  Periodic status interval in seconds (default: 10)\n");
    std::printf("  --quiet              Suppress periodic status lines (errors/shares still print)\n");
    std::printf("  --color MODE         Color output: auto | always | never (default: auto)\n");
    std::printf("  --exact-gpu          Mine with the official-compatible exact GPU path (default)\n");
    std::printf("  --fast-gpu           Mine with the recovered historical fast path\n");
    std::printf("  --gpu-engine MODE    Select GPU engine: exact | recovered | staged | cleanroom\n");
    std::printf("  --gpu-subbatch N     Staged engine sub-batch size (default: 32)\n");
    std::printf("  --perf               Enable per-batch GPU performance logs\n");
    std::printf("  --bench [N]          Run offline synthetic GPU benchmark iterations (default: 50)\n");
    std::printf("  --verbose-jobs       Log every job notification with jobid/gen\n");
    std::printf("  --submit-log PATH    Write structured submit events to PATH\n");
    std::printf("  --verify-dero-vectors\n");
    std::printf("  --verify-gpu-parity  Verify the exact GPU path against the CPU oracle\n");
    std::printf("  --verify-recovered-gpu-parity  Compare the recovered GPU path to the exact oracle\n");
    std::printf("  --verify-fast-gpu-parity  Deprecated alias for --verify-recovered-gpu-parity\n");
    std::printf("  --verify-staged-gpu-parity  Compare the staged GPU path to the exact oracle\n");
    std::printf("  --trace-dero INPUT\n");
    std::printf("  --replay-submit FILE\n");
    std::printf("  -h, --help\n");
}

namespace net {
struct Connection {
    socket_t sock = SOCKET_INVALID;
#ifdef HAS_OPENSSL
    SSL_CTX* ssl_ctx = nullptr;
    SSL*     ssl = nullptr;
#endif
    bool connected = false;
    std::vector<uint8_t> read_buf;
};

static int tls_send(Connection& c, const void* data, int len) {
#ifdef HAS_OPENSSL
    if (c.ssl) return SSL_write(c.ssl, data, len);
#endif
    return send(c.sock, (const char*)data, len, 0);
}

static int tls_recv(Connection& c, void* data, int len) {
#ifdef HAS_OPENSSL
    if (c.ssl) return SSL_read(c.ssl, data, len);
#endif
    return recv(c.sock, (char*)data, len, 0);
}

static int tls_recv_all(Connection& c, void* data, int len) {
    int received = 0; uint8_t* dst = (uint8_t*)data;
    if (!c.read_buf.empty()) { int from_buf = std::min(len, (int)c.read_buf.size()); std::memcpy(dst, c.read_buf.data(), from_buf); c.read_buf.erase(c.read_buf.begin(), c.read_buf.begin() + from_buf); received += from_buf; }
    while (received < len) { int r = tls_recv(c, dst + received, len - received); if (r <= 0) return -1; received += r; }
    return received;
}

static Connection connect_tls(const std::string& host, int port) {
    Connection c; struct addrinfo hints = {}, *result = nullptr; hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
    std::string port_str = std::to_string(port); if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result) != 0) return c;
    c.sock = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (c.sock == SOCKET_INVALID) { freeaddrinfo(result); return c; }
    if (connect(c.sock, result->ai_addr, (int)result->ai_addrlen) != 0) { CLOSE_SOCKET(c.sock); c.sock = SOCKET_INVALID; freeaddrinfo(result); return c; }
    freeaddrinfo(result);
#ifdef HAS_OPENSSL
    c.ssl_ctx = SSL_CTX_new(TLS_client_method()); if (!c.ssl_ctx) { CLOSE_SOCKET(c.sock); c.sock = SOCKET_INVALID; return c; }
    c.ssl = SSL_new(c.ssl_ctx); SSL_set_fd(c.ssl, (int)c.sock); SSL_set_tlsext_host_name(c.ssl, host.c_str());
    if (SSL_connect(c.ssl) != 1) { SSL_free(c.ssl); c.ssl = nullptr; SSL_CTX_free(c.ssl_ctx); c.ssl_ctx = nullptr; CLOSE_SOCKET(c.sock); c.sock = SOCKET_INVALID; return c; }
#endif
    c.connected = true; return c;
}

static void disconnect(Connection& c) {
#ifdef HAS_OPENSSL
    if (c.ssl) { SSL_shutdown(c.ssl); SSL_free(c.ssl); c.ssl = nullptr; }
    if (c.ssl_ctx) { SSL_CTX_free(c.ssl_ctx); c.ssl_ctx = nullptr; }
#endif
    if (c.sock != SOCKET_INVALID) { CLOSE_SOCKET(c.sock); c.sock = SOCKET_INVALID; }
    c.connected = false;
}

static bool ws_upgrade(Connection& c, const std::string& host, int port, const std::string& path) {
    char req[2048]; snprintf(req, sizeof(req), "GET %s HTTP/1.1\r\nHost: %s:%d\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n", path.c_str(), host.c_str(), port);
    if (tls_send(c, req, (int)strlen(req)) <= 0) return false;
    char resp[8192]; int total = 0; for (int attempt = 0; attempt < 20 && total < (int)sizeof(resp) - 1; attempt++) { int n = tls_recv(c, resp + total, (int)sizeof(resp) - 1 - total); if (n <= 0) break; total += n; resp[total] = '\0'; if (strstr(resp, "\r\n\r\n") != nullptr) break; }
    if (total <= 0) return false;
    char* header_end = strstr(resp, "\r\n\r\n");
    if (!header_end) return false;
    const int header_len = (int)((header_end + 4) - resp);
    if (strstr(resp, "101") == nullptr) return false;
    const int leftover = total - header_len;
    if (leftover > 0) {
        c.read_buf.insert(c.read_buf.end(),
                          reinterpret_cast<const uint8_t*>(resp + header_len),
                          reinterpret_cast<const uint8_t*>(resp + total));
    }
    return true;
}

static bool ws_send_frame(Connection& c, uint8_t opcode, const uint8_t* payload, size_t len) {
    uint8_t header[14]; int hlen = 0; header[0] = (uint8_t)(0x80 | (opcode & 0x0F));
    if (len < 126) { header[1] = (uint8_t)(0x80 | len); hlen = 2; }
    else if (len < 65536) { header[1] = 0x80 | 126; header[2] = (uint8_t)(len >> 8); header[3] = (uint8_t)(len); hlen = 4; }
    else return false;
    uint8_t mask[4] = {0x12, 0x34, 0x56, 0x78}; memcpy(&header[hlen], mask, 4); hlen += 4;
    if (tls_send(c, header, hlen) != hlen) return false;
    std::vector<uint8_t> masked(len);
    for (size_t i = 0; i < len; i++) masked[i] = payload[i] ^ mask[i & 3];
    return len == 0 || tls_send(c, masked.data(), (int)len) == (int)len;
}

static bool ws_send(Connection& c, const std::string& msg) {
    return ws_send_frame(c, 0x1, reinterpret_cast<const uint8_t*>(msg.data()), msg.size());
}

static std::string ws_recv(Connection& c) {
    std::string assembled;
    bool expecting_continuation = false;
    while (true) {
        uint8_t header[2];
        if (tls_recv_all(c, header, 2) != 2) return "";
        const bool fin = (header[0] & 0x80) != 0;
        const uint8_t opcode = header[0] & 0x0F;
        size_t payload_len = header[1] & 0x7F;
        const bool has_mask = (header[1] & 0x80) != 0;
        if (payload_len == 126) {
            uint8_t ext[2];
            if (tls_recv_all(c, ext, 2) != 2) return "";
            payload_len = ((size_t)ext[0] << 8) | ext[1];
        } else if (payload_len == 127) {
            uint8_t ext[8];
            if (tls_recv_all(c, ext, 8) != 8) return "";
            payload_len = 0;
            for (int i = 0; i < 8; i++) payload_len = (payload_len << 8) | ext[i];
        }

        uint8_t mask[4] = {};
        if (has_mask && tls_recv_all(c, mask, 4) != 4) return "";

        std::string data(payload_len, '\0');
        if (payload_len > 0) {
            if (tls_recv_all(c, &data[0], (int)payload_len) != (int)payload_len) return "";
            if (has_mask) {
                for (size_t i = 0; i < payload_len; i++) data[i] ^= mask[i & 3];
            }
        }

        if (opcode == 0x8) {
            c.connected = false;
            return "";
        }
        if (opcode == 0x9) {
            ws_send_frame(c, 0xA, reinterpret_cast<const uint8_t*>(data.data()), data.size());
            continue;
        }
        if (opcode == 0xA) continue;

        if (opcode == 0x1) {
            assembled = std::move(data);
            expecting_continuation = !fin;
            if (fin) return assembled;
            continue;
        }
        if (opcode == 0x0 && expecting_continuation) {
            assembled += data;
            if (fin) return assembled;
            continue;
        }
    }
}
}

namespace json {
static std::string get_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\""; size_t pos = json.find(search); if (pos == std::string::npos) return "";
    pos = json.find(':', pos); if (pos == std::string::npos) return ""; pos = json.find('"', pos + 1); if (pos == std::string::npos) return "";
    size_t end = json.find('"', pos + 1); if (end == std::string::npos) return ""; return json.substr(pos + 1, end - pos - 1);
}
static int64_t get_int(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\""; size_t pos = json.find(search); if (pos == std::string::npos) return 0;
    pos = json.find(':', pos); if (pos == std::string::npos) return 0; pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++; return std::atoll(json.c_str() + pos);
}
static std::string hex_encode(const uint8_t* data, int len) {
    static const char hex[] = "0123456789abcdef"; std::string out(len * 2, '\0');
    for (int i = 0; i < len; i++) { out[i*2] = hex[data[i] >> 4]; out[i*2+1] = hex[data[i] & 0xf]; } return out;
}
static void hex_decode(const std::string& hex, uint8_t* out, int max_len) {
    int len = (int)hex.size() / 2; if (len > max_len) len = max_len;
    for (int i = 0; i < len; i++) { auto c2i = [](char c) -> int { if (c >= '0' && c <= '9') return c - '0'; if (c >= 'a' && c <= 'f') return c - 'a' + 10; if (c >= 'A' && c <= 'F') return c - 'A' + 10; return 0; }; out[i] = (uint8_t)((c2i(hex[i*2]) << 4) | c2i(hex[i*2+1])); }
}
}

static uint64_t now_ms() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static std::string effective_blob_key(const std::string& blob_hex) {
    static constexpr size_t kIgnoredTailHex = 12U * 2U;
    if (blob_hex.size() <= kIgnoredTailHex) return blob_hex;
    return blob_hex.substr(0, blob_hex.size() - kIgnoredTailHex);
}

static std::string json_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 16);
    for (char ch : in) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char)ch < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)ch);
                    out += buf;
                } else {
                    out += ch;
                }
                break;
        }
    }
    return out;
}

static void append_submit_log_line(const std::string& line) {
    std::lock_guard<std::mutex> lock(g_submit_log_mutex);
    FILE* fp = std::fopen(g_config.submit_log_path.c_str(), "ab");
    if (!fp) return;
    std::fwrite(line.data(), 1, line.size(), fp);
    std::fwrite("\n", 1, 1, fp);
    std::fclose(fp);
}

static void log_submit_event(const char* kind, const MiningJob* mined_job,
                             const uint8_t* work, const uint8_t* hash,
                             bool is_latest_job, const std::string& raw = "") {
    std::string line = "{";
    line += "\"ts_ms\":" + std::to_string(now_ms());
    line += ",\"kind\":\"" + std::string(kind) + "\"";
    if (mined_job) {
        line += ",\"jobid\":\"" + json_escape(mined_job->jobid) + "\"";
        line += ",\"job_generation\":" + std::to_string(mined_job->generation);
        line += ",\"job_blob\":\"" + mined_job->blockhashing_blob_hex + "\"";
        line += ",\"difficulty\":" + std::to_string(mined_job->difficulty);
    }
    line += ",\"is_latest_job\":" + std::string(is_latest_job ? "true" : "false");
    if (work) line += ",\"mbl_blob\":\"" + json::hex_encode(work, MINIBLOCK_SIZE) + "\"";
    if (hash) line += ",\"local_hash\":\"" + json::hex_encode(hash, 32) + "\"";
    if (!raw.empty()) line += ",\"raw\":\"" + json_escape(raw) + "\"";
    line += "}";
    append_submit_log_line(line);
}

struct AsciiVector {
    const char* input;
    const char* expected_hex;
};

struct HexVector {
    const char* input_hex;
    const char* expected_hex;
};

static bool parse_trace_input(const std::string& spec, std::vector<uint8_t>& out) {
    out.clear();
    if (spec.rfind("hex:", 0) == 0) {
        const std::string hex = spec.substr(4);
        if ((hex.size() & 1U) != 0U) return false;
        out.resize(hex.size() / 2);
        json::hex_decode(hex, out.data(), (int)out.size());
        return true;
    }
    out.assign(spec.begin(), spec.end());
    return true;
}

static int run_verify_dero_vectors() {
    const std::array<AsciiVector, 10> kAsciiVectors = {{
        {"a",         "54e2324ddacc3f0383501a9e5760f85d63e9bc6705e9124ca7aef89016ab81ea"},
        {"ab",        "faeaff767be60134f0bcc5661b5f25413791b4df8ad22ff6732024d35ec4e7d0"},
        {"abc",       "715c3d8c61a967b7664b1413f8af5a2a9ba0005922cb0ba4fac8a2d502b92cd6"},
        {"abcd",      "74cc16efc1aac4768eb8124e23865da4c51ae134e29fa4773d80099c8bd39ab8"},
        {"abcde",     "d080d0484272d4498bba33530c809a02a4785368560c5c3eac17b5dacd357c4b"},
        {"abcdef",    "813e89e0484cbd3fbb3ee059083af53ed761b770d9c245be142c676f669e4607"},
        {"abcdefg",   "3972fe8fe2c9480e9d4eff383b160e2f05cc855dc47604af37bc61fdf20f21ee"},
        {"abcdefgh",  "f96191b7e39568301449d75d42d05090e41e3f79a462819473a62b1fcc2d0997"},
        {"abcdefghi", "8c76af6a57dfed744d5b7467fa822d9eb8536a851884aa7d8e3657028d511322"},
        {"abcdefghij","f838568c38f83034b2ff679d5abf65245bd2be1b27c197ab5fbac285061cf0a7"},
    }};
    const HexVector kRepeatVector = {
        "419ebb000000001bbdc9bf2200000000635d6e4e24829b4249fe0e67878ad4350000000043f53e5436cf610000086b00",
        "c392762a462fd991ace791bfe858c338c10c23c555796b50f665b636cb8c8440",
    };

    int failures = 0;
    for (const auto& tv : kAsciiVectors) {
        astrobwt::WorkerState worker{};
        uint8_t out[32] = {};
        astrobwt::hash(reinterpret_cast<const uint8_t*>(tv.input), (int)std::strlen(tv.input), out, worker);
        const std::string actual = json::hex_encode(out, 32);
        const bool ok = (actual == tv.expected_hex);
        std::printf("%s ascii \"%s\"\n", ok ? "PASS" : "FAIL", tv.input);
        if (!ok) {
            std::printf("  expected: %s\n", tv.expected_hex);
            std::printf("  actual  : %s\n", actual.c_str());
            failures++;
        }
    }

    std::vector<uint8_t> repeat_input(std::strlen(kRepeatVector.input_hex) / 2);
    json::hex_decode(kRepeatVector.input_hex, repeat_input.data(), (int)repeat_input.size());
    astrobwt::WorkerState repeat_worker{};
    uint8_t repeat_out[32] = {};
    astrobwt::hash(repeat_input.data(), (int)repeat_input.size(), repeat_out, repeat_worker);
    const std::string repeat_actual = json::hex_encode(repeat_out, 32);
    const bool repeat_ok = (repeat_actual == kRepeatVector.expected_hex);
    std::printf("%s repeat-vector\n", repeat_ok ? "PASS" : "FAIL");
    if (!repeat_ok) {
        std::printf("  input   : %s\n", kRepeatVector.input_hex);
        std::printf("  expected: %s\n", kRepeatVector.expected_hex);
        std::printf("  actual  : %s\n", repeat_actual.c_str());
        failures++;
    }

    std::printf("SUMMARY: %d/%d passed\n", 11 - failures, 11);
    return failures == 0 ? 0 : 1;
}

static int run_trace_dero(const std::string& spec) {
    std::vector<uint8_t> input;
    if (!parse_trace_input(spec, input)) {
        std::fprintf(stderr, "Invalid trace input: %s\n", spec.c_str());
        return 1;
    }
    astrobwt::WorkerState worker{};
    astrobwt::HashTrace trace;
    uint8_t out[32] = {};
    astrobwt::hash_with_trace(input.data(), (int)input.size(), out, worker, &trace);

    std::printf("{\n");
    std::printf("  \"input_hex\": \"%s\",\n", json::hex_encode(input.data(), (int)input.size()).c_str());
    std::printf("  \"sha256_hex\": \"%s\",\n", json::hex_encode(trace.sha256, 32).c_str());
    std::printf("  \"salsa256_hex\": \"%s\",\n", json::hex_encode(trace.salsa, 256).c_str());
    std::printf("  \"rc4_256_hex\": \"%s\",\n", json::hex_encode(trace.rc4, 256).c_str());
    std::printf("  \"initial_lhash\": %llu,\n", (unsigned long long)trace.initial_lhash);
    std::printf("  \"initial_prev_lhash\": %llu,\n", (unsigned long long)trace.initial_prev_lhash);
    std::printf("  \"data_len\": %d,\n", trace.data_len);
    std::printf("  \"sdata_sha256_hex\": \"%s\",\n", json::hex_encode(trace.sdata_sha256, 32).c_str());
    std::printf("  \"sa_sha256_hex\": \"%s\",\n", json::hex_encode(trace.sa_sha256, 32).c_str());
    std::printf("  \"final_hash_hex\": \"%s\",\n", json::hex_encode(out, 32).c_str());
    std::printf("  \"iterations\": [\n");
    for (size_t i = 0; i < trace.iterations.size(); ++i) {
        const auto& it = trace.iterations[i];
        std::printf("    {\"try\": %d, \"random_switcher\": %llu, \"op\": %u, \"pos1\": %u, \"pos2\": %u, \"a\": %u, \"chunk_255\": %u, \"lhash\": %llu, \"prev_lhash\": %llu, \"exit_reason\": %u, \"chunk_digest\": \"%s\"}%s\n",
                    it.try_index,
                    (unsigned long long)it.random_switcher,
                    (unsigned)it.op,
                    (unsigned)it.pos1,
                    (unsigned)it.pos2,
                    (unsigned)it.a,
                    (unsigned)it.chunk_255,
                    (unsigned long long)it.lhash,
                    (unsigned long long)it.prev_lhash,
                    (unsigned)it.exit_reason,
                    json::hex_encode(it.chunk_digest, 32).c_str(),
                    (i + 1 == trace.iterations.size()) ? "" : ",");
    }
    std::printf("  ]\n}\n");
    return 0;
}

static int run_replay_submit(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "Cannot open replay file: %s\n", path.c_str());
        return 1;
    }
    int c2s = 0;
    int latest = 0;
    int stale = 0;
    int malformed = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.find("\"dir\": \"c2s\"") == std::string::npos) continue;
        c2s++;
        if (line.find("\"jobid\":") == std::string::npos || line.find("\"mbl_blob_len\": 96") == std::string::npos) malformed++;
        if (line.find("\"submit_matches_latest\": true") != std::string::npos) latest++;
        if (line.find("\"submit_matches_latest\": false") != std::string::npos) stale++;
    }
    std::printf("{\"file\":\"%s\",\"c2s\":%d,\"latest\":%d,\"stale\":%d,\"malformed\":%d}\n",
                path.c_str(), c2s, latest, stale, malformed);
    return malformed == 0 ? 0 : 1;
}

static std::mutex g_pool_mutex;
static net::Connection* g_pool_conn = nullptr;

static void pool_thread_fn() {
#ifdef HAS_OPENSSL
    SSL_library_init(); SSL_load_error_strings(); OpenSSL_add_all_algorithms();
#endif
    uint64_t logged_diff = 0;
    int64_t logged_height = 0;
    bool logged_job_valid = false;
    while (g_running) {
        { std::string path = "/ws/" + g_config.wallet; if (!g_config.worker_name.empty()) path += "." + g_config.worker_name;
          printf("%s%s%s %sConnecting to wss://%s:%d%s%s\n", CC(UI_DIM), ts().c_str(), CC(UI_RESET), CC(UI_GOOD), g_config.pool_host.c_str(), g_config.pool_port, path.c_str(), CC(UI_RESET)); fflush(stdout); }
        net::Connection conn = net::connect_tls(g_config.pool_host, g_config.pool_port);
        if (!conn.connected) { std::this_thread::sleep_for(std::chrono::seconds(5)); continue; }
        bool upgraded = false;
        if (!g_config.worker_name.empty()) {
            upgraded = net::ws_upgrade(conn, g_config.pool_host, g_config.pool_port, "/ws/" + g_config.wallet + "." + g_config.worker_name);
            if (!upgraded) {
                net::disconnect(conn);
                conn = net::connect_tls(g_config.pool_host, g_config.pool_port);
                if (!conn.connected) { std::this_thread::sleep_for(std::chrono::seconds(5)); continue; }
            }
        }
        if (!upgraded) {
            upgraded = net::ws_upgrade(conn, g_config.pool_host, g_config.pool_port, "/ws/" + g_config.wallet);
        }
        if (!upgraded) { net::disconnect(conn); std::this_thread::sleep_for(std::chrono::seconds(5)); continue; }
        printf("[POOL] Connected!\n"); fflush(stdout);
        { std::lock_guard<std::mutex> lock(g_pool_mutex); g_pool_conn = &conn; }
        while (g_running) {
            std::string msg = net::ws_recv(conn); if (msg.empty()) break;
            if (msg.find("status") != std::string::npos && (msg.find("OK") != std::string::npos || msg.find("accepted") != std::string::npos)) {
                printf(COL_GREEN "[POOL] Share accepted!\n" COL_RESET); fflush(stdout);
                g_accepted++;
                log_submit_event("pool_message", nullptr, nullptr, nullptr, true, msg);
                continue;
            }
            
            std::string blob = json::get_string(msg, "blockhashing_blob");
            if (blob.empty()) {
                // If it's not a job and not a simple OK, print it to see if it's an error or rejection!
                if (msg.find("error") != std::string::npos || msg.find("rejected") != std::string::npos) {
                    printf(COL_RED "[POOL MESSAGE] %s\n" COL_RESET, msg.c_str()); fflush(stdout);
                    log_submit_event("pool_message", nullptr, nullptr, nullptr, false, msg);
                }
                continue;
            }

            int64_t height = json::get_int(msg, "height");
            uint64_t diff = (uint64_t)json::get_int(msg, "difficultyuint64");
            // DERO getwork reports acceptance as running counters embedded in the job push
            // (miniblocks=accepted, rejected=rejected). There is NO separate "share accepted" ack.
            if (msg.find("\"miniblocks\"") != std::string::npos)
                g_accepted.store((uint64_t)json::get_int(msg, "miniblocks"), std::memory_order_relaxed);
            if (msg.find("\"rejected\"") != std::string::npos)
                g_rejected.store((uint64_t)json::get_int(msg, "rejected"), std::memory_order_relaxed);
            std::string jobid = json::get_string(msg, "jobid"); if (jobid.empty()) jobid = json::get_string(msg, "job_id");
            if (!blob.empty() && !jobid.empty()) {
                const std::string effective_blob = effective_blob_key(blob);
                bool should_log_job = false;
                {
                    std::lock_guard<std::mutex> lock(g_job_mutex);
                    if (g_has_job && g_current_job.valid &&
                        g_current_job.difficulty == diff &&
                        g_current_job.height == height) {
                        if (g_current_job.blockhashing_blob_hex == blob &&
                            g_current_job.jobid == jobid) {
                            continue;
                        }
                        if (g_current_job.effective_blob_hex == effective_blob) {
                            g_current_job.jobid = jobid;
                            g_current_job.blockhashing_blob_hex = blob;
                            continue;
                        }
                    }
                    should_log_job = !logged_job_valid ||
                                     logged_height != height ||
                                     logged_diff != diff;
                }

                MiningJob new_job{};
                std::memset(new_job.work, 0, sizeof(new_job.work)); json::hex_decode(blob, new_job.work, MINIBLOCK_SIZE);
                uint8_t random_tail[12]; for (int i = 0; i < 12; i++) random_tail[i] = (uint8_t)(rand() & 0xFF);
                std::memcpy(&new_job.work[MINIBLOCK_SIZE - 12], random_tail, 12);
                new_job.jobid = jobid; new_job.blockhashing_blob_hex = blob; new_job.effective_blob_hex = effective_blob; new_job.difficulty = diff; new_job.height = height; new_job.valid = true;
                new_job.generation = (uint64_t)(g_job_counter.load(std::memory_order_relaxed) + 1);

                {
                    std::lock_guard<std::mutex> lock(g_job_mutex);
                    g_current_job = new_job;
                    g_has_job = true;
                }
                g_job_counter.store((int64_t)new_job.generation, std::memory_order_relaxed);
                if (g_config.verbose_jobs) {
                    printf("[JOB] Height: %lld  Diff: %llu  JobID: %s  Gen: %llu\n",
                           (long long)new_job.height,
                           (unsigned long long)new_job.difficulty,
                           new_job.jobid.c_str(),
                           (unsigned long long)new_job.generation);
                    fflush(stdout);
                } else if (should_log_job) {
                    logged_job_valid = true;
                    logged_height = new_job.height;
                    logged_diff = new_job.difficulty;
                    printf("[JOB] Height: %lld  Diff: %llu\n",
                           (long long)new_job.height,
                           (unsigned long long)new_job.difficulty);
                    fflush(stdout);
                }
            }
        }
        { std::lock_guard<std::mutex> lock(g_pool_mutex); g_pool_conn = nullptr; }
        net::disconnect(conn); if (g_running) std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}
static void gpu_mine_thread_fn(int gpu_id, int batch_override) {
    GPUMinerConfig cfg;
    cfg.device_id = gpu_id;
    cfg.batch_size = batch_override;
    cfg.engine_mode = g_config.gpu_engine;
    cfg.staged_subbatch = g_config.staged_subbatch;
    cfg.perf_logging = g_config.perf_logging;
    GPUContext* ctx = gpu_create_context(cfg);
    if (!ctx) return;

    const int bs = gpu_get_batch_size(ctx);
    printf("[GPU %d] Pure GPU %s mode | %s | Batch size: %d",
           gpu_id,
           engine_cli_name(g_config.gpu_engine),
           engine_banner_desc(g_config.gpu_engine),
           bs);
    if (g_config.gpu_engine == GPUEngineMode::Staged) {
        printf(" | Sub-batch: %d", g_config.staged_subbatch);
    }
    printf("\n");
    fflush(stdout);

    uint64_t last_job_generation = 0;

    auto submit_solutions_async = [gpu_id](std::vector<GPUSolution> sols, MiningJob job) {
        if (sols.empty()) return;
        std::thread([sols = std::move(sols), job = std::move(job), gpu_id]() {
            for (auto& sol : sols) {
                uint8_t work[MINIBLOCK_SIZE];
                std::memcpy(work, job.work, MINIBLOCK_SIZE);
                work[43] = (uint8_t)(sol.nonce >> 24);
                work[44] = (uint8_t)(sol.nonce >> 16);
                work[45] = (uint8_t)(sol.nonce >> 8);
                work[46] = (uint8_t)sol.nonce;
                work[47] = (uint8_t)gpu_id;

                std::string submit_jobid;
                std::scoped_lock lock(g_pool_mutex, g_job_mutex);
                const bool is_latest_job = g_has_job && g_current_job.valid &&
                                           g_current_job.generation == job.generation &&
                                           g_current_job.effective_blob_hex == job.effective_blob_hex &&
                                           g_current_job.difficulty == job.difficulty &&
                                           g_current_job.height == job.height;
                if (!is_latest_job) {
                    log_submit_event("submit_dropped_stale", &job, work, sol.hash, false);
                    continue;
                }
                submit_jobid = g_current_job.jobid;
                const std::string submit = "{\"jobid\":\"" + submit_jobid + "\",\"mbl_blob\":\"" + json::hex_encode(work, 48) + "\"}\n";
                if (g_pool_conn && g_pool_conn->connected) {
                    g_shares_found.fetch_add(1, std::memory_order_relaxed);
                    if (g_config.verbose_jobs)
                        printf("%s[GPU %d] share found (nonce %u)%s\n", CC(UI_DIM), gpu_id, sol.nonce, CC(UI_RESET));
                    log_submit_event("submit_sent", &job, work, sol.hash, true);
                    net::ws_send(*g_pool_conn, submit);
                } else {
                    log_submit_event("submit_dropped_disconnected", &job, work, sol.hash, true);
                }
            }
        }).detach();
    };

    while (g_running) {
        MiningJob job;
        {
            std::lock_guard<std::mutex> lock(g_job_mutex);
            if (!g_has_job || !g_current_job.valid) {
                job.valid = false;
            } else {
                job = g_current_job;
            }
        }

        if (!job.valid) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        if (job.generation != last_job_generation) {
            last_job_generation = job.generation;
            job.work[MINIBLOCK_SIZE - 1] = (uint8_t)gpu_id;
            gpu_set_work(ctx, job.work, job.difficulty);
        }

        const uint32_t nonce_start = g_global_nonce.fetch_add(bs, std::memory_order_relaxed);
        std::vector<GPUSolution> sols;
        gpu_mine_batch(ctx, nonce_start, sols);
        g_total_hashes.fetch_add(bs, std::memory_order_relaxed);
        if (gpu_id >= 0 && gpu_id < MAX_GPUS) g_gpu_hashes[gpu_id].fetch_add(bs, std::memory_order_relaxed);
        submit_solutions_async(std::move(sols), job);
    }
    gpu_destroy_context(ctx);
}

static std::string fmt_si(double v) {
    char b[32];
    if (v >= 1e9)      snprintf(b, sizeof b, "%.1fG", v / 1e9);
    else if (v >= 1e6) snprintf(b, sizeof b, "%.1fM", v / 1e6);
    else if (v >= 1e3) snprintf(b, sizeof b, "%.1fK", v / 1e3);
    else               snprintf(b, sizeof b, "%.0f", v);
    return std::string(b);
}

// Snapshot of enumerated GPUs (name/BDF) for the stats block; set in main().
static std::vector<GPUDeviceInfo> g_gpus;

struct GpuTelem { double pwr = -1, clk = -1, mem = -1, temp = -1; };

// Per-GPU telemetry via nvidia-smi (off the mining path; stats thread only).
static void query_telem(std::vector<GpuTelem>& out) {
    if (FILE* f = popen("nvidia-smi --query-gpu=index,power.draw,clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader,nounits 2>/dev/null", "r")) {
        char line[256];
        while (fgets(line, sizeof line, f)) {
            int idx; double p, c, m, t;
            if (sscanf(line, "%d, %lf, %lf, %lf, %lf", &idx, &p, &c, &m, &t) == 5 &&
                idx >= 0 && idx < (int)out.size()) {
                out[idx] = GpuTelem{p, c, m, t};
            }
        }
        pclose(f);
    }
}

// astronv-style running status line + per-GPU "GPU Stats" block, timestamped, per interval.
// HR is the last-interval window; per-GPU rates come from g_gpu_hashes[] so multi-GPU shows
// each card plus a TOTAL.
static void stats_thread_fn() {
    using sclock = std::chrono::steady_clock;
    const int N = std::max(1, (int)g_gpus.size());
    const auto t_start = sclock::now();
    auto t_win = t_start;
    bool mining = false;
    uint64_t last = 0;
    uint64_t mb_base = 0, rej_base = 0;   // pool miniblocks/rejected at this run's start -> session deltas
    std::vector<uint64_t> glast(N, 0);
    const std::string pool = g_config.pool_host + ":" + std::to_string(g_config.pool_port);
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::seconds(g_config.log_interval));
        if (!g_running) break;
        const auto now = sclock::now();
        const uint64_t total = g_total_hashes.load();
        if (!mining) {                       // waiting on the first pool job -> nothing to report yet
            if (total > 0) { mining = true; t_win = now; last = total; mb_base = g_accepted.load(); rej_base = g_rejected.load(); for (int i = 0; i < N; ++i) glast[i] = g_gpu_hashes[i].load(); }
            continue;
        }
        const double win = std::chrono::duration<double>(now - t_win).count();
        const double inst = win > 0 ? (double)(total - last) / win : 0.0;   // total H/s this window
        last = total; t_win = now;
        if (g_config.quiet) { for (int i = 0; i < N; ++i) glast[i] = g_gpu_hashes[i].load(); continue; }

        int64_t height = 0; uint64_t diff = 0;
        { std::lock_guard<std::mutex> lk(g_job_mutex); if (g_has_job) { height = g_current_job.height; diff = g_current_job.difficulty; } }
        const long up = (long)std::chrono::duration<double>(now - t_start).count();

        const char* d  = CC(UI_DIM);
        const char* r  = CC(UI_RESET);
        const char* g  = CC(UI_GOOD);
        const char* hr = CC(UI_HR);
        const char* rc = CC(g_rejected.load() ? UI_BAD : UI_DIM);

        // running status line -- "Mined" = DERO miniblocks accepted for this wallet (the mined unit),
        // shown as this-session count plus the pool/daemon lifetime total. Works pool or solo.
        const uint64_t acc = g_accepted.load();
        const uint64_t rej = g_rejected.load();
        const uint64_t mined_sess = acc >= mb_base ? acc - mb_base : acc;
        const uint64_t rej_sess   = rej >= rej_base ? rej - rej_base : rej;
        printf("%s%s%s [%s] %sMined %lu (%lu total)%s | %sRejected %lu (%lu total)%s | Height %lld | Diff %s | Uptime %02ld:%02ld:%02ld | %sHashrate %.2f KH/s%s\n",
            d, ts().c_str(), r,
            pool.c_str(),
            g, (unsigned long)mined_sess, (unsigned long)acc, r,
            rc, (unsigned long)rej_sess, (unsigned long)rej, r,
            (long long)height, fmt_si((double)diff).c_str(),
            up / 3600, (up / 60) % 60, up % 60,
            hr, inst / 1000.0, r);

        // GPU Stats block
        std::vector<GpuTelem> tel(N);
        query_telem(tel);
        printf("%s%s%s %s*************** GPU Stats ***************%s\n", d, ts().c_str(), r, g, r);
        double tot_hr = 0.0, tot_pwr = 0.0;
        for (int i = 0; i < N; ++i) {
            const uint64_t gh = g_gpu_hashes[i].load();
            const double ghr = win > 0 ? (double)(gh - glast[i]) / win : 0.0;
            glast[i] = gh;
            tot_hr += ghr;
            const GpuTelem& tt = tel[i];
            const double eff = tt.pwr > 0 ? ghr / tt.pwr : 0.0;
            if (tt.pwr > 0) tot_pwr += tt.pwr;
            const char* nm  = (i < (int)g_gpus.size()) ? g_gpus[i].name.c_str() : "GPU";
            const char* bdf = (i < (int)g_gpus.size()) ? g_gpus[i].pci_bus_id.c_str() : "";
            printf("%s%s%s GPU%d(%s) %s| HR:%s%.2f KH/s%s | Pwr:%.0fw | Clk:%.0fMhz | Mem:%.0fMhz | Temp:%.0fC | Eff:%.0fH/watt\n",
                d, ts().c_str(), r, i, bdf, nm,
                hr, ghr / 1000.0, r,
                tt.pwr < 0 ? 0.0 : tt.pwr,
                tt.clk < 0 ? 0.0 : tt.clk,
                tt.mem < 0 ? 0.0 : tt.mem,
                tt.temp < 0 ? 0.0 : tt.temp,
                eff);
        }
        const double tot_eff = tot_pwr > 0 ? tot_hr / tot_pwr : 0.0;
        printf("%s%s%s %s----------------------------------------%s\n", d, ts().c_str(), r, d, r);
        printf("%s%s%s TOTAL | HR:%s%.2f KH/s%s | Pwr:%.0fw | Eff:%.0fH/watt\n",
            d, ts().c_str(), r, hr, tot_hr / 1000.0, r, tot_pwr, tot_eff);
        printf("%s%s%s %s****************************************%s\n", d, ts().c_str(), r, g, r);
        fflush(stdout);
    }
}

static void signal_handler(int sig) { (void)sig; g_running = false; }

// ---- batch-size auto-detect (memory guesstimate + small speed check) ----
// per-hash VRAM footprint, matching gpu_create_context's allocations.
static size_t per_hash_bytes() { return (size_t)71429 * (8 + 8 + 4 + 4) + (size_t)72 * 1024; }

static std::string batch_cache_path(const std::string& gpu_name) {
    const char* home = getenv("HOME");
    std::string base = home ? std::string(home) : std::string("/tmp");
    std::string safe; for (char c : gpu_name) safe += (std::isalnum((unsigned char)c) ? c : '_');
    return base + "/.derocuda_batch_" + safe;
}
static int batch_cache_read(const std::string& gpu_name) {
    std::ifstream f(batch_cache_path(gpu_name)); int b = 0;
    if (f >> b && b >= 256 && b <= 65535) return b;
    return 0;
}
static void batch_cache_write(const std::string& gpu_name, int batch) {
    std::ofstream f(batch_cache_path(gpu_name)); if (f) f << batch << "\n";
}

// astronv guesses batch from free VRAM; we do the same for the ceiling, then time the ceiling and
// two steps below it and keep the fastest -- our engine has a perf cliff near the top that a pure
// memory pick would fall into. Returns the chosen batch.
static int gpu_autotune_batch(int device_id, GPUEngineMode engine, size_t total_mem, size_t free_mem) {
    const size_t ph = per_hash_bytes();
    // Ceiling from TOTAL VRAM: cudaMemGetInfo "free" under-reports what cudaMalloc can reclaim from
    // the driver reserve, so a free-based ceiling lands too low. The 0.88 factor puts the top
    // candidate at our ~4224 peak on an 8GB card (just below the refine cliff), so the tune never
    // grinds the slow cliffed batches. Round to nearest 128.
    int ceiling = (int)((double)total_mem * 0.85 / (double)ph);  // ~4096 on 8GB: keep the top candidate below the exact-tiny resource cliff (~4128+) so the tune never grinds a cliffed batch
    ceiling = ((ceiling + 64) / 128) * 128;
    if (ceiling < 512) ceiling = 512;
    const int cands[5] = { ceiling, ceiling - 128, ceiling - 256, ceiling - 384, ceiling - 512 };
    uint8_t work[112] = {}; for (int i = 0; i < 48; ++i) work[i] = (uint8_t)(i * 7 + 13); work[0] = 0x71;
    // Pre-warmup so the GPU is at boost clocks before timing -- otherwise the cold first (largest)
    // candidate loses to later warmer ones and we pick too small a batch.
    {
        GPUMinerConfig wc; wc.device_id = device_id; wc.engine_mode = engine; wc.batch_size = 2048;
        if (GPUContext* wctx = gpu_create_context(wc)) {
            gpu_set_work(wctx, work, (uint64_t)1ULL << 62);
            std::vector<GPUSolution> s; for (int i = 0; i < 10; ++i) { s.clear(); gpu_mine_batch(wctx, (uint32_t)(i * 2048), s); }
            gpu_destroy_context(wctx);
        }
    }
    int best_batch = 0; double best_khs = 0.0;
    for (int ci = 0; ci < 5; ++ci) {
        const int cb = cands[ci];
        if (cb < 256) continue;
        // OOM-safety: CUDA_CHECK only logs (does not abort/return null), so a too-big cudaMalloc
        // would corrupt rather than fail. Only test candidates that comfortably fit.
        if ((double)cb * (double)ph > (double)total_mem * 0.94) continue;
        if ((double)cb * (double)ph > (double)free_mem * 1.10) continue;
        GPUMinerConfig cfg; cfg.device_id = device_id; cfg.engine_mode = engine; cfg.batch_size = cb;
        GPUContext* ctx = gpu_create_context(cfg);
        if (!ctx) continue;
        const int bs = gpu_get_batch_size(ctx);
        gpu_set_work(ctx, work, (uint64_t)1ULL << 62);
        std::vector<GPUSolution> s;
        for (int i = 0; i < 3; ++i) { s.clear(); gpu_mine_batch(ctx, (uint32_t)(i * bs), s); }
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 6; ++i) { s.clear(); gpu_mine_batch(ctx, (uint32_t)(1000 + i * bs), s); }
        auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        const double khs = secs > 0 ? (double)bs * 6.0 / secs / 1000.0 : 0.0;
        gpu_destroy_context(ctx);
        // Candidates run largest-first; only switch to a smaller one if it is clearly (>5%) faster,
        // which only happens when the larger batch hit the refine cliff. Within-noise differences
        // keep the larger, astronv-like batch (more VRAM, same hashrate).
        if (khs > best_khs * 1.05) { best_khs = khs; best_batch = bs; }
    }
    return best_batch > 0 ? best_batch : 4224;
}

// astronv-style startup banner + GPU enumeration.
static void print_startup_banner(const std::vector<GPUDeviceInfo>& gpus) {
    const char* d = CC(UI_DIM); const char* r = CC(UI_RESET);
    auto line = [&](const std::string& s) { printf("%s%s%s %s\n", d, ts().c_str(), r, s.c_str()); };
    __builtin_cpu_init();
    auto yn = [](bool b) { return b ? "Yes" : "No"; };
    const std::string sep = "--------------------------------------------";
    const std::string pool = g_config.pool_host + ":" + std::to_string(g_config.pool_port);
    line(sep);
    line("| MINER_VERSION: " MINER_NAME " " MINER_VER);
    line("| COMPILED_TIME: " __DATE__);
    line("| MINER_AUTHOR: " MINER_AUTHOR);
    line("| Pool: " + pool);
    line("| Pool (failover 1): null");
    line("| Pool (failover 2): null");
    line("| Pool type: getwork");
    line("| Miner address: " + g_config.wallet);
    line("| GPU count: " + std::to_string(gpus.size()));
    line(std::string("| ") + CC(UI_GOOD) + "Fee: 0%" + CC(UI_RESET));
    line("| " + cpu_brand());
    line(std::string("| avx2: ") + yn(__builtin_cpu_supports("avx2")));
    line(std::string("| avx512: ") + yn(__builtin_cpu_supports("avx512f")));
    line(std::string("| sha: ") + yn(__builtin_cpu_supports("sha")));
    line(sep);
    line("Found " + std::to_string(gpus.size()) + " gpus");
    for (const auto& gpu : gpus) {
        char buf[256];
        snprintf(buf, sizeof buf, "GPU #%d(%s) | %s | SM%d.%d | FreeMem: %zuMB",
                 gpu.device_id, gpu.pci_bus_id.c_str(), gpu.name.c_str(),
                 gpu.compute_major, gpu.compute_minor, (size_t)(gpu.free_mem / (1024 * 1024)));
        line(buf);
    }
    fflush(stdout);
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0); srand((unsigned int)time(NULL));
    g_color_enabled = (isatty(1) != 0);   // --color {always,never} overrides during arg parse
    std::string trace_spec;
    std::string replay_path;
    bool verify_vectors = false;
    bool verify_gpu_parity = false;
    bool verify_recovered_gpu_parity = false;
    bool verify_staged_gpu_parity = false;
    bool show_help = false;
    int bench_iters = 0;
    int bench_probe_iters = 0;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") show_help = true;
        else if (arg == "-w" && i+1 < argc) g_config.wallet = argv[++i];
        else if (arg == "-d" && i+1 < argc) { std::string addr = argv[++i]; size_t colon = addr.rfind(':'); if (colon != std::string::npos) { g_config.pool_host = addr.substr(0, colon); g_config.pool_port = std::atoi(addr.substr(colon+1).c_str()); } else g_config.pool_host = addr; }
        else if ((arg == "-b" || arg == "--batch-size") && i+1 < argc) g_config.batch_size = std::max(1, std::atoi(argv[++i]));
        else if ((arg == "--worker" || arg == "-worker") && i+1 < argc) g_config.worker_name = argv[++i];
        else if ((arg == "--status-interval" || arg == "-log-interval") && i+1 < argc) g_config.log_interval = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--quiet") g_config.quiet = true;
        else if (arg == "--auto-batch") g_config.auto_batch = true;
        else if (arg == "--color" && i+1 < argc) { std::string m = argv[++i]; if (m == "never") g_color_enabled = false; else if (m == "always") g_color_enabled = true; else g_color_enabled = (isatty(1) != 0); }
        else if (arg == "--exact-gpu") g_config.gpu_engine = GPUEngineMode::Exact;
        else if (arg == "--fast-gpu") g_config.gpu_engine = GPUEngineMode::Recovered;
        else if (arg == "--gpu-engine" && i+1 < argc) {
            const std::string mode = argv[++i];
            if (mode == "exact") g_config.gpu_engine = GPUEngineMode::Exact;
            else if (mode == "legacy" || mode == "recovered") g_config.gpu_engine = GPUEngineMode::Recovered;
            else if (mode == "staged") g_config.gpu_engine = GPUEngineMode::Staged;
            else if (mode == "cleanroom") g_config.gpu_engine = GPUEngineMode::Cleanroom;
            else {
                std::fprintf(stderr, "Unknown GPU engine: %s\n", mode.c_str());
                print_usage(argc > 0 ? argv[0] : "openastronv_v3");
                return 1;
            }
        }
        else if (arg == "--gpu-subbatch" && i+1 < argc) g_config.staged_subbatch = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--perf") g_config.perf_logging = true;
        else if (arg == "--verbose-jobs") g_config.verbose_jobs = true;
        else if (arg == "--verify-dero-vectors") verify_vectors = true;
        else if (arg == "--verify-gpu-parity") verify_gpu_parity = true;
        else if (arg == "--verify-recovered-gpu-parity" || arg == "--verify-fast-gpu-parity") verify_recovered_gpu_parity = true;
        else if (arg == "--verify-staged-gpu-parity") verify_staged_gpu_parity = true;
        else if (arg == "--bench") { bench_iters = (i+1 < argc && argv[i+1][0] != '-') ? std::max(1, std::atoi(argv[++i])) : 50; }
        else if (arg == "--bench-probe") { bench_probe_iters = (i+1 < argc && argv[i+1][0] != '-') ? std::max(1, std::atoi(argv[++i])) : 20; }
        else if (arg == "--trace-dero" && i+1 < argc) trace_spec = argv[++i];
        else if (arg == "--replay-submit" && i+1 < argc) replay_path = argv[++i];
        else if (arg == "--submit-log" && i+1 < argc) g_config.submit_log_path = argv[++i];
        else {
            std::fprintf(stderr, "Unknown or incomplete argument: %s\n", arg.c_str());
            print_usage(argc > 0 ? argv[0] : "openastronv_v3");
            return 1;
        }
    }
    if (show_help) {
        print_usage(argc > 0 ? argv[0] : "openastronv_v3");
        return 0;
    }
    if (g_config.staged_subbatch <= 0) g_config.staged_subbatch = 32;
    if (bench_probe_iters > 0) {
        // STREAM-OVERLAP FEASIBILITY PROBE. Two half-batch contexts (own buffers,
        // own stream). Compare: (1) single full-batch ref, (2) 2x half serial
        // (one host thread, back-to-back -> split cost), (3) 2x half parallel
        // (two host threads -> scheduler overlaps sort||sort then refine||refine).
        auto gpus = gpu_enumerate();
        if (gpus.empty()) { std::fprintf(stderr, "No CUDA GPUs found\n"); return 1; }
        const int full_bs = g_config.batch_size > 0 ? g_config.batch_size : 3328;
        const int half_bs = full_bs / 2;
        GPUEngineMode emode = (g_config.gpu_engine == GPUEngineMode::Exact) ? GPUEngineMode::Recovered : g_config.gpu_engine;
        uint8_t work[112] = {};
        for (int i = 0; i < 48; ++i) work[i] = (uint8_t)(i * 7 + 13);
        work[0] = 0x71;
        const int N = bench_probe_iters;
        GPUMinerConfig cfgF; cfgF.device_id = gpus[0].device_id; cfgF.engine_mode = emode; cfgF.batch_size = full_bs;
        GPUContext* ctxF = gpu_create_context(cfgF);
        if (!ctxF) return 1;
        const int fbs = gpu_get_batch_size(ctxF);
        gpu_set_work(ctxF, work, (uint64_t)1ULL << 62);
        { std::vector<GPUSolution> s; for (int i=0;i<3;++i){ s.clear(); gpu_mine_batch(ctxF,(uint32_t)(i*fbs),s);} }
        auto ft0 = std::chrono::steady_clock::now();
        { std::vector<GPUSolution> s; for (int i=0;i<N;++i){ s.clear(); gpu_mine_batch(ctxF,(uint32_t)(1000+i*fbs),s);} }
        auto ft1 = std::chrono::steady_clock::now();
        const double fsecs = std::chrono::duration<double>(ft1-ft0).count();
        const double fkhs = (double)fbs * (double)N / fsecs / 1000.0;
        gpu_destroy_context(ctxF);
        GPUMinerConfig cfgA; cfgA.device_id = gpus[0].device_id; cfgA.engine_mode = emode; cfgA.batch_size = half_bs;
        GPUMinerConfig cfgB = cfgA;
        GPUContext* ctxA = gpu_create_context(cfgA);
        GPUContext* ctxB = gpu_create_context(cfgB);
        if (!ctxA || !ctxB) { std::fprintf(stderr, "probe: 2nd context alloc failed (VRAM?)\n"); return 1; }
        const int hbs = gpu_get_batch_size(ctxA);
        gpu_set_work(ctxA, work, (uint64_t)1ULL << 62);
        gpu_set_work(ctxB, work, (uint64_t)1ULL << 62);
        { std::vector<GPUSolution> sa, sb; for (int i=0;i<3;++i){ sa.clear(); sb.clear(); gpu_mine_batch(ctxA,(uint32_t)(i*hbs),sa); gpu_mine_batch(ctxB,(uint32_t)(0x40000000u+i*hbs),sb);} }
        auto st0 = std::chrono::steady_clock::now();
        { std::vector<GPUSolution> sa, sb; for (int i=0;i<N;++i){ sa.clear(); sb.clear(); gpu_mine_batch(ctxA,(uint32_t)(1000+i*hbs),sa); gpu_mine_batch(ctxB,(uint32_t)(0x40000000u+1000+i*hbs),sb);} }
        auto st1 = std::chrono::steady_clock::now();
        const double ssecs = std::chrono::duration<double>(st1-st0).count();
        const double skhs = 2.0 * (double)hbs * (double)N / ssecs / 1000.0;
        auto worker = [&](GPUContext* c, uint32_t base){ std::vector<GPUSolution> s; for (int i=0;i<N;++i){ s.clear(); gpu_mine_batch(c,(uint32_t)(base+i*hbs),s);} };
        auto pt0 = std::chrono::steady_clock::now();
        { std::thread ta(worker, ctxA, (uint32_t)1000); std::thread tb(worker, ctxB, (uint32_t)(0x40000000u+1000)); ta.join(); tb.join(); }
        auto pt1 = std::chrono::steady_clock::now();
        const double psecs = std::chrono::duration<double>(pt1-pt0).count();
        const double pkhs = 2.0 * (double)hbs * (double)N / psecs / 1000.0;
        std::printf("[PROBE] iters=%d  full_bs=%d half_bs=%d (x2)%c", N, fbs, hbs, 10);
        std::printf("[PROBE] (1) 1x full          : %.3f s  %.3f KH/s%c", fsecs, fkhs, 10);
        std::printf("[PROBE] (2) 2x half serial   : %.3f s  %.3f KH/s%c", ssecs, skhs, 10);
        std::printf("[PROBE] (3) 2x half parallel : %.3f s  %.3f KH/s%c", psecs, pkhs, 10);
        std::printf("[PROBE] overlap (3)vs(2): %.1f%% faster | (3)vs(1): %+.1f%% KH/s%c", (ssecs/psecs-1.0)*100.0, (pkhs/fkhs-1.0)*100.0, 10);
        gpu_destroy_context(ctxA);
        gpu_destroy_context(ctxB);
        return 0;
    }
    if (bench_iters > 0) {
        // Wall-clock throughput bench (no pool). Synthetic work; sweeps nonces.
        auto gpus = gpu_enumerate();
        if (gpus.empty()) { std::fprintf(stderr, "No CUDA GPUs found\n"); return 1; }
        GPUMinerConfig cfg;
        cfg.device_id = gpus[0].device_id;
        cfg.engine_mode = (g_config.gpu_engine == GPUEngineMode::Exact) ? GPUEngineMode::Recovered : g_config.gpu_engine;
        cfg.batch_size = g_config.batch_size;
        cfg.staged_subbatch = g_config.staged_subbatch;
        cfg.perf_logging = g_config.perf_logging;  // --perf prints per-stage [PERF] breakdown
        GPUContext* ctx = gpu_create_context(cfg);
        if (!ctx) return 1;
        const int bs = gpu_get_batch_size(ctx);
        uint8_t work[112] = {};
        for (int i = 0; i < 48; ++i) work[i] = (uint8_t)(i * 7 + 13);
        work[0] = 0x71;
        gpu_set_work(ctx, work, (uint64_t)1ULL << 62);  // tiny target -> ~no solutions, pure SA+hash timing
        std::vector<GPUSolution> sols;
        for (int i = 0; i < 3; ++i) { sols.clear(); gpu_mine_batch(ctx, (uint32_t)(i * bs), sols); }  // warmup
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < bench_iters; ++i) { sols.clear(); gpu_mine_batch(ctx, (uint32_t)(1000 + i * bs), sols); }
        auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        const double khs = (double)bs * (double)bench_iters / secs / 1000.0;
        std::printf("[BENCH] engine=%d batch=%d iters=%d  %.3f s  %.1f H/batch  %.3f KH/s\n",
                    (int)cfg.engine_mode, bs, bench_iters, secs, (double)bs, khs);
        gpu_destroy_context(ctx);
        return 0;
    }
    if (verify_vectors) return run_verify_dero_vectors();
    if (!trace_spec.empty()) return run_trace_dero(trace_spec);
    if (!replay_path.empty()) return run_replay_submit(replay_path);
    if (verify_gpu_parity || verify_recovered_gpu_parity || verify_staged_gpu_parity) {
        auto gpus = gpu_enumerate();
        if (gpus.empty()) {
            std::fprintf(stderr, "No CUDA GPUs found\n");
            return 1;
        }
        bool ok = true;
        for (const auto& gpu : gpus) {
            std::printf("[VERIFY] GPU %d: %s\n", gpu.device_id, gpu.name.c_str());
            GPUMinerConfig cfg;
            cfg.device_id = gpu.device_id;
            cfg.engine_mode = verify_recovered_gpu_parity ? GPUEngineMode::Recovered
                                                     : (verify_staged_gpu_parity ? GPUEngineMode::Staged
                                                                                 : GPUEngineMode::Exact);
            cfg.batch_size = g_config.batch_size;  // honor -b so verify exercises the high-batch scaled-cap path
            cfg.staged_subbatch = g_config.staged_subbatch;
            GPUContext* ctx = gpu_create_context(cfg);
            if (!ctx) {
                ok = false;
                continue;
            }
            const bool suite_ok = verify_recovered_gpu_parity ? gpu_verify_recovered_parity_suite(ctx)
                                 : verify_staged_gpu_parity ? gpu_verify_staged_parity_suite(ctx)
                                                            : gpu_verify_parity_suite(ctx);
            ok = suite_ok && ok;
            gpu_destroy_context(ctx);
        }
        return ok ? 0 : 1;
    }
    if (g_config.wallet.empty() || g_config.pool_host.empty()) {
        print_usage(argc > 0 ? argv[0] : "openastronv_v3");
        std::fprintf(stderr, "Both -d and -w are required for mining.\n");
        return 1;
    }
#ifdef _WIN32
    WSADATA wsa; WSAStartup(MAKEWORD(2,2), &wsa);
#endif
    signal(SIGINT, signal_handler); signal(SIGTERM, signal_handler);
    auto gpus = gpu_enumerate();
    if (gpus.empty()) {
        std::fprintf(stderr, "No CUDA GPUs found\n");
        return 1;
    }
    g_gpus = gpus;
    print_startup_banner(gpus);
    // Resolve per-GPU batch (-b override > cache > --auto-batch hunt > engine default) and print
    // the astronv-style auto-detect + consuming lines.
    std::vector<int> batches(gpus.size(), g_config.batch_size);
    {
        const size_t ph = per_hash_bytes();
        const bool tunable = (g_config.gpu_engine == GPUEngineMode::Recovered || g_config.gpu_engine == GPUEngineMode::Staged);
        const int eng_default = (g_config.gpu_engine == GPUEngineMode::Recovered) ? 4224
                              : (g_config.gpu_engine == GPUEngineMode::Staged) ? 2954 : 96;
        const char* d = CC(UI_DIM); const char* r = CC(UI_RESET);
        for (size_t i = 0; i < gpus.size(); ++i) {
            int b = g_config.batch_size;   // 0 unless -b given
            if (b <= 0 && g_config.auto_batch && tunable) {
                b = batch_cache_read(gpus[i].name);
                if (b <= 0) { b = gpu_autotune_batch(gpus[i].device_id, g_config.gpu_engine, gpus[i].total_mem, gpus[i].free_mem); batch_cache_write(gpus[i].name, b); }
            }
            batches[i] = b;   // 0 lets gpu_create_context apply the engine default
            const int eff_b = b > 0 ? b : eng_default;
            printf("%s%s%s GPU #%d auto detect batch size %d\n", d, ts().c_str(), r, gpus[i].device_id, eff_b);
            printf("%s%s%s GPU #%d consuming %.0fMB\n", d, ts().c_str(), r, gpus[i].device_id, (double)eff_b * (double)ph / (1024.0 * 1024.0));
        }
        fflush(stdout);
    }
    bool exact_ok_all = true;
    bool recovered_ok_all = true;
    bool staged_ok_all = true;
    for (const auto& gpu : gpus) {
        std::printf("[VERIFY] Preflight GPU %d: %s\n", gpu.device_id, gpu.name.c_str());
        GPUMinerConfig cfg;
        cfg.device_id = gpu.device_id;
        cfg.engine_mode = GPUEngineMode::Exact;
        cfg.staged_subbatch = g_config.staged_subbatch;
        GPUContext* ctx = gpu_create_context(cfg);
        if (!ctx) return 1;
        const bool ok = gpu_verify_parity_smoke_suite(ctx);
        exact_ok_all = exact_ok_all && ok;
        bool recovered_ok = true;
        bool staged_ok = true;
        if (g_config.gpu_engine == GPUEngineMode::Recovered) {
            recovered_ok = gpu_verify_recovered_smoke_suite(ctx);
            recovered_ok_all = recovered_ok_all && recovered_ok;
        } else if (g_config.gpu_engine == GPUEngineMode::Staged) {
            staged_ok = gpu_verify_staged_smoke_suite(ctx);
            staged_ok_all = staged_ok_all && staged_ok;
        }
        gpu_destroy_context(ctx);
        if (!ok) {
            std::fprintf(stderr, "[VERIFY] GPU %d exact smoke FAILED.\n", gpu.device_id);
            std::fprintf(stderr, "[VERIFY] Refusing to mine until the startup smoke matches the CPU oracle.\n");
            return 1;
        }
        if (!recovered_ok || !staged_ok) {
            std::fprintf(stderr, "[VERIFY] GPU %d selected-engine smoke FAILED.\n", gpu.device_id);
            std::fprintf(stderr, "[VERIFY] Run the full parity command for that engine before mining again.\n");
            return 1;
        }
    }
    std::printf("[MODE] Selected %s GPU mode: %s\n",
                engine_cli_name(g_config.gpu_engine),
                engine_banner_desc(g_config.gpu_engine));
    std::printf("[MODE] Exact startup smoke: %s\n", exact_ok_all ? "PASS" : "FAIL");
    if (g_config.gpu_engine == GPUEngineMode::Recovered) {
        std::printf("[MODE] Recovered startup smoke: %s\n", recovered_ok_all ? "PASS" : "FAIL");
        std::printf("[MODE] Full recovered parity remains opt-in via --verify-recovered-gpu-parity.\n");
    } else if (g_config.gpu_engine == GPUEngineMode::Staged) {
        std::printf("[MODE] Staged startup smoke: %s\n", staged_ok_all ? "PASS" : "FAIL");
        std::printf("[MODE] Staged GPU sub-batch: %d\n", g_config.staged_subbatch);
        std::printf("[MODE] Full staged parity remains opt-in via --verify-staged-gpu-parity.\n");
    } else if (g_config.gpu_engine == GPUEngineMode::Cleanroom) {
        std::printf("[MODE] Cleanroom startup smoke: %s\n", exact_ok_all ? "PASS" : "FAIL");
        std::printf("[MODE] Cleanroom currently mines through the exact fallback path.\n");
    }
    std::fflush(stdout);
    std::thread pt(pool_thread_fn), st(stats_thread_fn); std::vector<std::thread> gt;
    for (int i=0; i<(int)gpus.size(); i++) gt.emplace_back(gpu_mine_thread_fn, i, batches[i]);
    pt.join(); st.join(); for (auto& t : gt) t.join();
    return g_exit_code.load(std::memory_order_relaxed);
}
