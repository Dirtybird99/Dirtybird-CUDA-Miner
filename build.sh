#!/usr/bin/env bash
# Build the Dirtybird-CUDA-Miner (openastronv_v3) AstroBWTv3 DERO GPU miner.
# Requires: CUDA toolkit (nvcc), gcc, OpenSSL dev headers (libssl-dev).
#   nvcc is found on PATH, or via $CUDA_HOME. Default arch is sm_89 (Ada / RTX 40-series);
#   override with CUDA_ARCH=sm_86 (Ampere) etc. Output goes to ./bin/openastronv_v3.
#   SKIP_SELFTEST=1 skips the GPU parity+speed gate (use when building without a GPU, e.g. CI).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NVCC="$(command -v nvcc 2>/dev/null || true)"
if [ -z "$NVCC" ] && [ -n "${CUDA_HOME:-}" ]; then NVCC="$CUDA_HOME/bin/nvcc"; fi
if [ -z "$NVCC" ] || [ ! -x "$NVCC" ]; then
  echo "ERROR: nvcc not found. Install the CUDA toolkit and put nvcc on PATH, or set CUDA_HOME." >&2
  exit 1
fi
CUDA_ROOT="${CUDA_HOME:-$(dirname "$(dirname "$NVCC")")}"
# Per-target CUDA include/lib dir (x86_64-linux on amd64; sbsa-linux / aarch64-linux on arm64).
# Glob picks the dir this toolkit installed; fall back to x86_64-linux.
CUDA_TGT=""
for d in "$CUDA_ROOT"/targets/*-linux; do [ -d "$d" ] && CUDA_TGT="$d" && break; done
: "${CUDA_TGT:=$CUDA_ROOT/targets/x86_64-linux}"
ARCH="${CUDA_ARCH:-sm_89}"
# Expand CUDA_ARCH (comma-separated, e.g. "sm_87,sm_90") into nvcc -gencode flags.
GENCODE=""
IFS=',' read -ra _archs <<< "$ARCH"
for a in "${_archs[@]}"; do n="${a#sm_}"; GENCODE="$GENCODE -gencode arch=compute_${n},code=sm_${n}"; done
OUT="${OUT:-./bin/openastronv_v3}"
TMP="$OUT.new"
mkdir -p ./obj "$(dirname "$OUT")"

echo "[1/3] libsais.c -> obj (as C)"
gcc -O2 -c extern/libsais/libsais.c -Iextern/libsais -o obj/libsais.o || { echo BUILD_FAIL_GCC; exit 1; }

echo "[2/3] nvcc link (slow) -> $TMP  (arch=$ARCH, cuda=$CUDA_ROOT)"
"$NVCC" -O2 -std=c++17 $GENCODE -DHAS_OPENSSL \
  -Isrc -Iextern/libsais \
  -I"$CUDA_ROOT/include" -I"$CUDA_TGT/include" \
  src/main.cpp src/gpu/astrobwt_gpu.cu obj/libsais.o \
  -L"$CUDA_ROOT/lib" -L"$CUDA_ROOT/lib64" -L"$CUDA_TGT/lib" \
  -lssl -lcrypto \
  -Xlinker -rpath -Xlinker '$ORIGIN/lib' \
  -o "$TMP" 2>&1 | tail -30
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ] || [ ! -x "$TMP" ]; then echo "BUILD_FAIL_NVCC rc=$rc"; rm -f "$TMP"; exit 1; fi

# Gate the artifact before deploying: parity vs CPU oracle + a speed floor (needs a CUDA GPU).
if [ "${SKIP_SELFTEST:-0}" = "1" ]; then
  mv -f "$TMP" "$OUT"; ls -la "$OUT" && echo "BUILD_OK (self-test skipped)"; exit 0
fi
echo "[3/3] self-check: parity + speed gate"
if ! "$TMP" --verify-recovered-gpu-parity >/dev/null 2>&1; then echo BUILD_FAIL_PARITY; rm -f "$TMP"; exit 1; fi
KHS=$("$TMP" --bench 40 2>/dev/null | grep -oE '[0-9.]+ KH/s' | grep -oE '[0-9.]+' | tail -1)
echo "  self-bench: ${KHS:-0} KH/s (gate >= 8.0)"
if awk "BEGIN{exit !(${KHS:-0} >= 8.0)}"; then
  mv -f "$TMP" "$OUT"; ls -la "$OUT" && echo BUILD_OK
else
  echo "BUILD_FAIL_SLOW: ${KHS:-0} KH/s < 8.0 -- not deploying"; rm -f "$TMP"; exit 1
fi
