#!/usr/bin/env bash
# HiveOS entry point for Dirtybird-CUDA-Miner.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
. h-manifest.conf

# Use the runtime libs bundled next to the binary (libssl/libcrypto/libstdc++) so it starts on
# rigs whose system libs are too old. Belt-and-suspenders with the binary's $ORIGIN/lib rpath.
[ -d "$PWD/lib" ] && export LD_LIBRARY_PATH="$PWD/lib:${LD_LIBRARY_PATH:-}"

# (Re)generate the args line from the current flight sheet.
[ -f h-config.sh ] && bash h-config.sh

ARGS="$(cat "$CUSTOM_CONFIG_FILENAME" 2>/dev/null)"
mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"

# stdout is tee'd to the log so h-stats.sh / mmp-stats.sh can parse the hashrate line.
./openastronv_v3 $ARGS 2>&1 | tee -a "${CUSTOM_LOG_BASENAME}.log"
