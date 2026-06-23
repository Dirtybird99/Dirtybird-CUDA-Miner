#!/usr/bin/env bash
# Generate the Dirtybird-CUDA-Miner run-args line from the HiveOS flight sheet.
# EXPERIMENTAL HiveOS/MMPOS integration (see README) — not verified on a live rig.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
. h-manifest.conf

# Community defaults (used when the flight sheet leaves a field empty). Change %WAL% in the
# flight sheet to mine to your own DERO wallet.
DEF_URL="community-pools.mysrv.cloud:10300"
DEF_WALLET="dero1qyvpht6yfyfm6p896vw3yq32w972unmp63xmfsyehjahj7tplwdmkqqvg95j7"

URL="${CUSTOM_URL:-$DEF_URL}"
URL="${URL#stratum+tcp://}"; URL="${URL#http://}"; URL="${URL#https://}"
TEMPLATE="${CUSTOM_TEMPLATE:-$DEF_WALLET}"
WALLET="${TEMPLATE%%.*}"
WORKER="${TEMPLATE#*.}"
[ "$WORKER" = "$TEMPLATE" ] && WORKER="$(hostname -s 2>/dev/null || echo rig)"
[ -z "$WALLET" ] && WALLET="$DEF_WALLET"

ARGS="-d ${URL} -w ${WALLET} --worker ${WORKER} --fast-gpu --auto-batch --color never --status-interval 10"
[ -n "${CUSTOM_USER_CONFIG:-}" ] && ARGS="$ARGS ${CUSTOM_USER_CONFIG}"

echo "$ARGS" > "$CUSTOM_CONFIG_FILENAME"
