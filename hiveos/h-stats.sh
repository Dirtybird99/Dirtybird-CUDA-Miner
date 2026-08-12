#!/usr/bin/env bash
# HiveOS stats hook. The agent SOURCES this inside a function and reads $khs and
# $stats afterwards, so it must not `set -e` and must not `exit` -- either would
# take the agent down with it.
#
# This miner exposes no stats API. It prints a status line per --status-interval
# (default 10s) which h-run.sh tees into the log; parse the freshest one:
#
#   HH:MM:SS [pool:port] Mined 3 (597 total) | Rejected 0 (0 total) | Height 1234569 \
#     | Diff 12.3M | Uptime 00:08:09 | Hashrate 68.28 KH/s
#
# followed by a per-GPU block:
#
#   HH:MM:SS GPU0(0000:01:00.0) NVIDIA RTX 4070 | HR:12.34 KH/s | Pwr:120w | Clk:2505Mhz \
#     | Mem:10501Mhz | Temp:61C | Eff:103H/watt
#
# HIVE_MANIFEST is an override for test-h-stats.sh; rigs use the sibling file.
# BASH_SOURCE, not $0 -- $0 is the agent, not this script.
# shellcheck source=hiveos/h-manifest.conf disable=SC1091
. "${HIVE_MANIFEST:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/h-manifest.conf}"

khs=0
stats="null"

_log="${CUSTOM_LOG_BASENAME}.log"
# Bounded read: this runs on every agent poll and the log grows all session.
# 64K covers many status intervals even with the GPU block interleaved.
#
# The `|| true` on these two is load-bearing, not decoration: a missing log makes
# tail exit 1 and an unmatched grep exits 1, and either aborts the whole hook if
# the agent happens to source it under `set -e`. Both are ordinary states here --
# the miner has simply not printed a status line yet.
# The tr/sed pair is insurance, not a working part: the miner ends every status
# line with \n and wraps whole fields in colour, so escapes currently fall
# outside every captured group. They cost nothing and cover the format drifting.
_tail="$(tail -c 65536 "$_log" 2>/dev/null | tr '\r' '\n' | sed $'s/\x1b\\[[0-9;]*[a-zA-Z]//g' || true)"

_line="$(grep -E 'Hashrate [0-9.]+ KH/s' <<< "$_tail" | tail -n 1 || true)"

_rate=0
_acc=0
_rej=0
_uptime=0
if [[ -n $_line ]]; then
    # Anchor on the labels rather than on position: the last "KH/s" in the log
    # belongs to the per-GPU TOTAL row, not to the status line.
    _rate="$(sed -nE 's/.*Hashrate ([0-9.]+) KH\/s.*/\1/p' <<< "$_line")"
    # The parenthesised figure is the pool's lifetime count, which is what the
    # C miner reports and what the dashboard's A column already shows.
    _acc="$(sed -nE 's/.*Mined [0-9]+ \(([0-9]+) total\).*/\1/p' <<< "$_line")"
    _rej="$(sed -nE 's/.*Rejected [0-9]+ \(([0-9]+) total\).*/\1/p' <<< "$_line")"
    _hms="$(sed -nE 's/.*Uptime ([0-9]+):([0-9]{2}):([0-9]{2}).*/\1 \2 \3/p' <<< "$_line")"
    if [[ -n $_hms ]]; then
        read -r _h _m _s <<< "$_hms"
        # 10# so a zero-padded :08 is not read as octal.
        _uptime=$((10#$_h * 3600 + 10#$_m * 60 + 10#$_s))
    fi
fi

[[ $_rate =~ ^[0-9]+(\.[0-9]+)?$ ]] || _rate=0
[[ $_acc  =~ ^[0-9]+$ ]] || _acc=0
[[ $_rej  =~ ^[0-9]+$ ]] || _rej=0

# Per-GPU rows from the newest stats block, so the dashboard shows each card
# instead of one total split evenly. Reset at GPU0: the block repeats per interval.
_gpu_block="$(awk '/ GPU0\(/ { buf = "" } / GPU[0-9]+\(/ { buf = buf $0 "\n" } END { printf "%s", buf }' <<< "$_tail" || true)"

_hs=""
_temp=""
while IFS= read -r _g; do
    [[ -z $_g ]] && continue
    _ghr="$(sed -nE 's/.*\| HR:([0-9.]+) KH\/s.*/\1/p' <<< "$_g")"
    _gtemp="$(sed -nE 's/.*Temp:([0-9.]+)C.*/\1/p' <<< "$_g")"
    [[ $_ghr   =~ ^[0-9]+(\.[0-9]+)?$ ]] || _ghr=0
    [[ $_gtemp =~ ^[0-9]+(\.[0-9]+)?$ ]] || _gtemp=0
    _hs="${_hs:+$_hs,}$_ghr"
    _temp="${_temp:+$_temp,}$_gtemp"
done <<< "$_gpu_block"

# No GPU block yet (startup, or --quiet): report the total as a single card so
# the row still shows a live figure rather than an empty array.
if [[ -z $_hs ]]; then
    _hs="$_rate"
    _temp=0
fi

khs="$_rate"
# Zeros, not "null", on the miss path: pre-first-job and reconnecting the miner
# is genuinely at 0.00, and a live zero beats a "not reporting" row.
stats=$(cat <<-END
{
    "hs": [$_hs],
    "hs_units": "khs",
    "temp": [$_temp],
    "uptime": $_uptime,
    "ar": [$_acc, $_rej],
    "algo": "astrobwtv3",
    "ver": "${CUSTOM_VERSION:-dev}"
}
END
)
