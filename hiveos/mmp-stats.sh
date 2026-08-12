#!/usr/bin/env bash
# MMPOS agent stats hook. Executed (not sourced) by the agent; prints one JSON
# object on stdout.
#
# The log parse lives in h-stats.sh and is reused here by sourcing it, so the
# two hooks cannot drift apart. That sets $khs plus the per-GPU $_hs / $_temp
# arrays and the $_acc / $_rej counters; only the JSON shape differs.
cd "$(dirname "$(readlink -f "$0")")" || exit 1

# shellcheck source=hiveos/h-stats.sh disable=SC1091
. ./h-stats.sh

# shellcheck disable=SC2154  # set by h-stats.sh
jq -nc \
  --argjson hs "[$_hs]" \
  --argjson temp "[$_temp]" \
  --arg units "khs" \
  --argjson acc "$_acc" \
  --argjson rej "$_rej" \
  --arg algo "astrobwtv3" \
  --arg miner_name "$CUSTOM_NAME" \
  --arg miner_version "${CUSTOM_VERSION:-dev}" \
  '{$hs, hs_units: $units, $temp, ar: [$acc, $rej], $algo, $miner_name, $miner_version}'
