#!/usr/bin/env bash
# MMPOS agent stats hook for Dirtybird-CUDA-Miner. Prints the stats JSON to stdout.
# This miner exposes no API, so stats are PARSED FROM THE LOG. EXPERIMENTAL — untested on a rig.
cd "$(dirname "$(readlink -f "$0")")" 2>/dev/null
. h-manifest.conf 2>/dev/null

log="${CUSTOM_LOG_BASENAME}.log"
get_miner_stats() {
  local khs acc rej ngpu hs_arr
  khs=$(grep -oE '[0-9]+(\.[0-9]+)? KH/s' "$log" 2>/dev/null | tail -1 | grep -oE '[0-9]+(\.[0-9]+)?')
  [ -z "$khs" ] && khs=0
  acc=$(grep -oE 'Mined [0-9]+' "$log" 2>/dev/null | tail -1 | grep -oE '[0-9]+'); [ -z "$acc" ] && acc=0
  rej=$(grep -oE 'Rejected [0-9]+' "$log" 2>/dev/null | tail -1 | grep -oE '[0-9]+'); [ -z "$rej" ] && rej=0
  ngpu=$(grep -oE 'GPU count: [0-9]+' "$log" 2>/dev/null | tail -1 | grep -oE '[0-9]+')
  [ -z "$ngpu" ] || [ "$ngpu" = "0" ] && ngpu=1
  hs_arr=$(awk -v k="$khs" -v n="$ngpu" 'BEGIN{for(i=0;i<n;i++){printf (i?",%.3f":"%.3f"),k/n}}')

  jq -nc \
    --argjson hs "[$hs_arr]" \
    --arg units "khs" \
    --argjson acc "$acc" \
    --argjson rej "$rej" \
    --arg miner_name "$CUSTOM_NAME" \
    --arg miner_version "$CUSTOM_VERSION" \
    --arg algo "astrobwtv3" \
    '{$hs, hs_units: $units, ar: [$acc, $rej], $algo, $miner_name, $miner_version}'
}
get_miner_stats
