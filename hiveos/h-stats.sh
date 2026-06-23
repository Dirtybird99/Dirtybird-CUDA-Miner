#!/usr/bin/env bash
# HiveOS stats hook. Sourced by the agent; it reads $khs and $stats.
# This miner exposes no API, so stats are PARSED FROM THE LOG. EXPERIMENTAL — untested on a rig.
[ -z "${CUSTOM_MINER:-}" ] && CUSTOM_MINER=dirtybird-cuda
. /hive/miners/custom/$CUSTOM_MINER/h-manifest.conf 2>/dev/null

khs=0
stats="null"
log="${CUSTOM_LOG_BASENAME}.log"

if [ -f "$log" ]; then
  khs=$(grep -oE '[0-9]+(\.[0-9]+)? KH/s' "$log" | tail -1 | grep -oE '[0-9]+(\.[0-9]+)?')
  [ -z "$khs" ] && khs=0
  acc=$(grep -oE 'Mined [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+'); [ -z "$acc" ] && acc=0
  rej=$(grep -oE 'Rejected [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+'); [ -z "$rej" ] && rej=0
  ngpu=$(grep -oE 'GPU count: [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+')
  [ -z "$ngpu" ] || [ "$ngpu" = "0" ] && ngpu=1
  hs_arr=$(awk -v k="$khs" -v n="$ngpu" 'BEGIN{for(i=0;i<n;i++){printf (i?",%.3f":"%.3f"),k/n}}')
  uptime=$(cut -d. -f1 /proc/uptime 2>/dev/null); [ -z "$uptime" ] && uptime=0

  stats=$(jq -nc \
    --argjson hs "[$hs_arr]" \
    --arg units "khs" \
    --argjson uptime "$uptime" \
    --argjson acc "$acc" \
    --argjson rej "$rej" \
    --arg algo "astrobwtv3" \
    '{hs: $hs, hs_units: $units, uptime: $uptime, ar: [$acc, $rej], algo: $algo}' 2>/dev/null)
fi

[ -z "$khs" ] && khs=0
[ -z "$stats" ] && stats="null"
