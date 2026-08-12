#!/usr/bin/env bash
# Exercise h-stats.sh against captured miner output.
#
# Usage: test-h-stats.sh [dir]   (default: this script's directory)
#
# The optional directory lets CI point the test at an EXTRACTED RELEASE PACKAGE
# rather than the repo, so the parser under test is the one that actually ships.
# A manifest is injected per case, so this does not check the packaged manifest;
# the release workflow asserts its CUSTOM_VERSION separately.
set -euo pipefail

SUT_DIR="$(cd "${1:-$(dirname "$0")}" && pwd)"
[[ -f $SUT_DIR/h-stats.sh ]] || { echo "no h-stats.sh in $SUT_DIR" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail() { echo "FAIL: $1" >&2; echo "  stats=$2" >&2; exit 1; }

# The agent feeds $stats straight to the dashboard, so a malformed object is as
# bad as a wrong number. Checked on every path, not just the interesting ones.
assert_json() {
    if command -v jq > /dev/null 2>&1; then
        jq -e . > /dev/null <<< "$1" || fail "malformed JSON" "$1"
    fi
}

# Build a manifest pointing at a scratch log. CUSTOM_VERSION is deliberately not
# set here when testing a package: the packaged manifest supplies it.
make_manifest() {
    cat > "$TMP/manifest" <<-EOF
	CUSTOM_NAME=dirtybird-cuda
	CUSTOM_VERSION=${1}
	CUSTOM_LOG_BASENAME=$TMP/miner
	EOF
}

run_sut() { HIVE_MANIFEST="$TMP/manifest" . "$SUT_DIR/h-stats.sh"; }

# ---------------------------------------------------------------- no log yet
# Pre-first-poll the log does not exist. The contract is a live zero, not
# "null": a null row reads as "miner not reporting" on the dashboard.
make_manifest 9.9.9
rm -f "$TMP/miner.log"
(
    run_sut
    [[ $khs == 0 ]]                  || fail "no-log khs should be 0, got $khs" "$stats"
    [[ $stats != "null" ]]           || fail "no-log stats must not be null" "$stats"
    [[ $stats == *'"hs": [0]'*    ]] || fail "no-log hs" "$stats"
    [[ $stats == *'"uptime": 0'*  ]] || fail "no-log uptime" "$stats"
    [[ $stats == *'"ar": [0, 0]'* ]] || fail "no-log ar" "$stats"
    assert_json "$stats"
)

# An empty log file is a different path from a missing one; same contract.
: > "$TMP/miner.log"
(
    run_sut
    [[ $khs == 0 ]]               || fail "empty-log khs should be 0, got $khs" "$stats"
    [[ $stats == *'"hs": [0]'* ]] || fail "empty-log hs" "$stats"
    assert_json "$stats"
)

# ------------------------------------------------------- a real mining session
# Two status blocks: the parser must report the SECOND. Startup chatter and the
# TOTAL row are both traps -- TOTAL is the last "KH/s" in the file but is not
# the status line, and the banner's "GPU count" is not a hashrate.
cat > "$TMP/miner.log" <<-'EOF'
	| GPU count: 2
	12:00:00 [dero.rabidmining.com:10300] Mined 1 (595 total) | Rejected 0 (2 total) | Height 1234000 | Diff 11.0M | Uptime 00:30:00 | Hashrate 60.00 KH/s
	12:00:00 *************** GPU Stats ***************
	12:00:00 GPU0(0000:01:00.0) NVIDIA GeForce RTX 4070| HR:30.00 KH/s | Pwr:120w | Clk:2505Mhz | Mem:10501Mhz | Temp:55C | Eff:250H/watt
	12:00:00 GPU1(0000:02:00.0) NVIDIA GeForce RTX 4070| HR:30.00 KH/s | Pwr:118w | Clk:2500Mhz | Mem:10500Mhz | Temp:54C | Eff:254H/watt
	12:00:00 ----------------------------------------
	12:00:00 TOTAL | HR:60.00 KH/s | Pwr:238w | Eff:252H/watt
	12:00:00 ****************************************
	12:34:56 [dero.rabidmining.com:10300] Mined 3 (597 total) | Rejected 0 (4 total) | Height 1234569 | Diff 12.3M | Uptime 01:02:08 | Hashrate 68.28 KH/s
	12:34:56 *************** GPU Stats ***************
	12:34:56 GPU0(0000:01:00.0) NVIDIA GeForce RTX 4070| HR:34.10 KH/s | Pwr:120w | Clk:2505Mhz | Mem:10501Mhz | Temp:61C | Eff:284H/watt
	12:34:56 GPU1(0000:02:00.0) NVIDIA GeForce RTX 4070| HR:34.18 KH/s | Pwr:118w | Clk:2500Mhz | Mem:10500Mhz | Temp:59C | Eff:289H/watt
	12:34:56 ----------------------------------------
	12:34:56 TOTAL | HR:68.28 KH/s | Pwr:238w | Eff:287H/watt
	12:34:56 ****************************************
EOF
(
    run_sut
    [[ $khs == 68.28 ]]                        || fail "khs should be 68.28, got $khs" "$stats"
    [[ $stats == *'"hs": [34.10,34.18]'*    ]] || fail "per-GPU hs" "$stats"
    [[ $stats == *'"temp": [61,59]'*        ]] || fail "per-GPU temp" "$stats"
    # 01:02:08 -- the :08 also proves the 10# guard against octal.
    [[ $stats == *'"uptime": 3728'*         ]] || fail "uptime" "$stats"
    [[ $stats == *'"ar": [597, 4]'*         ]] || fail "ar (lifetime totals)" "$stats"
    [[ $stats == *'"ver": "9.9.9"'*         ]] || fail "ver from manifest" "$stats"
    assert_json "$stats"
)

# --------------------------------------------------------------- colour output
# h-config.sh passes --color never, but a flight sheet can add --color always
# through the extra-args field, and the log then carries SGR escapes.
#
# What this pins is that escapes do not break the match. It does NOT exercise the
# ANSI strip in h-stats.sh: the miner wraps whole fields, so every escape falls
# outside the captured groups and deleting the strip still passes. The strip is
# insurance against that changing -- verified by mutation, not asserted here.
printf '\033[2m13:00:00\033[0m [pool:10300] \033[32mMined 4 (600 total)\033[0m | \033[2mRejected 0 (4 total)\033[0m | Height 1234570 | Diff 12.3M | Uptime 00:00:09 | \033[36mHashrate 70.00 KH/s\033[0m\n' \
    > "$TMP/miner.log"
(
    run_sut
    [[ $khs == 70.00 ]]                || fail "ANSI khs, got $khs" "$stats"
    [[ $stats == *'"ar": [600, 4]'* ]] || fail "ANSI ar" "$stats"
    [[ $stats == *'"uptime": 9'*    ]] || fail "ANSI uptime" "$stats"
    # No GPU block in this sample: fall back to one card carrying the total.
    [[ $stats == *'"hs": [70.00]'*  ]] || fail "ANSI hs fallback" "$stats"
    assert_json "$stats"
)

# ------------------------------------------------------------- garbage in log
# Never emit malformed JSON, whatever is in the log.
printf 'starting up\nsome unrelated line with KH/s in it\n' > "$TMP/miner.log"
(
    run_sut
    [[ $khs == 0 ]] || fail "garbage khs should be 0, got $khs" "$stats"
    assert_json "$stats"
)

echo "h-stats.sh: all cases passed ($SUT_DIR)"
