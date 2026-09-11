#!/bin/bash
# Run one K-Search per tunable, concurrently, against one GPU.
#
# A round is ~40 s of which only ~1-1.5 s needs the device exclusively, so four
# searches leave the device idle most of the time. What makes that safe is the
# referee's flock (flash_rt/kopt/devlock.py): measured regions never overlap,
# and each one scrubs L2 on entry so it does not inherit the previous holder's
# resident weights. Without that lock these numbers would be meaningless.
#
#   scripts/ksearch-fanout.sh start [tunable ...]
#   scripts/ksearch-fanout.sh status
#   scripts/ksearch-fanout.sh stop
#
# Per tunable: a PID file (refuses to start a second search on a tunable that
# already has one -- the one real shared-state collision), its own log, and a
# staggered start so the logs are legible.
set -uo pipefail

KSEARCH_ROOT="${KSEARCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACTS="${ARTIFACTS:-$KSEARCH_ROOT/.ksearch-flashrt}"
RUNDIR="$ARTIFACTS/fanout"
TARGET="${TARGET:-local-a100}"
STAGGER_S="${STAGGER_S:-10}"

# The four BF16 decoder GEMM shapes the A100 path actually executes. n_8192 is
# the merged FP8 gate/up shape (never executed here) and the fused ada-norm
# tunable is reachable only from pipeline_rtx_fp16, so neither is included.
DEFAULT_TUNABLES=(
  "pi05.decoder.gemm_bf16.dtype_bf16-k_2048-m_10-n_1024"
  "pi05.decoder.gemm_bf16.dtype_bf16-k_4096-m_10-n_1024"
  "pi05.decoder.gemm_bf16.dtype_bf16-k_1024-m_10-n_2560"
  "pi05.decoder.gemm_bf16.dtype_bf16-k_1024-m_10-n_4096"
)

mkdir -p "$RUNDIR"

pidfile() { echo "$RUNDIR/$1.pid"; }
logfile() { echo "$RUNDIR/$1.log"; }

alive() {
  local pf; pf="$(pidfile "$1")"
  [ -f "$pf" ] || return 1
  local pid; pid="$(cat "$pf" 2>/dev/null)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

cmd_start() {
  local tunables=("$@")
  [ ${#tunables[@]} -eq 0 ] && tunables=("${DEFAULT_TUNABLES[@]}")
  for t in "${tunables[@]}"; do
    if alive "$t"; then
      echo "SKIP  $t (already running, pid $(cat "$(pidfile "$t")"))"
      continue
    fi
    local seed_var="SEED_${t//[^a-zA-Z0-9]/_}"
    local seed="${!seed_var:-${SEED_DEFAULT:-}}"
    (
      export FLASHRT_ENV_SCRIPT="${FLASHRT_ENV_SCRIPT:-$HOME/flashrt-sm80-env.sh}"
      # shellcheck disable=SC1090
      . "$FLASHRT_ENV_SCRIPT"
      export TUNABLE="$t" TARGET="$TARGET" TARGET_GPU="${TARGET_GPU:-A100}"
      export MAX_ROUNDS="${MAX_ROUNDS:-80}"
      export WM_MAX_DIFFICULTY="${WM_MAX_DIFFICULTY:-5}"
      [ -n "$seed" ] && export CONTINUE_SOLUTION="$seed"
      exec bash "$KSEARCH_ROOT/scripts/run_flashrt_search.sh"
    ) > "$(logfile "$t")" 2>&1 &
    echo $! > "$(pidfile "$t")"
    echo "START $t  pid $!  log $(logfile "$t")${seed:+  seed $seed}"
    sleep "$STAGGER_S"
  done
}

cmd_status() {
  printf '%-52s %-8s %-7s %s\n' TUNABLE STATE ROUNDS BEST
  for pf in "$RUNDIR"/*.pid; do
    [ -e "$pf" ] || continue
    local t; t="$(basename "$pf" .pid)"
    local state="dead"; alive "$t" && state="running"
    local lg; lg="$(logfile "$t")"
    local rounds best
    rounds="$(grep -cE 'Round [0-9]+:' "$lg" 2>/dev/null || echo 0)"
    best="$(grep -oE 'speedup=[0-9.]+x' "$lg" 2>/dev/null | sed 's/speedup=//;s/x//' \
            | sort -g | tail -1)"
    printf '%-52s %-8s %-7s %s\n' "${t##*.}" "$state" "$rounds" "${best:--}"
  done
}

cmd_stop() {
  for pf in "$RUNDIR"/*.pid; do
    [ -e "$pf" ] || continue
    local t; t="$(basename "$pf" .pid)"
    if alive "$t"; then
      local pid; pid="$(cat "$pf")"
      # TERM the group and let an in-flight referee finish its locked region;
      # killing mid-eval would leave the next holder measuring a warm device.
      kill -TERM "-$(ps -o pgid= "$pid" | tr -d ' ')" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
      echo "STOP  $t (pid $pid)"
    fi
    rm -f "$pf"
  done
}

case "${1:-status}" in
  start) shift; cmd_start "$@" ;;
  status) cmd_status ;;
  stop) cmd_stop ;;
  *) echo "usage: $0 {start [tunable ...]|status|stop}" >&2; exit 2 ;;
esac
