#!/usr/bin/env bash
# Re-evaluate saved checkpoints, one at a time, with the eval_ckpt preset
# (val-only; AIME24 + AIME25 + HMMT Feb 2025, n=32 by default), then print a summary.
#
#   bash scripts/eval_sweep.sh [KEY=VALUE ...]          # KEY=VALUE is passed to every eval run
#   RUNS="p1_direct_pi_union p1_direct_opd" STEPS="60 120" BASE=0 bash scripts/eval_sweep.sh
#   bash scripts/eval_sweep.sh --summary                # only print the summary of existing logs
#
# Checkpoints must be loaded on the same number of GPUs they were saved with (GPUS_PER_NODE, default 8 here).
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

RUNS=${RUNS:-"p1_direct_pi_union p1_direct_opd p1_nopi_union"}
STEPS=${STEPS:-"120"}
BASE=${BASE:-1}                      # 1 = also evaluate the untrained Qwen3-4B
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints}
LOG_ROOT=${LOG_ROOT:-logs}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
mkdir -p "${LOG_ROOT}"

targets=()
[ "${BASE}" = "1" ] && targets+=(base)
for r in ${RUNS}; do
  for s in ${STEPS}; do
    targets+=("${OUTPUT_ROOT}/${r}/global_step_${s}")
  done
done

log_name() {  # same naming as the eval_ckpt preset
  if [ "$1" = "base" ]; then
    echo "eval_base"
  else
    echo "eval_$(basename "$(dirname "$1")")_s${1##*global_step_}"
  fi
}

summary() {
  printf '%-40s %s\n' "checkpoint" "mean@N per benchmark"
  for t in "${targets[@]}"; do
    f="${LOG_ROOT}/$(log_name "$t").out"
    [ -f "$f" ] || continue
    scores=$(grep -oE 'val-core/[^/ ]+/acc/mean@[0-9]+:(np\.float64\()?[0-9.]+' "$f" \
      | sed -E 's#val-core/([^/]+)/acc/mean@[0-9]+:(np\.float64\()?([0-9.]+)#\1=\3#' | sort -u | tr '\n' ' ')
    printf '%-40s %s\n' "$(log_name "$t")" "${scores:-(no result yet)}"
  done
}

if [ "${1:-}" = "--summary" ]; then
  summary
  exit 0
fi

for t in "${targets[@]}"; do
  if [ "$t" != "base" ] && [ ! -d "$t/actor" ]; then
    echo "[eval_sweep] skip (no checkpoint): $t"
    continue
  fi
  out="${LOG_ROOT}/$(log_name "$t").out"
  echo "[eval_sweep] $(date '+%F %T') $t -> $out"
  bash "${SCRIPT_DIR}/experiments.sh" eval_ckpt CKPT="$t" "$@" > "$out" 2>&1 \
    || echo "[eval_sweep] FAILED: $t (see $out)"
done

summary
