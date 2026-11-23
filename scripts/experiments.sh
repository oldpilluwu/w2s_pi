#!/usr/bin/env bash
# Named experiment presets for scripts/train_prob_delta_qwen3_4b.sh (Qwen3-4B student, 120 steps).
#
#   bash scripts/experiments.sh <name>            # run one preset
#   bash scripts/experiments.sh --list            # show presets
#
# Wrap with your scheduler, e.g.:  sbatch --gpus=8 --wrap "bash scripts/experiments.sh p1_direct_pi_union"
# Any variable set in the environment still overrides the preset (e.g. TOTAL_TRAINING_STEPS=2 for a smoke run).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PASSRATE_DATASET=${PASSRATE_DATASET:-datasets/train/skywork-or1-math-dapo-original.passrate.parquet}

preset() {
  case "$1" in
    # ---- priority 1: core comparison --------------------------------------------------------
    p1_direct_opd)          # Direct-OPD baseline, same teacher inputs; step-0 validation = Qwen3-4B base score
      export REWARD_MODE=delta_opd VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True} ;;
    p1_direct_pi_union)     # p* = S0 + (T_gold - R), union support
      export TARGET_FORM=direct_pi VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False} ;;
    p1_nopi_union)          # p* = S0 + (T - R), union support (probability vs log difference)
      export PI_GATE=none VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False} ;;
    # ---- priority 2: support and target ablations --------------------------------------------
    p2_direct_pi_student)   # p* = S0 + (T_gold - R), student top-k only
      export TARGET_FORM=direct_pi SUPPORT_FROM="[student]" VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False} ;;
    p2_nopi_student)        # p* = S0 + (T - R), student top-k only
      export PI_GATE=none SUPPORT_FROM="[student]" VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False} ;;
    p2_decomposed_union)    # p* = S0 + (T - R) + (T_gold - T), keeps the RL shift on the answer line
      export TARGET_FORM=decomposed VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False} ;;
    # ---- priority 3: need teacher pass rates (scripts/compute_teacher_pass_rate.py) ------------
    p3_direct_pi_fail)
      export TARGET_FORM=direct_pi PI_GATE=fail TRAIN_DATASET=${TRAIN_DATASET:-$PASSRATE_DATASET} VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False} ;;
    p3_direct_pi_soft)
      export TARGET_FORM=direct_pi PI_GATE=soft TRAIN_DATASET=${TRAIN_DATASET:-$PASSRATE_DATASET} VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False} ;;
    p3_direct_pi_pass)
      export TARGET_FORM=direct_pi PI_GATE=pass TRAIN_DATASET=${TRAIN_DATASET:-$PASSRATE_DATASET} VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False} ;;
    # ---- optional ----------------------------------------------------------------------------
    opt_direct_pi_fkl)      # explicit forward-KL loss (gradient-identical to pg; sanity check)
      export TARGET_FORM=direct_pi LOSS_FORM=forward_kl VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False} ;;
    *)
      echo "unknown preset: $1" >&2; return 1 ;;
  esac
  export EXPERIMENT_NAME=${EXPERIMENT_NAME:-$1}
}

if [ "${1:-}" = "--list" ] || [ -z "${1:-}" ]; then
  grep -E '^    [a-z0-9_]+\)' "${BASH_SOURCE[0]}" | sed -E 's/^ +//; s/\)//'
  exit 0
fi

preset "$1"
exec bash "${SCRIPT_DIR}/train_prob_delta_qwen3_4b.sh"
