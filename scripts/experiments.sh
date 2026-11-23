#!/usr/bin/env bash
# Named experiment presets for scripts/train_prob_delta_qwen3_4b.sh (Qwen3-4B student).
#
#   bash scripts/experiments.sh <name> [KEY=VALUE ...]   # run one preset, optional overrides
#   bash scripts/experiments.sh --list                   # show presets
#
# Example:
#   bash scripts/experiments.sh p1_direct_pi_union VAL_BEFORE_TRAIN=True
#   bash scripts/experiments.sh p1_direct_pi_union TOTAL_TRAINING_STEPS=2      # smoke run
# Wrap with your scheduler if you have one, e.g. sbatch --gpus=2 --wrap "bash scripts/experiments.sh <name>".
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

if [ "${1:-}" = "--list" ] || [ -z "${1:-}" ]; then
  grep -E '^    [a-z0-9_]+\)' "${BASH_SOURCE[0]}" | sed -E 's/^ +//; s/\)//'
  exit 0
fi

NAME=$1
shift

# ---- KEY=VALUE overrides from the command line (applied before the defaults below) -----------
for arg in "$@"; do
  case "$arg" in
    *=*) export "${arg%%=*}=${arg#*=}" ;;
    *) echo "expected KEY=VALUE, got: $arg" >&2; exit 1 ;;
  esac
done

# ---- eval_ckpt: evaluation-only defaults (must be set before the shared defaults below) ------
if [ "$NAME" = "eval_ckpt" ]; then
  export TEST_DATASET=${TEST_DATASET:-"['${REPO_ROOT}/datasets/eval/aime24.parquet','${REPO_ROOT}/datasets/eval/aime25.parquet','${REPO_ROOT}/datasets/eval/hmmt_feb.parquet']"}
  export VAL_N=${VAL_N:-32}
fi

# ---- run settings shared by every preset (explicit; override with KEY=VALUE) -----------------
export GPUS_PER_NODE=${GPUS_PER_NODE:-2}                 # 2x A100 80GB pod
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}            # prompts per step (paper: 64), x N_RESPONSES rollouts
export N_RESPONSES=${N_RESPONSES:-4}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-120}
export SAVE_FREQ=${SAVE_FREQ:-120}                       # one checkpoint at the end
export CKPT_CONTENTS=${CKPT_CONTENTS:-full}              # full (resumable) | weights (eval only, ~1/3 the size)
export TEST_FREQ=${TEST_FREQ:-30}                        # eval at steps 30, 60, 90, 120
export VAL_N=${VAL_N:-16}                                # mean@16
export TEST_DATASET=${TEST_DATASET:-"['${REPO_ROOT}/datasets/eval/aime25.parquet']"}   # AIME25 only
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-2048}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-32768}
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}
export ENABLE_THINKING=${ENABLE_THINKING:-True}
export TEACHER_INPUT_MODE=${TEACHER_INPUT_MODE:-own_template}
export ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}  # set True if the 2-GPU run OOMs
export TEACHER_DTYPE=${TEACHER_DTYPE:-fp32}                       # set bf16 if the 2-GPU run OOMs
export PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
export PASSRATE_DATASET=${PASSRATE_DATASET:-datasets/train/skywork-or1-math-dapo-original.passrate.parquet}

preset() {
  case "$1" in
    # ---- priority 1: core comparison --------------------------------------------------------
    p1_direct_pi_union)     # p* = S0 + (T_gold - R), union support
      export REWARD_MODE=prob_delta TARGET_FORM=direct_pi PI_GATE=${PI_GATE:-all} SUPPORT_FROM=${SUPPORT_FROM:-"[student,teacher,teacher_pi,ref]"} LOSS_FORM=${LOSS_FORM:-pg}
      export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True} ;;   # first run: step-0 eval = Qwen3-4B base
    p1_direct_opd)          # Direct-OPD baseline, same teacher inputs and eval
      export REWARD_MODE=delta_opd
      export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False} ;;
    p1_nopi_union)          # p* = S0 + (T - R), union support (probability vs log difference)
      export REWARD_MODE=prob_delta TARGET_FORM=decomposed PI_GATE=none SUPPORT_FROM=${SUPPORT_FROM:-"[student,teacher,teacher_pi,ref]"} LOSS_FORM=${LOSS_FORM:-pg}
      export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False} ;;
    # ---- priority 2: support and target ablations --------------------------------------------
    p2_direct_pi_student)   # p* = S0 + (T_gold - R), student top-k only
      export REWARD_MODE=prob_delta TARGET_FORM=direct_pi PI_GATE=${PI_GATE:-all} SUPPORT_FROM="[student]" LOSS_FORM=${LOSS_FORM:-pg}
      export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False} ;;
    p2_nopi_student)        # p* = S0 + (T - R), student top-k only
      export REWARD_MODE=prob_delta TARGET_FORM=decomposed PI_GATE=none SUPPORT_FROM="[student]" LOSS_FORM=${LOSS_FORM:-pg}
      export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False} ;;
    p2_decomposed_union)    # p* = S0 + (T - R) + (T_gold - T), keeps the RL shift on the answer line
      export REWARD_MODE=prob_delta TARGET_FORM=decomposed PI_GATE=${PI_GATE:-all} SUPPORT_FROM=${SUPPORT_FROM:-"[student,teacher,teacher_pi,ref]"} LOSS_FORM=${LOSS_FORM:-pg}
      export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False} ;;
    # ---- priority 3: need teacher pass rates (scripts/compute_teacher_pass_rate.py) ------------
    p3_direct_pi_fail)
      export REWARD_MODE=prob_delta TARGET_FORM=direct_pi PI_GATE=fail TRAIN_DATASET=${TRAIN_DATASET:-$PASSRATE_DATASET}
      export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False} ;;
    p3_direct_pi_soft)
      export REWARD_MODE=prob_delta TARGET_FORM=direct_pi PI_GATE=soft TRAIN_DATASET=${TRAIN_DATASET:-$PASSRATE_DATASET}
      export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False} ;;
    p3_direct_pi_pass)
      export REWARD_MODE=prob_delta TARGET_FORM=direct_pi PI_GATE=pass TRAIN_DATASET=${TRAIN_DATASET:-$PASSRATE_DATASET}
      export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False} ;;
    # ---- optional ----------------------------------------------------------------------------
    opt_direct_pi_fkl)      # explicit forward-KL loss (gradient-identical to pg; sanity check)
      export REWARD_MODE=prob_delta TARGET_FORM=direct_pi PI_GATE=${PI_GATE:-all} LOSS_FORM=forward_kl
      export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False} ;;
    # ---- evaluation only ---------------------------------------------------------------------
    eval_ckpt)              # val-only: CKPT=checkpoints/<run>/global_step_N or CKPT=base; AIME24+AIME25+HMMT, n=32
      if [ -z "${CKPT:-}" ]; then
        echo "eval_ckpt needs CKPT=checkpoints/<run>/global_step_N (or CKPT=base)" >&2; return 1
      fi
      # reward settings only decide which workers get built; nothing is trained
      export REWARD_MODE=prob_delta TARGET_FORM=direct_pi PI_GATE=all LOSS_FORM=pg
      export VAL_ONLY=True VAL_BEFORE_TRAIN=True
      export LOAD_CONTENTS="['model']"   # works for both full and weights-only checkpoints
      if [ "$CKPT" = "base" ]; then
        export RESUME_MODE=disable
        export EXPERIMENT_NAME=${EXPERIMENT_NAME:-eval_base}
      else
        if [ ! -d "$CKPT/actor" ]; then
          echo "no actor checkpoint at $CKPT/actor" >&2; return 1
        fi
        CKPT_ABS=$(cd "$CKPT" && pwd)
        export RESUME_MODE=resume_path RESUME_FROM_PATH=$CKPT_ABS
        export EXPERIMENT_NAME=${EXPERIMENT_NAME:-eval_$(basename "$(dirname "$CKPT_ABS")")_s${CKPT_ABS##*global_step_}}
      fi ;;
    *)
      echo "unknown preset: $1 (see --list)" >&2; return 1 ;;
  esac
  export EXPERIMENT_NAME=${EXPERIMENT_NAME:-$1}
}

preset "$NAME"

echo "[experiments] ${EXPERIMENT_NAME}: REWARD_MODE=${REWARD_MODE} TARGET_FORM=${TARGET_FORM:-n/a} PI_GATE=${PI_GATE:-n/a}" \
     "SUPPORT_FROM=${SUPPORT_FROM:-default} LOSS_FORM=${LOSS_FORM:-n/a} GPUS=${GPUS_PER_NODE} BATCH=${MINI_BATCH_SIZE}" \
     "STEPS=${TOTAL_TRAINING_STEPS} TEST_FREQ=${TEST_FREQ} VAL_N=${VAL_N} SAVE_FREQ=${SAVE_FREQ} VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN}" \
     "VAL_ONLY=${VAL_ONLY:-False} RESUME_MODE=${RESUME_MODE:-auto} RESUME_FROM_PATH=${RESUME_FROM_PATH:-none}" \
     "CKPT_CONTENTS=${CKPT_CONTENTS} LOAD_CONTENTS=${LOAD_CONTENTS:-same}"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[experiments] DRY_RUN=1: TEST_DATASET=${TEST_DATASET} TRAIN_DATASET=${TRAIN_DATASET:-default} PYTHON_BIN=${PYTHON_BIN}"
  exit 0
fi
exec bash "${SCRIPT_DIR}/train_prob_delta_qwen3_4b.sh"
