#!/usr/bin/env bash
set -euo pipefail

if [ "${DEBUG:-0}" = "1" ]; then
  set -x
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.99}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3.12}
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}
PYTHON_BIN_DIR=$(cd "$(dirname "${PYTHON_BIN}")" && pwd)
RAY_BIN=${RAY_BIN:-"${PYTHON_BIN_DIR}/ray"}
if [ ! -x "${RAY_BIN}" ]; then
  RAY_BIN=ray
fi

MODEL_ROOT=${MODEL_ROOT:-models}
DATA_ROOT=${DATA_ROOT:-datasets}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints}
LOG_ROOT=${LOG_ROOT:-logs}

PROJECT_NAME=${PROJECT_NAME:-Direct-OPD}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${REWARD_MODE:-prob_delta}_${TARGET_FORM:-decomposed}_${LOSS_FORM:-pg}_qwen3_4b_pi${PI_GATE:-all}}
ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-"${MODEL_ROOT}/Qwen3-4B"}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-"${MODEL_ROOT}/JustRL-DeepSeek-1.5B"}
TEACHER_REF_MODEL_PATH=${TEACHER_REF_MODEL_PATH:-"${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B"}
TRAIN_DATASET=${TRAIN_DATASET:-"${DATA_ROOT}/train/skywork-or1-math-dapo-original.parquet"}
TEST_DATASET=${TEST_DATASET:-"['${REPO_ROOT}/datasets/eval/aime25.parquet']"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${OUTPUT_ROOT}/${EXPERIMENT_NAME}"}
OUTPUTS_DIR=${OUTPUTS_DIR:-"${CHECKPOINT_DIR}/outputs"}
TRAIN_LOG=${TRAIN_LOG:-"${LOG_ROOT}/${EXPERIMENT_NAME}.log"}

ADV_ESTIMATOR=${ADV_ESTIMATOR:-token_reward_direct}
# ---- prob_delta: probability-space delta + privileged gold answer ----
# REWARD_MODE=delta_opd runs the Direct-OPD baseline with identical data/model/teacher-input settings
REWARD_MODE=${REWARD_MODE:-prob_delta}
if [ "${REWARD_MODE}" = "delta_opd" ]; then
  USE_KL_LOSS=${USE_KL_LOSS:-True}; ADAPTIVE_KL=${ADAPTIVE_KL:-True}; KL_LOSS_COEF=${KL_LOSS_COEF:-2.5}
else
  USE_KL_LOSS=${USE_KL_LOSS:-False}; ADAPTIVE_KL=${ADAPTIVE_KL:-False}; KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}
fi
SUPPORT_FROM=${SUPPORT_FROM:-"[student,teacher,teacher_pi,ref]"}  # "[student]" = student top-k only
TARGET_FORM=${TARGET_FORM:-decomposed}  # decomposed: S0+a_RL(T-R)+a_PI*w(Tgold-T) | direct_pi: S0+a_PI*w(Tgold-R)
LOSS_FORM=${LOSS_FORM:-pg}              # pg: A=p*-pi_old surrogate | forward_kl: explicit KL(p*||pi_theta)
ALPHA_RL=${ALPHA_RL:-1.0}
ALPHA_PI=${ALPHA_PI:-1.0}
PI_GATE=${PI_GATE:-all}  # none | all | fail | pass | soft (fail/pass/soft need extra_info.teacher_pass_rate)
PI_GATE_THRESHOLD=${PI_GATE_THRESHOLD:-0.5}
MASK_ANSWER_SPAN_FOR_PI=${MASK_ANSWER_SPAN_FOR_PI:-True}
# PI-OPD (REWARD_MODE=delta_opd only): reward += OPD_PI_ALPHA * pi_hat * clip(log T(.|gold) - log T, +-OPD_PI_CLIP)
# on the student's top-k, answer span masked (MASK_ANSWER_SPAN_FOR_PI), gated by PI_GATE.
# OPD_PI_SHADOW=True logs opd_pi/* without changing the reward (preflight).
OPD_PI_ALPHA=${OPD_PI_ALPHA:-0.0}
OPD_PI_CLIP=${OPD_PI_CLIP:-2.0}
OPD_PI_SHADOW=${OPD_PI_SHADOW:-False}
ENABLE_THINKING=${ENABLE_THINKING:-True}  # Direct-OPD Table 1 Qwen3 baselines match thinking mode
TEACHER_INPUT_MODE=${TEACHER_INPUT_MODE:-own_template}  # own_template | student_tokens
GRPO_OUTCOME_WEIGHT=${GRPO_OUTCOME_WEIGHT:-1.0}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-2048}
MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-32768}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$(( MAX_RESP_LENGTH + MAX_PROMPT_LENGTH > MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ? MAX_RESP_LENGTH + MAX_PROMPT_LENGTH : MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ))}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-128}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
TEMPERATURE=${TEMPERATURE:-1.0}
TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
N_RESPONSES=${N_RESPONSES:-4}
LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}
TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}
ADAPTIVE_KL_LOSS_REWARD_KEY=${ADAPTIVE_KL_LOSS_REWARD_KEY:-delta_opd/weighted_reward_mean}
ADAPTIVE_KL_LOSS_EPS=${ADAPTIVE_KL_LOSS_EPS:-0.01}
ADAPTIVE_KL_LOSS_MIN_COEF=${ADAPTIVE_KL_LOSS_MIN_COEF:-0.5}
ADAPTIVE_KL_LOSS_MAX_COEF=${ADAPTIVE_KL_LOSS_MAX_COEF:-2.5}
MODEL_DTYPE=${MODEL_DTYPE:-fp32}
# memory knobs for small nodes (e.g. 2x A100 80GB): offload the student's Adam states to CPU,
# store teacher weights in bf16 (their forward already runs under bf16 autocast)
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
TEACHER_DTYPE=${TEACHER_DTYPE:-${MODEL_DTYPE}}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
PARALLEL_SIZE=${PARALLEL_SIZE:-1}
OPTIM_LR=${OPTIM_LR:-1e-6}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-True}
ENFORCE_EAGER=${ENFORCE_EAGER:-False}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1024}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-True}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-True}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=${NUM_NODES:-1}
LOGGER=${LOGGER:-"['console']"}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
# eval-only runs (scripts/experiments.sh eval_ckpt): VAL_ONLY=True + RESUME_MODE=resume_path
# + RESUME_FROM_PATH=<abs path>/global_step_N; the checkpoint must be loaded on the GPU count it was saved with
VAL_ONLY=${VAL_ONLY:-False}
RESUME_MODE=${RESUME_MODE:-auto}  # auto | disable | resume_path
RESUME_FROM_PATH=${RESUME_FROM_PATH:-}
# what each actor checkpoint holds:
#   full    = weights + optimizer + scheduler/RNG (resumable; ~48 GB per checkpoint on 1 GPU for the 4B student)
#   weights = FSDP weight shards only (enough for eval_ckpt on the same GPU count; ~16 GB; not resumable)
#   hf      = a plain Hugging Face model in <ckpt>/actor/huggingface (safetensors; evaluate anywhere with
#             scripts/eval_vllm.py on any GPU count; ~16 GB; not resumable)
CKPT_CONTENTS=${CKPT_CONTENTS:-full}
case "${CKPT_CONTENTS}" in
  full) SAVE_CONTENTS="['model','optimizer','extra']" ;;
  weights) SAVE_CONTENTS="['model']" ;;
  hf) SAVE_CONTENTS="['hf_model']" ;;
  *) echo "CKPT_CONTENTS must be 'full', 'weights' or 'hf', got '${CKPT_CONTENTS}'" >&2; exit 1 ;;
esac
LOAD_CONTENTS=${LOAD_CONTENTS:-${SAVE_CONTENTS}}
if [ "${CKPT_CONTENTS}" != "full" ] && [ "${RESUME_MODE}" = "auto" ] && [ "${VAL_ONLY}" != "True" ]; then
  # auto-resume from a weights-only checkpoint would silently restart Adam and the LR schedule
  RESUME_MODE=disable
  echo "[ckpt] CKPT_CONTENTS=${CKPT_CONTENTS}: auto-resume disabled; rerunning starts from step 0" >&2
fi
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-2}
VAL_N=${VAL_N:-16}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-30}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-120}  # paper uses 300; all our runs (incl. baseline) use 120
IS_PLOT=${IS_PLOT:-True}
MANAGE_RAY=${MANAGE_RAY:-True}
VLLM_USE_INDUCTOR=${VLLM_USE_INDUCTOR:-False}
VLLM_CUDAGRAPH_MODE=${VLLM_CUDAGRAPH_MODE:-FULL_DECODE_ONLY}
VLLM_CUDAGRAPH_CAPTURE_SIZES=${VLLM_CUDAGRAPH_CAPTURE_SIZES:-}
VLLM_COMPILE_SIZES=${VLLM_COMPILE_SIZES:-}

PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}
ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-16384}
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-16384}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}
mkdir -p "${LOG_ROOT}" "${CHECKPOINT_DIR}" "${OUTPUTS_DIR}/validation_log"

RESUME_ARGS=(trainer.resume_mode="${RESUME_MODE}" trainer.val_only="${VAL_ONLY}"
  actor_rollout_ref.actor.checkpoint.save_contents="${SAVE_CONTENTS}"
  actor_rollout_ref.actor.checkpoint.load_contents="${LOAD_CONTENTS}")
if [ -n "${RESUME_FROM_PATH}" ]; then
  RESUME_ARGS+=(trainer.resume_from_path="${RESUME_FROM_PATH}")
fi

VLLM_ENGINE_OVERRIDES=()
if [ -n "${VLLM_USE_INDUCTOR}" ]; then
  VLLM_ENGINE_OVERRIDES+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.use_inductor="${VLLM_USE_INDUCTOR}")
fi
if [ -n "${VLLM_CUDAGRAPH_MODE}" ]; then
  VLLM_ENGINE_OVERRIDES+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="${VLLM_CUDAGRAPH_MODE}")
fi
if [ -n "${VLLM_CUDAGRAPH_CAPTURE_SIZES}" ]; then
  VLLM_ENGINE_OVERRIDES+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="${VLLM_CUDAGRAPH_CAPTURE_SIZES}")
fi
if [ -n "${VLLM_COMPILE_SIZES}" ]; then
  VLLM_ENGINE_OVERRIDES+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.compile_sizes="${VLLM_COMPILE_SIZES}")
fi

if [ "${MANAGE_RAY}" = "True" ]; then
  "${RAY_BIN}" stop --force || true
  # Ray has no authentication: anyone who can reach its ports can run code on this machine.
  # Single-node runs only need loopback, so bind everything Ray can bind to 127.0.0.1.
  # A host firewall that blocks all inbound traffic except SSH is still required.
  RAY_BIND_LOCAL=${RAY_BIND_LOCAL:-True}
  RAY_START_ARGS=(--head --disable-usage-stats)
  if [ "${RAY_BIND_LOCAL}" = "True" ]; then
    RAY_START_ARGS+=(--node-ip-address=127.0.0.1 --dashboard-host=127.0.0.1)
  fi
  "${RAY_BIN}" start "${RAY_START_ARGS[@]}"
  sleep 5
  # Warn about anything listening on a non-loopback address apart from SSH (22) and Jupyter (8888)
  if command -v ss >/dev/null 2>&1; then
    EXPOSED=$(ss -Htlnp 2>/dev/null | awk '{print $4, $6}' \
      | grep -vE '^(127\.[0-9.]+|\[::1\]|\[::ffff:127\.[0-9.]+\]):[0-9]+ ' \
      | grep -vE ':(22|8888) ' || true)
    if [ -n "${EXPOSED}" ]; then
      echo "WARNING: ports listening on non-loopback addresses (Ray has no auth; block them with a firewall):" >&2
      echo "${EXPOSED}" >&2
    fi
  fi
fi

set +e
"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator="${ADV_ESTIMATOR}" \
  algorithm.grpo_outcome_weight="${GRPO_OUTCOME_WEIGHT}" \
  data.shuffle=False \
  data.train_files="${TRAIN_DATASET}" \
  data.val_files="${TEST_DATASET}" \
  data.train_batch_size="${MINI_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESP_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING}" \
  actor_rollout_ref.model.path="${ACTOR_MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_activation_offload=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${OPTIM_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${PARALLEL_SIZE}" \
  actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.adaptive_kl_loss_coef="${ADAPTIVE_KL}" \
  actor_rollout_ref.actor.adaptive_kl_loss_reward_key="${ADAPTIVE_KL_LOSS_REWARD_KEY}" \
  actor_rollout_ref.actor.adaptive_kl_loss_eps="${ADAPTIVE_KL_LOSS_EPS}" \
  actor_rollout_ref.actor.adaptive_kl_loss_min_coef="${ADAPTIVE_KL_LOSS_MIN_COEF}" \
  actor_rollout_ref.actor.adaptive_kl_loss_max_coef="${ADAPTIVE_KL_LOSS_MAX_COEF}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype="${MODEL_DTYPE}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.fsdp_config.model_dtype="${MODEL_DTYPE}" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" \
  +actor_rollout_ref.rollout.reward_mode="${REWARD_MODE}" \
  +actor_rollout_ref.rollout.log_prob_top_k="${LOG_PROB_TOP_K}" \
  +actor_rollout_ref.rollout.top_k_strategy="${TOP_K_STRATEGY}" \
  +actor_rollout_ref.rollout.reward_weight_mode="${REWARD_WEIGHT_MODE}" \
  +actor_rollout_ref.rollout.teacher_temperature="${TEACHER_TEMPERATURE}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.free_cache_engine="${FREE_CACHE_ENGINE}" \
  actor_rollout_ref.rollout.enforce_eager="${ENFORCE_EAGER}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL}" \
  actor_rollout_ref.rollout.enable_prefix_caching="${ENABLE_PREFIX_CACHING}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.n="${N_RESPONSES}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  +actor_rollout_ref.rollout.val_kwargs.max_tokens="${MAX_VAL_RESP_LENGTH}" \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.repetition_penalty="${REPETITION_PENALTY}" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  reward_model.enable=True \
  +reward_model.reward_kwargs.enable_format_reward=False \
  +reward_model.teacher_input_mode="${TEACHER_INPUT_MODE}" \
  +reward_model.student_tokenizer="${ACTOR_MODEL_PATH}" \
  +prob_delta.alpha_rl="${ALPHA_RL}" \
  +prob_delta.alpha_pi="${ALPHA_PI}" \
  +prob_delta.pi_gate="${PI_GATE}" \
  +prob_delta.pi_gate_threshold="${PI_GATE_THRESHOLD}" \
  +prob_delta.mask_answer_span_for_pi="${MASK_ANSWER_SPAN_FOR_PI}" \
  +prob_delta.support_from="${SUPPORT_FROM}" \
  +prob_delta.target_form="${TARGET_FORM}" \
  +prob_delta.loss_form="${LOSS_FORM}" \
  +opd_pi.alpha="${OPD_PI_ALPHA}" \
  +opd_pi.clip="${OPD_PI_CLIP}" \
  +opd_pi.shadow="${OPD_PI_SHADOW}" \
  +opd_pi.mask_answer_span="${MASK_ANSWER_SPAN_FOR_PI}" \
  +opd_pi.pi_gate="${PI_GATE}" \
  +opd_pi.pi_gate_threshold="${PI_GATE_THRESHOLD}" \
  reward_model.model.path="${REWARD_MODEL_PATH}" \
  reward_model.model.input_tokenizer=null \
  reward_model.model.use_remove_padding=True \
  reward_model.model.fsdp_config.param_offload=False \
  +reward_model.model.dtype="${TEACHER_DTYPE}" \
  reward_model.micro_batch_size_per_gpu=24 \
  +teacher_ref_reward_model.enable=True \
  +teacher_ref_reward_model.model.path="${TEACHER_REF_MODEL_PATH}" \
  +teacher_ref_reward_model.model.input_tokenizer=null \
  +teacher_ref_reward_model.model.use_remove_padding=True \
  +teacher_ref_reward_model.model.fsdp_config.param_offload=False \
  +teacher_ref_reward_model.model.dtype="${TEACHER_DTYPE}" \
  +teacher_ref_reward_model.micro_batch_size_per_gpu=24 \
  custom_reward_function.path="${REPO_ROOT}/verl/verl/utils/reward_score/ttrl_math/__init__.py" \
  custom_reward_function.name=reward_func \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.log_val_generations="${LOG_VAL_GENERATIONS}" \
  trainer.logger="${LOGGER}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.validation_data_dir="${OUTPUTS_DIR}/validation_log/${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
  trainer.nnodes="${NUM_NODES}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  trainer.is_plot="${IS_PLOT}" \
  "${RESUME_ARGS[@]}" \
  "${VLLM_ENGINE_OVERRIDES[@]}" \
  2>&1 | tee -a "${TRAIN_LOG}"
STATUS=${PIPESTATUS[0]}
set -e

if [ "${MANAGE_RAY}" = "True" ]; then
  "${RAY_BIN}" stop --force || true
fi
exit "${STATUS}"
