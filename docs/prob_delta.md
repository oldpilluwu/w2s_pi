# prob_delta: probability-space weak-to-strong distillation with privileged information

Fork of Direct-OPD. The student is trained toward a target built from **probability** differences
(not log-ratios), and the post-RL teacher can additionally see the gold answer (privileged
information, PI).

## Objective

At every position of the student's own rollout, on a token support `U`:

```
p* = normalize( max( pi_S0 + a_RL (pi_T - pi_R) + a_PI w_PI (pi_T(.|gold) - pi_T), 0 ) )
loss = KL(p* || pi_theta)          (forward KL, on-policy)
```

- `pi_S0`: frozen initial student. `pi_T`: post-RL teacher (JustRL-1.5B). `pi_R`: its base (R1-Distill-1.5B).
- `w_PI` = question gate x (not in the final-answer span).
- `U` = union of the top-k of student, teacher, teacher-with-gold-answer and reference, with every
  model's probability gathered on all of `U` (two teacher passes).
- The update reuses Direct-OPD's top-k policy-gradient path with advantage `A = p* - pi_old_bar`
  on `U`; with one on-policy update this is exactly the forward-KL gradient
  (`verl/tests/trainer/ppo/test_prob_delta_on_cpu.py::test_pg_gradient_equals_forward_kl_gradient`).
- No KL loss: the anchor `pi_S0` is inside `p*`.

## Teacher inputs

`reward_model.teacher_input_mode=own_template` renders the prompt in the teacher's own chat
template (plus the gold-answer hint for the PI pass) and appends the student's response token ids
mapped into the teacher vocabulary — no decode/re-encode, so positions line up exactly. A prompt
suffix that duplicates the start of the response (R1's injected `<think>\n`) is removed. Ids with no
teacher counterpart are logged as `prob_delta/teacher_unmappable_response_frac`.
`student_tokens` reproduces Direct-OPD (teachers score the student's Qwen3-templated tokens).

## Running (Qwen3-4B student, thinking mode)

```bash
# PI on all questions (arm A2, default)
bash scripts/train_prob_delta_qwen3_4b.sh

# A0: probability delta only, no PI
PI_GATE=none bash scripts/train_prob_delta_qwen3_4b.sh

# A1 / A3 / A4 need teacher pass rates first
python scripts/compute_teacher_pass_rate.py --model models/JustRL-DeepSeek-1.5B \
  --input datasets/train/skywork-or1-math-dapo-original.parquet \
  --output datasets/train/skywork-or1-math-dapo-original.passrate.parquet --k 8
TRAIN_DATASET=datasets/train/skywork-or1-math-dapo-original.passrate.parquet PI_GATE=fail bash scripts/train_prob_delta_qwen3_4b.sh
#   PI_GATE=pass (A3), PI_GATE=soft (A4)

# Direct-OPD baseline, same teacher inputs (KL 2.5 adaptive as in the paper)
REWARD_MODE=delta_opd bash scripts/train_prob_delta_qwen3_4b.sh
# Direct-OPD exactly as released (teachers read the student's template)
REWARD_MODE=delta_opd TEACHER_INPUT_MODE=student_tokens bash scripts/train_prob_delta_qwen3_4b.sh

# Ablation: student-top-k support only
SUPPORT_FROM="[student]" bash scripts/train_prob_delta_qwen3_4b.sh
```

Other knobs: `ALPHA_RL`, `ALPHA_PI`, `PI_GATE_THRESHOLD`, `MASK_ANSWER_SPAN_FOR_PI`,
`ENABLE_THINKING`, `LOG_PROB_TOP_K` (per-model top-k; support is up to 4x this).

Target / loss variants:

- `TARGET_FORM=direct_pi`: `p* = S0 + a_PI w (T_gold - R)`. Skips the plain-teacher passes
  (4 teacher passes instead of 6; 2 with `SUPPORT_FROM="[student]"`). With a_RL = a_PI = 1, gate all
  and no answer mask it equals the decomposed form.
- `LOSS_FORM=forward_kl`: explicit `KL(p* || pi_theta)` over the support instead of the
  policy-gradient surrogate. Same gradient for one on-policy update (tested); logs
  `actor/prob_delta_forward_kl`.

## Submitting

Named presets live in `scripts/experiments.sh` (`bash scripts/experiments.sh --list`). Every run
setting is written out there (2 GPUs, 64 prompts x 4 rollouts, 120 steps, eval on AIME25 with
n=16 every 30 steps, one checkpoint at the end); override any of them as `KEY=VALUE` arguments:

```bash
bash scripts/experiments.sh p1_direct_pi_union                      # first run (step-0 eval on)
bash scripts/experiments.sh p1_direct_opd
bash scripts/experiments.sh p1_nopi_union
bash scripts/experiments.sh p1_direct_pi_union TOTAL_TRAINING_STEPS=2  # smoke run
DRY_RUN=1 bash scripts/experiments.sh p1_direct_opd                 # print settings only
```

Priority 1 = `p1_*`; priority 2 = `p2_*`; priority 3 = `p3_*` (needs teacher pass rates).

## Re-evaluating saved checkpoints

The in-training eval (AIME25 only, 30 problems, n=16) has a standard error of about +-0.015 on
mean@16, too coarse to separate methods. `eval_ckpt` reloads a saved checkpoint and only runs
validation, by default on AIME24 + AIME25 + HMMT Feb 2025 (90 problems) with n=32:

```bash
bash scripts/experiments.sh eval_ckpt CKPT=checkpoints/p1_direct_opd/global_step_120 GPUS_PER_NODE=8
bash scripts/experiments.sh eval_ckpt CKPT=base GPUS_PER_NODE=8          # untrained Qwen3-4B

# sweep: base + step 120 of the three p1 runs, then a summary table
bash scripts/eval_sweep.sh "LOGGER=['console','wandb']" PROJECT_NAME=w2s-pi WANDB_ENTITY=<entity>
RUNS="p1_direct_pi_union p1_direct_opd" STEPS="30 60 90 120" BASE=0 bash scripts/eval_sweep.sh
bash scripts/eval_sweep.sh --summary                                      # re-print the table
```

A checkpoint must be loaded on the same GPU count it was saved with (`eval_sweep.sh` defaults to 8).

### Evaluating on a separate machine

To keep the training GPU busy, turn in-training eval off, save plain Hugging Face weights, and
evaluate them elsewhere with `scripts/eval_vllm.py` (plain vLLM: no Ray, FSDP, teachers, or matching
GPU count; same prompts, sampling and checker as the in-training eval):

```bash
# training machine
bash scripts/experiments.sh p1_nopi_union GPUS_PER_NODE=1 ... CKPT_CONTENTS=hf SAVE_FREQ=30 TEST_FREQ=-1 VAL_BEFORE_TRAIN=False
# -> checkpoints/p1_nopi_union/global_step_{30,60,90,120}/actor/huggingface  (~16 GB each, safetensors)

# eval machine (same conda env; only the checkpoint and datasets/eval are needed)
python scripts/eval_vllm.py --model models/Qwen3-4B                                 # base model
python scripts/eval_vllm.py --model checkpoints/p1_nopi_union/global_step_60 --n 32
```

Results go to `logs/eval_vllm/<run>_s<step>.json` (mean@N, maj@N, best@N, truncation rate per
benchmark) plus per-sample `.samples.jsonl`.

To move checkpoints through a private Hugging Face repo automatically:

```bash
# training machine, next to training (token with write access to this one repo)
python scripts/hf_ckpt_upload.py --repo <user>/w2s-ckpts --delete-local
# eval machine (read token); one copy per GPU if you have several
python scripts/hf_ckpt_eval.py --repo <user>/w2s-ckpts --delete-local
```

The uploader converts each finished checkpoint to bf16 (~8 GB), uploads it to
`<run>/global_step_N/actor/huggingface` in one commit, checks every file arrived, and leaves a
`.uploaded` marker. The evaluator downloads checkpoints it has no result for (earliest step first),
runs `eval_vllm.py` (arguments after `--` are passed on) and marks failures with `<name>.failed`
(delete it to retry). On a multi-GPU eval machine, run one checkpoint per GPU
(`CUDA_VISIBLE_DEVICES=i ... &`). `CKPT_CONTENTS=hf` (and `weights`) checkpoints cannot resume a
crashed run, so auto-resume is off for them.
Results go to `logs/eval_<run>_s<step>.out` and to W&B as run `eval_<run>_s<step>`, logged at the
checkpoint's step.

## PI-OPD: Direct-OPD plus the privileged answer in log space

Direct-OPD's top-k reward is `pi_hat_k (log T_k - log R_k)` on the student's top-k, where `pi_hat` is
the student's renormalised top-k probability. With one on-policy update this is the policy gradient of
`E_pi[log T - log R]`, the teacher's implicit RL reward, applied to the student's own distribution
(`test_opd_reward_gradient_is_expected_log_ratio_gradient`). PI-OPD adds one term:

```
A_k = pi_hat_k [ (log T_k - log R_k) + a * w_t * clip(log T_k(.|gold) - log T_k, -c, c) ]
```

`w_t` = question gate (`PI_GATE`) x not-in-answer-span (`MASK_ANSWER_SPAN_FOR_PI`). The KL loss and its
adaptive coefficient are Direct-OPD's, and the coefficient keeps following the RL-only reward
(`delta_opd/weighted_reward_mean`). With `a = 1`, no clip and no mask the reward is
`pi_hat (log T_gold - log R)`. Cost: one extra teacher pass (gold-answer hint) on the student top-k.

```bash
# preflight: Direct-OPD unchanged, the PI term only measured (opd_pi/*), 10 steps, no eval/checkpoint
bash scripts/experiments.sh pre_opd_pi GPUS_PER_NODE=8
# the run (90 steps by default); set OPD_PI_ALPHA / OPD_PI_CLIP from the preflight
bash scripts/experiments.sh opd_pi_2k GPUS_PER_NODE=8 OPD_PI_ALPHA=1.0 OPD_PI_CLIP=2.0
```

Preflight metrics (`opd_pi/*`, all weighted by `pi_hat`, per response position):

| metric | meaning |
|---|---|
| `pi_abs_wmean`, `rl_abs_wmean` | size of the PI term vs the RL term |
| `pi_abs_wmean_reasoning` / `_answer_span` | PI size outside / inside the answer span |
| `pi_gt_rl_frac` | share of positions where the PI term is larger than the RL term |
| `pi_wmean`, `rl_wmean`, `corr_pi_rl` | signed means and their per-position correlation |
| `pi_abs_wmean_p99`, `pi_raw_absmax_p99` | tails, to choose the clip |
| `clipped_weight_frac` | probability mass whose PI log-ratio exceeds the clip |
| `applied_abs_mean` | the PI reward actually added (0 in the preflight) |

## Run order (original plan)

| # | Run | Command (env vars) |
|---|-----|--------------------|
| 0 | Eval only: Qwen3-4B base and released `JustRL-Qwen3-4B` | validation-only run, no training |
| 1 | Smoke test (~5 steps, small batch) | `TOTAL_TRAINING_STEPS=5 MINI_BATCH_SIZE=16` |
| 2 | Loss check, 40 steps each | `LOSS_FORM=pg` vs `LOSS_FORM=forward_kl` |
| 3 | Direct-OPD baseline (same teacher inputs) | `REWARD_MODE=delta_opd` |
| 4 | Probability delta, no PI, union | `PI_GATE=none` |
| 5 | Direct PI target, union | `TARGET_FORM=direct_pi` |
| 6 | Direct PI target, student top-k | `TARGET_FORM=direct_pi SUPPORT_FROM="[student]"` |
| 7 | Probability delta, no PI, student top-k | `PI_GATE=none SUPPORT_FROM="[student]"` |
| 8 | Decomposed PI target (keeps RL shift on the answer line) | default |
| 9 | Gates (after `compute_teacher_pass_rate.py`) | best form + `PI_GATE=fail / soft / pass` |
| 10 | Alpha sweep on the best run | `ALPHA_PI` / `ALPHA_RL` in {0.5, 1, 1.5} |

## Logged diagnostics (`prob_delta/*`)

`tv_rl`, `tv_pi` (size of each shift), `tv_pi_reasoning_span` vs `tv_pi_answer_span_unmasked`,
`clip_frac`, `clipped_mass`, `support_mass_{s0,t,r,t_pi}` (how much of each distribution `U` covers),
`support_size`, `forward_kl_to_target`, `pi_gate_frac_sequences`, `answer_span_frac_tokens`.

## Cost vs Direct-OPD

Per step: 6 teacher forwards of 1.5B models (vs 2), one extra 4B student forward and one 4B frozen
student forward on `U`. The sampled-token reference forward is skipped when the KL loss is off.

## Not yet verified

The CPU unit tests cover the math and the input assembly. The distributed path (Ray workers, FSDP,
vLLM) has not been run; do a short smoke run (a few steps, small batch) before a full run.
