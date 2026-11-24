# Copyright 2026 the W2S prob-delta authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Probability-space weak-to-strong distillation with privileged information ("prob_delta").

Target at every student-visited state s_t, restricted to a token support U_t:

    p*(.|s_t) = normalize( max( pi_S0 + a_RL * (pi_T - pi_R) + a_PI * w_PI * (pi_T(.|f) - pi_T), 0 ) )

  pi_S0     frozen initial student
  pi_T      post-RL weak teacher           pi_R   its pre-RL reference
  pi_T(.|f) post-RL teacher with privileged information f (the gold answer) in its prompt
  w_PI      per-position gate in [0, 1] (question-level gate x answer-span mask)

The student minimises the forward KL(p* || pi_theta) on its own rollouts. We implement it through
the existing top-k policy-gradient path: with advantage A(v) = p*(v) - pi_old_bar(v) over U
(both renormalised over U, so sum_v A(v) = 0) and a single on-policy update, the gradient of
-sum_v A(v) log pi_theta(v) w.r.t. the logits on U equals the forward-KL gradient.

This module is pure torch so it can be unit-tested on CPU.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

_EPS = 1e-12

PI_GATES = ("none", "all", "fail", "pass", "soft")


# --------------------------------------------------------------------------------------------
# Support construction
# --------------------------------------------------------------------------------------------
def build_union_support(id_tensors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate candidate token ids and mark the first valid occurrence of each id.

    Args:
        id_tensors: tensors of shape (B, T, K_i) holding token ids; -1 marks "no candidate".
            The first tensor must be the student's top-k (always valid), it is used to fill holes.

    Returns:
        union_ids: (B, T, sum K_i) ids, every entry a real token id (holes filled with the
            student's top-1 id so that any gather is safe).
        valid: (B, T, sum K_i) bool, True exactly once per distinct id.
    """
    if len(id_tensors) == 0:
        raise ValueError("build_union_support needs at least one id tensor")
    ids = torch.cat([t.long() for t in id_tensors], dim=-1)
    exists = ids >= 0
    # Put invalid ids at the end of the sort order so they never shadow a real id.
    sort_key = torch.where(exists, ids, torch.full_like(ids, torch.iinfo(torch.long).max))
    sorted_ids, order = torch.sort(sort_key, dim=-1, stable=True)
    dup_sorted = torch.zeros_like(sorted_ids, dtype=torch.bool)
    dup_sorted[..., 1:] = sorted_ids[..., 1:] == sorted_ids[..., :-1]
    first = torch.empty_like(dup_sorted)
    first.scatter_(-1, order, ~dup_sorted)
    valid = first & exists
    filler = id_tensors[0][..., :1].long().expand_as(ids)
    union_ids = torch.where(exists, ids, filler)
    return union_ids, valid


def _masked_probs(log_probs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    lp = torch.where(torch.isfinite(log_probs), log_probs, torch.full_like(log_probs, -float("inf")))
    p = lp.float().exp()
    return torch.where(valid, p, torch.zeros_like(p))


def _normalize(p: torch.Tensor) -> torch.Tensor:
    return p / p.sum(dim=-1, keepdim=True).clamp_min(_EPS)


# --------------------------------------------------------------------------------------------
# Target and advantage
# --------------------------------------------------------------------------------------------
def compute_prob_delta_target(
    logp_s0: torch.Tensor,
    logp_t: torch.Tensor,
    logp_r: torch.Tensor,
    valid: torch.Tensor,
    alpha_rl: float = 1.0,
    alpha_pi: float = 0.0,
    logp_t_pi: torch.Tensor | None = None,
    pi_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build the clipped additive target p* on the support.

    All log-prob tensors are (B, T, K) values of the *full* (un-truncated) distributions at the
    support ids; ``valid`` (B, T, K) marks the distinct support entries.
    ``pi_weight`` is (B, T) in [0, 1].

    Returns p_star (B, T, K) (zero on invalid entries, sums to 1 over valid entries) and a dict of
    per-position diagnostics, each (B, T).
    """
    p_s0 = _masked_probs(logp_s0, valid)
    p_t = _masked_probs(logp_t, valid)
    p_r = _masked_probs(logp_r, valid)

    delta_rl = p_t - p_r
    target = p_s0 + alpha_rl * delta_rl

    use_pi = alpha_pi != 0.0 and logp_t_pi is not None and pi_weight is not None
    if use_pi:
        p_t_pi = _masked_probs(logp_t_pi, valid)
        delta_pi = p_t_pi - p_t
        target = target + alpha_pi * pi_weight.to(target.dtype).unsqueeze(-1) * delta_pi
    else:
        delta_pi = torch.zeros_like(delta_rl)

    p_star, diagnostics = _clip_and_normalize(target, valid, p_s0)
    diagnostics.update(
        {
            "tv_rl": 0.5 * delta_rl.abs().sum(dim=-1),
            "tv_pi": 0.5 * delta_pi.abs().sum(dim=-1),
            "support_mass_s0": p_s0.sum(dim=-1),
            "support_mass_t": p_t.sum(dim=-1),
            "support_mass_r": p_r.sum(dim=-1),
        }
    )
    if use_pi:
        diagnostics["support_mass_t_pi"] = p_t_pi.sum(dim=-1)
    return p_star, diagnostics


def compute_direct_pi_target(
    logp_s0: torch.Tensor,
    logp_t_pi: torch.Tensor,
    logp_r: torch.Tensor,
    valid: torch.Tensor,
    alpha: float = 1.0,
    weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Direct form  p* = normalize( max( pi_S0 + alpha * w * (pi_T(.|gold) - pi_R), 0 ) ).

    The plain teacher pi_T is not needed. ``weight`` (B, T) in [0, 1]; where it is 0 the target is
    the anchor pi_S0 (e.g. on the masked answer span)."""
    p_s0 = _masked_probs(logp_s0, valid)
    p_t_pi = _masked_probs(logp_t_pi, valid)
    p_r = _masked_probs(logp_r, valid)
    delta = p_t_pi - p_r
    w = torch.ones(delta.shape[:-1], dtype=delta.dtype) if weight is None else weight.to(delta.dtype)
    target = p_s0 + alpha * w.unsqueeze(-1) * delta
    p_star, diagnostics = _clip_and_normalize(target, valid, p_s0)
    diagnostics.update(
        {
            "tv_direct": 0.5 * delta.abs().sum(dim=-1),
            "support_mass_s0": p_s0.sum(dim=-1),
            "support_mass_t_pi": p_t_pi.sum(dim=-1),
            "support_mass_r": p_r.sum(dim=-1),
        }
    )
    return p_star, diagnostics


def _clip_and_normalize(
    target: torch.Tensor, valid: torch.Tensor, p_s0: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    neg = (target < 0) & valid
    clipped = torch.where(valid, target.clamp_min(0.0), torch.zeros_like(target))
    mass = clipped.sum(dim=-1, keepdim=True)
    # Degenerate case (all mass clipped away): fall back to the anchor.
    p_star = torch.where(mass > _EPS, clipped / mass.clamp_min(_EPS), _normalize(p_s0))
    n_valid = valid.sum(dim=-1).clamp_min(1)
    return p_star, {
        "clip_frac": neg.sum(dim=-1) / n_valid,
        "clipped_mass": (-target).clamp_min(0.0).mul(valid).sum(dim=-1),
        "support_size": valid.sum(dim=-1).float(),
    }


def forward_kl_token_loss(
    log_probs_on_support: torch.Tensor, p_star: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Explicit forward KL(p* || pi_bar) per position, pi_bar = student renormalised over the support.

    log_probs_on_support: (B, T, K) student log-probs (with grad) at the support ids.
    Returns (B, T); aggregate with the usual response mask."""
    lp = log_probs_on_support.masked_fill(~valid, -float("inf"))
    log_pi_bar = lp - torch.logsumexp(lp, dim=-1, keepdim=True)
    log_pi_bar = torch.where(valid, log_pi_bar, torch.zeros_like(log_pi_bar))
    p = torch.where(valid, p_star, torch.zeros_like(p_star)).detach()
    log_p = torch.log(p.clamp_min(_EPS))
    return torch.where(p > 0, p * (log_p - log_pi_bar), torch.zeros_like(p)).sum(dim=-1)


def compute_prob_delta_advantage(
    p_star: torch.Tensor,
    logp_old: torch.Tensor,
    valid: torch.Tensor,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """A(v) = p*(v) - pi_old_bar(v) on the support (zero elsewhere); sums to zero per position.

    Returns advantages (B, T, K) and diagnostics (B, T): forward KL(p* || pi_old_bar).
    """
    p_old = _normalize(_masked_probs(logp_old, valid))
    adv = torch.where(valid, p_star - p_old, torch.zeros_like(p_star))
    if response_mask is not None:
        adv = adv * response_mask.to(adv.dtype).unsqueeze(-1)
    log_ratio = torch.log(p_star.clamp_min(_EPS)) - torch.log(p_old.clamp_min(_EPS))
    fkl = torch.where(valid & (p_star > 0), p_star * log_ratio, torch.zeros_like(p_star)).sum(dim=-1)
    return adv, {"forward_kl_to_target": fkl}


# --------------------------------------------------------------------------------------------
# PI-OPD: Direct-OPD's log-ratio reward plus a privileged-information log-ratio term
# --------------------------------------------------------------------------------------------
def student_topk_weights(student_logp: torch.Tensor) -> torch.Tensor:
    """Direct-OPD's detached weights: softmax of the student's top-k log-probs (renormalised)."""
    w = torch.softmax(student_logp.detach().float(), dim=-1)
    return torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)


def compute_opd_pi_terms(
    student_logp: torch.Tensor,
    logp_t: torch.Tensor,
    logp_t_pi: torch.Tensor,
    pi_weight: torch.Tensor,
    alpha: float,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PI term added to Direct-OPD's top-k reward.

        rm_pi[b, t, k] = alpha * w_PI[b, t] * pi_hat[b, t, k] * clip(log T(k | gold) - log T(k), -clip, clip)

    All log-prob tensors are (B, T, K) on the student's top-k ids; pi_hat are Direct-OPD's student
    weights, so tokens the student finds unlikely get almost no push (guards against answer leakage).
    ``pi_weight`` is (B, T) in [0, 1] (question gate x not-answer-span). ``clip <= 0`` disables clipping.
    With alpha = 1, no clip and w_PI = 1, Direct-OPD's reward plus this term equals
    pi_hat * (log T(.|gold) - log R).

    Returns (rm_pi, delta_pi) where delta_pi is the unclipped log-ratio, both (B, T, K) float32.
    """
    delta_pi = torch.nan_to_num((logp_t_pi.float() - logp_t.float()).detach(), nan=0.0, posinf=0.0, neginf=0.0)
    clipped = delta_pi.clamp(-clip, clip) if clip > 0 else delta_pi
    w = student_topk_weights(student_logp)
    rm_pi = float(alpha) * pi_weight.float().unsqueeze(-1) * w * clipped
    return rm_pi, delta_pi


def opd_pi_metrics(
    student_logp: torch.Tensor,
    logp_t: torch.Tensor,
    logp_r: torch.Tensor,
    delta_pi: torch.Tensor,
    rm_pi: torch.Tensor,
    response_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    clip: float,
) -> dict[str, float]:
    """Size of the PI log-ratio next to Direct-OPD's RL log-ratio, both weighted by pi_hat.

    Logged with or without the PI term in the loss (shadow mode), to choose alpha and the clip."""
    w = student_topk_weights(student_logp)
    delta_rl = (logp_t.float() - logp_r.float()).detach()
    pi_abs = (w * delta_pi.abs()).sum(-1)  # (B, T)
    rl_abs = (w * delta_rl.abs()).sum(-1)
    pi_signed = (w * delta_pi).sum(-1)
    rl_signed = (w * delta_rl).sum(-1)
    m = response_mask.bool()
    reason = m & ~answer_mask.bool()
    ans = m & answer_mask.bool()

    out = {
        "opd_pi/pi_abs_wmean": masked_position_mean(pi_abs, m),
        "opd_pi/rl_abs_wmean": masked_position_mean(rl_abs, m),
        "opd_pi/pi_wmean": masked_position_mean(pi_signed, m),
        "opd_pi/rl_wmean": masked_position_mean(rl_signed, m),
        "opd_pi/pi_abs_wmean_reasoning": masked_position_mean(pi_abs, reason),
        "opd_pi/pi_abs_wmean_answer_span": masked_position_mean(pi_abs, ans),
        "opd_pi/pi_gt_rl_frac": masked_position_mean((pi_abs > rl_abs).float(), m),
        "opd_pi/answer_span_frac_tokens": masked_position_mean(answer_mask.float(), m),
        "opd_pi/applied_abs_mean": masked_position_mean(rm_pi.abs().sum(-1), m),
    }
    if clip > 0:
        out["opd_pi/clipped_weight_frac"] = masked_position_mean((w * (delta_pi.abs() > clip).float()).sum(-1), m)
    if m.any():
        vals = pi_abs[m]
        raw_max = delta_pi.abs().amax(-1)[m]
        if vals.numel() > 1_000_000:  # torch.quantile input limit; a strided sample is enough
            stride = vals.numel() // 1_000_000 + 1
            vals, raw_max = vals[::stride], raw_max[::stride]
        out["opd_pi/pi_abs_wmean_p99"] = float(torch.quantile(vals, 0.99))
        out["opd_pi/pi_raw_absmax_p99"] = float(torch.quantile(raw_max, 0.99))
        x, y = pi_signed[m].double(), rl_signed[m].double()
        if x.numel() > 1 and x.std() > 0 and y.std() > 0:
            out["opd_pi/corr_pi_rl"] = float(torch.corrcoef(torch.stack([x, y]))[0, 1])
    return out


# --------------------------------------------------------------------------------------------
# Privileged-information gating
# --------------------------------------------------------------------------------------------
def compute_pi_gate(gate: str, teacher_pass_rate: torch.Tensor | None, threshold: float = 0.5) -> torch.Tensor | None:
    """Per-question weight in [0, 1] for the PI term.

    none: 0 everywhere | all: 1 everywhere
    fail: 1 where teacher pass rate < threshold | pass: 1 where pass rate >= threshold
    soft: 1 - pass rate
    Returns None for "none"/"all" when no pass rates are supplied (caller broadcasts).
    """
    if gate not in PI_GATES:
        raise ValueError(f"Unknown pi_gate={gate!r}; expected one of {PI_GATES}")
    if gate in ("none", "all"):
        if teacher_pass_rate is None:
            return None
        return torch.full_like(teacher_pass_rate, 0.0 if gate == "none" else 1.0, dtype=torch.float32)
    if teacher_pass_rate is None:
        raise ValueError(f"pi_gate={gate!r} needs per-question teacher pass rates (extra_info.teacher_pass_rate)")
    rate = teacher_pass_rate.float()
    if torch.isnan(rate).any():
        raise ValueError("teacher_pass_rate contains NaN; every training question needs a pass rate")
    if gate == "fail":
        return (rate < threshold).float()
    if gate == "pass":
        return (rate >= threshold).float()
    return (1.0 - rate).clamp(0.0, 1.0)


def answer_span_mask(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    vocab_pieces: Sequence[str],
    markers: Sequence[str],
) -> torch.Tensor:
    """(B, T) bool, True from the start of the token containing the LAST marker occurrence onward.

    ``vocab_pieces[i]`` is the decoded string of token id i (precomputed once per tokenizer).
    """
    bsz, resp_len = responses.shape
    out = torch.zeros((bsz, resp_len), dtype=torch.bool)
    lengths = response_mask.sum(dim=-1).tolist()
    n_pieces = len(vocab_pieces)
    for i in range(bsz):
        n = int(lengths[i])
        if n == 0:
            continue
        ids = responses[i, :n].tolist()
        pieces = [vocab_pieces[t] if 0 <= t < n_pieces else "" for t in ids]
        text = "".join(pieces)
        start_char = max((text.rfind(m) for m in markers), default=-1)
        if start_char < 0:
            continue
        pos = 0
        for j, piece in enumerate(pieces):
            if pos + len(piece) > start_char:
                out[i, j:n] = True
                break
            pos += len(piece)
    return out.to(responses.device)


# --------------------------------------------------------------------------------------------
# Teacher inputs in the teacher's own chat template
# --------------------------------------------------------------------------------------------
def build_vocab_maps(
    student_vocab: dict[str, int],
    teacher_vocab: dict[str, int],
    overrides: dict[str, str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map token ids between two tokenizers by token string. -1 where no counterpart exists.

    overrides: {student_token_string: teacher_token_string} for specials with different names.
    """
    s_size = max(student_vocab.values()) + 1
    t_size = max(teacher_vocab.values()) + 1
    s2t = torch.full((s_size,), -1, dtype=torch.long)
    t2s = torch.full((t_size,), -1, dtype=torch.long)
    for tok, sid in student_vocab.items():
        tid = teacher_vocab.get(tok)
        if tid is not None:
            s2t[sid] = tid
            if t2s[tid] < 0:
                t2s[tid] = sid
    for s_tok, t_tok in (overrides or {}).items():
        if s_tok in student_vocab and t_tok in teacher_vocab:
            s2t[student_vocab[s_tok]] = teacher_vocab[t_tok]
            t2s[teacher_vocab[t_tok]] = student_vocab[s_tok]
    return s2t, t2s


def remap_ids(ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Apply an id map; ids outside the table or negative map to -1."""
    table = table.to(ids.device)
    in_range = (ids >= 0) & (ids < table.numel())
    safe = torch.where(in_range, ids, torch.zeros_like(ids))
    return torch.where(in_range, table[safe], torch.full_like(ids, -1))


def strip_prompt_response_overlap(prompt_ids: list[int], response_ids: list[int], max_overlap: int = 8) -> list[int]:
    """Drop the longest prompt suffix that equals a response prefix (e.g. a template-injected
    ``<think>\\n`` that the student generated itself), so the response is scored token-for-token."""
    limit = min(max_overlap, len(prompt_ids), len(response_ids))
    for k in range(limit, 0, -1):
        if prompt_ids[-k:] == response_ids[:k]:
            return prompt_ids[:-k]
    return prompt_ids


def assemble_teacher_batch(
    prompt_ids: Sequence[Sequence[int]],
    responses: torch.Tensor,
    response_attention_mask: torch.Tensor,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """[left pad][teacher prompt][student response (teacher ids), right padded] — same layout verl
    uses, so logits[:, -R-1:-1] line up with the student's response positions exactly."""
    bsz, resp_len = responses.shape
    max_prompt = max(len(p) for p in prompt_ids)
    input_ids = torch.full((bsz, max_prompt + resp_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_prompt + resp_len), dtype=torch.long)
    for i, p in enumerate(prompt_ids):
        if len(p) > 0:
            input_ids[i, max_prompt - len(p) : max_prompt] = torch.as_tensor(list(p), dtype=torch.long)
            attention_mask[i, max_prompt - len(p) : max_prompt] = 1
    input_ids[:, max_prompt:] = responses.long().cpu()
    attention_mask[:, max_prompt:] = response_attention_mask.long().cpu()
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": responses.long().cpu(),
    }


def add_answer_hint(messages: Sequence[dict], answer: str, template: str) -> list[dict]:
    """Return a copy of the chat with the privileged hint appended to the last user turn."""
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user":
            m["content"] = str(m.get("content", "")) + template.format(answer=answer)
            return out
    raise ValueError("No user message to attach the privileged answer to")


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------
def masked_position_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask.to(x.dtype)
    return float((x * m).sum() / m.sum().clamp_min(1.0))
