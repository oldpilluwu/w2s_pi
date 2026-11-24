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
"""CPU tests for verl/trainer/ppo/prob_delta.py (imported by path: no ray/vllm needed)."""

import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).resolve().parents[3] / "verl" / "trainer" / "ppo" / "prob_delta.py"
_spec = importlib.util.spec_from_file_location("prob_delta", _PATH)
pd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pd)


def _logp(p):
    return torch.log(torch.as_tensor(p, dtype=torch.float32))


# ---------------------------------------------------------------------------------------- support
def test_union_support_dedupes_and_fills_holes():
    s = torch.tensor([[[5, 3]]])
    t = torch.tensor([[[3, 7]]])
    r = torch.tensor([[[-1, 5]]])
    ids, valid = pd.build_union_support([s, t, r])
    assert ids.tolist() == [[[5, 3, 3, 7, 5, 5]]]  # hole (-1) filled with student top-1 (5)
    assert valid.tolist() == [[[True, True, False, True, False, False]]]


def test_union_support_first_occurrence_is_kept():
    a = torch.tensor([[[9, 9, 2]]])
    ids, valid = pd.build_union_support([a])
    assert valid.tolist() == [[[True, False, True]]]


# ----------------------------------------------------------------------------------------- target
def _full_support(*probs):
    """Probabilities over a 4-token vocab, support = all 4 tokens."""
    valid = torch.ones(1, 1, 4, dtype=torch.bool)
    return [_logp(p).view(1, 1, 4) for p in probs], valid


def test_target_is_clipped_additive_mixture():
    (s0, t, r), valid = _full_support([0.4, 0.3, 0.2, 0.1], [0.1, 0.7, 0.1, 0.1], [0.5, 0.2, 0.2, 0.1])
    p_star, diag = pd.compute_prob_delta_target(s0, t, r, valid, alpha_rl=1.0)
    raw = torch.tensor([0.4 - 0.4, 0.3 + 0.5, 0.2 - 0.1, 0.1 + 0.0])
    expected = raw.clamp_min(0) / raw.clamp_min(0).sum()
    assert torch.allclose(p_star.view(-1), expected, atol=1e-6)
    assert torch.allclose(diag["tv_rl"], torch.tensor([[0.5]]), atol=1e-6)


def test_negative_mass_is_clipped():
    (s0, t, r), valid = _full_support([0.1, 0.6, 0.2, 0.1], [0.0001, 0.8, 0.1, 0.0999], [0.6, 0.2, 0.1, 0.1])
    p_star, diag = pd.compute_prob_delta_target(s0, t, r, valid, alpha_rl=1.0)
    assert p_star[0, 0, 0] == 0.0
    assert torch.isclose(p_star.sum(), torch.tensor(1.0))
    assert diag["clip_frac"].item() > 0
    assert diag["clipped_mass"].item() > 0


def test_pi_with_unit_alphas_equals_s0_plus_tpi_minus_r():
    (s0, t, r, tf), valid = _full_support(
        [0.4, 0.3, 0.2, 0.1], [0.1, 0.6, 0.2, 0.1], [0.3, 0.3, 0.2, 0.2], [0.05, 0.25, 0.6, 0.1]
    )
    w = torch.ones(1, 1)
    p_star, _ = pd.compute_prob_delta_target(s0, t, r, valid, 1.0, 1.0, tf, w)
    raw = (s0.exp() + tf.exp() - r.exp()).clamp_min(0).view(-1)
    assert torch.allclose(p_star.view(-1), raw / raw.sum(), atol=1e-6)


def test_zero_pi_weight_matches_no_pi():
    (s0, t, r, tf), valid = _full_support(
        [0.4, 0.3, 0.2, 0.1], [0.1, 0.6, 0.2, 0.1], [0.3, 0.3, 0.2, 0.2], [0.05, 0.25, 0.6, 0.1]
    )
    a, _ = pd.compute_prob_delta_target(s0, t, r, valid, 1.0, 1.0, tf, torch.zeros(1, 1))
    b, _ = pd.compute_prob_delta_target(s0, t, r, valid, 1.0, 0.0)
    assert torch.allclose(a, b)


def test_invalid_slots_get_zero_target_and_neg_inf_is_safe():
    s0 = _logp([0.5, 0.5, 0.5]).view(1, 1, 3)
    t = torch.tensor([[[-0.5, -float("inf"), -0.5]]])
    r = _logp([0.4, 0.4, 0.4]).view(1, 1, 3)
    valid = torch.tensor([[[True, True, False]]])
    p_star, _ = pd.compute_prob_delta_target(s0, t, r, valid)
    assert p_star[0, 0, 2] == 0
    assert torch.isfinite(p_star).all()
    assert torch.isclose(p_star.sum(), torch.tensor(1.0))


# -------------------------------------------------------------------------------------- advantage
def test_advantage_sums_to_zero_and_respects_masks():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 6)
    logp_old = torch.log_softmax(logits, -1)
    valid = torch.ones(2, 3, 6, dtype=torch.bool)
    valid[..., -1] = False
    p_star = torch.softmax(torch.randn(2, 3, 6), -1) * valid
    p_star = p_star / p_star.sum(-1, keepdim=True)
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    adv, diag = pd.compute_prob_delta_advantage(p_star, logp_old, valid, response_mask)
    assert torch.allclose(adv.sum(-1), torch.zeros(2, 3), atol=1e-6)
    assert (adv[..., -1] == 0).all()
    assert (adv[0, 2] == 0).all()
    assert (diag["forward_kl_to_target"] >= -1e-6).all()


@pytest.mark.parametrize("restricted", [False, True])
def test_pg_gradient_equals_forward_kl_gradient(restricted):
    """-sum_U A(v) log pi(v), A = p* - pi_old_bar, has the same logit gradient as
    the forward KL(p* || pi_bar) with pi_bar = pi renormalised over U (at theta = theta_old)."""
    torch.manual_seed(1)
    vocab = 12
    support = torch.tensor([0, 3, 4, 7, 8, 11]) if restricted else torch.arange(vocab)
    z = torch.randn(vocab, requires_grad=True)
    p_star = torch.softmax(torch.randn(support.numel()), -1)

    logp_old = torch.log_softmax(z.detach(), -1)[support].view(1, 1, -1)
    valid = torch.ones_like(logp_old, dtype=torch.bool)
    adv, _ = pd.compute_prob_delta_advantage(p_star.view(1, 1, -1), logp_old, valid)

    pg_loss = -(adv.view(-1) * torch.log_softmax(z, -1)[support]).sum()
    (g_pg,) = torch.autograd.grad(pg_loss, z)

    log_pi_bar = torch.log_softmax(z[support], -1)
    fkl = (p_star * (torch.log(p_star) - log_pi_bar)).sum()
    (g_fkl,) = torch.autograd.grad(fkl, z)
    assert torch.allclose(g_pg, g_fkl, atol=1e-6)


def test_direct_target_matches_decomposed_with_unit_alphas():
    (s0, t, r, tf), valid = _full_support(
        [0.4, 0.3, 0.2, 0.1], [0.1, 0.6, 0.2, 0.1], [0.3, 0.3, 0.2, 0.2], [0.05, 0.25, 0.6, 0.1]
    )
    w = torch.ones(1, 1)
    direct, diag = pd.compute_direct_pi_target(s0, tf, r, valid, alpha=1.0, weight=w)
    decomposed, _ = pd.compute_prob_delta_target(s0, t, r, valid, 1.0, 1.0, tf, w)
    assert torch.allclose(direct, decomposed, atol=1e-6)
    assert "tv_direct" in diag


def test_direct_target_zero_weight_is_anchor():
    (s0, r, tf), valid = _full_support([0.4, 0.3, 0.2, 0.1], [0.3, 0.3, 0.2, 0.2], [0.05, 0.25, 0.6, 0.1])
    p_star, _ = pd.compute_direct_pi_target(s0, tf, r, valid, alpha=1.0, weight=torch.zeros(1, 1))
    assert torch.allclose(p_star, s0.exp(), atol=1e-6)


@pytest.mark.parametrize("restricted", [False, True])
def test_explicit_forward_kl_loss_matches_pg_surrogate(restricted):
    torch.manual_seed(2)
    vocab = 10
    support = torch.tensor([1, 2, 5, 9]) if restricted else torch.arange(vocab)
    z = torch.randn(vocab, requires_grad=True)
    p_star = torch.softmax(torch.randn(support.numel()), -1).view(1, 1, -1)
    valid = torch.ones_like(p_star, dtype=torch.bool)

    lp = torch.log_softmax(z, -1)[support].view(1, 1, -1)
    fkl = pd.forward_kl_token_loss(lp, p_star, valid).sum()
    (g_fkl,) = torch.autograd.grad(fkl, z)

    adv, diag = pd.compute_prob_delta_advantage(p_star, lp.detach(), valid)
    pg = -(adv * torch.log_softmax(z, -1)[support].view(1, 1, -1)).sum()
    (g_pg,) = torch.autograd.grad(pg, z)

    assert torch.allclose(g_fkl, g_pg, atol=1e-6)
    assert torch.allclose(fkl.detach(), diag["forward_kl_to_target"].sum(), atol=1e-6)


def test_explicit_forward_kl_ignores_invalid_slots():
    lp = torch.log(torch.tensor([[[0.5, 0.3, 0.2]]])).requires_grad_(True)
    p_star = torch.tensor([[[0.6, 0.4, 0.0]]])
    valid = torch.tensor([[[True, True, False]]])
    loss = pd.forward_kl_token_loss(lp, p_star, valid).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert lp.grad[0, 0, 2] == 0


# ------------------------------------------------------------------------------------------- gate
def test_pi_gate_variants():
    rates = torch.tensor([0.0, 0.25, 0.5, 1.0])
    assert pd.compute_pi_gate("fail", rates, 0.5).tolist() == [1, 1, 0, 0]
    assert pd.compute_pi_gate("pass", rates, 0.5).tolist() == [0, 0, 1, 1]
    assert pd.compute_pi_gate("soft", rates).tolist() == [1.0, 0.75, 0.5, 0.0]
    assert pd.compute_pi_gate("all", rates).tolist() == [1, 1, 1, 1]
    assert pd.compute_pi_gate("all", None) is None
    with pytest.raises(ValueError):
        pd.compute_pi_gate("fail", None)
    with pytest.raises(ValueError):
        pd.compute_pi_gate("fail", torch.tensor([float("nan")]))
    with pytest.raises(ValueError):
        pd.compute_pi_gate("bogus", rates)


def test_answer_span_mask_marks_last_answer_line():
    pieces = ["<pad>", "So", " x", "=", "3", "\n", "Answer", ":", " 3", "Answer:"]
    responses = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 0], [1, 2, 9, 8, 0, 0, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0]])
    out = pd.answer_span_mask(responses, mask, pieces, ["Answer:"])
    # row 0 pieces: So | x | = | 3 | \n | Answer | : | 3   -> span starts at "Answer" (position 5)
    assert out[0].tolist() == [False] * 5 + [True, True, True, False]
    assert out[1].tolist() == [False, False, True, True] + [False] * 5


def test_answer_span_mask_no_marker():
    out = pd.answer_span_mask(torch.tensor([[1, 2]]), torch.tensor([[1, 1]]), ["a", "b", "c"], ["Answer:"])
    assert not out.any()


# ---------------------------------------------------------------------------------- teacher input
def test_vocab_maps_and_remap():
    student = {"a": 0, "b": 1, "<|im_end|>": 2, "<think>": 3}
    teacher = {"a": 0, "b": 1, "<eos>": 5, "<think>": 4}
    s2t, t2s = pd.build_vocab_maps(student, teacher, overrides={"<|im_end|>": "<eos>"})
    assert s2t.tolist() == [0, 1, 5, 4]
    assert pd.remap_ids(torch.tensor([0, 3, 2, 9, -1]), s2t).tolist() == [0, 4, 5, -1, -1]
    assert pd.remap_ids(torch.tensor([4, 5, 2]), t2s).tolist() == [3, 2, -1]


def test_strip_overlap_removes_injected_think():
    assert pd.strip_prompt_response_overlap([1, 2, 9, 10], [9, 10, 11]) == [1, 2]
    assert pd.strip_prompt_response_overlap([1, 2, 9], [9, 10]) == [1, 2]
    assert pd.strip_prompt_response_overlap([1, 2, 3], [9, 10]) == [1, 2, 3]


def test_assemble_teacher_batch_aligns_response_at_the_end():
    responses = torch.tensor([[7, 8, 0], [7, 0, 0]])
    resp_attn = torch.tensor([[1, 1, 0], [1, 0, 0]])
    out = pd.assemble_teacher_batch([[1, 2, 3], [4]], responses, resp_attn, pad_token_id=99)
    assert out["input_ids"].tolist() == [[1, 2, 3, 7, 8, 0], [99, 99, 4, 7, 0, 0]]
    assert out["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0]]
    # the logit at position -R-1 (last prompt token) predicts response[0]
    r = responses.shape[1]
    assert out["input_ids"][:, -r:].tolist() == responses.tolist()
    assert out["position_ids"][1].tolist()[2:4] == [0, 1]


def test_add_answer_hint_targets_last_user_turn_and_copies():
    chat = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    out = pd.add_answer_hint(chat, "42", " [ans={answer}]")
    assert out[1]["content"] == "q [ans=42]"
    assert chat[1]["content"] == "q"
    with pytest.raises(ValueError):
        pd.add_answer_hint([{"role": "system", "content": "s"}], "1", "{answer}")


# ---------------------------------------------------------------------------------------- PI-OPD
def _rand_logp(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.log_softmax(torch.randn(*shape, generator=g) * 2, dim=-1)


def test_opd_pi_alpha_zero_adds_nothing_but_returns_delta():
    s, t, tpi = (_rand_logp(2, 3, 4, seed=i) for i in range(3))
    rm, delta = pd.compute_opd_pi_terms(s, t, tpi, torch.ones(2, 3), alpha=0.0, clip=2.0)
    assert torch.count_nonzero(rm) == 0
    assert torch.allclose(delta, tpi - t)


def test_opd_pi_plus_direct_opd_equals_direct_form():
    # alpha=1, no clip, gate 1: pi_hat*(T - R) + pi_hat*(T_gold - T) == pi_hat*(T_gold - R)
    s, t, tpi, r = (_rand_logp(2, 5, 6, seed=i) for i in range(4))
    w = torch.softmax(s, -1)
    rm_rl = w * (t - r)  # Direct-OPD (dp_actor._compute_delta_opd_rm_scores)
    rm_pi, _ = pd.compute_opd_pi_terms(s, t, tpi, torch.ones(2, 5), alpha=1.0, clip=0.0)
    assert torch.allclose(rm_rl + rm_pi, w * (tpi - r), atol=1e-6)


def test_opd_pi_clip_and_position_weight():
    s = torch.log(torch.tensor([[[0.5, 0.5], [0.5, 0.5]]]))
    t = torch.log(torch.tensor([[[0.5, 0.5], [0.5, 0.5]]]))
    tpi = torch.tensor([[[0.0, -20.0], [0.0, -20.0]]])  # log-ratios ~ +0.69 and ~ -19.3
    weight = torch.tensor([[1.0, 0.0]])  # second position masked (e.g. answer span)
    rm, delta = pd.compute_opd_pi_terms(s, t, tpi, weight, alpha=2.0, clip=2.0)
    assert torch.allclose(rm[0, 0], torch.tensor([2.0 * 0.5 * (0.0 - torch.log(torch.tensor(0.5))), 2.0 * 0.5 * -2.0]))
    assert torch.count_nonzero(rm[0, 1]) == 0
    assert delta[0, 0, 1] < -19  # delta is returned unclipped


def test_opd_pi_unmappable_and_nonfinite_give_zero():
    s = _rand_logp(1, 2, 3, seed=0)
    t = torch.tensor([[[-1e4, -1.0, -2.0], [-float("inf"), -1.0, -2.0]]])
    tpi = torch.tensor([[[-1e4, -0.5, -2.0], [-float("inf"), -0.5, -2.0]]])
    rm, delta = pd.compute_opd_pi_terms(s, t, tpi, torch.ones(1, 2), alpha=1.0, clip=2.0)
    assert torch.isfinite(rm).all() and torch.isfinite(delta).all()
    assert delta[0, 0, 0] == 0 and delta[0, 1, 0] == 0


def test_opd_reward_gradient_is_expected_log_ratio_gradient():
    # On-policy, a top-k reward pi_hat * D with surrogate -sum_k A_k log pi_k has the gradient of
    # -E_pi[D] (exact expectation over the support): Direct-OPD / PI-OPD maximise the expected log-ratio.
    torch.manual_seed(0)
    logits = torch.randn(3, 5, requires_grad=True)
    d = torch.randn(3, 5)
    logp = torch.log_softmax(logits, -1)
    adv = torch.softmax(logits.detach(), -1) * d
    (g_surrogate,) = torch.autograd.grad(-(adv * logp).sum(), logits)
    (g_expected,) = torch.autograd.grad(-(torch.softmax(logits, -1) * d).sum(), logits)
    assert torch.allclose(g_surrogate, g_expected, atol=1e-6)


def test_opd_pi_metrics_values():
    s = torch.log(torch.tensor([[[0.75, 0.25], [0.75, 0.25], [0.75, 0.25]]]))
    t = torch.zeros(1, 3, 2)
    r = torch.full((1, 3, 2), -0.1)  # rl log-ratio 0.1 everywhere
    tpi = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [0.0, 0.0]]])  # pi log-ratio 1, 3 (clipped at 2), 0
    mask = torch.tensor([[1, 1, 0]])  # third position is padding
    answer = torch.tensor([[False, True, False]])
    rm, delta = pd.compute_opd_pi_terms(s, t, tpi, mask.float() * (~answer).float(), alpha=1.0, clip=2.0)
    m = pd.opd_pi_metrics(s, t, r, delta, rm, mask, answer, clip=2.0)
    assert m["opd_pi/pi_abs_wmean"] == pytest.approx((0.75 + 2.25) / 2)
    assert m["opd_pi/rl_abs_wmean"] == pytest.approx(0.1)
    assert m["opd_pi/pi_abs_wmean_reasoning"] == pytest.approx(0.75)
    assert m["opd_pi/pi_abs_wmean_answer_span"] == pytest.approx(2.25)
    assert m["opd_pi/pi_gt_rl_frac"] == pytest.approx(1.0)
    assert m["opd_pi/answer_span_frac_tokens"] == pytest.approx(0.5)
    assert m["opd_pi/clipped_weight_frac"] == pytest.approx(0.75 / 2)  # only the 3.0 entry, weight 0.75
    assert m["opd_pi/applied_abs_mean"] == pytest.approx(0.75 / 2)  # answer position masked
    assert m["opd_pi/pi_raw_absmax_p99"] == pytest.approx(3.0, abs=0.05)
