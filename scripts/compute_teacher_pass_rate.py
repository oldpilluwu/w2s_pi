#!/usr/bin/env python3
"""Pre-generate the post-RL teacher's answers and store a per-question pass rate.

Adds ``extra_info.teacher_pass_rate`` (fraction of k samples the verifier marks correct) to a
Direct-OPD training parquet. Needed for prob_delta.pi_gate in {fail, pass, soft}.

The teacher sees the prompt in ITS OWN chat template, as it does during prob_delta training.

Example:
    python scripts/compute_teacher_pass_rate.py \
        --model models/JustRL-DeepSeek-1.5B \
        --input datasets/train/skywork-or1-math-dapo-original.parquet \
        --output datasets/train/skywork-or1-math-dapo-original.passrate.parquet \
        --k 8 --max-tokens 8192
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "verl"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="post-RL teacher (e.g. JustRL-DeepSeek-1.5B)")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--k", type=int, default=8, help="samples per question")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--limit", type=int, default=None, help="only the first N rows (debugging)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from verl.utils.reward_score.ttrl_math import compute_score

    df = pd.read_parquet(args.input)
    if args.limit is not None:
        df = df.iloc[: args.limit].copy()

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tok.apply_chat_template(list(msgs), add_generation_prompt=True, tokenize=False) for msgs in df["prompt"]
    ]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    params = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    outputs = llm.generate(prompts, params)

    rates = []
    for out, rm in zip(outputs, df["reward_model"], strict=True):
        gt = rm["ground_truth"]
        correct = [bool(compute_score(o.text, str(gt))["acc"]) for o in out.outputs]
        rates.append(sum(correct) / len(correct))

    def with_rate(extra, rate):
        extra = dict(extra) if extra is not None else {}
        extra["teacher_pass_rate"] = float(rate)
        return extra

    df["extra_info"] = [with_rate(e, r) for e, r in zip(df["extra_info"], rates, strict=True)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    s = pd.Series(rates)
    summary = {
        "rows": len(s),
        "mean_pass_rate": float(s.mean()),
        "frac_all_fail": float((s == 0).mean()),
        "frac_all_pass": float((s == 1).mean()),
        "frac_below_0.5": float((s < 0.5).mean()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
