#!/usr/bin/env python3
"""Evaluate a model or a saved checkpoint with plain vLLM, outside the training job.

Same protocol as the in-training validation of scripts/train_prob_delta_qwen3_4b.sh: the student's
chat template with thinking on, temperature 0.7, top-p 0.95, up to 32768 new tokens, scored with the
ttrl_math checker on the `acc` field. Needs vLLM and this repo's environment, but no Ray, FSDP,
teacher models or matching GPU count, so it runs on any machine with one or more GPUs.

--model takes a Hugging Face model directory, or a checkpoint saved with CKPT_CONTENTS=hf
(checkpoints/<run>/global_step_N; actor/huggingface is appended automatically).

Examples:
    python scripts/eval_vllm.py --model models/Qwen3-4B                                  # base model
    python scripts/eval_vllm.py --model checkpoints/p1_nopi_union/global_step_60 --n 32
    CUDA_VISIBLE_DEVICES=1 python scripts/eval_vllm.py --model ... &                     # one checkpoint per GPU
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = [str(REPO_ROOT / "datasets" / "eval" / f) for f in ("aime24.parquet", "aime25.parquet", "hmmt_feb.parquet")]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model dir, or checkpoints/<run>/global_step_N saved with CKPT_CONTENTS=hf")
    p.add_argument("--data", nargs="+", default=DEFAULT_DATA, help="eval parquets (default: AIME24, AIME25, HMMT Feb)")
    p.add_argument("--n", type=int, default=32, help="samples per problem")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--max-prompt-len", type=int, default=1024)
    p.add_argument("--no-thinking", action="store_true", help="render prompts with enable_thinking=False")
    p.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="result name (default: <run>_s<step> or the model dir name)")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "logs" / "eval_vllm")
    p.add_argument("--save-text", action="store_true", help="also store every response text (large)")
    p.add_argument("--grade-workers", type=int, default=16)
    return p.parse_args(argv)


def resolve_model_dir(path: str) -> Path:
    p = Path(path)
    if re.fullmatch(r"global_step_\d+", p.name):
        p = p / "actor" / "huggingface"
    elif p.name == "actor":
        p = p / "huggingface"
    if not (p / "config.json").exists():
        raise SystemExit(f"no config.json in {p}")
    if not any(p.glob("*.safetensors")) and not any(p.glob("*.bin")):
        raise SystemExit(
            f"no weights in {p}: training checkpoints need CKPT_CONTENTS=hf "
            "(CKPT_CONTENTS=weights/full keep FSDP shards; use `experiments.sh eval_ckpt` for those)"
        )
    return p


def default_name(model_dir: Path) -> str:
    parts = model_dir.resolve().parts
    for i, part in enumerate(parts):
        m = re.fullmatch(r"global_step_(\d+)", part)
        if m and i > 0:
            return f"{parts[i - 1]}_s{m.group(1)}"
    return model_dir.name


_REWARD_FUNC = None


def _reward_func():
    """The training checker (ttrl_math), loaded on its own so graders don't import verl/ray."""
    global _REWARD_FUNC
    if _REWARD_FUNC is None:
        pkg_dir = REPO_ROOT / "verl" / "verl" / "utils" / "reward_score" / "ttrl_math"
        spec = importlib.util.spec_from_file_location(
            "ttrl_math", pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)]
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ttrl_math"] = mod
        spec.loader.exec_module(mod)
        _REWARD_FUNC = mod.reward_func
    return _REWARD_FUNC


def _grade(task):
    data_source, text, ground_truth = task
    reward_func = _reward_func()  # an import failure must stop the eval, not score everything 0
    try:
        res = reward_func(data_source, text, ground_truth)
    except Exception:
        return False, ""
    if isinstance(res, dict):
        return bool(res.get("acc", False)), str(res.get("pred", "") or "")
    return bool(res), ""


def summarize(records: list[dict], n: int) -> dict:
    """records: one per sample with keys data_source, problem, acc, pred, n_tokens, truncated."""
    by_source = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_source[r["data_source"]][r["problem"]].append(r)
    out = {}
    for source, problems in sorted(by_source.items()):
        per_problem_mean, best, maj = [], [], []
        for samples in problems.values():
            accs = [s["acc"] for s in samples]
            per_problem_mean.append(sum(accs) / len(accs))
            best.append(any(accs))
            preds = [s["pred"] for s in samples if s["pred"]]
            if preds:
                top = Counter(preds).most_common(1)[0][0]
                maj.append(next(s["acc"] for s in samples if s["pred"] == top))
            else:
                maj.append(False)
        samples = [s for ss in problems.values() for s in ss]
        k = len(per_problem_mean)
        out[source] = {
            "problems": k,
            f"mean@{n}": sum(per_problem_mean) / k,
            f"maj@{n}": sum(maj) / k,
            f"best@{n}": sum(best) / k,
            "truncated_frac": sum(s["truncated"] for s in samples) / len(samples),
            "mean_response_tokens": sum(s["n_tokens"] for s in samples) / len(samples),
        }
    return out


def main(argv=None) -> None:
    args = parse_args(argv)
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    # fail fast if the checker is broken here (it needs Linux: it times out with SIGALRM)
    if not _grade(("AIME25", "so the answer is \\boxed{70}", "70"))[0] or _grade(("AIME25", "\\boxed{71}", "70"))[0]:
        raise SystemExit("ttrl_math checker self-test failed; check the environment before evaluating")

    model_dir = resolve_model_dir(args.model)
    name = args.name or default_name(model_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.concat([pd.read_parquet(f) for f in args.data], ignore_index=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    prompts, meta = [], []
    for i, row in df.iterrows():
        ids = tok.apply_chat_template(
            list(row["prompt"]), add_generation_prompt=True, tokenize=True, enable_thinking=not args.no_thinking
        )
        if len(ids) > args.max_prompt_len:
            raise SystemExit(f"prompt {i} has {len(ids)} tokens > --max-prompt-len {args.max_prompt_len}")
        prompts.append(TokensPrompt(prompt_token_ids=list(ids)))
        meta.append((str(row["data_source"]), i, str(row["reward_model"]["ground_truth"])))

    # bf16 like the training rollout; HF checkpoints from training are written in fp32
    llm = LLM(
        model=str(model_dir),
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_len + args.max_tokens,
        enable_prefix_caching=True,
        seed=args.seed,
    )
    params = SamplingParams(
        n=args.n, temperature=args.temperature, top_p=args.top_p, top_k=-1, max_tokens=args.max_tokens
    )
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    gen_s = time.time() - t0

    tasks, records = [], []
    for (source, problem, gt), out in zip(meta, outputs, strict=True):
        for j, o in enumerate(out.outputs):
            tasks.append((source, o.text, gt))
            rec = {
                "data_source": source,
                "problem": problem,
                "sample": j,
                "n_tokens": len(o.token_ids),
                "truncated": o.finish_reason == "length",
            }
            if args.save_text:
                rec["text"] = o.text
            records.append(rec)
    with ProcessPoolExecutor(args.grade_workers, mp_context=mp.get_context("spawn")) as pool:
        for rec, (acc, pred) in zip(records, pool.map(_grade, tasks, chunksize=8), strict=True):
            rec["acc"], rec["pred"] = acc, pred

    summary = summarize(records, args.n)
    result = {
        "name": name,
        "model": str(model_dir),
        "settings": {k: v for k, v in vars(args).items() if k not in ("out_dir",)} | {"thinking": not args.no_thinking},
        "generation_seconds": gen_s,
        "results": summary,
    }
    (args.out_dir / f"{name}.json").write_text(json.dumps(result, indent=2, default=str))
    with open(args.out_dir / f"{name}.samples.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[eval_vllm] {name}: {len(prompts)} problems x {args.n} samples, generation {gen_s / 60:.1f} min")
    for source, m in summary.items():
        print(
            f"val-core/{source}/acc/mean@{args.n}:{m[f'mean@{args.n}']:.4f}  maj@{args.n}:{m[f'maj@{args.n}']:.4f}"
            f"  best@{args.n}:{m[f'best@{args.n}']:.4f}  truncated:{m['truncated_frac']:.3f}"
            f"  mean_tokens:{m['mean_response_tokens']:.0f}"
        )
    print(f"[eval_vllm] wrote {args.out_dir / (name + '.json')}")


if __name__ == "__main__":
    main()
