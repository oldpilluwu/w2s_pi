#!/usr/bin/env python3
"""Diagnostics for comparing training runs: why does one method beat another?

Three parts, all read-only:

  1. training metrics from logs/<run>.out that the plain summary leaves out (Direct-OPD's own
     delta_opd/* metrics, advantages, KL loss), as means over step windows;
  2. the in-training validation dumps (checkpoints/<run>/outputs/validation_log/<run>/<step>.jsonl):
     per run and step, accuracy, truncation at the length limit, missing answers, looping, length of
     right vs wrong answers; then a per-problem table (pooled over the eval steps after 0) so you can
     see which problems a method gains or loses against the base model and against the others;
  3. with --drift (one free GPU): how far each saved checkpoint moved from the base model, token by
     token, on the base model's own validation text: KL(ckpt || base), entropy, top-1 agreement,
     probability mass moved outside the base model's top-16, split by position (inside / beyond the
     2048-token training length) and by how confident the base model was.

    python scripts/analyze_runs.py > logs/analysis.txt
    python scripts/analyze_runs.py --drift --drift-steps 30 150 > logs/analysis_drift.txt
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ORDER = ["p1_direct_pi_union", "p1_nopi_union", "p1_direct_opd", "p2_direct_pi_student", "p2_decomposed_union"]
SHORT = {
    "p1_direct_pi_union": "dpiU",
    "p1_nopi_union": "nopiU",
    "p1_direct_opd": "OPD",
    "p2_direct_pi_student": "dpiS",
    "p2_decomposed_union": "decU",
    "p2_nopi_student": "nopiS",
}
NUM = re.compile(r"^(?:np\.float64\()?(-?(?:[0-9.]+(?:e[-+]?\d+)?|nan|inf))\)?$")
WINDOWS = [(1, 5), (6, 30), (31, 60), (61, 90), (91, 120), (121, 150)]
TRAIN_PREFIXES = (
    "delta_opd/",
    "critic/advantages/",
    "critic/rewards/",
    "teacher",
    "actor/kl_loss",
    "actor/kl_coef",
    "actor/entropy",
    "actor/grad_norm",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="*", default=None, help="run names (default: every run found)")
    p.add_argument("--logs", type=Path, default=REPO_ROOT / "logs")
    p.add_argument("--ckpt-root", type=Path, default=REPO_ROOT / "checkpoints")
    p.add_argument("--eval-data", nargs="+", default=[str(REPO_ROOT / "datasets/eval/aime24.parquet"),
                                                     str(REPO_ROOT / "datasets/eval/aime25.parquet")])
    p.add_argument("--max-val-len", type=int, default=32768)
    p.add_argument("--excerpts", type=int, default=2, help="problems to show answer tails for")
    p.add_argument("--drift", action="store_true", help="part 3: checkpoint-vs-base drift (needs a GPU)")
    p.add_argument("--base-model", type=Path, default=REPO_ROOT / "models/Qwen3-4B")
    p.add_argument("--drift-steps", nargs="+", type=int, default=[30, 150])
    p.add_argument("--drift-tokens", type=int, default=4096, help="response tokens scored per text")
    p.add_argument("--drift-samples", type=int, default=2, help="base-model texts per problem")
    return p.parse_args(argv)


def ordered(names):
    return sorted(names, key=lambda r: (RUN_ORDER.index(r) if r in RUN_ORDER else 99, r))


# ---------------------------------------------------------------------------------------------
# 1. training metrics
# ---------------------------------------------------------------------------------------------
def parse_log(path: Path) -> dict[int, dict[str, float]]:
    steps = {}
    for line in open(path, errors="replace"):
        m = re.search(r"step:(\d+) - (.*)$", line.rstrip())
        if not m:
            continue
        d = {}
        for kv in m.group(2).split(" - "):
            if ":" not in kv:
                continue
            k, v = kv.split(":", 1)
            n = NUM.match(v.strip())
            if n:
                d[k] = float(n.group(1))
        steps[int(m.group(1))] = d
    return steps


def training_section(args, runs):
    print("=" * 100)
    print("1. TRAINING METRICS (window means; prob_delta/* already reported separately)")
    print("windows: " + " ".join(f"{a}-{b}" for a, b in WINDOWS))
    for run in runs:
        path = args.logs / f"{run}.out"
        if not path.exists():
            continue
        s = parse_log(path)
        keys = sorted(
            {k for d in s.values() for k in d if k.startswith(TRAIN_PREFIXES) and "_chunk_" not in k}
        )
        print(f"\n--- {run}")
        for k in keys:
            vals = []
            for a, b in WINDOWS:
                xs = [s[t][k] for t in range(a, b + 1) if t in s and k in s[t]]
                vals.append(f"{sum(xs) / len(xs):.4g}" if xs else "-")
            print(f"  {k}: " + " ".join(vals))


# ---------------------------------------------------------------------------------------------
# 2. validation dumps
# ---------------------------------------------------------------------------------------------
def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_problems(files):
    import pandas as pd

    problems = []
    for f in files:
        df = pd.read_parquet(f)
        for _, row in df.iterrows():
            msgs = [dict(m) for m in list(row["prompt"])]
            user = next(m["content"] for m in reversed(msgs) if m.get("role") == "user")
            problems.append(
                {
                    "source": str(row["data_source"]),
                    "key": norm(user)[:200],
                    "gt": str(row["reward_model"]["ground_truth"]),
                    "messages": msgs,
                }
            )
    return problems


def find_dumps(ckpt_root: Path, runs):
    dumps = defaultdict(dict)  # run -> step -> path
    for f in glob.glob(str(ckpt_root / "*" / "outputs" / "validation_log" / "*" / "*.jsonl")):
        p = Path(f)
        run = p.parent.name
        if runs and run not in runs:
            continue
        if p.stem.isdigit():
            dumps[run][int(p.stem)] = p
    return dumps


def load_dump(path: Path, problems, cache: dict):
    """Records grouped by problem index. cache maps normalized input -> problem index."""
    by_problem = defaultdict(list)
    for line in open(path, errors="replace"):
        if not line.strip():
            continue
        r = json.loads(line)
        inp = r.get("input", "")
        idx = cache.get(inp)
        if idx is None:
            n_inp = norm(inp)
            idx = next((i for i, pr in enumerate(problems) if pr["key"] in n_inp), -1)
            cache[inp] = idx
        if idx < 0:
            continue
        by_problem[idx].append(r)
    return by_problem


def looping(text: str) -> bool:
    tail = text[-6000:].encode("utf-8", "ignore")
    return len(tail) >= 3000 and len(zlib.compress(tail)) / len(tail) < 0.12


def dump_stats(by_problem, problems, max_len):
    """Per source: dict of aggregate stats."""
    out = {}
    per_source = defaultdict(list)
    for idx, recs in by_problem.items():
        per_source[problems[idx]["source"]].append((idx, recs))
    for src, items in sorted(per_source.items()):
        recs = [r for _, rs in items for r in rs]
        n = len(recs)
        acc = [bool(r.get("acc", r.get("score", 0) > 0)) for r in recs]
        lens = [int(r.get("response_length", 0)) for r in recs]
        trunc = [L >= max_len for L in lens]
        noans = [not str(r.get("pred", "")).strip() for r in recs]
        loop = [looping(r.get("output", "")) for r in recs]
        ok_len = [L for L, a in zip(lens, acc) if a]
        bad_len = [L for L, a in zip(lens, acc) if not a]
        wrong = n - sum(acc)
        npred = [len({str(r.get("pred", "")) for r in rs if str(r.get("pred", "")).strip()}) for _, rs in items]
        out[src] = {
            "n": n,
            "acc": sum(acc) / n,
            "trunc": sum(trunc) / n,
            "noans": sum(noans) / n,
            "wrong_trunc": sum(t and not a for t, a in zip(trunc, acc)) / max(wrong, 1),
            "wrong_noans": sum(x and not a for x, a in zip(noans, acc)) / max(wrong, 1),
            "loop": sum(loop) / n,
            "len_ok": sum(ok_len) / max(len(ok_len), 1),
            "len_bad": sum(bad_len) / max(len(bad_len), 1),
            "npred": sum(npred) / max(len(npred), 1),
        }
    return out


def problem_acc(by_problem):
    return {
        idx: sum(bool(r.get("acc", r.get("score", 0) > 0)) for r in recs) / len(recs)
        for idx, recs in by_problem.items()
        if recs
    }


def tail(text: str, n: int = 400) -> str:
    return text[-n:].replace("\n", " \\n ")


def validation_section(args, runs, problems, dumps):
    print("\n" + "=" * 100)
    print("2. VALIDATION DUMPS")
    print(
        "acc | trunc = hit the length limit | noans = no extractable answer | of wrong answers: "
        "share truncated / share without answer | loop = repetitive tail | mean tokens of right / "
        "wrong answers | distinct answers per problem"
    )
    cache = {}
    loaded = {}  # (run, step) -> by_problem
    hdr = f"{'run':>6} {'step':>4} {'src':>7} {'acc':>6} {'trunc':>6} {'noans':>6} {'w_trc':>6} {'w_noa':>6} {'loop':>6} {'len_ok':>7} {'len_bad':>7} {'npred':>6}"
    print(hdr)
    for run in runs:
        for step in sorted(dumps.get(run, {})):
            bp = load_dump(dumps[run][step], problems, cache)
            loaded[(run, step)] = bp
            for src, st in dump_stats(bp, problems, args.max_val_len).items():
                print(
                    f"{SHORT.get(run, run[:6]):>6} {step:>4} {src:>7} {st['acc']:6.3f} {st['trunc']:6.3f} {st['noans']:6.3f}"
                    f" {st['wrong_trunc']:6.3f} {st['wrong_noans']:6.3f} {st['loop']:6.3f} {st['len_ok']:7.0f}"
                    f" {st['len_bad']:7.0f} {st['npred']:6.2f}"
                )

    # base = any run's step 0
    base = next((bp for (run, step), bp in loaded.items() if step == 0), None)
    if base is None:
        print("\n(no step-0 dump found: per-problem table skipped)")
        return loaded
    base_acc = problem_acc(base)

    # pooled per-problem accuracy over steps > 0
    pooled = {}
    for run in runs:
        recs = defaultdict(list)
        for (r, step), bp in loaded.items():
            if r == run and step > 0:
                for idx, rs in bp.items():
                    recs[idx].extend(rs)
        if recs:
            pooled[run] = problem_acc(recs)
    cols = [r for r in runs if r in pooled]
    opd = "p1_direct_opd" if "p1_direct_opd" in pooled else None
    pd_runs = [r for r in cols if r != opd]

    print("\nPER-PROBLEM ACCURACY (pooled over evals after step 0; base = step 0)")
    print(f"{'#':>3} {'src':>7} {'gt':>8} {'base':>5} " + " ".join(f"{SHORT.get(r, r[:6]):>6}" for r in cols)
          + ("  OPD-pd" if opd and pd_runs else ""))
    rows = []
    for idx, pr in enumerate(problems):
        if idx not in base_acc:
            continue
        vals = [pooled[r].get(idx, float("nan")) for r in cols]
        diff = None
        if opd and pd_runs:
            diff = pooled[opd].get(idx, float("nan")) - sum(pooled[r].get(idx, 0.0) for r in pd_runs) / len(pd_runs)
        rows.append((idx, pr, vals, diff))
    for idx, pr, vals, diff in rows:
        line = f"{idx:>3} {pr['source']:>7} {pr['gt'][:8]:>8} {base_acc[idx]:5.2f} " + " ".join(f"{v:6.2f}" for v in vals)
        if diff is not None:
            line += f"  {diff:+6.2f}"
        print(line)

    # where do the differences come from?
    print("\nCHANGE VS BASE (problems): gained = base<=0.25 & run>=0.5 | lost = base>=0.75 & run<=0.5 |"
          " net = sum over problems of (run - base) in problems")
    for r in cols:
        gained = [i for i, _, _, _ in rows if base_acc[i] <= 0.25 and pooled[r].get(i, 0) >= 0.5]
        lost = [i for i, _, _, _ in rows if base_acc[i] >= 0.75 and pooled[r].get(i, 1) <= 0.5]
        net = sum(pooled[r].get(i, 0) - base_acc[i] for i, _, _, _ in rows)
        by_src = defaultdict(float)
        for i, pr, _, _ in rows:
            by_src[pr["source"]] += pooled[r].get(i, 0) - base_acc[i]
        src_txt = " ".join(f"{s}:{v:+.2f}" for s, v in sorted(by_src.items()))
        print(f"  {SHORT.get(r, r):>6}: gained {gained} lost {lost} net {net:+.2f} ({src_txt})")
    if opd and pd_runs:
        diffs = sorted(((d, i) for i, _, _, d in rows if d == d), reverse=True)
        pos = sum(d for d, _ in diffs if d > 0)
        neg = sum(d for d, _ in diffs if d < 0)
        print(f"\nOPD minus prob_delta mean: total {pos + neg:+.2f} problems = {pos:+.2f} where OPD is ahead,"
              f" {neg:+.2f} where it is behind")
        print("  largest OPD leads: " + ", ".join(f"#{i} {d:+.2f}" for d, i in diffs[:6]))
        print("  largest OPD deficits: " + ", ".join(f"#{i} {d:+.2f}" for d, i in diffs[-4:]))

        # answer tails for the problems where OPD leads most
        last = {r: max(s for (rr, s) in loaded if rr == r) for r in cols}
        for d, i in diffs[: args.excerpts]:
            pr = problems[i]
            print(f"\n--- problem #{i} ({pr['source']}, gt {pr['gt']}), OPD lead {d:+.2f}")
            for r in pd_runs[:1] + [opd]:
                recs = loaded[(r, last[r])].get(i, [])
                preds = Counter(str(x.get("pred", "")) or "<none>" for x in recs)
                print(f"  {SHORT.get(r, r)} step {last[r]}: answers {dict(preds.most_common(5))}")
                wrong = [x for x in recs if not x.get("acc")]
                right = [x for x in recs if x.get("acc")]
                ex = (wrong or right)[:1] if r != opd else (right or wrong)[:1]
                for x in ex:
                    print(f"    [{'right' if x.get('acc') else 'wrong'}, {x.get('response_length')} tokens] ...{tail(x.get('output', ''))}")
    return loaded


# ---------------------------------------------------------------------------------------------
# 3. checkpoint drift vs base
# ---------------------------------------------------------------------------------------------
def drift_section(args, runs, problems, loaded):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 100)
    print(f"3. DRIFT FROM BASE on the base model's own validation text ({args.drift_samples} per problem,"
          f" first {args.drift_tokens} response tokens)")
    base = next((bp for (run, step), bp in loaded.items() if step == 0), None)
    if base is None:
        print("no step-0 dump: cannot pick base-model texts")
        return
    tok = AutoTokenizer.from_pretrained(args.base_model)
    seqs = []  # (input_ids tensor, prompt_len)
    for idx, recs in sorted(base.items()):
        for r in recs[: args.drift_samples]:
            prompt = tok.apply_chat_template(problems[idx]["messages"], add_generation_prompt=True, tokenize=True,
                                             enable_thinking=True)
            resp = tok(r.get("output", ""), add_special_tokens=False)["input_ids"][: args.drift_tokens]
            if len(resp) < 16:
                continue
            seqs.append((torch.tensor(list(prompt) + resp), len(prompt)))
    print(f"{len(seqs)} texts")

    dev = "cuda"
    load = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **load).to(dev).eval()

    ckpts = []
    for run in runs:
        for step in args.drift_steps:
            d = args.ckpt_root / run / f"global_step_{step}" / "actor" / "huggingface"
            if (d / "config.json").exists():
                ckpts.append((run, step, d))
            else:
                print(f"  (missing {d})")

    buckets = [("pos<2048", lambda pos, hb: pos < 2048), ("pos>=2048", lambda pos, hb: pos >= 2048),
               ("baseH<0.1", lambda pos, hb: hb < 0.1), ("baseH>=0.1", lambda pos, hb: hb >= 0.1)]
    print(f"{'run':>6} {'step':>4} {'bucket':>10} {'frac':>5} {'KL(c||b)':>9} {'H_ckpt':>7} {'H_base':>7}"
          f" {'top1agr':>7} {'out16':>7} {'dlogp':>7}")
    for run, step, d in ckpts:
        model = AutoModelForCausalLM.from_pretrained(d, **load).to(dev).eval()
        acc = {name: defaultdict(float) for name, _ in buckets}
        with torch.no_grad():
            for ids, plen in seqs:
                ids = ids.to(dev)
                lb_all = base_model(input_ids=ids[None]).logits[0]
                lc_all = model(input_ids=ids[None]).logits[0]
                tgt = ids[plen:]
                for s in range(0, len(tgt), 1024):
                    e = min(s + 1024, len(tgt))
                    lb = torch.log_softmax(lb_all[plen - 1 + s: plen - 1 + e].float(), -1)
                    lc = torch.log_softmax(lc_all[plen - 1 + s: plen - 1 + e].float(), -1)
                    pc, pb = lc.exp(), lb.exp()
                    kl = (pc * (lc - lb)).sum(-1)
                    hc = -(pc * lc).sum(-1)
                    hb = -(pb * lb).sum(-1)
                    agree = (lc.argmax(-1) == lb.argmax(-1)).float()
                    out16 = 1 - pc.gather(-1, lb.topk(16, -1).indices).sum(-1)
                    t = tgt[s:e, None]
                    dlogp = (lc.gather(-1, t) - lb.gather(-1, t))[:, 0]
                    pos = torch.arange(s, e, device=dev)
                    for name, f in buckets:
                        m = f(pos, hb).float()
                        a = acc[name]
                        a["n"] += m.sum().item()
                        for k, v in (("kl", kl), ("hc", hc), ("hb", hb), ("agree", agree), ("out16", out16),
                                     ("dlogp", dlogp)):
                            a[k] += (v * m).sum().item()
                del lb_all, lc_all
        total = acc["pos<2048"]["n"] + acc["pos>=2048"]["n"]
        for name, _ in buckets:
            a = acc[name]
            n = max(a["n"], 1)
            print(f"{SHORT.get(run, run[:6]):>6} {step:>4} {name:>10} {a['n'] / max(total, 1):5.2f} {a['kl'] / n:9.4f}"
                  f" {a['hc'] / n:7.3f} {a['hb'] / n:7.3f} {a['agree'] / n:7.3f} {a['out16'] / n:7.4f} {a['dlogp'] / n:7.3f}")
        del model
        torch.cuda.empty_cache()


def main(argv=None):
    args = parse_args(argv)
    dumps = find_dumps(args.ckpt_root, args.runs)
    log_runs = [p.stem for p in args.logs.glob("p[0-9]_*.out")]
    runs = ordered(set(args.runs or (list(dumps) + log_runs)))
    training_section(args, runs)
    problems = load_problems(args.eval_data)
    loaded = validation_section(args, runs, problems, dumps)
    if args.drift:
        drift_section(args, runs, problems, loaded)


if __name__ == "__main__":
    main()
