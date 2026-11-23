#!/usr/bin/env python3
"""Upload HF-format training checkpoints to a private Hugging Face repo as they are written.

Run next to training on the training machine (checkpoints saved with CKPT_CONTENTS=hf). For every
finished checkpoints/<run>/global_step_N/actor/huggingface it:
  1. converts the fp32 weights to bf16 (half the size; scripts/eval_vllm.py runs in bf16 anyway),
  2. uploads them to <repo>/<run>/global_step_N/actor/huggingface in one commit,
  3. writes a `.uploaded` marker, and with --delete-local removes the local fp32 copy.

Uses the token from `hf auth login` (or HF_TOKEN); a fine-grained token with write access to this one
repo is enough. Download and evaluate on another machine with scripts/hf_ckpt_eval.py.

    python scripts/hf_ckpt_upload.py --repo <user>/w2s-ckpts                    # watch until Ctrl-C
    python scripts/hf_ckpt_upload.py --repo <user>/w2s-ckpts --once             # upload what is ready, exit
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STEP_RE = re.compile(r"global_step_(\d+)")
MARKER = ".uploaded"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="private HF model repo, e.g. <user>/w2s-ckpts (created if missing)")
    p.add_argument("--ckpt-root", type=Path, default=REPO_ROOT / "checkpoints")
    p.add_argument("--runs", nargs="*", default=None, help="only these run names (default: all)")
    p.add_argument("--interval", type=int, default=120, help="seconds between scans")
    p.add_argument("--settle", type=int, default=60, help="a checkpoint must be unchanged this long before upload")
    p.add_argument("--once", action="store_true", help="upload what is ready now and exit")
    p.add_argument("--delete-local", action="store_true", help="delete the local fp32 HF copy after a verified upload")
    p.add_argument("--keep-fp32", action="store_true", help="upload as saved, without converting to bf16")
    return p.parse_args(argv)


def hf_dir_complete(hf_dir: Path, settle: int, now: float | None = None) -> bool:
    """True when save_pretrained has finished: config, tokenizer, every weight shard, nothing recently modified."""
    if not (hf_dir / "config.json").exists():
        return False
    index = hf_dir / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if not all((hf_dir / s).exists() for s in shards):
            return False
    elif not (hf_dir / "model.safetensors").exists():
        return False
    if not any(hf_dir.glob("tokenizer*")):
        return False
    now = time.time() if now is None else now
    newest = max(f.stat().st_mtime for f in hf_dir.iterdir() if f.is_file())
    return now - newest >= settle


def find_ready(ckpt_root: Path, runs, settle: int, now: float | None = None) -> list[tuple[str, int, Path]]:
    ready = []
    for hf_dir in sorted(ckpt_root.glob("*/global_step_*/actor/huggingface")):
        step_dir = hf_dir.parents[1]
        run = step_dir.parent.name
        m = STEP_RE.fullmatch(step_dir.name)
        if not m or (runs and run not in runs) or (step_dir / MARKER).exists():
            continue
        if hf_dir_complete(hf_dir, settle, now):
            ready.append((run, int(m.group(1)), hf_dir))
    return sorted(ready, key=lambda r: (r[0], r[1]))


def to_bf16(src: Path, dst: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.config.torch_dtype = torch.bfloat16
    model.save_pretrained(dst, safe_serialization=True)
    AutoTokenizer.from_pretrained(src).save_pretrained(dst)
    del model


def upload_one(api, repo: str, run: str, step: int, hf_dir: Path, keep_fp32: bool) -> str:
    path_in_repo = f"{run}/global_step_{step}/actor/huggingface"
    with tempfile.TemporaryDirectory(prefix=f"bf16_{run}_{step}_", dir=hf_dir.parents[2]) as tmp:
        folder = hf_dir
        if not keep_fp32:
            to_bf16(hf_dir, Path(tmp))
            folder = Path(tmp)
        for attempt in range(1, 4):
            try:
                api.upload_folder(
                    repo_id=repo,
                    repo_type="model",
                    folder_path=str(folder),
                    path_in_repo=path_in_repo,
                    commit_message=f"{run} step {step}",
                )
                break
            except Exception as e:  # network hiccups: retry, then give up on this checkpoint for now
                print(f"[upload] {run} step {step}: attempt {attempt} failed: {e}", file=sys.stderr)
                if attempt == 3:
                    raise
                time.sleep(30 * attempt)
        remote = set(api.list_repo_files(repo, repo_type="model"))
        expected = {f"{path_in_repo}/{f.name}" for f in folder.iterdir() if f.is_file()}
        missing = expected - remote
        if missing:
            raise RuntimeError(f"upload incomplete, missing on the hub: {sorted(missing)[:3]}")
    return path_in_repo


def main(argv=None) -> None:
    args = parse_args(argv)
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    print(f"[upload] watching {args.ckpt_root} -> {args.repo} (private)", flush=True)
    while True:
        for run, step, hf_dir in find_ready(args.ckpt_root, args.runs, args.settle):
            t0 = time.time()
            print(f"[upload] {run} step {step}: converting and uploading ...", flush=True)
            try:
                path = upload_one(api, args.repo, run, step, hf_dir, args.keep_fp32)
            except Exception as e:
                print(f"[upload] {run} step {step}: FAILED, will retry next scan: {e}", file=sys.stderr, flush=True)
                continue
            (hf_dir.parents[1] / MARKER).write_text(f"{args.repo}:{path}\n")
            if args.delete_local:
                shutil.rmtree(hf_dir)
            print(f"[upload] {run} step {step}: done in {time.time() - t0:.0f}s -> {args.repo}/{path}", flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
