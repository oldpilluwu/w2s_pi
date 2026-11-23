#!/usr/bin/env python3
"""Download checkpoints from the private HF repo and evaluate each one with scripts/eval_vllm.py.

Run on the eval machine. It polls <repo> for <run>/global_step_N/actor/huggingface folders (uploaded by
scripts/hf_ckpt_upload.py), skips any that already have logs/eval_vllm/<run>_s<N>.json, downloads the
next one, evaluates it and (with --delete-local) removes the download. Several copies can run at once,
one per GPU: each claims a checkpoint with a lock file first.

Uses the token from `hf auth login` (or HF_TOKEN); read access to the repo is enough. Arguments after
`--` go to eval_vllm.py.

    python scripts/hf_ckpt_eval.py --repo <user>/w2s-ckpts                       # watch until Ctrl-C
    CUDA_VISIBLE_DEVICES=1 python scripts/hf_ckpt_eval.py --repo <user>/w2s-ckpts &   # a second GPU
    python scripts/hf_ckpt_eval.py --repo <user>/w2s-ckpts --once -- --n 16 --data datasets/eval/aime25.parquet
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_RE = re.compile(r"^([^/]+)/global_step_(\d+)/actor/huggingface/config\.json$")


def parse_args(argv=None) -> tuple[argparse.Namespace, list[str]]:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True)
    p.add_argument("--download-root", type=Path, default=REPO_ROOT / "ckpts")
    p.add_argument("--results", type=Path, default=REPO_ROOT / "logs" / "eval_vllm")
    p.add_argument("--runs", nargs="*", default=None, help="only these run names (default: all)")
    p.add_argument("--interval", type=int, default=300, help="seconds between polls")
    p.add_argument("--once", action="store_true", help="evaluate what is on the hub now and exit")
    p.add_argument("--delete-local", action="store_true", help="delete each download after its eval succeeds")
    return p.parse_args(argv), extra


def list_checkpoints(files, runs=None) -> list[tuple[str, int]]:
    found = set()
    for f in files:
        m = CKPT_RE.match(f)
        if m and (not runs or m.group(1) in runs):
            found.add((m.group(1), int(m.group(2))))
    return sorted(found, key=lambda rs: (rs[1], rs[0]))  # earliest steps first, across runs


def claim(results: Path, name: str) -> bool:
    """Atomically claim a checkpoint; False if it is done, failed before, or another evaluator holds it."""
    results.mkdir(parents=True, exist_ok=True)
    if (results / f"{name}.json").exists() or (results / f"{name}.failed").exists():
        return False
    try:
        fd = os.open(results / f"{name}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    return True


def main(argv=None) -> None:
    args, extra = parse_args(argv)
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    print(f"[eval] polling {args.repo}; results in {args.results}", flush=True)
    while True:
        todo = [
            (run, step)
            for run, step in list_checkpoints(api.list_repo_files(args.repo, repo_type="model"), args.runs)
            if not (args.results / f"{run}_s{step}.json").exists()
            and not (args.results / f"{run}_s{step}.failed").exists()
        ]
        did_work = False
        for run, step in todo:
            name = f"{run}_s{step}"
            if not claim(args.results, name):
                continue
            did_work = True
            lock = args.results / f"{name}.lock"
            sub = f"{run}/global_step_{step}/actor/huggingface"
            try:
                print(f"[eval] {name}: downloading", flush=True)
                snapshot_download(args.repo, repo_type="model", allow_patterns=[f"{sub}/*"], local_dir=args.download_root)
                cmd = [sys.executable, str(REPO_ROOT / "scripts" / "eval_vllm.py"),
                       "--model", str(args.download_root / run / f"global_step_{step}"),
                       "--name", name, "--out-dir", str(args.results), *extra]
                print(f"[eval] {name}: {' '.join(cmd)}", flush=True)
                with open(args.results / f"{name}.log", "w") as log:
                    rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT).returncode
                if rc != 0:
                    # not retried automatically; delete <name>.failed to try again
                    (args.results / f"{name}.failed").write_text(f"eval_vllm.py exited {rc}\n")
                    print(f"[eval] {name}: FAILED (exit {rc}); see {args.results / (name + '.log')}", flush=True)
                    continue
                if args.delete_local:
                    shutil.rmtree(args.download_root / run / f"global_step_{step}", ignore_errors=True)
                print(f"[eval] {name}: done", flush=True)
            finally:
                lock.unlink(missing_ok=True)
            break  # re-list after each checkpoint so earlier steps from new uploads come first
        if args.once and not did_work:
            return
        if not did_work:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
