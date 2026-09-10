#!/usr/bin/env python
"""Push saves/unlearn/baselines/tofu/<method>/<model>/<split> checkpoints to the HF Hub.

One public repo per checkpoint, named like open-unlearning's:
    <namespace>/tofu_<model>_<split>_<method>
Resumable: a repo whose Hub file tree already matches the local files (path + size)
is skipped, so the script can simply be re-run after an interruption.

Usage (from repo root, inside the `unlearning` env):
    python scripts/push_baselines_to_hub.py --dry-run
    python scripts/push_baselines_to_hub.py --sizes 1B            # just 1B models
    python scripts/push_baselines_to_hub.py                       # 1B -> 3B -> 8B
"""
import argparse
import fnmatch
import json
import sys
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils import HfHubHTTPError

ROOT = Path("saves/unlearn/baselines/tofu")
SIZE_TO_MODEL = {
    "1B": "Llama-3.2-1B-Instruct",
    "3B": "Llama-3.2-3B-Instruct",
    "8B": "Llama-3.1-8B-Instruct",
}
LICENSE = {"Llama-3.2-1B-Instruct": "llama3.2", "Llama-3.2-3B-Instruct": "llama3.2",
           "Llama-3.1-8B-Instruct": "llama3.1"}
# Files shipped per checkpoint (glob patterns relative to the checkpoint dir).
INCLUDE = [
    "config.json", "generation_config.json",
    "*.safetensors", "model.safetensors.index.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    "trainer_state.json", "profiling.json",
    "*.log",                 # <method>.log
    ".hydra/*",              # config.yaml / hydra.yaml / overrides.yaml
    "evals/*",               # TOFU_EVAL.json, TOFU_SUMMARY.json, eval.log, profiling.json
]
README = "README.md"


def local_files(ckpt: Path) -> dict[str, int]:
    out = {}
    for f in ckpt.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(ckpt).as_posix()
        if any(fnmatch.fnmatch(rel, pat) for pat in INCLUDE):
            out[rel] = f.stat().st_size
    return out


def remote_files(api: HfApi, repo_id: str) -> dict[str, int] | None:
    try:
        tree = api.list_repo_tree(repo_id, recursive=True)
        return {t.path: (t.lfs.size if getattr(t, "lfs", None) else t.size)
                for t in tree if hasattr(t, "size")}
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return None
        raise


def model_card(method: str, model: str, split: str, ckpt: Path) -> str:
    cfg = {}
    hyd = ckpt / ".hydra" / "config.yaml"
    if hyd.exists():
        try:
            import yaml
            cfg = yaml.safe_load(hyd.read_text()) or {}
        except Exception:
            cfg = {}
    base = (cfg.get("model", {}).get("model_args", {})
            .get("pretrained_model_name_or_path", f"open-unlearning/tofu_{model}_full"))
    method_args = cfg.get("trainer", {}).get("method_args", {})
    summary = {}
    sp = ckpt / "evals" / "TOFU_SUMMARY.json"
    if sp.exists():
        try:
            summary = json.loads(sp.read_text())
        except Exception:
            summary = {}
    lines = [
        "---",
        f"license: {LICENSE[model]}",
        f"base_model: {base}",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "datasets:",
        "- locuslab/TOFU",
        "tags:",
        "- unlearning",
        "- tofu",
        f"- {method}",
        f"- {split}",
        "---",
        "",
        f"# tofu_{model}_{split}_{method}",
        "",
        f"`{base}` unlearned on the TOFU `{split}` split with **{method}**, "
        "trained with the [open-unlearning](https://github.com/locuslab/open-unlearning) framework. "
        "Used as a weight-unlearning baseline / draft model in the "
        "[Speculative-Decoding-Unlearning](https://github.com/JoaoVitorBoer/Speculative-Decoding-Unlearning) project.",
        "",
        "Full training config: `.hydra/config.yaml`. TOFU evaluation outputs: `evals/`.",
        "",
    ]
    if method_args:
        lines += ["## Method hyperparameters", "", "```yaml"]
        lines += [f"{k}: {v}" for k, v in method_args.items()]
        lines += ["```", ""]
    if summary:
        lines += ["## TOFU summary metrics", "", "| metric | value |", "|---|---|"]
        lines += [f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |"
                  for k, v in sorted(summary.items())]
        lines += [""]
    return "\n".join(lines)


def push_one(api: HfApi, ns: str, method: str, model: str, split: str, dry: bool) -> str:
    ckpt = ROOT / method / model / split
    repo_id = f"{ns}/tofu_{model}_{split}_{method}"
    files = local_files(ckpt)
    if not any(p.endswith(".safetensors") for p in files):
        return f"SKIP (no weights) {repo_id}"
    total_gb = sum(files.values()) / 1e9

    remote = remote_files(api, repo_id)
    if remote is not None:
        missing = {p: s for p, s in files.items() if remote.get(p) != s}
        if not missing:
            return f"DONE (already complete) {repo_id}"
        note = f"incomplete on hub, {len(missing)}/{len(files)} files differ"
    else:
        note = "new repo"
    if dry:
        return f"WOULD PUSH {repo_id}  [{note}; {len(files)} files, {total_gb:.1f} GB]"

    t0 = time.time()
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(ckpt / rel)) for rel in sorted(files)]
    ops.append(CommitOperationAdd(path_in_repo=README,
                                  path_or_fileobj=model_card(method, model, split, ckpt).encode()))
    api.create_commit(repo_id, operations=ops, repo_type="model",
                      commit_message=f"Upload {method} {model} {split} checkpoint")
    # verify
    remote = remote_files(api, repo_id) or {}
    bad = [p for p, s in files.items() if remote.get(p) != s]
    dt = time.time() - t0
    if bad:
        return f"FAILED verify {repo_id}: {bad}"
    return f"PUSHED {repo_id}  [{note}; {total_gb:.1f} GB in {dt/60:.1f} min]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", default="JoaoBoer")
    ap.add_argument("--sizes", nargs="+", default=["1B", "3B", "8B"], choices=list(SIZE_TO_MODEL))
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--splits", nargs="*", default=["forget01", "forget05", "forget10"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    api = HfApi()
    who = api.whoami()["name"]
    print(f"logged in as {who}; pushing to namespace {a.namespace}", flush=True)
    methods = a.methods or sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    n_ok = n_fail = 0
    for size in a.sizes:
        model = SIZE_TO_MODEL[size]
        for method in methods:
            for split in a.splits:
                if not (ROOT / method / model / split).is_dir():
                    continue
                for attempt in range(3):
                    try:
                        msg = push_one(api, a.namespace, method, model, split, a.dry_run)
                        break
                    except Exception as e:  # network hiccup: retry
                        msg = f"ERROR {method}/{model}/{split}: {type(e).__name__}: {e}"
                        print(f"[{time.strftime('%H:%M:%S')}] {msg} (attempt {attempt+1}/3)", flush=True)
                        time.sleep(30)
                print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
                if msg.startswith(("FAILED", "ERROR")):
                    n_fail += 1
                else:
                    n_ok += 1
    print(f"finished: {n_ok} ok, {n_fail} failed", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
