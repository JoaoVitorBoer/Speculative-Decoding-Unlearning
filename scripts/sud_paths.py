#!/usr/bin/env python3
"""Single source of truth for SUD results paths & naming.

Results layout (scheme A — combination folder on top, shortened model tags,
flat leaf):

    <root>/<combo>/target-<tag>/draft-<tag>/<forget_split>/a<alpha>_k<k>_s<seed>/
        TOFU_SUMMARY.json
        run_meta.json          # exact ids + config, so tools never parse paths

`<combo>` is one of the four Table-4.1 rows (role-only names):
    original-generic     original model  ×  generic small draft
    original-unlearned   original model  ×  unlearned draft
    original-retain      original model  ×  retain-only draft
    unlearned-retain     unlearned model ×  retain-only draft

`model_tag()` shortens a HF id / local path to a compact, unambiguous tag by
dropping the org prefix and the tofu role suffix (`_full` / `_retainNN` /
`_forgetNN`), while KEEPING the model family and — for unlearned checkpoints —
the unlearning method (e.g. `Llama-3.2-1B-Instruct_SimNPO`). The role that the
suffix used to carry is now carried by the `<combo>` folder, and the exact,
unshortened id is always preserved in `run_meta.json`.

Consumed by:
  * scripts/run_sud_*.sh        (via the `resolve` CLI below)
  * scripts/migrate_sud_results.py
  * scripts/summarize_sud_results.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# ── naming ────────────────────────────────────────────────────────────────────

def model_tag(model_id: str) -> str:
    """Shorten a model id / path to a compact, unambiguous folder tag.

    open-unlearning/tofu_Llama-3.2-3B-Instruct_full         -> Llama-3.2-3B-Instruct
    meta-llama/Llama-3.2-1B-Instruct                        -> Llama-3.2-1B-Instruct
    open-unlearning/tofu_Llama-3.2-1B-Instruct_retain95     -> Llama-3.2-1B-Instruct
    open-unlearning/unlearn_tofu_Llama-..._forget10_SimNPO_lr..._ep5
                                                            -> Llama-3.2-1B-Instruct_SimNPO
    saves/.../models/GradDiff_GDR/forget10/Llama-3.2-1B-Instruct/lr-1e-5_ep-10
                                                            -> Llama-3.2-1B-Instruct_GradDiff_GDR
    saves/unlearn/baselines/tofu/NPO_GDR/Llama-3.2-3B-Instruct/forget10
                                                            -> Llama-3.2-3B-Instruct_NPO_GDR
    muse-bench/MUSE-News_target                             -> MUSE-News
    muse-bench/MUSE-Books_retrain                           -> MUSE-Books
    saves/unlearn/baselines/muse/NPO_GDR/Llama-2-7b-hf/News -> Llama-2-7b-hf_NPO_GDR
    """
    mid = model_id.rstrip("/")

    # Unlearned hub checkpoint: keep family + unlearning method.
    m = re.search(r"unlearn_tofu_(Llama[^_]+)_forget\d+_([A-Za-z0-9]+)", mid)
    if m:
        return f"{m.group(1)}_{m.group(2)}"

    # MUSE weight-baseline checkpoint:
    #   .../baselines/muse/<METHOD>/<family>/<data_split>
    # (written by scripts/run_muse_weight_baselines.sh). The data split is
    # dropped — it is already carried by the <split> folder of the results path
    # and by the target tag (MUSE-News / MUSE-Books).
    m = re.search(r"/baselines/muse/([^/]+)/([^/]+)/(?:News|Books)$", mid)
    if m:
        return f"{m.group(2)}_{m.group(1)}"

    # TOFU weight-baseline checkpoint:
    #   .../baselines/tofu/<METHOD>/<family>/<forget_split>
    # (written by scripts/run_tofu_weight_baselines.sh). The split is dropped —
    # it is already carried by the <forget_split> folder of the results path.
    # The `tofu/` level is optional so checkpoints written before the benchmark
    # split of the baselines tree still resolve to the same tag.
    m = re.search(r"/baselines/(?:tofu/)?([^/]+)/(Llama[^/]+)/forget\d+$", mid)
    if m:
        return f"{m.group(2)}_{m.group(1)}"

    # Locally-trained checkpoint: .../models/<METHOD>/<split>/<family>/<hparams>
    m = re.search(r"/models/([^/]+)/[^/]+/(Llama[^/]+)/", mid + "/")
    if m:
        return f"{m.group(2)}_{m.group(1)}"

    base = mid.split("/")[-1]                                   # drop org / path prefix
    base = re.sub(r"^tofu_", "", base)                          # strip tofu_ prefix
    # Strip the role suffix — TOFU (_full / _retainNN / _forgetNN) and MUSE
    # (_target / _retrain) alike; the role is carried by the <combo> folder.
    base = re.sub(r"_(full|retain\d+|forget\d+|target|retrain)$", "", base)
    return base


def _is_unlearned(model_id: str) -> bool:
    # A weight-edited checkpoint: the hub name `unlearn_tofu_...`, a locally
    # trained `.../models/<METHOD>/...`, or a weight baseline
    # `.../baselines/<METHOD>/...`. NB: the org `open-unlearning` also
    # contains "unlearn", so match the checkpoint name, not a bare substring.
    return ("unlearn_tofu_" in model_id
            or "/models/" in model_id
            or "/baselines/" in model_id)


def target_role(model_id: str) -> str:
    """`original` (tofu _full / muse _target), `unlearned` (weight-edited), else `other`."""
    if _is_unlearned(model_id):
        return "unlearned"
    if model_id.rstrip("/").split("/")[-1].endswith(("_full", "_target")):
        return "original"
    return "other"


def draft_role(model_id: str) -> str:
    """`generic` / `retain` / `unlearned`, else `other`."""
    if _is_unlearned(model_id):
        return "unlearned"
    # TOFU retain-only checkpoint (_retainNN) or MUSE retrained-without-forget
    # checkpoint (_retrain) — both are the "retain draft" role.
    if re.search(r"_retain\d+|_retrain$", model_id.rstrip("/")):
        return "retain"
    # Off-the-shelf model: neither benchmark's finetuned checkpoint.
    if "tofu_" not in model_id and "MUSE-" not in model_id:
        return "generic"
    return "other"


_COMBO = {
    ("original", "generic"): "original-generic",
    ("original", "unlearned"): "original-unlearned",
    ("original", "retain"): "original-retain",
    ("unlearned", "retain"): "unlearned-retain",
}

COMBOS = tuple(_COMBO.values())


def combo_of(target_id: str, draft_id: str) -> str:
    """Infer the Table-4.1 combo folder from a (target, draft) pair.

    The run_sud_*.sh scripts pass their combo explicitly; this is used for
    migrating legacy runs that predate the combo folder.
    """
    tr, dr = target_role(target_id), draft_role(draft_id)
    return _COMBO.get((tr, dr), f"other-{tr}-{dr}")


def family_of(model_id: str) -> str | None:
    """HF model family (e.g. `Llama-3.2-1B-Instruct`) from an id or a tag."""
    m = re.search(r"(Llama-[0-9.]+-[0-9]+B-Instruct)", model_id)
    return m.group(1) if m else None


# ── paths ─────────────────────────────────────────────────────────────────────

def leaf_name(alpha, k, seed) -> str:
    return f"a{alpha}_k{k}_s{seed}"


def run_dir(root, combo: str, target_id: str, draft_id: str,
            split: str, alpha, k, seed) -> Path:
    return (
        Path(root)
        / combo
        / f"target-{model_tag(target_id)}"
        / f"draft-{model_tag(draft_id)}"
        / split
        / leaf_name(alpha, k, seed)
    )


def run_meta(combo: str, target_id: str, draft_id: str,
             split: str, alpha, k, seed) -> dict:
    """Full, unshortened run metadata written into every run dir."""
    return {
        "combo": combo,
        "target": target_id,
        "draft": draft_id,
        "target_tag": model_tag(target_id),
        "draft_tag": model_tag(draft_id),
        "forget_split": split,
        "alpha": float(alpha),
        "k_sud": int(k),
        "seed": int(seed),
    }


# ── CLI (used by run_sud_*.sh) ────────────────────────────────────────────────

def _cmd_resolve(a: argparse.Namespace) -> None:
    """Create the run dir, write run_meta.json, print the dir path to stdout."""
    combo = a.combo if a.combo and a.combo != "auto" else combo_of(a.target, a.draft)
    d = run_dir(a.root, combo, a.target, a.draft, a.split, a.alpha, a.k, a.seed)
    d.mkdir(parents=True, exist_ok=True)
    meta = run_meta(combo, a.target, a.draft, a.split, a.alpha, a.k, a.seed)
    (d / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(d)


def _cmd_tag(a: argparse.Namespace) -> None:
    print(model_tag(a.model))


def _cmd_combo(a: argparse.Namespace) -> None:
    print(combo_of(a.target, a.draft))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("resolve", help="mkdir run dir + write run_meta.json, print dir")
    r.add_argument("--root", required=True)
    r.add_argument("--combo", default="auto",
                   help="combo folder name, or 'auto' to infer from target/draft")
    r.add_argument("--target", required=True)
    r.add_argument("--draft", required=True)
    r.add_argument("--split", required=True)
    r.add_argument("--alpha", required=True)
    r.add_argument("--k", required=True)
    r.add_argument("--seed", required=True)
    r.set_defaults(func=_cmd_resolve)

    t = sub.add_parser("tag", help="print the short tag for a model id")
    t.add_argument("model")
    t.set_defaults(func=_cmd_tag)

    c = sub.add_parser("combo", help="print the inferred combo for target/draft")
    c.add_argument("--target", required=True)
    c.add_argument("--draft", required=True)
    c.set_defaults(func=_cmd_combo)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
