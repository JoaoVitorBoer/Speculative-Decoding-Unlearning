#!/usr/bin/env python3
"""One-shot migration of legacy SUD result folders into the combo-first scheme.

Legacy layout (two variants that coexist today):
    results/target-<slug>/draft-<slug>/<split>/alpha-<a>_k-<k>_seed-<s>/...   (flat)
    results/target-<slug>/draft-<slug>/<split>/alpha-<a>_k-<k>/seed-<s>/...   (nested)

New layout (see scripts/sud_paths.py):
    results/<combo>/target-<tag>/draft-<tag>/<split>/a<a>_k<k>_s<s>/
        <all original files, e.g. TOFU_SUMMARY.json, TOFU_EVAL.json, .hydra/>
        run_meta.json      (backfilled)

The combo is inferred from the (target, draft) roles. Each run directory is
moved wholesale (preserving every file it holds), run_meta.json is backfilled,
and now-empty legacy parent dirs are pruned.

Dry-run by default; pass --apply to perform the moves.

    python scripts/migrate_sud_results.py                 # show the plan
    python scripts/migrate_sud_results.py --apply         # do it
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sud_paths import combo_of, model_tag, run_dir, run_meta  # noqa: E402


TARGET_PREFIX = "target-"
DRAFT_PREFIX = "draft-"

# alpha-<a>_k-<k> then a '_' (flat) or '/' (nested) before seed-<s>.
_LEAF_RE = re.compile(r"alpha-([0-9]*\.?[0-9]+)_k-(\d+)[/_]seed-(\d+)")

# Known org prefixes to restore the '/' that path-sanitisation replaced with '_'.
_ORGS = ("open-unlearning", "meta-llama")


def desanitize(slug: str) -> str:
    """Reverse `id.replace('/', '_')` for the known orgs so tag/role logic works.

    target-open-unlearning_tofu_Llama-3.2-3B-Instruct_full
        -> open-unlearning/tofu_Llama-3.2-3B-Instruct_full
    """
    for org in _ORGS:
        if slug.startswith(org + "_"):
            return org + "/" + slug[len(org) + 1:]
    return slug


class Plan:
    __slots__ = ("src", "dst", "combo", "target", "draft", "split", "alpha", "k", "seed")

    def __init__(self, src, dst, combo, target, draft, split, alpha, k, seed):
        self.src, self.dst, self.combo = src, dst, combo
        self.target, self.draft, self.split = target, draft, split
        self.alpha, self.k, self.seed = alpha, k, seed


def build_plans(results_root: Path) -> tuple[list[Plan], list[Path]]:
    """Return (plans, skipped_run_dirs).

    A run directory is identified by the `.hydra/` folder Hydra writes for every
    eval, so incomplete runs (no TOFU_SUMMARY.json yet) migrate too and nothing
    is left orphaned in the legacy layout.
    """
    plans: list[Plan] = []
    skipped: list[Path] = []

    for hydra_dir in sorted(results_root.glob("target-*/draft-*/**/.hydra")):
        run_src = hydra_dir.parent
        parts = run_src.relative_to(results_root).parts  # (target-dir, draft-dir, split, *leaf)
        if len(parts) < 4:
            skipped.append(run_src)
            continue
        target_dir, draft_dir, split = parts[0], parts[1], parts[2]
        leaf = "/".join(parts[3:])

        m = _LEAF_RE.search(leaf)
        if m is None:
            skipped.append(run_src)
            continue
        alpha, k, seed = m.group(1), m.group(2), m.group(3)

        target = desanitize(target_dir[len(TARGET_PREFIX):])
        draft = desanitize(draft_dir[len(DRAFT_PREFIX):])
        combo = combo_of(target, draft)

        dst = run_dir(results_root, combo, target, draft, split, alpha, k, seed)
        plans.append(Plan(run_src, dst, combo, target, draft, split, alpha, k, seed))

    return plans, skipped


def prune_empty(root: Path) -> int:
    """Remove empty dirs under each legacy top-level `target-*` tree. Returns count."""
    removed = 0
    for top in sorted(root.glob("target-*")):
        if not top.is_dir():
            continue
        for d in sorted(top.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
                removed += 1
        if top.is_dir() and not any(top.iterdir()):
            top.rmdir()
            removed += 1
    return removed


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--results-root", type=Path,
                   default=repo_root / "saves/unlearn/sud/results")
    p.add_argument("--apply", action="store_true",
                   help="perform the moves (default: dry-run)")
    args = p.parse_args()

    root = args.results_root
    if not root.exists():
        raise SystemExit(f"results-root does not exist: {root}")

    plans, skipped = build_plans(root)
    if not plans:
        print(f"No legacy runs found under {root} (nothing to migrate).")
        return

    # Group counts by combo, and detect destination collisions.
    by_combo: dict[str, int] = {}
    collisions: list[Plan] = []
    for pl in plans:
        by_combo[pl.combo] = by_combo.get(pl.combo, 0) + 1
        if pl.dst.exists():
            collisions.append(pl)

    print(f"{'APPLY' if args.apply else 'DRY-RUN'}: {len(plans)} legacy runs -> combo scheme")
    for combo in sorted(by_combo):
        print(f"  {combo:20s} {by_combo[combo]:3d} runs")
    if skipped:
        print(f"\nSKIPPED {len(skipped)} unparseable run dirs:")
        for s in skipped[:10]:
            print(f"  {s.relative_to(root)}")
    if collisions:
        print(f"\nWARNING: {len(collisions)} destinations already exist and will be SKIPPED:")
        for pl in collisions[:10]:
            print(f"  {pl.dst.relative_to(root)}")

    # Show a couple of example moves.
    print("\nExample moves:")
    for pl in plans[:3]:
        print(f"  {pl.src.relative_to(root)}")
        print(f"    -> {pl.dst.relative_to(root)}")

    if not args.apply:
        print("\n(dry-run) re-run with --apply to perform the migration.")
        return

    moved = 0
    for pl in plans:
        if pl.dst.exists():
            continue  # collision already reported
        pl.dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(pl.src), str(pl.dst))
        meta = run_meta(pl.combo, pl.target, pl.draft, pl.split, pl.alpha, pl.k, pl.seed)
        (pl.dst / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        moved += 1

    pruned = prune_empty(root)
    print(f"\nMoved {moved} runs; backfilled run_meta.json; pruned {pruned} empty legacy dirs.")


if __name__ == "__main__":
    main()
