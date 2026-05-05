"""Summarize SUD unlearning results into a CSV.

Walks `saves/unlearn/sud/results/target-*/draft-*/<split>/alpha-*/TOFU_SUMMARY.json`
(skipping the `baselines/` subtree), then for each run computes:
  - memorization        HM(1 - {extraction_strength, exact_memorization,
                                forget_Q_A_PARA_Prob, forget_truth_ratio})
  - model_utility       passed through from the summary
  - agg_memorization    HM(model_utility, memorization)
  - privacy_score       HM(s_MIA per {LOSS, ZLib, Min-k, Min-k++})

s_MIA uses the SUD paper's piecewise normalization against AUC_retain pulled
from saves/eval/tofu_<model>_retain{90,95,99}/TOFU_SUMMARY.json based on the
forget split (forget10->retain90, forget05->retain95, forget01->retain99).

Usage:
    python scripts/summarize_sud_results.py \
        --results-root saves/unlearn/sud/results \
        --eval-root    saves/eval \
        --output       saves/unlearn/sud/results/summary.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


MIA_KEYS = ["mia_loss", "mia_zlib", "mia_min_k", "mia_min_k_plus_plus"]

MEM_SOURCE_COLS = [
    "extraction_strength",
    "exact_memorization",
    "forget_Q_A_PARA_Prob",
    "forget_truth_ratio",
]

SPLIT_TO_RETAIN = {
    "forget10": "retain90",
    "forget05": "retain95",
    "forget01": "retain99",
}

TARGET_PREFIX = "target-open-unlearning_"
DRAFT_PREFIX = "draft-open-unlearning_"


def harmonic_mean_frame(df: pd.DataFrame) -> pd.Series:
    """Row-wise harmonic mean. NaN if any value is NaN; 0 if any value <= 0."""
    arr = df.to_numpy(dtype=float)
    n = arr.shape[1]
    out = np.full(arr.shape[0], np.nan)
    for i, row in enumerate(arr):
        if np.any(np.isnan(row)):
            continue
        if np.any(row <= 0):
            out[i] = 0.0
            continue
        out[i] = n / np.sum(1.0 / row)
    return pd.Series(out, index=df.index)


def compute_memorization(df: pd.DataFrame) -> pd.Series:
    missing = [c for c in MEM_SOURCE_COLS if c not in df.columns]
    if missing:
        print(f"WARNING: cannot compute memorization; missing columns: {missing}")
        return pd.Series(np.nan, index=df.index)
    return harmonic_mean_frame(1.0 - df[MEM_SOURCE_COLS])


def compute_agg_memorization(df: pd.DataFrame) -> pd.Series:
    if "model_utility" not in df.columns or "memorization" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return harmonic_mean_frame(df[["model_utility", "memorization"]])


def s_mia(auc: float, ref: float) -> float:
    """SUD paper's piecewise normalization: maps AUC=ref -> 1, falls off linearly."""
    if auc > ref:
        return 1.0 - (auc - ref) / (1.0 - ref)
    return 1.0 - (ref - auc) / ref


def model_name_from_target(target_dir_name: str) -> str | None:
    """target-open-unlearning_tofu_Llama-3.2-1B-Instruct_full -> Llama-3.2-1B-Instruct"""
    if not target_dir_name.startswith(TARGET_PREFIX):
        return None
    rest = target_dir_name[len(TARGET_PREFIX):]
    # strip leading "tofu_"
    if rest.startswith("tofu_"):
        rest = rest[len("tofu_"):]
    # strip trailing "_full"
    if rest.endswith("_full"):
        rest = rest[: -len("_full")]
    return rest or None


def parse_alpha(alpha_dir_name: str) -> float | None:
    m = re.fullmatch(r"alpha-([0-9]*\.?[0-9]+)", alpha_dir_name)
    return float(m.group(1)) if m else None


def load_retain_aucs(eval_root: Path, model: str, split: str) -> dict[str, float] | None:
    """Returns {mia_loss: auc, ...} for the matching retain model, or None if missing."""
    retain_split = SPLIT_TO_RETAIN.get(split)
    if retain_split is None:
        print(f"WARNING: unknown split '{split}' (no retain mapping)")
        return None
    summary_path = eval_root / f"tofu_{model}_{retain_split}" / "TOFU_SUMMARY.json"
    if not summary_path.exists():
        print(f"WARNING: retain summary not found: {summary_path}")
        return None
    data = json.loads(summary_path.read_text())
    aucs: dict[str, float] = {}
    for k in MIA_KEYS:
        if k not in data:
            print(f"WARNING: {summary_path} missing key {k}")
            return None
        aucs[k] = float(data[k])
    return aucs


def compute_privacy_score(summary: dict, retain_aucs: dict[str, float] | None) -> float:
    if retain_aucs is None:
        return float("nan")
    s_values: list[float] = []
    for k in MIA_KEYS:
        if k not in summary:
            print(f"WARNING: run summary missing {k}; privacy_score=NaN")
            return float("nan")
        s_values.append(s_mia(float(summary[k]), retain_aucs[k]))
    arr = np.array(s_values, dtype=float)
    if np.any(arr <= 0):
        return 0.0
    return float(len(arr) / np.sum(1.0 / arr))


def collect_rows(results_root: Path, eval_root: Path) -> list[dict]:
    retain_cache: dict[tuple[str, str], dict[str, float] | None] = {}
    rows: list[dict] = []

    for summary_path in sorted(results_root.glob("target-*/draft-*/*/alpha-*/TOFU_SUMMARY.json")):
        rel_parts = summary_path.relative_to(results_root).parts
        # parts: target-..., draft-..., <split>, alpha-..., TOFU_SUMMARY.json
        target_dir, draft_dir, split, alpha_dir, _ = rel_parts

        target = target_dir[len(TARGET_PREFIX):] if target_dir.startswith(TARGET_PREFIX) else target_dir
        draft = draft_dir[len(DRAFT_PREFIX):] if draft_dir.startswith(DRAFT_PREFIX) else draft_dir
        alpha = parse_alpha(alpha_dir)

        try:
            summary = json.loads(summary_path.read_text())
        except Exception as e:
            print(f"WARNING: failed to read {summary_path}: {e}")
            continue

        model = model_name_from_target(target_dir)
        cache_key = (model or "", split)
        if cache_key not in retain_cache:
            retain_cache[cache_key] = (
                load_retain_aucs(eval_root, model, split) if model else None
            )
        retain_aucs = retain_cache[cache_key]

        row: dict = {
            "target": target,
            "draft": draft,
            "split": split,
            "alpha": alpha,
            "path": str(summary_path.relative_to(results_root)),
        }
        # carry through everything from the summary, then overwrite/add derived metrics
        row.update(summary)
        row["privacy_score"] = compute_privacy_score(summary, retain_aucs)
        rows.append(row)

    return rows


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=repo_root / "saves/unlearn/sud/results")
    parser.add_argument("--eval-root", type=Path, default=repo_root / "saves/eval")
    parser.add_argument("--output", type=Path, default=repo_root / "saves/unlearn/sud/results/summary.csv")
    args = parser.parse_args()

    if not args.results_root.exists():
        raise SystemExit(f"results-root does not exist: {args.results_root}")

    rows = collect_rows(args.results_root, args.eval_root)
    if not rows:
        print(f"No TOFU_SUMMARY.json files matched under {args.results_root}")
        return

    df = pd.DataFrame(rows)
    df["memorization"] = compute_memorization(df)
    df["agg_memorization"] = compute_agg_memorization(df)

    leading = ["target", "draft", "split", "alpha",
               "model_utility", "memorization", "agg_memorization", "privacy_score"]
    cols = [c for c in leading if c in df.columns] + [c for c in df.columns if c not in leading]
    df = df[cols].sort_values(["target", "draft", "split", "alpha"], na_position="last")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
