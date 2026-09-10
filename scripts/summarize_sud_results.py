"""Summarize SUD unlearning results into a CSV.

Walks `saves/unlearn/sud/results/<combo>/target-*/draft-*/<split>/<leaf>/TOFU_SUMMARY.json`
(skipping any `baselines/` subtree -- the weight baselines live under
`saves/unlearn/baselines/` and are summarized by scripts/summarize_sud_baselines.py).
Run config (combo, exact target/draft ids,
alpha, k, seed) is read from the run_meta.json beside each summary, with a
path-parsing fallback. Then for each run computes the three OpenUnlearning
evaluation dimensions and their overall aggregate (arXiv:2506.12618, App. F.1),
all combined via harmonic mean (HM):
  - memorization        HM(1 - extraction_strength, 1 - exact_memorization,
                           1 - forget_Q_A_PARA_Prob, forget_truth_ratio)
                        The first three are "higher = more memorization" so they
                        are inverted (1 - x). forget_truth_ratio is stored via
                        the `closer_to_1_better` aggregator, min(tr, 1/tr) with
                        tr = false/true, where higher already means better
                        forgetting -- so it enters the HM directly.
  - privacy_score       HM(s_MIA per {LOSS, ZLib, Min-k, Min-k++})
  - utility             HM(model_utility, forget_Q_A_gibberish)
                        forget_Q_A_gibberish is P(clean) from the gibberish
                        classifier (class 0), i.e. fluency; higher = better.
  - aggregate           HM(memorization, privacy_score, utility)
  - aggregate_mu        HM(memorization, privacy_score, model_utility)
                        The same headline score with the `utility` dimension
                        swapped for raw TOFU Model Utility, i.e. WITHOUT the
                        forget-set fluency term. `utility` folds fluency in at
                        equal weight, so a run whose gibberish score dips drags
                        the aggregate down for a reason unrelated to either
                        forgetting or retained capability; this column isolates
                        that. Reported alongside, never instead of, `aggregate`.

s_MIA uses the SUD paper's piecewise normalization against AUC_retain pulled
from saves/eval/tofu_<model>_retain{90,95,99}/TOFU_SUMMARY.json based on the
forget split (forget10->retain90, forget05->retain95, forget01->retain99).

This module is also where the metric definitions live for everything else that
scores a run, so no formula is ever copied: scripts/summarize_sud_baselines.py
imports them for the weight baselines. The MUSE analogues of the three
dimensions sit beside the TOFU ones as `compute_muse_*` / `load_muse_retain_aucs`
(MUSE has a different metric set -- see the section comment); this script's own
walk is TOFU-only.

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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sud_paths import family_of  # noqa: E402


MIA_KEYS = ["mia_loss", "mia_zlib", "mia_min_k", "mia_min_k_plus_plus"]

# Memorization dimension (arXiv:2506.12618 App. F.1). These three are
# "higher = more memorization", so they are inverted (1 - x).
MEM_INVERT_COLS = [
    "extraction_strength",
    "exact_memorization",
    "forget_Q_A_PARA_Prob",
]

# forget_truth_ratio is stored via the `closer_to_1_better` aggregator --
# min(tr, 1/tr) with tr = false/true -- so higher already means the true and
# perturbed answers are equally likely, i.e. better forgetting. Used as-is.
MEM_DIRECT_COLS = ["forget_truth_ratio"]

# Utility dimension: TOFU Model Utility + forget-set fluency (P(clean) from the
# gibberish classifier, class 0). Both are "higher = better".
UTILITY_COLS = ["model_utility", "forget_Q_A_gibberish"]

# Every key a finished TOFU eval writes. `TOFU_SUMMARY.json` is written
# INCREMENTALLY -- the evaluator appends each metric as it finishes -- so a file
# that exists is not a file that is done. A run still on the GPU therefore parses
# fine and yields a row whose missing metrics quietly become NaN. Rows are checked
# against this set and marked in the `complete` column instead.
REQUIRED_TOFU_KEYS = [
    "exact_memorization",
    "extraction_strength",
    "forget_Q_A_PARA_Prob",
    "forget_Q_A_gibberish",
    "forget_quality",
    "forget_truth_ratio",
    "model_utility",
    "privleak",
    *MIA_KEYS,
]

SPLIT_TO_RETAIN = {
    "forget10": "retain90",
    "forget05": "retain95",
    "forget01": "retain99",
}

TARGET_PREFIX = "target-open-unlearning_"
DRAFT_PREFIX = "draft-open-unlearning_"


# ── extra columns read from the files beside the summary ─────────────────────
# `TOFU_SUMMARY.json` keeps only the metrics the eval config asked for at the
# top level; `TOFU_EVAL.json` beside it also carries the per-split sub-metrics
# that Model Utility aggregates away. Two reasons to lift a few of them into the
# CSV:
#
#   * the leakage-vs-utility frontier needs a utility axis that is a pure
#     teacher-forced likelihood. `model_utility` is not one -- 3 of its 9
#     sub-metrics are sampled ROUGE -- so `retain_Q_A_Prob` and friends are the
#     only honest x-axis for it.
#   * putting the `*_ROUGE` sub-metrics beside their `*_Prob` twins is what lets
#     a reader SEE the sampled-vs-greedy confound instead of being told about it:
#     the forward probes move with alpha, the ROUGE ones barely do.
#
# Every key below is in the intersection of all four eval sources (SUD runs,
# weight baselines, `_full` references, `retain*` references), so no column is
# half-populated. Deliberately short: this is not a mirror of the eval file.
EXTRA_EVAL_KEYS = (
    # The six forward terms of TOFU Model Utility, one per (split, quantity).
    # Kept individually so a reader can see which part of Model Utility moved.
    "retain_Q_A_Prob",          # forward  — retain split, the frontier's x axis
    "retain_Truth_Ratio",       # forward
    "ra_Q_A_Prob_normalised",   # forward  — real authors
    "ra_Truth_Ratio",           # forward
    "wf_Q_A_Prob_normalised",   # forward  — world facts
    "wf_Truth_Ratio",           # forward
    # The three sampled terms it drops, kept so the confound is visible rather
    # than described, plus the forget-set ROUGE for the same reason.
    "retain_Q_A_ROUGE",         # sampled  — the ROUGE twin of retain_Q_A_Prob
    "ra_Q_A_ROUGE",             # sampled
    "wf_Q_A_ROUGE",             # sampled
    "forget_Q_A_ROUGE",         # sampled
    # Remaining forward probes, useful as alternative frontier axes.
    "ra_Q_A_Prob",              # forward  — un-normalised real-authors probe
    "wf_Q_A_Prob",              # forward  — un-normalised world-facts probe
    "retain_Q_A_PARA_Prob",     # forward
    "retain_Q_A_PERT_Prob",     # forward
    "ra_Q_A_PERT_Prob",         # forward
    "wf_Q_A_PERT_Prob",         # forward
    "forget_Q_A_PERT_Prob",     # forward
)


def read_extra_metrics(eval_path: Path,
                       keys: tuple[str, ...] = EXTRA_EVAL_KEYS) -> dict:
    """{key: agg_value} for `keys` present in a *_EVAL.json; {} if unreadable.

    Best-effort by design: a missing file, a missing key or a non-scalar
    `agg_value` (the retain references store `forget_quality` that way) yields no
    column rather than a crash, so an in-flight eval still summarizes.
    """
    try:
        data = json.loads(eval_path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        print(f"WARNING: failed to read {eval_path}: {e}")
        return {}
    out = {}
    for k in keys:
        entry = data.get(k)
        if not isinstance(entry, dict):
            continue
        v = entry.get("agg_value")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
    return out


def read_profile(profile_path: Path, stage: str, prefix: str) -> dict:
    """`{<prefix>_seconds, <prefix>_vram_mb}` from a profiling.json; {} if absent.

    The one cost number the runs actually record. Wall-clock of a whole eval on
    shared cluster hardware is indicative, not a benchmark -- but SUD runs two
    forward passes per token where a weight baseline runs one, so the ratio
    between a run and its own draft is the closest thing on disk to the price of
    decode-time unlearning.
    """
    try:
        data = json.loads(profile_path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        print(f"WARNING: failed to read {profile_path}: {e}")
        return {}
    out = {}
    secs = (data.get("timings_sec") or {}).get(stage)
    vram = (data.get("vram_peak_mb") or {}).get(stage)
    if isinstance(secs, (int, float)):
        out[f"{prefix}_seconds"] = float(secs)
    if isinstance(vram, (int, float)):
        out[f"{prefix}_vram_mb"] = float(vram)
    return out


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


def _hm_dimension(df: pd.DataFrame, name: str,
                  invert_cols: list[str] | None = None,
                  direct_cols: list[str] | None = None) -> pd.Series:
    """HM over `1 - invert_cols` and `direct_cols`; NaN if any column is absent.

    Every dimension below is this same shape -- only which columns are inverted
    ("higher = more memorization") and which enter directly differ.
    """
    invert_cols, direct_cols = invert_cols or [], direct_cols or []
    missing = [c for c in invert_cols + direct_cols if c not in df.columns]
    if missing:
        print(f"WARNING: cannot compute {name}; missing columns: {missing}")
        return pd.Series(np.nan, index=df.index)
    parts = []
    if invert_cols:
        parts.append(1.0 - df[invert_cols])
    if direct_cols:
        parts.append(df[direct_cols])
    return harmonic_mean_frame(pd.concat(parts, axis=1))


def compute_memorization(df: pd.DataFrame) -> pd.Series:
    return _hm_dimension(df, "memorization", MEM_INVERT_COLS, MEM_DIRECT_COLS)


def compute_utility(df: pd.DataFrame) -> pd.Series:
    return _hm_dimension(df, "utility", direct_cols=UTILITY_COLS)


def compute_aggregate(df: pd.DataFrame) -> pd.Series:
    return _hm_dimension(df, "aggregate",
                         direct_cols=["memorization", "privacy_score", "utility"])


def compute_aggregate_mu(df: pd.DataFrame) -> pd.Series:
    """`aggregate` with the utility dimension replaced by raw Model Utility.

    `utility` is HM(model_utility, forget_Q_A_gibberish), so fluency enters the
    headline number at the same weight as all nine Model Utility sub-metrics
    combined. That makes a fluency dip on the forget set -- which is arguably the
    POINT of unlearning, not a regression -- indistinguishable from a loss of
    retained capability. This variant drops the fluency term so the two can be
    read apart. Fluency is still on the CSV as its own column.
    """
    return _hm_dimension(df, "aggregate_mu",
                         direct_cols=["memorization", "privacy_score",
                                      "model_utility"])


def s_mia(auc: float, ref: float) -> float:
    """SUD paper's piecewise normalization: maps AUC=ref -> 1, falls off linearly."""
    if auc > ref:
        return 1.0 - (auc - ref) / (1.0 - ref)
    return 1.0 - (ref - auc) / ref


_MODEL_RE = re.compile(r"_tofu_(Llama[^_]+)_(?:full|forget\d+|retain\d+)")


def model_name_from_target(target_dir_name: str) -> str | None:
    """Pull the HF model name out of a target dir.

    Handles both forms:
      target-open-unlearning_tofu_Llama-3.2-1B-Instruct_full         -> Llama-3.2-1B-Instruct
      target-open-unlearning_..._tofu_Llama-3.2-1B-Instruct_forget10_... -> Llama-3.2-1B-Instruct
    """
    m = _MODEL_RE.search(target_dir_name)
    return m.group(1) if m else None


def _read_aucs(summary_path: Path) -> dict[str, float] | None:
    """{mia_loss: auc, ...} from a *_SUMMARY.json, or None if unusable."""
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


def load_retain_aucs(eval_root: Path, model: str, split: str) -> dict[str, float] | None:
    """Returns {mia_loss: auc, ...} for the matching retain model, or None if missing."""
    retain_split = SPLIT_TO_RETAIN.get(split)
    if retain_split is None:
        print(f"WARNING: unknown split '{split}' (no retain mapping)")
        return None
    return _read_aucs(eval_root / f"tofu_{model}_{retain_split}" / "TOFU_SUMMARY.json")


# ── MUSE (arXiv:2407.06460) ──────────────────────────────────────────────────
# MUSE reports a different metric set: no model_utility, no paraphrase
# probability, no truth ratio and no gibberish classifier. The dimensions below
# are the MUSE-metric analogues of the TOFU ones above -- same shape (higher =
# better, harmonic mean over components), different underlying quantities:
#   - memorization  HM(1 - forget_verbmem_ROUGE, 1 - forget_knowmem_ROUGE,
#                      1 - exact_memorization, 1 - extraction_strength)
#                   VerbMem and KnowMem-on-forget are MUSE's two leakage
#                   metrics; the other two are shared with TOFU. All four are
#                   "higher = more memorization", so all four are inverted.
#   - utility       retain_knowmem_ROUGE, MUSE's only utility-preservation
#                   metric (its stand-in for TOFU's Model Utility).
#   - privacy_score identical to TOFU's s_MIA harmonic mean; only AUC_retain
#                   differs -- it comes from the MUSE retrain model.
# The aggregate is HM(memorization, privacy_score, utility) either way, so a
# MUSE row and a TOFU row are structurally comparable but never interchangeable.
MUSE_MEM_INVERT_COLS = [
    "forget_verbmem_ROUGE",
    "forget_knowmem_ROUGE",
    "exact_memorization",
    "extraction_strength",
]

MUSE_UTILITY_COLS = ["retain_knowmem_ROUGE"]


def compute_muse_memorization(df: pd.DataFrame) -> pd.Series:
    return _hm_dimension(df, "memorization", MUSE_MEM_INVERT_COLS)


def compute_muse_utility(df: pd.DataFrame) -> pd.Series:
    return _hm_dimension(df, "utility", direct_cols=MUSE_UTILITY_COLS)


def load_muse_retain_aucs(eval_root: Path, model: str, split: str) -> dict[str, float] | None:
    """MUSE's retain reference: the model retrained without the forget corpus.

    `split` is the MUSE corpus (News / Books) -- unlike TOFU there is no
    forget-fraction to map, each corpus has exactly one retrain checkpoint.
    """
    return _read_aucs(eval_root / f"muse_{model}_{split}_retrain" / "MUSE_SUMMARY.json")


def compute_privacy_score(summary: dict, retain_aucs: dict[str, float] | None) -> float:
    if retain_aucs is None:
        return float("nan")
    s_values: list[float] = []
    for k in MIA_KEYS:
        if k not in summary:
            return float("nan")          # reported by the completeness check
        s_values.append(s_mia(float(summary[k]), retain_aucs[k]))
    arr = np.array(s_values, dtype=float)
    if np.any(arr <= 0):
        return 0.0
    return float(len(arr) / np.sum(1.0 / arr))


def _build_row(summary_path: Path, *, row_base: dict, model: str | None, split: str,
               eval_root: Path,
               retain_cache: dict[tuple[str, str], dict[str, float] | None]) -> dict | None:
    try:
        summary = json.loads(summary_path.read_text())
    except Exception as e:
        print(f"WARNING: failed to read {summary_path}: {e}")
        return None

    cache_key = (model or "", split)
    if cache_key not in retain_cache:
        retain_cache[cache_key] = (
            load_retain_aucs(eval_root, model, split) if model else None
        )
    retain_aucs = retain_cache[cache_key]

    row: dict = dict(row_base)
    row.update(summary)
    row.update(read_extra_metrics(summary_path.with_name("TOFU_EVAL.json")))
    row.update(read_profile(summary_path.with_name("profiling.json"),
                            stage="evaluation", prefix="eval"))
    row["privacy_score"] = compute_privacy_score(summary, retain_aucs)
    missing = [k for k in REQUIRED_TOFU_KEYS if k not in summary]
    row["complete"] = not missing
    row["_missing_keys"] = missing
    row["_path"] = str(summary_path.parent)
    return row


_LEAF_RE = re.compile(r"a([0-9]*\.?[0-9]+)_k(\d+)_s(\d+)")


def collect_rows(results_root: Path, eval_root: Path) -> list[dict]:
    """Walk SUD runs: <combo>/target-*/draft-*/<split>/<leaf>/TOFU_SUMMARY.json.

    Prefers the run_meta.json written beside each summary (exact ids + config);
    falls back to parsing the path when it is absent.
    """
    retain_cache: dict[tuple[str, str], dict[str, float] | None] = {}
    rows: list[dict] = []

    for summary_path in sorted(results_root.glob("*/target-*/draft-*/*/*/TOFU_SUMMARY.json")):
        combo = summary_path.relative_to(results_root).parts[0]
        if combo == "baselines":
            continue

        meta_path = summary_path.with_name("run_meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            target_full = meta.get("target", "")
            split = meta.get("forget_split")
            row_base = {
                "combo": meta.get("combo", combo),
                "target": meta.get("target_tag") or target_full,
                "draft": meta.get("draft_tag") or meta.get("draft"),
                "split": split,
                "alpha": meta.get("alpha"),
                "k": meta.get("k_sud"),
                "seed": meta.get("seed"),
            }
            model = model_name_from_target(target_full) or family_of(target_full)
        else:
            _, target_dir, draft_dir, split, leaf, _ = summary_path.relative_to(results_root).parts
            target_tag = target_dir[len(TARGET_PREFIX):] if target_dir.startswith(TARGET_PREFIX) else target_dir
            draft_tag = draft_dir[len(DRAFT_PREFIX):] if draft_dir.startswith(DRAFT_PREFIX) else draft_dir
            m = _LEAF_RE.search(leaf)
            row_base = {
                "combo": combo,
                "target": target_tag,
                "draft": draft_tag,
                "split": split,
                "alpha": float(m.group(1)) if m else None,
                "k": int(m.group(2)) if m else None,
                "seed": int(m.group(3)) if m else None,
            }
            model = family_of(target_tag)

        row = _build_row(summary_path, row_base=row_base, model=model, split=split,
                         eval_root=eval_root, retain_cache=retain_cache)
        if row is not None:
            rows.append(row)

    return rows


def report_incomplete(df: pd.DataFrame, *, drop: bool) -> pd.DataFrame:
    """Print which runs are still mid-eval; drop them if asked. Strips helper cols.

    An incomplete run is not an error -- it is a job still on the GPU -- but it
    must never pass silently, because its missing metrics become NaN and a NaN
    aggregate is indistinguishable from "this configuration scored nothing".
    """
    incomplete = df[~df["complete"]]
    if len(incomplete):
        print(f"\n{len(incomplete)} of {len(df)} runs are INCOMPLETE "
              f"(TOFU_SUMMARY.json is written metric-by-metric, so these are "
              f"still evaluating or died mid-eval):")
        for _, r in incomplete.iterrows():
            print(f"  {r['target']} / {r['draft']} / {r['split']} / "
                  f"a{r['alpha']}_k{r['k']}_s{r['seed']}  "
                  f"missing: {', '.join(r['_missing_keys'])}")
        print(f"  -> {'dropped from' if drop else 'kept in'} the CSV; "
              f"filter on the `complete` column downstream.\n")
        if drop:
            df = df[df["complete"]]
    return df.drop(columns=["_missing_keys"])


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=repo_root / "saves/unlearn/sud/results")
    parser.add_argument("--eval-root", type=Path, default=repo_root / "saves/eval")
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV output path (default: <results-root>/summary.csv)")
    parser.add_argument("--drop-incomplete", action="store_true", default=False,
                        help="Omit runs whose TOFU_SUMMARY.json is not finished yet "
                             "(default: keep them, flagged in the `complete` column).")
    args = parser.parse_args()

    if not args.results_root.exists():
        raise SystemExit(f"results-root does not exist: {args.results_root}")

    if args.output is None:
        args.output = args.results_root / "summary.csv"

    rows = collect_rows(args.results_root, args.eval_root)
    if not rows:
        print(f"No TOFU_SUMMARY.json files matched under {args.results_root}")
        return

    df = pd.DataFrame(rows)
    df["memorization"] = compute_memorization(df)
    df["utility"] = compute_utility(df)
    df["aggregate"] = compute_aggregate(df)
    df["aggregate_mu"] = compute_aggregate_mu(df)
    df = report_incomplete(df, drop=args.drop_incomplete)

    leading = ["combo", "target", "draft", "split", "alpha", "k", "seed", "complete",
               "aggregate", "aggregate_mu", "memorization", "privacy_score",
               "utility", "model_utility", "forget_quality"]
    cols = [c for c in leading if c in df.columns] + [c for c in df.columns if c not in leading]
    sort_cols = [c for c in ["combo", "target", "draft", "split", "alpha", "k", "seed"]
                 if c in df.columns]
    df = df[cols].sort_values(sort_cols, na_position="last")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
