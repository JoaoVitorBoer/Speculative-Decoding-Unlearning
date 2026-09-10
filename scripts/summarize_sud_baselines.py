"""Summarize the weight-based unlearning baselines into one CSV per benchmark.

The baselines tree is split by benchmark, because TOFU and MUSE do not share a
metric set and their rows must never land in the same table:

    <baselines-root>/tofu/<method>/<model>/<forget_split>/evals/TOFU_SUMMARY.json
    <baselines-root>/muse/<method>/<model>/<data_split>/evals/MUSE_SUMMARY.json

written by `scripts/baselines/baseline_tofu_<method>.sh` and
`scripts/run_tofu_weight_baselines.sh` for TOFU, and
`scripts/run_muse_weight_baselines.sh` for MUSE.

The `<method>` level is whatever tag the writing script chose, and it is recorded
verbatim in the `method` column. The TOFU tree uses the bare method name (`NPO`),
one directory per method; the older `run_*_weight_baselines.sh` convention is
method + retain-loss mode (`NPO_GDR`). Either way the hyperparameters are broken
out into the `lr` / `epochs` / `retain_loss_type` columns from Hydra, so a row
stays queryable without parsing its name. Keeping the tag verbatim is what lets
`scripts/sud_results_data.py` join these rows to the SUD runs: a SUD draft tag is
`<model>_<method>` built from the same directory name.

Each subtree gets its own

    <baselines-root>/<benchmark>/summary_baselines.csv

Both CSVs carry the same four evaluation dimensions as
`scripts/summarize_sud_results.py` -- memorization / privacy_score / utility and
their harmonic-mean aggregate -- so the numbers are directly comparable to the
SUD runs of the same benchmark. A baseline row is identified by (method, model,
split) alone: the SUD-only knobs (target/draft pair, alpha, k, seed) do not apply
to a trained checkpoint and are not emitted.

The metric definitions live in `summarize_sud_results` and are imported, not
copied -- a weight baseline is scored exactly like a SUD run, only the run
config differs (a trained checkpoint, no draft model and no blend). The two
benchmarks differ only in which of those definitions apply, which is all the
BENCHMARKS table below encodes.

Training config (learning rate, epochs, retain-loss type) is recovered
from the `.hydra/overrides.yaml` Hydra writes into each checkpoint dir, so the
table records *how* each baseline was produced without re-deriving it from
task names.

NOTE: the `full` / `retain*` / `target` / `retrain` reference evals are NOT
included as rows -- neither the TOFU ones under `saves/eval/` nor the
`open-unlearning_tofu_*_full/` eval dirs that sit beside the checkpoints (those
have no `evals/` level, so the walk below never sees them). They were produced
with an older metric set and lack forget_Q_A_PARA_Prob, forget_truth_ratio and
forget_Q_A_gibberish, so their memorization and utility scores would be NaN.
They are still read for the retain-set MIA AUCs that privacy_score normalizes
against.

Usage:
    python scripts/summarize_sud_baselines.py                     # both benchmarks
    python scripts/summarize_sud_baselines.py --benchmark muse    # just one
    python scripts/summarize_sud_baselines.py \
        --baselines-root saves/unlearn/baselines \
        --eval-root      saves/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_sud_results import (  # noqa: E402
    MIA_KEYS,
    REQUIRED_TOFU_KEYS,
    compute_aggregate,
    compute_aggregate_mu,
    compute_memorization,
    compute_muse_memorization,
    compute_muse_utility,
    compute_privacy_score,
    compute_utility,
    load_muse_retain_aucs,
    load_retain_aucs,
    read_extra_metrics,
    read_profile,
)
from sud_paths import family_of  # noqa: E402


@dataclass(frozen=True)
class Benchmark:
    """What differs between TOFU and MUSE when scoring a weight baseline."""
    name: str
    summary_file: str
    retain_aucs: Callable[[Path, str, str], dict[str, float] | None]
    memorization: Callable[[pd.DataFrame], pd.Series]
    utility: Callable[[pd.DataFrame], pd.Series]
    # Extra leading columns, benchmark-specific, after the shared identity ones.
    extra_leading: tuple[str, ...] = ()
    # Keys a FINISHED eval of this benchmark writes. The summary JSON is appended
    # to metric-by-metric, so its existence does not mean the eval is done; rows
    # are checked against this and marked in the `complete` column.
    required: tuple[str, ...] = ()


# MUSE's finished-eval keys: the four leakage metrics, the one utility metric,
# and the MIA block shared with TOFU.
MUSE_MEM_INVERT_COLS_REQUIRED = (
    "forget_verbmem_ROUGE", "forget_knowmem_ROUGE",
    "exact_memorization", "extraction_strength", "retain_knowmem_ROUGE",
)

BENCHMARKS: dict[str, Benchmark] = {
    "tofu": Benchmark(
        name="tofu",
        summary_file="TOFU_SUMMARY.json",
        retain_aucs=load_retain_aucs,
        memorization=compute_memorization,
        utility=compute_utility,
        extra_leading=("model_utility",),
        required=tuple(REQUIRED_TOFU_KEYS),
    ),
    "muse": Benchmark(
        name="muse",
        summary_file="MUSE_SUMMARY.json",
        retain_aucs=load_muse_retain_aucs,
        memorization=compute_muse_memorization,
        utility=compute_muse_utility,
        extra_leading=("forget_verbmem_ROUGE", "forget_knowmem_ROUGE",
                       "retain_knowmem_ROUGE"),
        required=(*MUSE_MEM_INVERT_COLS_REQUIRED, *MIA_KEYS, "privleak"),
    ),
}


# Hydra override key -> CSV column, for the training config of each checkpoint.
# `trainer` itself is deliberately absent: it is redundant with the `method`
# directory tag, and the scripts that set it up do not keep the two in sync.
OVERRIDE_COLS = {
    "trainer.args.learning_rate": "lr",
    "trainer.args.num_train_epochs": "epochs",
    "trainer.method_args.retain_loss_type": "retain_loss_type",
    "model.model_args.pretrained_model_name_or_path": "target_id",
}


def read_train_overrides(ckpt_dir: Path) -> dict:
    """Training config from `<ckpt>/.hydra/overrides.yaml`; {} if unavailable.

    The file is a flat YAML list of `- key=value` entries, so it is parsed
    line-wise rather than pulling in a YAML dependency for one shape.
    """
    overrides_path = ckpt_dir / ".hydra" / "overrides.yaml"
    if not overrides_path.exists():
        print(f"WARNING: no training overrides at {overrides_path}")
        return {}
    try:
        lines = overrides_path.read_text().splitlines()
    except Exception as e:
        print(f"WARNING: failed to read {overrides_path}: {e}")
        return {}

    out: dict = {}
    for line in lines:
        entry = line.strip()
        if not entry.startswith("- "):
            continue
        key, sep, value = entry[2:].strip().partition("=")
        if not sep or key not in OVERRIDE_COLS:
            continue
        out[OVERRIDE_COLS[key]] = value.strip().strip("'\"")
    return out


def collect_baseline_rows(bench_root: Path, eval_root: Path,
                          bench: Benchmark) -> list[dict]:
    """Walk <method>/<model>/<split>/evals/<BENCH>_SUMMARY.json under one benchmark."""
    retain_cache: dict[tuple[str, str], dict[str, float] | None] = {}
    rows: list[dict] = []

    for summary_path in sorted(bench_root.glob(f"*/*/*/evals/{bench.summary_file}")):
        method, model, split, _, _ = summary_path.relative_to(bench_root).parts
        ckpt_dir = summary_path.parent.parent

        try:
            summary = json.loads(summary_path.read_text())
        except Exception as e:
            print(f"WARNING: failed to read {summary_path}: {e}")
            continue

        overrides = read_train_overrides(ckpt_dir)
        # The family drives the retain-model lookup; prefer the exact target id
        # recorded at training time, fall back to the <model> dir name. MUSE's
        # single family (Llama-2-7b-hf) is not matched by family_of, so there it
        # is always the dir name -- which is exactly the eval path's family.
        target_id = overrides.get("target_id", "")
        family = family_of(target_id) or family_of(model) or model

        cache_key = (family, split)
        if cache_key not in retain_cache:
            retain_cache[cache_key] = bench.retain_aucs(eval_root, family, split)

        row: dict = {
            "benchmark": bench.name,
            "combo": "weight-baseline",
            "method": method,
            "model": family,
            "split": split,
            "lr": overrides.get("lr"),
            "epochs": overrides.get("epochs"),
            "retain_loss_type": overrides.get("retain_loss_type"),
            "target_id": target_id or None,
            "path": str(ckpt_dir),
        }
        row.update(summary)
        # Same two side-files as a SUD run, read from the same place relative to
        # the summary -- so a baseline row and a SUD row carry identical extra
        # columns and can share a chart axis. MUSE has none of the TOFU
        # sub-metric keys, so there it simply contributes nothing.
        row.update(read_extra_metrics(
            summary_path.with_name(bench.summary_file.replace("_SUMMARY", "_EVAL"))))
        row.update(read_profile(summary_path.with_name("profiling.json"),
                                stage="evaluation", prefix="eval"))
        # ...plus the cost a decode-time method never pays: training the weights.
        row.update(read_profile(ckpt_dir / "profiling.json",
                                stage="training", prefix="train"))
        row["privacy_score"] = compute_privacy_score(summary, retain_cache[cache_key])
        missing = [k for k in bench.required if k not in summary]
        row["complete"] = not missing
        if missing:
            print(f"WARNING: incomplete eval (still running or died mid-eval): "
                  f"{method}/{model}/{split} missing {', '.join(missing)}")
        rows.append(row)

    return rows


def summarize_benchmark(bench: Benchmark, bench_root: Path, eval_root: Path,
                        output: Path) -> int:
    """Write one benchmark's summary CSV. Returns the number of rows written."""
    if not bench_root.exists():
        print(f"[{bench.name}] no such directory: {bench_root} -- skipped")
        return 0

    rows = collect_baseline_rows(bench_root, eval_root, bench)
    if not rows:
        print(f"[{bench.name}] no evals/{bench.summary_file} files matched under "
              f"{bench_root} -- nothing written (has the eval run yet?)")
        return 0

    df = pd.DataFrame(rows)
    df["memorization"] = bench.memorization(df)
    df["utility"] = bench.utility(df)
    df["aggregate"] = compute_aggregate(df)
    # TOFU only: MUSE has no Model Utility to substitute for the utility
    # dimension in `aggregate_mu`.
    if bench.name == "tofu":
        df["aggregate_mu"] = compute_aggregate_mu(df)

    leading = ["benchmark", "combo", "method", "model", "split", "complete",
               "aggregate", "aggregate_mu", "memorization", "privacy_score",
               "utility", *bench.extra_leading]
    cols = [c for c in leading if c in df.columns] + [c for c in df.columns if c not in leading]
    sort_cols = [c for c in ["model", "split", "method"] if c in df.columns]
    df = df[cols].sort_values(sort_cols, na_position="last")

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"[{bench.name}] wrote {len(df)} rows to {output}")
    return len(df)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--baselines-root", type=Path,
                        default=repo_root / "saves/unlearn/baselines",
                        help="parent of the per-benchmark subtrees (tofu/, muse/)")
    parser.add_argument("--eval-root", type=Path, default=repo_root / "saves/eval")
    parser.add_argument("--benchmark", choices=[*BENCHMARKS, "all"], default="all",
                        help="which benchmark subtree to summarize (default: all)")
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV output path; only valid with a single --benchmark "
                             "(default: <baselines-root>/<benchmark>/summary_baselines.csv)")
    args = parser.parse_args()

    if not args.baselines_root.exists():
        raise SystemExit(f"baselines-root does not exist: {args.baselines_root}")

    names = list(BENCHMARKS) if args.benchmark == "all" else [args.benchmark]
    if args.output is not None and len(names) > 1:
        raise SystemExit("--output needs a single --benchmark; "
                         "without it each benchmark writes its own CSV")

    total = 0
    for name in names:
        bench = BENCHMARKS[name]
        bench_root = args.baselines_root / name
        output = args.output or bench_root / "summary_baselines.csv"
        total += summarize_benchmark(bench, bench_root, args.eval_root, output)

    if total == 0:
        print(f"No baseline evals found under {args.baselines_root}")


if __name__ == "__main__":
    main()
