#!/usr/bin/env python3
"""Loader + metric semantics for SUD results. Shared by two consumers.

  * dashboard/app.py         interactive Streamlit page
  * scripts/plot_results.py  static PNG/PDF figures

Everything here is the one combination still being run:

    original-unlearned   p / target = the TOFU-finetuned ORIGINAL (memorised) model
                         q / draft  = that model after weight-based unlearning

SUD decodes from log pi = (1-alpha) log p + alpha log q, so alpha sweeps a path
from p to q. `COMBO` is enforced on load; a run from any other combination is
dropped with a note rather than quietly averaged in.

Two sources are read, and the comparison is only ever against the second:

    SUD sweeps   saves/unlearn/sud/results/summary.csv
    q baselines  saves/unlearn/baselines/tofu/summary_baselines.csv

A SUD run's draft IS the weight baseline it is measured against -- same
checkpoint, same split, same method -- so that row is the only fair comparison
and the only one drawn. The original model p is not read at all (`load_all` below
still resolves it, because scripts/plot_results.py draws it).

One caveat worth carrying: SUD's decoder samples at temperature 1 while both
reference evals decode greedily, so a GENERATION metric (model_utility,
forget_Q_A_gibberish, and the `utility` / `aggregate` / `aggregate_mu`
composites that contain them) compares two decoding regimes and understates SUD
by roughly 0.02-0.03. The teacher-forced metrics carry no such handicap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_sud_results import (  # noqa: E402
    EXTRA_EVAL_KEYS,
    compute_aggregate_mu,
    compute_memorization,
    compute_privacy_score,
    load_retain_aucs,
    read_extra_metrics,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO / "saves/unlearn/sud/results"
DEFAULT_BASELINES = REPO / "saves/unlearn/baselines/tofu/summary_baselines.csv"
DEFAULT_EVAL = REPO / "saves/eval"

#: The only Table-4.1 row still in play. See scripts/sud_paths.py for the others.
COMBO = "original-unlearned"


# ── metric semantics ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    direction: str          # up | down | zero
    group: str
    help: str
    log_scale: bool = False

    @property
    def higher_is_better(self) -> bool:
        return self.direction == "up"

    @property
    def note(self) -> str:
        return {"up": "higher is better", "down": "lower is better",
                "zero": "0 is ideal"}[self.direction]


_M = [
    Metric("aggregate", "Aggregate", "up", "Composite",
           "HM(memorization, privacy, utility) — the headline score"),
    Metric("aggregate_mu", "Aggregate (model utility)", "up", "Composite",
           "HM(memorization, privacy, model_utility) — the headline score with "
           "the utility dimension swapped for raw Model Utility, i.e. with the "
           "forget-set fluency term dropped"),
    Metric("memorization", "Memorization", "up", "Composite",
           "HM of 1−extraction, 1−exact_mem, 1−para_prob, truth_ratio; "
           "higher = less memorised"),
    Metric("privacy_score", "Privacy", "up", "Composite",
           "HM of the four per-attack MIA scores, normalised against the retain "
           "model's AUC. Hard-clamped to 0 whenever any AUC hits 0 or 1"),
    Metric("utility", "Utility", "up", "Composite",
           "HM(model_utility, forget-set fluency) — both generation-based"),

    Metric("exact_memorization", "Exact memorization", "down", "Forgetting",
           "Verbatim greedy recall of the forget answer. Computed from argmax(π), "
           "a decode SUD never performs, so it overstates what SUD exposes"),
    Metric("extraction_strength", "Extraction strength", "down", "Forgetting",
           "Longest greedy-extractable suffix of the forget answer. Also an "
           "argmax quantity"),
    Metric("forget_Q_A_PARA_Prob", "Forget prob", "down", "Forgetting",
           "Likelihood of the paraphrased forget answer — a teacher-forced "
           "quantity, faithful to what SUD emits"),
    Metric("forget_truth_ratio", "Truth ratio", "up", "Forgetting",
           "min(tr, 1/tr) with tr = false/true on the forget set; higher = true "
           "and perturbed answers are equally likely"),
    Metric("forget_quality", "Forget quality", "up", "Forgetting",
           "KS-test p-value against the retain model's truth-ratio distribution. "
           "Spans ~200 orders of magnitude, so it is compared as a ratio (in "
           "decades) and plotted on a log axis",
           log_scale=True),

    Metric("model_utility", "Model utility", "up", "Utility",
           "TOFU Model Utility over 9 sub-metrics. 3 of the 9 are sampled ROUGE, "
           "and SUD samples where every baseline decodes greedily — so this "
           "understates SUD by roughly 0.02–0.03"),
    Metric("retain_Q_A_Prob", "Retain prob", "up", "Utility",
           "Likelihood of the correct answer on the retain split"),
    Metric("forget_Q_A_gibberish", "Fluency", "up", "Utility",
           "P(clean) from the gibberish classifier on generated forget answers"),

    Metric("privleak", "PrivLeak", "zero", "Privacy",
           "Relative Min-K% MIA gap against the retain model. 0 is ideal, "
           "negative = still leaking, positive = over-forgotten"),
]
METRICS: dict[str, Metric] = {m.key: m for m in _M}
METRIC_KEYS = list(METRICS)

#: The default comparison set for the p / q / SUD bars — one metric per thing a
#: reader actually asks about, not every column in the CSV.
BAR_METRICS = ["aggregate", "aggregate_mu", "memorization", "privacy_score",
               "utility", "model_utility", "exact_memorization",
               "extraction_strength"]


def get(row, key: str):
    """Value or None. A missing column, None and NaN are all None."""
    if row is None:
        return None
    try:
        v = row[key] if key in row else None
    except (KeyError, TypeError, IndexError):
        v = None
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else f


def best_row(df: pd.DataFrame, key: str):
    """Best row by the metric's own direction; None if unavailable."""
    if key not in df.columns:
        return None
    d = df.dropna(subset=[key])
    if d.empty:
        return None
    direction = METRICS[key].direction if key in METRICS else "up"
    if direction == "up":
        return d.loc[d[key].idxmax()]
    if direction == "down":
        return d.loc[d[key].idxmin()]
    return d.loc[d[key].abs().idxmin()]          # zero


def improvement(key: str, sud_v, base_v):
    """SUD minus its baseline, signed so positive always means SUD is better.

    A log-scale metric is compared as a RATIO, in decades: forget quality is a
    p-value spanning some two hundred orders of magnitude, and subtracting two of
    them calls 1e-3 and 1e-217 indistinguishable.
    """
    if sud_v is None or base_v is None:
        return None
    M = METRICS[key]
    if M.log_scale:
        if sud_v <= 0 or base_v <= 0:
            return None
        ratio = float(np.log10(sud_v / base_v))
        return ratio if M.direction == "up" else -ratio
    if M.direction == "up":
        return sud_v - base_v
    if M.direction == "down":
        return base_v - sud_v
    return abs(base_v) - abs(sud_v)              # zero: closer wins


def pretty_method(method: str) -> str:
    """`GradDiff_GDR` -> `GradDiff +GDR` (the retain-loss suffix is a variant)."""
    return method[:-4] + " +GDR" if str(method).endswith("_GDR") else str(method)


def pretty_model(model: str) -> str:
    """`Llama-3.2-1B-Instruct` -> `1B`."""
    for prefix in ("Llama-3.2-", "Llama-3.1-", "Llama-2-"):
        if str(model).startswith(prefix):
            return str(model)[len(prefix):].replace("-Instruct", "")
    return str(model)


def method_of(draft: str, target: str) -> str:
    """`Llama-3.2-1B-Instruct_GradDiff_GDR` + target -> `GradDiff_GDR`."""
    draft, target = str(draft), str(target)
    return draft[len(target) + 1:] if draft.startswith(target + "_") else draft


# ── is summary.csv still a picture of what is on disk? ───────────────────────

@dataclass
class Freshness:
    """The CSVs are build artefacts, refreshed only by running the summarizers.

    Checked on every load because a stale one looks completely normal, and this
    project has silently shown 16 rows against 122 finished runs before. Two
    checks, not one: a count alone misses a run RE-EVALUATED in place.
    """
    csv_rows: int
    finished_on_disk: int
    pending_on_disk: int
    newest_run: float = 0.0
    csv_written: float = 0.0

    @property
    def stale(self) -> bool:
        return (self.csv_rows != self.finished_on_disk
                or self.newest_run > self.csv_written + 1.0)

    @property
    def message(self) -> str:
        rebuild = "rerun `make summaries`"
        gap = self.csv_rows - self.finished_on_disk
        if gap > 0:
            # The one direction where rebuilding is the WRONG move: those rows
            # are real measurements whose output directories were removed, and a
            # rebuild would delete them.
            return (f"summary.csv holds {gap} run(s) that are no longer on disk "
                    f"— it is their only record, so do NOT rebuild it unless "
                    f"you meant to drop them")
        if gap < 0:
            return (f"{-gap} finished run(s) on disk are missing from "
                    f"summary.csv — {rebuild}")
        if self.newest_run > self.csv_written + 1.0:
            return f"a run was written after summary.csv was built — {rebuild}"
        extra = (f" · {self.pending_on_disk} more still evaluating"
                 if self.pending_on_disk else "")
        return f"{self.csv_rows} runs, matching what is on disk{extra}"


def check_freshness(results_root: Path, combo: str = COMBO) -> Freshness:
    csv = results_root / "summary.csv"
    rows = 0
    if csv.exists():
        rows = int(pd.read_csv(csv, usecols=["combo"]).eq(combo).sum().iloc[0])
    dirs = [d for d in (results_root / combo).glob("target-*/draft-*/*/*")
            if d.is_dir()]
    present = [d / "TOFU_SUMMARY.json" for d in dirs
               if (d / "TOFU_SUMMARY.json").exists()]
    return Freshness(
        csv_rows=rows, finished_on_disk=len(present),
        pending_on_disk=len(dirs) - len(present),
        newest_run=max((f.stat().st_mtime for f in present), default=0.0),
        csv_written=csv.stat().st_mtime if csv.exists() else 0.0)


# ── loading ──────────────────────────────────────────────────────────────────

@dataclass
class Dataset:
    runs: pd.DataFrame                      # one row per SUD run
    slices: pd.DataFrame                    # one row per (target, split, method)
    baselines: dict[tuple, pd.Series]       # the weight baseline per slice
    freshness: Freshness
    notes: list[str] = field(default_factory=list)

    def sweep(self, target: str, split: str, method: str) -> pd.DataFrame:
        """One slice's alpha sweep, endpoints included, ordered by alpha."""
        d = self.runs
        return (d[(d.target == target) & (d.split == split) & (d.method == method)]
                .sort_values("alpha"))

    def baseline(self, target: str, split: str, method: str):
        """The weight-unlearned checkpoint this slice's runs decode from.

        None when it is no longer on disk — reported, never substituted with a
        neighbouring method or size.
        """
        return self.baselines.get((target, split, method))


def load_dataset(results_root: Path = DEFAULT_RESULTS,
                 baselines_csv: Path = DEFAULT_BASELINES,
                 combo: str = COMBO) -> Dataset:
    """Read the two CSVs and pair every sweep with its own weight baseline."""
    notes: list[str] = []
    sud = pd.read_csv(results_root / "summary.csv")

    other = sud[sud["combo"] != combo]
    if len(other):
        notes.append(f"{len(other)} run(s) from other combinations are not shown "
                     f"— this page is {combo} only.")
    sud = sud[sud["combo"] == combo].copy()
    if "complete" in sud.columns and not sud["complete"].astype(bool).all():
        n = int((~sud["complete"].astype(bool)).sum())
        notes.append(f"{n} run(s) are still evaluating; their missing metrics "
                     f"read as blank, not as zero.")

    sud["method"] = [method_of(d, t) for d, t in zip(sud["draft"], sud["target"])]
    base = pd.read_csv(baselines_csv)

    baselines: dict[tuple, pd.Series] = {}
    rows = []
    for (target, split, method), g in sud.groupby(["target", "split", "method"]):
        # Exact match on all three, never approximate: a 3B NPO baseline pairs
        # only with a 3B NPO draft on the same split.
        qw = base[(base.model == target) & (base.split == split)
                  & (base.method == method)]
        if len(qw):
            baselines[(target, split, method)] = qw.iloc[0]
        rows.append({
            "target": target, "split": split, "method": method,
            "n_runs": len(g),
            "alphas": ", ".join(f"{v:g}" for v in sorted(g.alpha.unique())),
            "has_baseline": bool(len(qw)),
        })

    slices = (pd.DataFrame(rows).sort_values(["target", "split", "method"])
              .reset_index(drop=True))
    no_base = slices[~slices.has_baseline]
    if len(no_base):
        notes.append(
            f"{len(no_base)} slice(s) have no weight baseline on disk "
            f"({', '.join(sorted(no_base.method.unique()))}), so their metrics "
            f"are shown with a blank baseline column.")

    ks = sorted(sud.k.unique())
    if len(ks) == 1:
        notes.append(f"Every run uses k_sud = {int(ks[0])}, which is a throughput "
                     f"knob and provably does not change the sampled "
                     f"distribution — so there is nothing to compare across k.")

    return Dataset(runs=sud, slices=slices, baselines=baselines,
                   freshness=check_freshness(results_root, combo), notes=notes)


# ── the shape scripts/plot_results.py consumes ───────────────────────────────
# The dashboard compares against the weight baseline and nothing else. The static
# figures still draw the original model, so p is resolved here and only here.

REFERENCE_EXTRA_KEYS = ("forget_Q_A_PARA_Prob", "forget_truth_ratio")


def load_p_reference(eval_root: Path, family: str, split: str) -> dict | None:
    """The original model's downloaded eval for one (family, split), or None.

    Its SUMMARY file omits two memorization terms that its EVAL file carries, so
    both are read and `memorization` is computed the same way it is for a run.
    `aggregate_mu` is computed too -- it needs only model_utility, which every
    reference eval has, so unlike `aggregate` it can carry a p line on its panel.
    `utility` and `aggregate` stay absent: they need the gibberish score, which no
    reference eval has, and a partial harmonic mean is a different quantity.
    """
    path = (eval_root / f"tofu_{family}_full" / f"evals_{split}"
            / "TOFU_SUMMARY.json")
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, OSError, ValueError):
        return None
    data.update(read_extra_metrics(
        path.with_name("TOFU_EVAL.json"), EXTRA_EVAL_KEYS + REFERENCE_EXTRA_KEYS))
    data["privacy_score"] = compute_privacy_score(
        data, load_retain_aucs(eval_root, family, split))
    one = pd.DataFrame([data])
    v = compute_memorization(one).iloc[0]
    if np.isfinite(v):
        data["memorization"] = float(v)
        # aggregate_mu drops the fluency term, so it is the one composite a
        # reference eval can actually be scored on. Feed it the memorization
        # just computed -- compute_aggregate_mu reads it off the frame.
        one["memorization"] = float(v)
        a = compute_aggregate_mu(one).iloc[0]
        if np.isfinite(a):
            data["aggregate_mu"] = float(a)
    return data

def load_frames(results_root: Path = DEFAULT_RESULTS,
                baselines_csv: Path = DEFAULT_BASELINES):
    """-> (sud_df, baselines_df), the two summary CSVs, combo-filtered."""
    sud = pd.read_csv(results_root / "summary.csv")
    return sud[sud["combo"] == COMBO].copy(), pd.read_csv(baselines_csv)


def load_all(results_root: Path = DEFAULT_RESULTS,
             baselines_csv: Path = DEFAULT_BASELINES,
             eval_root: Path = DEFAULT_EVAL):
    """-> (sud, base, p_refs, retain_refs) keyed by (model, split)."""
    sud, base = load_frames(results_root, baselines_csv)
    sud["method"] = [method_of(d, t) for d, t in zip(sud["draft"], sud["target"])]
    pairs = sorted(set(zip(sud["target"], sud["split"]))
                   | set(zip(base["model"], base["split"])))
    p_refs, retain_refs = {}, {}
    for model, split in pairs:
        p = load_p_reference(eval_root, model, split)
        if p is not None:
            p_refs[(model, split)] = p
        aucs = load_retain_aucs(eval_root, model, split)
        if aucs is not None:
            retain_refs[(model, split)] = dict(aucs)
    return sud, base, p_refs, retain_refs


def q_row(base: pd.DataFrame, model: str, split: str, method: str):
    """The weight baseline that is also this SUD run's draft; None if absent."""
    m = base[(base.model == model) & (base.split == split) & (base.method == method)]
    return m.iloc[0] if len(m) else None
