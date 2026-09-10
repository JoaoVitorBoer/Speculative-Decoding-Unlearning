#!/usr/bin/env python
"""k-sweep figures for SUD (1B / forget10 / NPO draft), EMNLP / ACL-ready.

Two figures, all panels vs the draft window k (log2 axis, k = 1 ... 256):
  fig_k_invariance   (a) model_utility            seed 0 at every k, one line per alpha
                     (b) forget_Q_A_gibberish     the gibberish classifier's P(clean) on the
                         forget-question generations (higher = fewer gibberish answers)
  fig_k_throughput   tokens / s                   from plots/bench/spec_bench.json: measured points with a
                               bootstrap 95% CI over prompts, dotted theory
                                   T(k) = R / F(k),
                                   F(k) = sum_rounds (k_i + 1) / sum_rounds (1 - a^k_i)/(1 - a)
                               with ONE shared forward rate R (median forwards/s over every
                               cell: forward cost does not depend on alpha or k), a = that
                               alpha's pooled acceptance, and k_i the drafted length of each
                               round (<= k: the draft stops at its own EOS and at the 200-token
                               budget, so the window saturates near the draft's answer length).
                               The ratio of sums matters: the closed form at the mean k~ overstates
                               throughput once rounds are heterogeneous (see theory_fwd_per_tok).
With --with-baselines, panels (a)/(b) also draw horizontal reference lines for the fixed
models of this cell (plots/baselines.json, built by plots/baseline_refs.py): target p
(alpha = 0 limit), draft q (alpha = 1 limit), retain model (TOFU gold reference). Off by
default: they answer a different question (SUD's level, not its k-invariance) and they
are greedy-decoded evals while SUD samples at T = 1 (measured model_utility handicap ~0.015).
The values are still tabulated in the README.

k provably never changes the sampled distribution (draft-verify sampling from pi, no bonus
token); (a)/(b) are the empirical check, (c) is what k buys. Seed-to-seed noise (SD over
seeds 0-3 at k = 1) is reported in the README / CSV as a yardstick, not drawn.

Conference styling: Okabe-Ito colour-blind-safe palette, a distinct dash pattern AND marker
per alpha (identity never rides on colour alone), grey-scale reference lines with direct
labels, STIX (Times-like) text with fonts embedded as TrueType (pdf.fonttype 42 — ACL's
pubcheck rejects Type 3), no in-figure subtitles (the README carries a suggested caption).
Sizes: fig_k_invariance.* (single ACL column, 3.03 in, (a) above (b)), fig_k_invariance_wide.*
(full text width, 6.3 in, side by side), fig_k_throughput.* (single column).

Inputs   plots/results/**/TOFU_SUMMARY.json (+ run_meta.json)  [Part 1]
         plots/bench/spec_bench.json                             [Part 2]
         plots/baselines.json                                    [plots/baseline_refs.py, optional]
Outputs  plots/figs/fig_k_invariance.{pdf,png}, plots/figs/fig_k_invariance_wide.{pdf,png},
         plots/figs/fig_k_throughput.{pdf,png}, plots/fig_k_sweep_data.csv, plots/README.md

Run in the `unlearning` env (matplotlib lives only there):
    conda run -n unlearning python plots/make_figure.py
Another draft method (same target / split; its own results sub-tree, bench and baselines file,
suffixed output names; figures to --out-dir / --pdf-dir, README/CSV to --doc-dir):
    conda run -n unlearning python plots/make_figure.py --method SimNPO
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedFormatter, NullLocator  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PLOTS = REPO / "plots"
FIGS = PLOTS / "figs"        # every figure (PNG + PDF) lands here; README / CSV stay in plots/
N_METRICS_COMPLETE = 12       # keys a finished TOFU_SUMMARY.json of this eval config has
DEFAULT_KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DEFAULT_ALPHAS = [0.5, 0.8, 0.9]
SEED_YARDSTICK = [0, 1, 2, 3]  # seeds available at k = 1 (noise yardstick, not drawn)
MAX_NEW_TOKENS = 200           # eval generation budget; k above it is clipped per round
DEFAULT_METHOD = "NPO"         # the weight-unlearning method of the draft q
TARGET_TAG = "Llama-3.2-1B-Instruct"


def method_paths(method: str) -> dict:
    """Per-draft-method defaults. NPO keeps the original, unsuffixed file names; any other
    method gets `_<method>` appended so several drafts can live side by side in plots/.
    The results root is the draft's OWN sub-tree: globbing plots/results as a whole would
    merge the runs of every draft into one (alpha, k, seed) table."""
    sfx = "" if method == DEFAULT_METHOD else f"_{method}"
    return dict(
        method=method, suffix=sfx,
        results_root=PLOTS / "results" / "original-unlearned" / f"target-{TARGET_TAG}" / f"draft-{TARGET_TAG}_{method}",
        bench=PLOTS / "bench" / f"spec_bench{sfx}.json",
        alpha_bench=PLOTS / "bench" / f"alpha_bench{sfx}.json",
        baselines=PLOTS / f"baselines{sfx}.json",
        draft=f"saves/unlearn/baselines/tofu/{method}/{TARGET_TAG}/forget10",
        draft_tag=f"{TARGET_TAG}_{method}",
    )


def bench_for_method(path: Path, method: str):
    """The bench JSON at `path` if it was recorded with this method's draft, else None."""
    d = read_json(path)
    if not isinstance(d, dict):
        return None
    draft = str(d.get("meta", {}).get("draft", "")).replace("\\", "/")
    return d if f"/{method}/" in draft or draft.startswith(f"{method}/") else None


def rel(p: Path, start: Path) -> str:
    """POSIX relative path for README links."""
    import os
    return Path(os.path.relpath(Path(p), Path(start))).as_posix()

# (summary key, panel tag, panel title, y label, README description)
METRIC_ROWS = [
    ("model_utility", "(a)", "Model utility", "Model utility",
     "TOFU aggregate utility (retain / real-authors / world-facts), seed 0 at every k"),
    ("forget_Q_A_gibberish", "(b)", "Gibberish", "Gibberish score",
     "gibberish classifier P(clean) of the forget-question generations (higher = fewer gibberish answers), seed 0 at every k"),
]
BASELINE_ORDER = ["target", "retain", "draft"]

# ── conference styling ───────────────────────────────────────────────────────
# Okabe & Ito (2008) colour-blind-safe palette; alpha is ordered small -> large.
OKABE_ITO = dict(blue="#0072B2", vermillion="#D55E00", green="#009E73", orange="#E69F00",
                 sky="#56B4E9", purple="#CC79A7", yellow="#F0E442", black="#000000")
# Each alpha gets a colour + a dash pattern + a marker, so the three series stay apart for
# every kind of colour-vision deficiency and in greyscale print.
SERIES_STYLES = [
    dict(color=OKABE_ITO["blue"], ls="-", marker="o"),
    dict(color=OKABE_ITO["vermillion"], ls=(0, (4.0, 2.0)), marker="s"),
    dict(color=OKABE_ITO["green"], ls=(0, (4.0, 1.6, 1.0, 1.6)), marker="^"),
]
# Reference lines: grey-scale (so they never compete with the series), no markers,
# distinct patterns, direct text labels.
BASELINE_STYLES = {
    "target": dict(color="#000000", ls=(0, (1.0, 1.4))),
    "retain": dict(color="#4d4d4d", ls=(0, (6.0, 2.0))),
    "draft": dict(color="#8a8a8a", ls="-"),
}
LEGEND_LABEL = {"target": "target $p$", "retain": "retain model", "draft": "draft $q$ (NPO)"}
THEORY_STYLE = dict(ls=(0, (1.0, 1.4)), lw=0.9, alpha=0.85)
INK, INK2, MUTED, GRID, AXIS = "#000000", "#333333", "#666666", "#e4e4e4", "#999999"

mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,        # embed TrueType, never Type 3 (ACL pubcheck)
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})

LEAF_RE = re.compile(r"a(?P<alpha>[0-9.]+)_k(?P<k>\d+)_s(?P<seed>\d+)$")


# ── loading ─────────────────────────────────────────────────────────────────
def read_json(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def load_runs(root: Path) -> list[dict]:
    """One dict per run dir holding a TOFU_SUMMARY.json (alpha, k, seed, metrics)."""
    runs = []
    for sp in sorted(Path(root).glob("**/TOFU_SUMMARY.json")):
        d = sp.parent
        summ = read_json(sp)
        meta = read_json(d / "run_meta.json") or {}
        m = LEAF_RE.search(d.name)
        alpha = meta.get("alpha", float(m["alpha"]) if m else None)
        k = meta.get("k_sud", int(m["k"]) if m else None)
        seed = meta.get("seed", int(m["seed"]) if m else None)
        if alpha is None or k is None or seed is None:
            continue
        summ = summ if isinstance(summ, dict) else {}
        runs.append(dict(
            dir=str(d), alpha=float(alpha), k=int(k), seed=int(seed),
            metrics={key: summ.get(key) for key, *_ in METRIC_ROWS},
            n_metrics=len(summ), complete=len(summ) >= N_METRICS_COMPLETE,
        ))
    return runs


def run_table(runs):
    """{(alpha, k, seed): {metric: value}}."""
    return {(r["alpha"], r["k"], r["seed"]): r["metrics"] for r in runs}


def load_baselines(path: Path) -> dict:
    """{name: {label, model_utility, forget_Q_A_gibberish, ...}} in BASELINE_ORDER (may be empty)."""
    d = read_json(path) or {}
    models = d.get("models", {})
    return {name: models[name] for name in BASELINE_ORDER if name in models}


# ── panels (a)/(b) statistics ───────────────────────────────────────────────
def metric_stats(table: dict, metric: str, alphas, ks, seeds):
    out = {}
    for a in alphas:
        sweep = {k: table[(a, k, 0)][metric] for k in ks
                 if (a, k, 0) in table and table[(a, k, 0)].get(metric) is not None}
        seed_vals = {s: table[(a, 1, s)][metric] for s in seeds
                     if (a, 1, s) in table and table[(a, 1, s)].get(metric) is not None}
        st = dict(alpha=a, sweep=sweep, seed_vals=seed_vals, n_seeds=len(seed_vals))
        if len(seed_vals) > 1:
            v = np.array(list(seed_vals.values()))
            st.update(seed_mean=float(v.mean()), seed_sd=float(v.std(ddof=1)),
                      seed_range=float(v.max() - v.min()))
        v1 = sweep.get(1)
        st["dev_from_k1"] = {k: x - v1 for k, x in sweep.items()} if v1 is not None else {}
        st["max_abs_dev"] = max((abs(d) for d in st["dev_from_k1"].values()), default=None)
        if st.get("seed_sd") and sweep:
            st["z_by_k"] = {k: (x - st["seed_mean"]) / st["seed_sd"] for k, x in sweep.items()}
            st["max_abs_z"] = max(abs(z) for z in st["z_by_k"].values())
        if len(sweep) > 1:
            xs = np.array(sorted(sweep), float)
            ys = np.array([sweep[k] for k in sorted(sweep)], float)
            st["slope_per_doubling"] = float(np.polyfit(np.log2(xs), ys, 1)[0])
            st["sweep_range"] = float(ys.max() - ys.min())
        out[a] = st
    return out


# ── panel (c) statistics ────────────────────────────────────────────────────
def theory_tps(ke, a, R):
    """No-bonus speculative sampling with an equal-size draft: per round k~ draft
    forwards + 1 target forward, E[committed] = (1 - a^k~)/(1 - a). k~ is the
    EFFECTIVE window (mean drafted tokens per round), which is <= k because the
    draft stops at its own EOS (and at the max_new_tokens budget)."""
    if a >= 1.0:
        return R * ke / (ke + 1)
    return R * (1.0 - a ** ke) / ((1.0 - a) * (ke + 1))


def theory_fwd_per_tok(cell: dict, a: float):
    """Predicted forwards per committed token for one benchmark cell, from the
    acceptance a and the DISTRIBUTION of drafted lengths — a ratio of sums over
    rounds, sum_i (k_i + 1) / sum_i (1 - a^k_i)/(1 - a). Evaluating the formula at
    the mean k~ instead overstates throughput once rounds are heterogeneous (the
    draft's EOS / the budget cut some rounds short): long rounds burn forwards
    linearly while their accepted tokens saturate at 1/(1 - a).
    Returns (forwards_per_token, source) with source in
      "rounds"  — exact per-round histogram (cells benchmarked with draft_len_hist)
      "prompts" — per-prompt mean lengths (older cells: heterogeneity across prompts only)
      "mean"    — the single cell mean k~ (fallback)."""
    def committed(k):
        return k if a >= 1.0 else (1.0 - a ** k) / (1.0 - a)
    hist = cell.get("draft_len_hist")
    if hist:
        fw = sum(n * (int(k) + 1) for k, n in hist.items())
        tok = sum(n * committed(int(k)) for k, n in hist.items())
        return fw / tok, "rounds"
    pp = [q for q in cell.get("per_prompt", []) if q.get("rounds") and q.get("proposed") is not None]
    if pp:
        fw = sum(q["proposed"] + q["rounds"] for q in pp)
        tok = sum(q["rounds"] * committed(q["proposed"] / q["rounds"]) for q in pp)
        return fw / tok, "prompts"
    ke = cell["proposed"] / cell["rounds"]
    return (ke + 1) / committed(ke), "mean"


def theory_curve(theory_by_k: dict):
    """k -> theory tokens/s, the per-cell values interpolated in log2 k between the
    measured k's and held flat beyond the last one (the window is EOS-capped there)."""
    xs = np.array(sorted(theory_by_k), float)
    ys = np.array([theory_by_k[k] for k in sorted(theory_by_k)], float)

    def fn(k):
        return float(np.interp(np.log2(float(k)), np.log2(xs), ys))
    return fn


def window_interp(kbar: dict):
    """k -> effective window k~: the measured mean drafted tokens per round at the
    measured k's, linearly interpolated in log2 k between them, held flat beyond
    the last measured k (the draft's EOS has capped the window by then)."""
    xs = np.array(sorted(kbar), float)
    ys = np.array([kbar[k] for k in sorted(kbar)], float)

    def fn(k):
        k = float(k)
        if k <= xs[0]:
            return min(k, ys[0])
        return float(np.interp(np.log2(k), np.log2(xs), ys))
    return fn


def bootstrap_tps(per_prompt, n_boot=4000, seed=0):
    sec = np.array([p["seconds"] for p in per_prompt], float)
    tok = np.array([p["new_tokens"] for p in per_prompt], float)
    n = len(sec)
    if n < 2 or sec.sum() <= 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    r = tok[idx].sum(1) / sec[idx].sum(1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def bench_stats(bench: dict | None, alphas, ks):
    recs = {(r["alpha"], r["k"]): r for r in (bench or {}).get("records", [])
            if r.get("tokens_per_sec")}
    # shared hardware constant: forwards per second, pooled over every cell
    fps_all = []
    for r in recs.values():
        fw = r["proposed"] + r["rounds"]
        if r["total_seconds"] > 0 and fw > 0:
            fps_all.append(fw / r["total_seconds"])
    shared = dict(R=(float(np.median(fps_all)) if fps_all else None),
                  R_min=(min(fps_all) if fps_all else None), R_max=(max(fps_all) if fps_all else None),
                  R_spread=((max(fps_all) - min(fps_all)) / np.median(fps_all) if fps_all else None),
                  n_cells=len(recs))
    out = {}
    for a in alphas:
        cells = {k: recs[(a, k)] for k in ks if (a, k) in recs}
        st = dict(alpha=a, cells=cells)
        if not cells:
            out[a] = st
            continue
        st["tps"] = {k: c["tokens_per_sec"] for k, c in cells.items()}
        st["ci"] = {k: bootstrap_tps(c.get("per_prompt", [])) for k, c in cells.items()}
        st["acceptance"] = {k: c["acceptance"] for k, c in cells.items() if c.get("acceptance") is not None}
        st["acc_over_proposed"] = {k: c.get("acceptance_over_proposed") for k, c in cells.items()}
        st["fwd_per_tok"] = {k: (c["proposed"] + c["rounds"]) / c["committed"] for k, c in cells.items() if c["committed"]}
        st["fwd_per_sec"] = {k: (c["proposed"] + c["rounds"]) / c["total_seconds"] for k, c in cells.items() if c["total_seconds"] > 0}
        st["kbar"] = {k: c["proposed"] / c["rounds"] for k, c in cells.items() if c["rounds"]}
        st["trunc_frac"] = {k: (c["rounds_draft_truncated"] / c["rounds"]) for k, c in cells.items()
                            if c.get("rounds_draft_truncated") is not None and c["rounds"]}
        st["tokens_per_round"] = {k: c.get("tokens_per_round") for k, c in cells.items()}
        st["vram_mb"] = {k: c.get("max_memory_allocated_mb") for k, c in cells.items()}
        st["algo_speedup"] = {k: 2.0 / v for k, v in st["fwd_per_tok"].items()}   # exact: k = 1 costs 2 forwards/token
        acc_n = sum(c["accepted"] for c in cells.values())
        ver_n = sum(c["verified"] for c in cells.values())
        if ver_n:
            st["a_mean"] = acc_n / ver_n
            av = np.array(list(st["acceptance"].values()))
            st.update(a_min=float(av.min()), a_max=float(av.max()), a_range=float(av.max() - av.min()))
            ses = [math.sqrt(max(c["acceptance"] * (1 - c["acceptance"]), 1e-12) / c["verified"])
                   for c in cells.values() if c.get("verified")]
            st["a_se"] = float(np.mean(ses)) if ses else None
        if st["kbar"]:
            st["window"] = window_interp(st["kbar"])
            kmax = max(st["kbar"])
            st["K_sat"] = st["kbar"][kmax]                 # window at the largest k
            st["K_sat_k"] = kmax
            st["window_saturated"] = st["K_sat"] < 0.7 * kmax   # EOS, not k, limits the draft
        tps = st["tps"]
        k_star = max(tps, key=tps.get)
        st.update(k_star_meas=k_star, tps_max=tps[k_star])
        if 1 in tps:
            st["speedup"] = {k: v / tps[1] for k, v in tps.items()}
            st["best_speedup"] = tps[k_star] / tps[1]
        if shared["R"] and "a_mean" in st and "window" in st:
            R, a_m, win = shared["R"], st["a_mean"], st["window"]
            # per measured cell: R / predicted forwards per token, from the round-length
            # distribution (see theory_fwd_per_tok); the mean-k~ closed form only where a
            # k was not benchmarked at all
            st["theory_src"], st["theory_mean_k"] = {}, {}
            th = {}
            for k in ks:
                if k in cells:
                    fpt, src = theory_fwd_per_tok(cells[k], a_m)
                    th[k], st["theory_src"][k] = R / fpt, src
                    st["theory_mean_k"][k] = theory_tps(st["kbar"][k], a_m, R)
                else:
                    th[k], st["theory_src"][k] = theory_tps(win(k), a_m, R), "interp"
            st["theory"] = th
            st["theory_fn"] = theory_curve({k: v for k, v in th.items() if k in cells} or th)
            grid = np.arange(1, max(ks) + 1)
            tg = np.array([st["theory_fn"](k) for k in grid])
            st["k_star_theory"] = int(grid[tg.argmax()])
            st["theory_best_speedup"] = float(tg.max() / st["theory_fn"](1))
            st["theory_rel_err"] = {k: (st["theory"][k] - tps[k]) / tps[k] for k in tps}
            st["theory_mean_k_rel_err"] = {k: (st["theory_mean_k"][k] - tps[k]) / tps[k] for k in tps if k in st["theory_mean_k"]}
        vr = [v for v in st["vram_mb"].values() if v is not None]
        if vr:
            st.update(vram_min=min(vr), vram_max=max(vr), vram_range=max(vr) - min(vr),
                      vram_rel_range=(max(vr) - min(vr)) / (sum(vr) / len(vr)))
        out[a] = st
    return out, shared


# ── figure ──────────────────────────────────────────────────────────────────
def log2_axis(ax, ks, label=True):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.xaxis.set_major_formatter(FixedFormatter([str(k) for k in ks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(min(ks) / 1.45, max(ks) * 1.45)
    if label:
        ax.set_xlabel("Draft window $k$")
    ax.tick_params(axis="both", which="major", pad=2)


def panel_title(ax, tag, title):
    ax.set_title(f"{tag} {title}", loc="left", fontsize=8, weight="bold", pad=4)


def series_kw(i, lw=1.1, ms=3.6):
    s = SERIES_STYLES[i % len(SERIES_STYLES)]
    return dict(color=s["color"], ls=s["ls"], lw=lw, marker=s["marker"], ms=ms,
                mec="white", mew=0.6, solid_capstyle="round", solid_joinstyle="round", dash_capstyle="round")


def draw_baselines(ax, baselines: dict, metric: str):
    """Horizontal grey reference lines (identity via the shared legend: distinct grey
    patterns; direct labels would collide with the series at the right edge). Returns
    the values drawn, for the y-range."""
    vals = [(b[metric], name) for name, b in baselines.items() if b.get(metric) is not None]
    for v, name in vals:
        ax.axhline(v, lw=0.8, zorder=1.5, **BASELINE_STYLES[name])
    return [v for v, _ in vals]


def draw_metric_panel(ax, S, metric, alphas, ks, baselines):
    ys_all = []
    for i, a in enumerate(alphas):
        st = S[a]
        if not st["sweep"]:
            continue
        xs = sorted(st["sweep"])
        ys = [st["sweep"][k] for k in xs]
        ax.plot(xs, ys, zorder=3, **series_kw(i))
        ys_all += ys
    ys_all += draw_baselines(ax, baselines, metric)
    if ys_all:
        lo, hi = min(ys_all), max(ys_all)
        pad = max(0.12 * (hi - lo), 0.004)
        ax.set_ylim(lo - pad, hi + pad)


def draw_throughput_panel(ax, B, shared, alphas, ks):
    have_b = any(B[a].get("tps") for a in alphas)
    if not have_b:
        ax.text(0.5, 0.5, "benchmark not run yet", ha="center", va="center", transform=ax.transAxes, color=MUTED)
        ax.grid(False)
        return
    ymax = 0.0
    kmax_meas = max(k for a in alphas for k in B[a].get("tps", {}))
    kk = np.geomspace(min(ks), kmax_meas, 400)
    for i, a in enumerate(alphas):
        st = B[a]
        if not st.get("tps"):
            continue
        c = SERIES_STYLES[i]["color"]
        xs = sorted(st["tps"])
        ys = [st["tps"][k] for k in xs]
        if "theory_fn" in st:
            ax.plot(kk, [st["theory_fn"](k) for k in kk], color=c, zorder=2, **THEORY_STYLE)
        cis = [st["ci"].get(k) for k in xs]
        if all(cis):
            lo = [y - ci[0] for y, ci in zip(ys, cis)]
            hi = [ci[1] - y for y, ci in zip(ys, cis)]
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=c, elinewidth=0.7,
                        capsize=1.6, capthick=0.7, zorder=2.5)
            ymax = max(ymax, max(ci[1] for ci in cis))
        ax.plot(xs, ys, zorder=3, **series_kw(i))
        ymax = max(ymax, max(ys))
    ax.set_ylim(0, ymax * 1.2)
    # mark each series' fastest k (the extreme) with a ring; the speed-ups live in the caption
    for i, a in enumerate(alphas):
        st = B[a]
        if "k_star_meas" in st:
            ax.plot([st["k_star_meas"]], [st["tps_max"]], ls="none", marker="o", ms=8.5, mfc="none",
                    mec=SERIES_STYLES[i]["color"], mew=0.9, zorder=4)
    # where the draft's own EOS caps the window (only once the largest k shows it)
    caps = [B[a]["K_sat"] for a in alphas if B[a].get("window_saturated")]
    if caps:
        kc = float(np.mean(caps))
        ax.axvline(kc, color=MUTED, lw=0.6, ls=(0, (1.0, 1.8)), zorder=1)
        ax.text(kc * 1.08, ymax * 1.17, f"draft EOS caps the\nwindow at ≈{kc:.0f} tokens",
                fontsize=6.0, color=INK2, ha="left", va="top", linespacing=1.2)
    ax.set_ylabel("Tokens / s")
    if any(B[a].get("theory") for a in alphas):
        # the theory belongs to this panel only -> its own small legend in the free lower-left corner
        ax.legend(handles=[Line2D([], [], color=INK2, label="theory (shared $R$)", **THEORY_STYLE)],
                  loc="lower left", fontsize=6.5, handlelength=2.4, borderaxespad=0.3, handletextpad=0.5)


def legend_handles(alphas, B, baselines):
    """(series + theory, reference lines) as two lists so layouts can arrange them."""
    main = [Line2D([], [], label=f"$\\alpha$ = {a:g}", **series_kw(i, lw=1.1, ms=3.4)) for i, a in enumerate(alphas)]
    refs = [Line2D([], [], lw=0.9, label=LEGEND_LABEL.get(name, b["label"]), **BASELINE_STYLES[name])
            for name, b in baselines.items()]
    return main, refs


def interleave_for_rows(row1, row2):
    """matplotlib fills a legend column-major; this order yields row1 above row2."""
    out = []
    for i in range(max(len(row1), len(row2))):
        if i < len(row1):
            out.append(row1[i])
        if i < len(row2):
            out.append(row2[i])
    return out


def save(fig, out_stem: Path, pdf_dir: Path | None = None):
    """PNG next to `out_stem`; the PDF there too, or under `pdf_dir` if given."""
    pdf = (Path(pdf_dir) / out_stem.name if pdf_dir else out_stem).with_suffix(".pdf")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf)
    fig.savefig(out_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def make_invariance_figure(layout, M, B, baselines, alphas, ks, out_stem: Path, pdf_dir=None):
    """(a) model_utility and (b) gibberish vs k. Reference lines only with --with-baselines."""
    main, refs = legend_handles(alphas, B, baselines)
    two_rows = bool(refs) and layout == "column"
    if layout == "column":
        fig, axes = plt.subplots(2, 1, figsize=(3.03, 4.1 if not two_rows else 4.3), sharex=True,
                                 gridspec_kw=dict(hspace=0.36))
        fig.subplots_adjust(left=0.17, right=0.985, top=0.9 if not two_rows else 0.87, bottom=0.115)
        handles, ncol = (interleave_for_rows(main, refs) if refs else main), max(len(main), len(refs))
    else:  # full text width, side by side
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.4))
        fig.subplots_adjust(left=0.085, right=0.99, top=0.8, bottom=0.2, wspace=0.32)
        handles, ncol = main + refs, len(main) + len(refs)
    for ax, (metric, tag, title, ylabel, _) in zip(axes, METRIC_ROWS):
        draw_metric_panel(ax, M[metric], metric, alphas, ks, baselines)
        panel_title(ax, tag, title)
        ax.set_ylabel(ylabel)
        log2_axis(ax, ks, label=(layout != "column" or ax is axes[-1]))
    fig.legend(handles=handles, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 0.997),
               handlelength=2.4, columnspacing=1.2, handletextpad=0.5, labelspacing=0.35)
    save(fig, out_stem, pdf_dir)


def make_throughput_figure(B, shared, alphas, ks, out_stem: Path, pdf_dir=None):
    """tokens/s vs k, one single-column panel."""
    main, _ = legend_handles(alphas, B, {})
    fig, ax = plt.subplots(1, 1, figsize=(3.03, 2.5))
    fig.subplots_adjust(left=0.15, right=0.985, top=0.87, bottom=0.185)
    draw_throughput_panel(ax, B, shared, alphas, ks)
    log2_axis(ax, ks, label=True)
    fig.legend(handles=main, loc="upper center", ncol=len(main), bbox_to_anchor=(0.5, 0.997),
               handlelength=2.4, columnspacing=1.2, handletextpad=0.5)
    save(fig, out_stem, pdf_dir)


# ── CSV + README ─────────────────────────────────────────────────────────────
def f(x, nd=4, none="—"):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return none
    return f"{x:.{nd}f}"


def write_csv(path, M, B, alphas, ks):
    mu, gb = M["model_utility"], M["forget_Q_A_gibberish"]
    cols = ["alpha", "k", "model_utility", "forget_Q_A_gibberish", "acceptance", "tok_per_s",
            "tok_per_s_ci_lo", "tok_per_s_ci_hi", "speedup_vs_k1", "algo_speedup_2_over_fwd_per_tok",
            "theory_tok_per_s", "forwards_per_token", "forwards_per_sec", "mean_drafted_per_round",
            "draft_truncated_frac", "acceptance_over_proposed", "vram_peak_mb",
            "mu_dev_from_k1", "mu_seed_sd_k1", "mu_z_vs_seeds",
            "gib_dev_from_k1", "gib_seed_sd_k1", "gib_z_vs_seeds"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for a in alphas:
            sm, sg, sb = mu[a], gb[a], B[a]
            for k in ks:
                ci = sb.get("ci", {}).get(k) or ("", "")
                w.writerow([
                    a, k, sm["sweep"].get(k, ""), sg["sweep"].get(k, ""),
                    sb.get("acceptance", {}).get(k, ""), sb.get("tps", {}).get(k, ""), ci[0], ci[1],
                    sb.get("speedup", {}).get(k, ""), sb.get("algo_speedup", {}).get(k, ""),
                    sb.get("theory", {}).get(k, ""), sb.get("fwd_per_tok", {}).get(k, ""),
                    sb.get("fwd_per_sec", {}).get(k, ""), sb.get("kbar", {}).get(k, ""),
                    sb.get("trunc_frac", {}).get(k, ""), sb.get("acc_over_proposed", {}).get(k, ""),
                    sb.get("vram_mb", {}).get(k, ""),
                    sm["dev_from_k1"].get(k, ""), sm.get("seed_sd", ""), sm.get("z_by_k", {}).get(k, ""),
                    sg["dev_from_k1"].get(k, ""), sg.get("seed_sd", ""), sg.get("z_by_k", {}).get(k, ""),
                ])


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def metric_section(L, S, metric, tag, alphas, ks, baselines, nd=4):
    L.append(f"## Panel {tag} — {metric} across k\n")
    hdr = ["α"] + [f"k={k}" for k in ks]
    L.append(md_table(hdr, [[f"{a:g}"] + [f(S[a]["sweep"].get(k), nd) for k in ks] for a in alphas]))
    L.append("")
    L.append(md_table(["α", "max abs Δ vs k=1", "range over k", "slope / doubling of k", "seeds at k=1", "seed SD (k=1)", "max abs z vs seeds"], [
        [f"{a:g}", f(S[a].get("max_abs_dev"), nd), f(S[a].get("sweep_range"), nd),
         (f"{S[a]['slope_per_doubling']:+.5f}" if "slope_per_doubling" in S[a] else "—"),
         S[a]["n_seeds"], f(S[a].get("seed_sd"), nd), f(S[a].get("max_abs_z"), 2)] for a in alphas]))
    L.append("")
    verdict = []
    for a in alphas:
        st = S[a]
        if st.get("max_abs_z") is not None:
            verdict.append(f"α={a:g}: max abs z = {st['max_abs_z']:.2f}" +
                           (f", slope {st['slope_per_doubling']:+.5f} per doubling of k vs seed SD {st['seed_sd']:.4f}" if "slope_per_doubling" in st else ""))
    if verdict:
        L.append("- z = (value at k − mean over seeds at k=1) / seed SD; abs z ≲ 2–3 is indistinguishable from seed "
                 "noise. The slope is a least-squares trend in log2 k; a k effect would show as a slope ≫ seed SD. "
                 + "; ".join(verdict) + ".")
    refs = [(b["label"], b[metric]) for b in baselines.values() if b.get(metric) is not None]
    if refs:
        L.append("- Reference values (greedy-decoded standard evals, same prompts; not drawn by default): "
                 + ", ".join(f"{lab} = {v:.4f}" for lab, v in refs) + ".")
    L.append("")


def write_readme(path, M, B, shared, refs, drawn, alphas, ks, runs, bench, args, P, names):
    baselines = refs
    method = P["method"]
    doc_dir = Path(path).parent
    png = {k: rel(v.with_suffix(".png"), doc_dir) for k, v in names.items()}
    pdf = {k: (Path(args.pdf_dir) / v.name).with_suffix(".pdf").name for k, v in names.items()}
    results_rel = rel(P["results_root"], REPO)
    bench_rel = rel(P["bench"], REPO)
    baselines_rel = rel(P["baselines"], REPO)
    rebuild = (f"conda run -n unlearning python plots/make_figure.py --method {method}"
               + (f" --out-dir {rel(args.out_dir, REPO)}" if Path(args.out_dir) != FIGS else "")
               + (f" --pdf-dir {rel(args.pdf_dir, REPO)}" if Path(args.pdf_dir) != Path(args.out_dir) else "")
               + (f" --doc-dir {rel(args.doc_dir, REPO)}" if Path(args.doc_dir) != PLOTS else ""))
    now = datetime.now().isoformat(timespec="minutes")
    tab = run_table(runs)
    n_sweep = sum(1 for a in alphas for k in ks if (a, k, 0) in tab)
    n_seed = sum(1 for a in alphas for s in SEED_YARDSTICK[1:] if (a, 1, s) in tab)
    incomplete = [r["dir"] for r in runs if not r["complete"]]
    missing = [f"a{a:g}_k{k}_s0" for a in alphas for k in ks if (a, k, 0) not in tab]
    n_cells = sum(len(B[a].get("tps", {})) for a in alphas)
    meta = (bench or {}).get("meta", {})

    L = []
    L.append(f"# k-sweep figures for SUD ({method} draft) — model utility and gibberish (invariance) and throughput vs the draft window k\n")
    L.append(f"_Generated {now} by `plots/make_figure.py --method {method}`. Everything for these figures lives in `plots/`._\n")
    L.append(f"**Figure 1 — invariance** (`{pdf['col']}`, one ACL column; `{pdf['wide']}`, full text width):\n")
    L.append(f"![k invariance, column]({png['col']})\n")
    L.append(f"![k invariance, wide]({png['wide']})\n")
    if n_cells:
        L.append(f"**Figure 2 — throughput** (`{pdf['tps']}`, one ACL column):\n")
        L.append(f"![k throughput]({png['tps']})\n")
    else:
        L.append(f"**Figure 2 — throughput**: not drawn — `{bench_rel}` is missing or holds no cells for this draft "
                 f"(run `plots/bench_speculative.py --draft {P['draft']} --results-root {results_rel} --out {bench_rel} --resume`).\n")

    L.append("## Suggested captions\n")
    best = max(((a, B[a]["k_star_meas"], B[a]["best_speedup"]) for a in alphas if "best_speedup" in B[a]),
               key=lambda t: t[2], default=None)
    sds = {m: [M[m][a]["seed_sd"] for a in alphas if M[m][a].get("seed_sd")] for m, *_ in METRIC_ROWS}
    cap1 = ("**Figure 1. The draft window k does not change SUD's output.** TOFU forget10, target p = Llama-3.2-1B-Instruct "
            f"fine-tuned on TOFU, draft q = its {method}-unlearned checkpoint, seed 0 at every k. "
            "(a) Model utility and (b) gibberish score (classifier P(clean) of the forget answers; higher = fewer gibberish answers) "
            "are flat in k for every α: SUD samples each token exactly from π ∝ p^(1−α) q^α by draft-verify speculative sampling "
            "with no bonus token, so k only sets how many tokens are proposed per round. "
            + (f"Seed-to-seed SD at k = 1 is {min(sds['model_utility']):.3f}–{max(sds['model_utility']):.3f} (utility) and "
               f"{min(sds['forget_Q_A_gibberish']):.3f}–{max(sds['forget_Q_A_gibberish']):.3f} (gibberish)." if all(sds.values()) else "")
            + (f" Grey lines: the original model (α = 0 limit), the {method} draft (α = 1 limit) and the retain model, greedy-decoded "
               "standard evals on the same prompts (SUD samples at T = 1)." if drawn else ""))
    cap2 = ("**Figure 2. What k buys in speed.** Throughput of the same cell on 100 forget prompts (batch 1, no KV cache; "
            "error bars: bootstrap 95% CI over prompts; rings mark each α's fastest k). "
            "Dotted: T(k) = R(1−a^k̃)/((1−a)(k̃+1)) with one shared forward rate R, per-α acceptance a, and k̃ the measured mean "
            "drafted tokens per round, which saturates once the draft reaches its own end-of-sequence before k. "
            + (f"Best measured speed-up over k = 1: {best[2]:.2f}× at α = {best[0]:g}, k = {best[1]}." if best else ""))
    L.append(cap1)
    L.append("")
    L.append(cap2)
    L.append("")

    L.append("## Status\n")
    L.append(f"- Panels (a)/(b) eval runs: **{n_sweep}/{len(alphas) * len(ks)}** sweep points (seed 0) under `{results_rel}/`; "
             f"{n_seed}/{len(alphas) * (len(SEED_YARDSTICK) - 1)} extra k=1 seed runs (seeds {SEED_YARDSTICK[1:]}) available as the noise yardstick.")
    if missing:
        L.append(f"  - missing: {', '.join(missing)}")
    if incomplete:
        L.append(f"  - {len(incomplete)} run dir(s) have a partial TOFU_SUMMARY.json (< {N_METRICS_COMPLETE} metrics); "
                 f"metrics are used only where present.")
    L.append(f"- Figure 2 benchmark cells: **{n_cells}/{len(alphas) * len(ks)}**"
             + (f" from `{meta.get('bench_file', '?')}` (`{Path(meta.get('prompt_source', '')).name or 'n/a'}` prompts, n={meta.get('n_prompts', '?')}, "
                f"max_new_tokens={meta.get('max_new_tokens', '?')}, GPU: {meta.get('gpu', '?')})" if meta else
                f" — no `{bench_rel}` recorded with this draft yet."))
    L.append("- Reference lines (target / draft / retain): "
             + ("**drawn** (`--with-baselines`)" if drawn else "**not drawn** (default; pass `--with-baselines` to add them)")
             + (f"; values from `{baselines_rel}` ({', '.join(b['label'] for b in baselines.values())})" if baselines else
                f"; `{baselines_rel}` missing — run `plots/baseline_refs.py --method {method}`") + ".")
    L.append("")

    L.append("## Conference styling (EMNLP / ACL)\n")
    L.append(f"- Sizes: `{pdf['col']}` is one ACL column (3.03 in) with (a) above (b) on a shared k-axis, "
             f"`{pdf['wide']}` the full text width (6.3 in) side by side; `{pdf['tps']}` is one column. "
             "Text 7–8 pt at print size.")
    L.append("- Colour: Okabe–Ito colour-blind-safe palette (blue #0072B2, vermillion #D55E00, bluish green #009E73). "
             "Every α also has its own dash pattern (solid / dashed / dash-dot) and marker (circle / square / triangle), so the "
             "series stay apart under any colour-vision deficiency and in greyscale print; the theory is dotted; optional "
             "reference lines are grey-scale with distinct patterns.")
    L.append("- Fonts: STIX / Times-like serif to match the ACL body text; embedded as TrueType (`pdf.fonttype = 42`), "
             "never Type 3, so the PDF passes ACL's pubcheck. No subtitles inside the figure — see the suggested caption.")
    L.append("")

    L.append("## Setup\n")
    L.append("- Cell: target p = `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`, draft q = "
             f"`{P['draft']}` ({method}), TOFU forget10 eval suite.")
    L.append("- SUD decodes from π = softmax((1−α)·log p + α·log q) by draft-verify speculative sampling "
             "(`src/model/sud.py`): the draft proposes up to k tokens from q, one target forward verifies them, "
             "each is accepted w.p. min(1, π/q), the first rejection is replaced by a residual sample and ends "
             "the round. No bonus token (it would follow p and leak forget content). k therefore never changes "
             "the sampled distribution — Figure 1 is the empirical check; Figure 2 is what k buys in speed.")
    L.append(f"- Grid: α ∈ {{{', '.join(f'{a:g}' for a in alphas)}}} × k ∈ {{{', '.join(str(k) for k in ks)}}}, seed 0. "
             f"Noise yardstick: seeds {SEED_YARDSTICK} at k = 1 (SD reported, no band drawn).")
    L.append(f"- The eval generates at most {MAX_NEW_TOKENS} new tokens, so a window k > {MAX_NEW_TOKENS} is clipped to the "
             "remaining budget each round; and the draft stops proposing at its own EOS, so for k beyond the "
             "draft's typical answer length (~35 tokens) the effective window no longer grows with k — "
             "k = 64, 128, 256 are then near-identical procedures (only the random draws differ).")
    L.append("- Figure 2: 100 prompts = the first 100 `forget_Q_A_ROUGE` inputs of a Part-1 TOFU_EVAL.json, rebuilt "
             "token-for-token through the eval's own dataset/collator path and asserted to decode to the stored "
             "`input` strings; batch size 1; eval generation config (max_new_tokens = 200, temperature 1); "
             "wall-clock around `generate`, CUDA-synchronised; no KV cache (the prototype re-runs full forwards). "
             "Error bars: bootstrap 95% CI (4000 resamples of the prompt set, ratio of total tokens to total seconds).")
    L.append("")

    if baselines:
        L.append("## Reference values (panels a/b) — " + ("drawn" if drawn else "not drawn by default; `--with-baselines` adds them") + "\n")
        L.append(md_table(["line", "model", "role", "model_utility", "forget_Q_A_gibberish", "gibberish source"], [
            [b["label"], f"`{b['model']}`", b["role"], f(b.get("model_utility")), f(b.get("forget_Q_A_gibberish")),
             (b.get("forget_Q_A_gibberish_source", "") + (f" (stored {b['forget_Q_A_gibberish_stored']:.4f}, abs diff {b['recompute_abs_diff']:.4f})"
                                                          if "forget_Q_A_gibberish_stored" in b else ""))]
            for b in baselines.values()]))
        L.append("")
        L.append("- These are the standard **greedy-decoded** evals (`do_sample: false`) on byte-identical prompts; SUD always "
                 "samples at T = 1, which by itself costs about 0.015 model_utility relative to greedy decoding (measured on "
                 "the main grid as the median generate-minus-forward gap: +0.022 sampled vs +0.037 greedy). Read the SUD-vs-reference "
                 "gap with that in mind; the k-invariance claim does not depend on it. Where the metric was not stored "
                 "(target, retain) it was recomputed from the stored generations with the exact metric config; the draft's "
                 "recomputation reproduces its stored value (see the abs diff column).")
        L.append("")

    metric_section(L, M["model_utility"], "model_utility", "(a)", alphas, ks, baselines)
    metric_section(L, M["forget_Q_A_gibberish"], "forget_Q_A_gibberish", "(b)", alphas, ks, baselines)

    # ── panel (c) ──────────────────────────────────────────────────────────
    L.append("## Figure 2 — throughput, acceptance, forward rate, VRAM vs k\n")
    if n_cells:
        hdr = ["α"] + [f"k={k}" for k in ks]
        L.append("**Tokens / s** (measured; bootstrap 95% CI over prompts):\n")
        L.append(md_table(hdr, [[f"{a:g}"] + [
            (f"{B[a]['tps'][k]:.2f} [{B[a]['ci'][k][0]:.1f}, {B[a]['ci'][k][1]:.1f}]" if k in B[a].get("tps", {}) and B[a]["ci"].get(k)
             else f(B[a].get("tps", {}).get(k), 2)) for k in ks] for a in alphas]))
        L.append("")
        L.append("**Speed-up vs k = 1** — measured wall-clock ratio, and in parentheses the exact algorithmic ratio "
                 "2 / (forwards per committed token), which is free of timing noise:\n")
        L.append(md_table(hdr + ["k* (meas.)", "best (meas.)", "k* (theory)", "theory ceiling"], [
            [f"{a:g}"] + [(f"{B[a]['speedup'][k]:.2f}× ({B[a]['algo_speedup'][k]:.2f}×)" if k in B[a].get("speedup", {}) else "—") for k in ks]
            + [B[a].get("k_star_meas", "—"), (f"{B[a]['best_speedup']:.2f}×" if "best_speedup" in B[a] else "—"),
               B[a].get("k_star_theory", "—"), (f"{B[a]['theory_best_speedup']:.2f}×" if "theory_best_speedup" in B[a] else "—")]
            for a in alphas]))
        L.append("")
        if best:
            L.append(f"- **Best measured speed-up: {best[2]:.2f}× at (α = {best[0]:g}, k\\* = {best[1]})**. "
                     "The ceiling for no-bonus speculative sampling with an equal-size draft is 2× over k = 1 "
                     "(T(1) = R/2, sup_k T = R as a → 1).")
        L.append("")
        srcs = sorted({v for a in alphas for v in B[a].get("theory_src", {}).values()})
        src_note = {"rounds": "exact per-round histogram (`draft_len_hist`)",
                    "prompts": "per-prompt mean lengths (cells benchmarked before the histogram was recorded; heterogeneity across prompts only)",
                    "mean": "the single cell mean", "interp": "interpolated window (k not benchmarked)"}
        L.append("**Theory** T(k) = R / F(k), F = predicted forwards per committed token = Σ_rounds (k_i+1) / Σ_rounds (1−a^k_i)/(1−a), "
                 "i.e. the round-length DISTRIBUTION, not its mean (k_i ≤ k: the draft stops at its own EOS and at the generation budget; "
                 "once rounds are heterogeneous, the closed form at the mean k̃ overstates throughput because long rounds burn forwards "
                 "linearly while their accepted tokens saturate at 1/(1−a)). One shared R = median forwards/s over all cells "
                 f"({f(shared['R'], 1)} forwards/s; per-cell range {f(shared['R_min'], 1)}–{f(shared['R_max'], 1)}, spread {shared['R_spread']:.0%}); "
                 "a = pooled acceptance per α. Round lengths per cell from: "
                 + "; ".join(f"**{k}** = {src_note.get(k, k)}" for k in srcs) + ". Tokens/s, measured / theory:\n")
        L.append(md_table(hdr + ["a", "k̃ at largest k"], [[f"{a:g}"] + [
            (f"{B[a]['tps'][k]:.1f} / {B[a]['theory'][k]:.1f}" if k in B[a].get("tps", {}) and k in B[a].get("theory", {})
             else f(B[a].get("tps", {}).get(k), 1)) for k in ks]
            + [f(B[a].get("a_mean"), 3), (f"{B[a]['K_sat']:.1f} (k={B[a]['K_sat_k']})" if "K_sat" in B[a] else "—")]
            for a in alphas]))
        errs = [abs(e) for a in alphas for e in B[a].get("theory_rel_err", {}).values()]
        errs_mean = [abs(e) for a in alphas for e in B[a].get("theory_mean_k_rel_err", {}).values()]
        if errs:
            L.append(f"\n- Theory vs measurement: mean abs relative error {np.mean(errs):.1%}, max {max(errs):.1%} over "
                     f"{len(errs)} cells (the closed form at the mean k̃ would give {np.mean(errs_mean):.1%} / {max(errs_mean):.1%}). "
                     "Nothing is fitted: R is shared and a is the pooled acceptance, so the residual is hardware-rate fluctuation plus "
                     "whatever round-length heterogeneity the available per-cell data cannot resolve (see the source note above).")
            L.append("\n  Signed relative error (theory − measured) / measured, per cell:\n")
            L.append(md_table(hdr, [[f"{a:g}"] + [(f"{B[a]['theory_rel_err'][k]:+.1%}" if k in B[a].get("theory_rel_err", {}) else "—")
                                                  for k in ks] for a in alphas]))
        L.append("")
        L.append("**Forward rate** (forwards / s = (proposed + rounds) / seconds; the hardware constant the theory assumes — should be flat in k and α):\n")
        L.append(md_table(hdr, [[f"{a:g}"] + [f(B[a].get("fwd_per_sec", {}).get(k), 1) for k in ks] for a in alphas]))
        L.append("")
        L.append("**Forwards per committed token** (exact, from the counters; 2 at k = 1 by construction):\n")
        L.append(md_table(hdr, [[f"{a:g}"] + [f(B[a].get("fwd_per_tok", {}).get(k), 3) for k in ks] for a in alphas]))
        L.append("")
        L.append("**Mean drafted tokens per round k̃** (= proposed / rounds ≤ k; saturates once the draft reaches its EOS before k):\n")
        L.append(md_table(hdr, [[f"{a:g}"] + [f(B[a].get("kbar", {}).get(k), 1) for k in ks] for a in alphas]))
        if any(B[a].get("trunc_frac") for a in alphas):
            L.append("\nFraction of rounds whose draft stopped before k (EOS or budget):\n")
            L.append(md_table(hdr, [[f"{a:g}"] + [(f"{B[a]['trunc_frac'][k]:.2f}" if k in B[a].get("trunc_frac", {}) else "—") for k in ks] for a in alphas]))
        L.append("")
        L.append("**Acceptance a = accepted / verified** (the per-token acceptance probability; should be constant in k):\n")
        L.append(md_table(hdr + ["pooled", "max−min"], [
            [f"{a:g}"] + [f(B[a].get("acceptance", {}).get(k), 3) for k in ks]
            + [f(B[a].get("a_mean"), 3), f(B[a].get("a_range"), 3)] for a in alphas]))
        const = [a for a in alphas if "a_range" in B[a]]
        if const:
            worst_a = max(const, key=lambda a: B[a]["a_range"])
            worst, se = B[worst_a]["a_range"], B[worst_a].get("a_se")
            verdict = ("approximately constant in k (within sampling noise)" if se is None or worst <= 5 * se
                       else "NOT constant in k")
            L.append(f"\n- Acceptance is {verdict}: largest max−min over k = {worst:.3f} at α={worst_a:g}"
                     + (f", vs a per-cell binomial SE of ≈{se:.3f}" if se else "") + ".")
        L.append("\nAccepted / *proposed* (the literal counter ratio; proposals after the first rejection are discarded "
                 "unverified, so it falls with k mechanically — not the process's acceptance rate):\n")
        L.append(md_table(hdr, [[f"{a:g}"] + [f(B[a].get("acc_over_proposed", {}).get(k), 3) for k in ks] for a in alphas]))
        L.append("")
        L.append("**Peak VRAM** (`torch.cuda.max_memory_allocated`, MiB; both models resident):\n")
        L.append(md_table(hdr + ["max−min", "rel."], [
            [f"{a:g}"] + [f(B[a].get("vram_mb", {}).get(k), 0) for k in ks]
            + [f(B[a].get("vram_range"), 0), (f"{B[a]['vram_rel_range']:.1%}" if "vram_rel_range" in B[a] else "—")]
            for a in alphas]))
        vr = [B[a]["vram_rel_range"] for a in alphas if "vram_rel_range" in B[a]]
        if vr:
            L.append(f"\n- VRAM is {'flat' if max(vr) < 0.05 else 'NOT flat'} across k "
                     f"(largest relative spread {max(vr):.1%}); the draft window only adds k̃ extra rows to one "
                     f"verification forward.")
    else:
        L.append(f"_Benchmark not run yet — `{bench_rel}` missing, empty, or recorded with another draft._")
    L.append("")

    L.append("## Files\n")
    L.append("- `run_k_sweep.sh` — copy of `scripts/run_sud_original_unlearned_draft.sh` narrowed to this cell "
             "(k outer, α inner; `RESULTS_ROOT` → `plots/results`). Resume-safe: re-run to fill missing cells.")
    L.append("- `bench_speculative.py` — panel (c) benchmark (monkeypatches the unmodified `src/model/sud.py` debugger seam; "
             "writes no debug files; `--resume` skips cells already in the JSON).")
    L.append("- `baseline_refs.py` — builds `baselines.json` (reference lines) from the standard evals of the target, draft and retain models.")
    L.append("- `make_figure.py` — both figures (invariance in two sizes, throughput) + `fig_k_sweep_data.csv` + this README; `--with-baselines` adds the reference lines.")
    L.append("- `k_sweep_job.sbatch` — the single SLURM job chaining Part 2 (bench) → Part 1 (evals) → Part 3 (figure); "
             "draws the figure from whatever exists if the time limit approaches; re-submit it to resume.")
    L.append("- `results/` — the eval run dirs; `bench/spec_bench.json` — one record per (α, k) with per-prompt detail; `logs/`.")
    L.append("")
    L.append(f"Rebuild the figure: `{rebuild}`")
    Path(path).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default=DEFAULT_METHOD,
                    help="weight-unlearning method of the draft q (sub-dir of saves/unlearn/baselines/tofu/); "
                         "sets the default results root / bench / baselines file and suffixes the output names")
    ap.add_argument("--results-root", type=Path, default=None,
                    help="default: this method's own sub-tree of plots/results (never the whole tree: drafts would mix)")
    ap.add_argument("--bench", type=Path, default=None, help="default: plots/bench/spec_bench[_<method>].json")
    ap.add_argument("--baselines", type=Path, default=None, help="default: plots/baselines[_<method>].json")
    ap.add_argument("--with-baselines", action="store_true",
                    help="draw the target / draft / retain reference lines in (a)/(b) (off by default; values are always tabulated in the README)")
    ap.add_argument("--out-dir", type=Path, default=FIGS, help="where the PNGs go (default: plots/figs)")
    ap.add_argument("--pdf-dir", type=Path, default=None, help="where the PDFs go (default: --out-dir)")
    ap.add_argument("--doc-dir", type=Path, default=PLOTS, help="where the README and CSV go")
    ap.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    ap.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    args = ap.parse_args()

    P = method_paths(args.method)
    args.results_root = args.results_root or P["results_root"]
    args.bench = args.bench or P["bench"]
    args.baselines = args.baselines or P["baselines"]
    args.pdf_dir = args.pdf_dir or args.out_dir
    P.update(results_root=args.results_root, bench=args.bench, baselines=args.baselines)
    LEGEND_LABEL["draft"] = f"draft $q$ ({args.method})"
    sfx = P["suffix"]
    names = dict(col=args.out_dir / f"fig_k_invariance{sfx}", wide=args.out_dir / f"fig_k_invariance_wide{sfx}",
                 tps=args.out_dir / f"fig_k_throughput{sfx}")

    alphas, ks = sorted(args.alphas), sorted(args.ks)
    if len(alphas) > len(SERIES_STYLES):
        raise SystemExit(f"at most {len(SERIES_STYLES)} alphas have a distinct colour+dash+marker style")
    if not args.results_root.exists():
        raise SystemExit(f"no eval runs for draft method {args.method!r}: {args.results_root} does not exist")

    runs = load_runs(args.results_root)
    tab = run_table(runs)
    M = {metric: metric_stats(tab, metric, alphas, ks, SEED_YARDSTICK) for metric, *_ in METRIC_ROWS}
    bench_path = args.bench
    bench = bench_for_method(bench_path, args.method)
    if bench is None:
        if bench_path.exists():
            print(f"note: {bench_path.name} was recorded with another draft — ignored for method {args.method}")
        # the live file is absent while a re-run is queued; fall back to the newest archived bench OF THIS DRAFT
        cands = [q for q in sorted(bench_path.parent.glob("spec_bench*.json"), key=lambda q: q.stat().st_mtime)
                 if bench_for_method(q, args.method) is not None]
        if cands:
            bench_path = cands[-1]
            bench = bench_for_method(bench_path, args.method)
            print(f"note: {args.bench.name} missing — using {bench_path.name} for panel (c)")
    if bench is not None:
        bench.setdefault("meta", {})["bench_file"] = str(bench_path.relative_to(REPO)) if bench_path.is_relative_to(REPO) else str(bench_path)
    B, shared = bench_stats(bench, alphas, ks)
    refs = load_baselines(args.baselines)               # always tabulated in the README
    drawn = refs if args.with_baselines else {}          # drawn only on request

    for d in (args.out_dir, args.pdf_dir, args.doc_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
    make_invariance_figure("column", M, B, drawn, alphas, ks, names["col"], args.pdf_dir)
    make_invariance_figure("wide", M, B, drawn, alphas, ks, names["wide"], args.pdf_dir)
    if shared["n_cells"]:
        make_throughput_figure(B, shared, alphas, ks, names["tps"], args.pdf_dir)
    else:
        print(f"note: no benchmark cells for method {args.method} — {names['tps'].name} not drawn")
    write_csv(args.doc_dir / f"fig_k_sweep_data{sfx}.csv", M, B, alphas, ks)
    write_readme(args.doc_dir / f"README{sfx}.md", M, B, shared, refs, drawn, alphas, ks, runs, bench, args, P, names)

    # console summary
    print(f"runs: {len(runs)} ({sum(r['complete'] for r in runs)} complete); bench cells: {shared['n_cells']}; "
          f"shared R = {f(shared['R'], 2)} fwd/s (spread {f(shared['R_spread'], 3)}); reference lines drawn: {list(drawn) or 'none'}")
    for metric, tag, *_ in METRIC_ROWS:
        for a in alphas:
            st = M[metric][a]
            print(f"{tag} {metric} α={a:g}: { {k: round(v, 4) for k, v in sorted(st['sweep'].items())} } "
                  f"seed_sd={f(st.get('seed_sd'))} maxΔ={f(st.get('max_abs_dev'))} max|z|={f(st.get('max_abs_z'), 2)} "
                  f"slope/doubling={f(st.get('slope_per_doubling'), 5)}")
    for a in alphas:
        sb = B[a]
        if sb.get("tps"):
            print(f"(c) α={a:g}: tps={ {k: round(v, 2) for k, v in sorted(sb['tps'].items())} } a={f(sb.get('a_mean'), 3)} "
                  f"k̃(kmax)={f(sb.get('K_sat'), 1)} k*={sb.get('k_star_meas')} speedup={f(sb.get('best_speedup'), 2)}× "
                  f"k*_theory={sb.get('k_star_theory')}")
    print(f"wrote {names['col'].name}.png / .pdf (+ _wide{'' if not shared['n_cells'] else ', + ' + names['tps'].name}) "
          f"-> png: {args.out_dir}, pdf: {args.pdf_dir}; fig_k_sweep_data{sfx}.csv + README{sfx}.md -> {args.doc_dir}")


if __name__ == "__main__":
    main()
