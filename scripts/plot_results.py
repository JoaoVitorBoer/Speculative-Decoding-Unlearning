#!/usr/bin/env python3
"""Plot SUD sweeps against the weight-unlearning baselines they decode from.

Writes static PNG + PDF figures (plus one combined PDF) to `figures/results/`,
so nothing needs a server or a browser -- open the files straight from VS Code.

Three sources are joined:
  * SUD sweeps     saves/unlearn/sud/results/summary.csv         (alpha x k grid)
  * q / baseline   saves/unlearn/baselines/tofu/summary_baselines.csv (the
                   weight-unlearned checkpoint -- which is ALSO the draft of the
                   SUD run)
  * p / reference  saves/eval/tofu_<model>_full/evals_<split>/TOFU_SUMMARY.json
                   (the original model, no unlearning)

The join: a SUD run's `draft` tag is `<model>_<METHOD>`, which is exactly the
baseline row keyed by (method, model, split). So each SUD sweep can be drawn
against both the weights it blends from (q) and the model it started as (p).

Data loading and metric semantics come from scripts/sud_results_data.py, shared with
the interactive dashboard (dashboard/app.py), so the two cannot drift apart.

NOTE: no reference eval runs the gibberish classifier, so p has no fluency and
therefore no `utility` and no `aggregate`. `aggregate_mu` needs only
model_utility, which every reference eval does have -- so it is the one composite
whose panel CAN show p. Panels that cannot omit the p line and say so rather than
implying p scores zero.

Usage (needs the `unlearning` env for matplotlib):
    conda run -n unlearning python scripts/plot_results.py
    conda run -n unlearning python scripts/plot_results.py --only forget01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")                              # headless cluster: no display needed
import matplotlib.pyplot as plt             # noqa: E402
from matplotlib.lines import Line2D         # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sud_results_data import (  # noqa: E402
    DEFAULT_BASELINES, METRICS, best_row as best_sud, get, load_all, q_row,
)

REPO = Path(__file__).resolve().parents[1]

# ── palette: same as figures/make_figure.py (dataviz slots 1-3, all-pairs safe) ──
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURF, AXIS = "#e1e0d9", "#fcfcfb", "#c3c2b7"
# Ordinal blue ramp (validated --ordinal, light): light -> dark.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]

mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.linewidth": 0.8, "font.size": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})

SWEEP_METRICS = ["aggregate", "aggregate_mu", "memorization", "privacy_score"]
BAR_METRICS = ["aggregate", "aggregate_mu", "memorization", "privacy_score",
               "utility", "model_utility", "exact_memorization",
               "extraction_strength"]


def refline(ax, v, color, label):
    """Horizontal p/q reference, labelled at the right edge."""
    if v is None:
        return False
    ax.axhline(v, color=color, lw=1.6, zorder=2)
    ax.annotate(label, xy=(1.0, v), xycoords=("axes fraction", "data"),
                xytext=(4, 0), textcoords="offset points",
                va="center", ha="left", fontsize=8.5, color=INK, weight="bold",
                annotation_clip=False)
    return True


# ── figure 1: alpha sweep, metrics x methods ─────────────────────────────────
def fig_sweep(sud, base, p_refs, model, split):
    methods = sorted(sud.method.unique())
    ks = sorted(sud.k.unique())
    nr, nc = len(SWEEP_METRICS), len(methods)
    fig, axes = plt.subplots(nr, nc, figsize=(3.9 * nc + 1.1, 2.5 * nr + 1.0),
                             squeeze=False, sharex=True)
    p = p_refs.get((model, split))

    for r, key in enumerate(SWEEP_METRICS):
        label, hib = METRICS[key].label, METRICS[key].higher_is_better
        for c, meth in enumerate(methods):
            ax = axes[r][c]
            d = sud[sud.method == meth]
            for i, k in enumerate(ks):
                dk = d[d.k == k].sort_values("alpha").dropna(subset=[key])
                if dk.empty:
                    continue
                ax.plot(dk.alpha, dk[key], "-o", color=RAMP[i % len(RAMP)],
                        lw=1.8, ms=4.5, mec=SURF, mew=1.2, zorder=3,
                        label=f"k={k}" if r == 0 and c == 0 else None)
            q = q_row(base, model, split, meth)
            refline(ax, get(q, key), ORANGE, "q")
            refline(ax, get(p, key), AQUA, "p")
            # Metrics computed from teacher-forced likelihoods are identical across
            # k, so the four lines land on top of each other and only the last drawn
            # is visible. Say so, or it reads as three missing series.
            spread = max((d[d.alpha == a][key].max() - d[d.alpha == a][key].min()
                          for a in d.alpha.unique() if d[d.alpha == a][key].notna().any()),
                         default=0.0)
            # Relative to the panel's own data range: an absolute cutoff flags the
            # same visual phenomenon inconsistently across differently-scaled panels.
            span = d[key].max() - d[key].min() if d[key].notna().any() else 0.0
            coincide = np.isfinite(spread) and spread <= max(1e-9, 0.02 * span)
            if coincide:
                ax.annotate("k-lines coincide", xy=(0.03, 0.95), xycoords="axes fraction",
                            fontsize=7.5, color=MUTED, va="top")
            if r == 0:
                ax.set_title(meth.replace("_GDR", "") + ("  +GDR" if meth.endswith("_GDR") else ""),
                             fontsize=11, pad=8)
            if c == 0:
                ax.set_ylabel(f"{label}\n{'higher better' if hib else 'lower better'}",
                              fontsize=9.5)
            if r == nr - 1:
                ax.set_xlabel("α  (blend strength)")
            ax.set_xticks(sorted(sud.alpha.unique()))
            ax.margins(x=0.10)

    handles = [Line2D([], [], color=RAMP[i % len(RAMP)], lw=1.8, marker="o", ms=4.5,
                      label=f"SUD  k={k}") for i, k in enumerate(ks)]
    handles += [Line2D([], [], color=ORANGE, lw=1.8, label="q · weight baseline (= the draft)"),
                Line2D([], [], color=AQUA, lw=1.8, label="p · original model, no unlearning")]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)),
               fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"SUD α sweep vs. the weights it blends from — {model} · {split}",
                 fontsize=13, y=0.995)
    missing = [METRICS[k].label for k in SWEEP_METRICS if get(p, k) is None]
    if missing:
        fig.text(0.5, 0.962,
                 f"p has no {' / '.join(missing)} — it needs the forget-set fluency score, "
                 f"which no reference eval runs, so those panels have no p line.",
                 ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.055, 0.985, 0.958))
    return fig


# ── figure 2: trade-off frontier ─────────────────────────────────────────────
def fig_frontier(sud, base, p_refs, model, split, xk="model_utility", yk="privacy_score"):
    methods = sorted(sud.method.unique())
    alphas = sorted(sud.alpha.unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(4.3 * len(methods) + 0.6, 4.3),
                             squeeze=False)
    p = p_refs.get((model, split))

    for c, meth in enumerate(methods):
        ax = axes[0][c]
        d = sud[sud.method == meth].dropna(subset=[xk, yk])
        # one faint path per k, so the sweep reads as a trajectory in α
        for k in sorted(d.k.unique()):
            dk = d[d.k == k].sort_values("alpha")
            ax.plot(dk[xk], dk[yk], "-", color=RAMP[1], lw=1.0, alpha=0.45, zorder=2)
        for i, a in enumerate(alphas):
            da = d[d.alpha == a]
            if da.empty:
                continue
            ax.scatter(da[xk], da[yk], s=52, color=RAMP[i % len(RAMP)],
                       edgecolor=SURF, linewidth=1.3, zorder=4,
                       label=f"α={a}" if c == 0 else None)
        q = q_row(base, model, split, meth)
        ax.margins(x=0.12, y=0.14)          # headroom so anchors clear the frame
        anchors = []
        for row, col, name in ((q, ORANGE, "q"), (p, AQUA, "p")):
            xv, yv = get(row, xk), get(row, yk)
            if xv is None or yv is None:
                continue
            ax.scatter([xv], [yv], s=150, facecolor="none", edgecolor=col,
                       linewidth=2.4, zorder=5)
            anchors.append((xv, yv, name))
        # Label below the marker when it sits high in the panel, else above --
        # an anchor near the top edge otherwise collides with the panel title.
        for xv, yv, name in anchors:
            y0, y1 = ax.get_ylim()
            high = (yv - y0) / (y1 - y0) > 0.75 if y1 > y0 else False
            ax.annotate(name, (xv, yv), xytext=(10, -15 if high else 7),
                        textcoords="offset points", fontsize=11, weight="bold",
                        color=INK, va="top" if high else "baseline")
        ax.set_title(meth.replace("_GDR", "") + ("  +GDR" if meth.endswith("_GDR") else ""),
                     fontsize=11, pad=9)
        ax.set_xlabel(f"{METRICS[xk].label}  (higher better) →")
        if c == 0:
            ax.set_ylabel(f"↑ {METRICS[yk].label}  (higher better)")

    handles = [Line2D([], [], marker="o", ls="", ms=8, color=RAMP[i % len(RAMP)],
                      label=f"SUD  α={a}") for i, a in enumerate(alphas)]
    handles += [Line2D([], [], marker="o", ls="", ms=10, mfc="none", mec=ORANGE, mew=2.2,
                       label="q · weight baseline"),
                Line2D([], [], marker="o", ls="", ms=10, mfc="none", mec=AQUA, mew=2.2,
                       label="p · original model")]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), fontsize=9.5)
    fig.suptitle(f"Trade-off frontier: {METRICS[yk].label} vs {METRICS[xk].label} — {model} · {split}",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    return fig


# ── figure 3: p vs q vs best SUD, per metric ──────────────────────────────────
def fig_bars(sud, base, p_refs, model, split):
    methods = sorted(sud.method.unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(4.6 * len(methods) + 0.6, 4.4),
                             squeeze=False, sharey=True)
    p = p_refs.get((model, split))
    x = np.arange(len(BAR_METRICS))
    w = 0.26

    for c, meth in enumerate(methods):
        ax = axes[0][c]
        d = sud[sud.method == meth]
        q = q_row(base, model, split, meth)
        series = [("p", p, AQUA, -w), ("q", q, ORANGE, 0.0), ("SUD", None, BLUE, w)]
        for name, row, col, off in series:
            vals, missing, tiny = [], [], []
            for i, key in enumerate(BAR_METRICS):
                v = get(best_sud(d, key), key) if name == "SUD" else get(row, key)
                vals.append(np.nan if v is None else v)
                if v is None:
                    missing.append(i)
                elif v < 0.02:
                    tiny.append((i, v))
            ax.bar(x + off, vals, w, color=col, zorder=3,
                   label=name if c == 0 else None)
            for i in missing:       # say "n/a" instead of drawing nothing
                ax.annotate("n/a", (i + off, 0.012), ha="center", va="bottom",
                            fontsize=7.5, color=MUTED, rotation=90)
            # A real 0.00 draws as no bar at all, which reads identically to
            # missing data -- print the value so zero is legible as zero.
            for i, v in tiny:
                ax.annotate(f"{v:.2f}", (i + off, 0.012), ha="center", va="bottom",
                            fontsize=7.5, color=INK2, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([METRICS[k].label + ("  ↑" if METRICS[k].higher_is_better else "  ↓")
                            for k in BAR_METRICS], rotation=38, ha="right", fontsize=8.5)
        ax.set_title(meth.replace("_GDR", "") + ("  +GDR" if meth.endswith("_GDR") else ""),
                     fontsize=11)
        ax.set_ylim(0, 1.06)
        ax.grid(axis="x", visible=False)
        if c == 0:
            ax.set_ylabel("score")

    handles = [Line2D([], [], marker="s", ls="", ms=9, color=AQUA, label="p · original, no unlearning"),
               Line2D([], [], marker="s", ls="", ms=9, color=ORANGE, label="q · weight baseline"),
               Line2D([], [], marker="s", ls="", ms=9, color=BLUE, label="SUD · best α×k per metric")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9.5)
    fig.suptitle(f"p vs. q vs. best SUD — {model} · {split}", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    return fig


# ── figure 4: one-glance overview across every slice ──────────────────────────
def fig_overview(sud, base, p_refs, key="aggregate"):
    rows = []
    for (model, split, meth), d in sud.groupby(["target", "split", "method"]):
        q = q_row(base, model, split, meth)
        b = best_sud(d, key)
        rows.append(dict(
            label=f"{model.replace('Llama-3.2-', '')} · {split} · {meth.replace('_GDR', '')}",
            q=get(q, key), sud=get(b, key), n=len(d),
            alpha=None if b is None else b.alpha, k=None if b is None else b.k))
    rows = [r for r in rows if r["sud"] is not None or r["q"] is not None]

    # Biggest SUD win first. A slice with no q has no delta at all -- ranking it
    # by (sud - 0) would float it to the top as if it were the best result.
    def rank(r):
        if r["q"] is None or r["sud"] is None:
            return (1, 0.0)                      # incomparable -> after the rest
        return (0, -(r["sud"] - r["q"]))
    rows.sort(key=rank)

    fig, ax = plt.subplots(figsize=(10.6, 0.52 * len(rows) + 2.3))
    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        if r["q"] is not None and r["sud"] is not None:
            ax.plot([r["q"], r["sud"]], [i, i], color=AXIS, lw=1.8, zorder=2)
        if r["q"] is not None:
            ax.scatter([r["q"]], [i], s=72, color=ORANGE, edgecolor=SURF, lw=1.3, zorder=4)
        if r["sud"] is not None:
            ax.scatter([r["sud"]], [i], s=72, color=BLUE, edgecolor=SURF, lw=1.3, zorder=4)
        if r["q"] is not None and r["sud"] is not None:
            d = r["sud"] - r["q"]
            ax.annotate(f"{d:+.3f}", xy=(1.0, i), xycoords=("axes fraction", "data"),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=("#0ca30c" if d > 0 else MUTED),
                        annotation_clip=False)
        elif r["q"] is None:
            ax.annotate("no q baseline", xy=(1.0, i), xycoords=("axes fraction", "data"),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8, color=MUTED, annotation_clip=False)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
    ax.invert_yaxis()                     # rank order reads top-to-bottom
    ax.set_xlabel(f"{METRICS[key].label}  (higher better) →")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Every slice: weight baseline (q) → best SUD α×k, by {METRICS[key].label}",
                 fontsize=12.5, pad=10)
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=8, color=ORANGE, label="q · weight baseline"),
        Line2D([], [], marker="o", ls="", ms=8, color=BLUE, label="SUD · best α×k"),
    ], loc="lower center", ncol=2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.055, 0.90, 1))
    return fig


# ── text summary, so numbers are readable without opening a file ─────────────
def print_table(sud, base, p_refs, key="aggregate"):
    print(f"\n{'slice':52s} {'p':>8s} {'q':>8s} {'bestSUD':>8s} {'Δ(SUD-q)':>9s}  best@   runs")
    print("-" * 104)
    for (model, split, meth), d in sud.groupby(["target", "split", "method"]):
        q, b = q_row(base, model, split, meth), best_sud(d, key)
        p = p_refs.get((model, split))
        pv, qv, bv = get(p, key), get(q, key), get(b, key)
        delta = f"{bv - qv:+.4f}" if (bv is not None and qv is not None) else "—"
        at = "—" if b is None else f"α{b.alpha}/k{int(b.k)}"
        f = lambda v: "  n/a  " if v is None else f"{v:8.4f}"
        label = f"{model.replace('Llama-3.2-', '')} · {split} · {meth}"
        print(f"{label:52s} {f(pv)} {f(qv)} {f(bv)} {delta:>9s}  {at:9s} {len(d):3d}")
    print(f"\np n/a on {METRICS[key].label}: the `_full` reference evals lack "
          f"forget_Q_A_PARA_Prob / forget_truth_ratio / forget_Q_A_gibberish,\n"
          f"so the composite dimensions cannot be computed for the original model.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=REPO / "saves/unlearn/sud/results")
    ap.add_argument("--baselines-csv", type=Path, default=DEFAULT_BASELINES)
    ap.add_argument("--eval-root", type=Path, default=REPO / "saves/eval")
    ap.add_argument("--outdir", type=Path, default=REPO / "figures/results")
    ap.add_argument("--only", default=None,
                    help="substring filter on '<model>_<split>' (e.g. forget01, 3B)")
    ap.add_argument("--metric", default="aggregate", help="metric for the overview figure")
    args = ap.parse_args()

    sud, base, p_refs, _retain = load_all(args.results_root, args.baselines_csv, args.eval_root)
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"loaded {len(sud)} SUD runs · {len(base)} baselines · {len(p_refs)} p references")

    slices = sorted({(m, s) for m, s in zip(sud.target, sud.split)})
    made = []
    pdf_path = args.outdir / "all_figures.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = fig_overview(sud, base, p_refs, args.metric)
        for ext in ("png", "pdf"):
            fig.savefig(args.outdir / f"overview.{ext}", dpi=200)
        pdf.savefig(fig); plt.close(fig); made.append("overview")

        for model, split in slices:
            tag = f"{model}_{split}"
            if args.only and args.only not in tag:
                continue
            d = sud[(sud.target == model) & (sud.split == split)]
            for name, builder in (("sweep", fig_sweep), ("frontier", fig_frontier),
                                  ("bars", fig_bars)):
                fig = builder(d, base, p_refs, model, split)
                for ext in ("png", "pdf"):
                    fig.savefig(args.outdir / f"{name}_{tag}.{ext}", dpi=200)
                pdf.savefig(fig); plt.close(fig); made.append(f"{name}_{tag}")

    print_table(sud, base, p_refs, args.metric)
    print(f"\nwrote {len(made)} figures (PNG + PDF) to {args.outdir}")
    print(f"combined: {pdf_path}")


if __name__ == "__main__":
    main()
