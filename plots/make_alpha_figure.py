#!/usr/bin/env python
"""alpha-tradeoff figure for SUD (1B / forget10 / NPO draft): speed AND unlearning vs alpha.

One figure, four panels on a shared linear alpha axis (0 = target p only, 1 = draft q only):
  (a) acceptance rate a          accepted / verified, pooled over the k = 1 and k = 4 benchmark
                                 cells of each alpha (a is a per-token quantity, k-invariant)
  (b) speed-up of k = 4 over k = 1  measured tokens/s ratio (paired bootstrap 95% CI over prompts),
                                 dotted theory 2 / F(4) with F from the round-length histogram
                                 (see make_figure.theory_fwd_per_tok): speed is a function of a alone
  (c) model utility              SUD at k = 1, seed 0 (sampled at T = 1) — the main grid where it has
                                 the alpha, plots/results (pass c of run_k_sweep.sh) otherwise
  (d) privleak                   same runs; 0 = calibrated; band = alphas whose forget_quality >= 0.01
Grey hollow markers at alpha = 0 / 1: the standard GREEDY evals of the target / draft (reference,
~0.015 model_utility above a sampled decode); grey dashed: the retain model (TOFU gold reference).

Reading down one alpha gives the whole tradeoff: how much unlearning, what utility it costs, how
much faster it decodes. Weight-based methods have no such axis — their inference cost is fixed.

Inputs   plots/bench/alpha_bench.json                        bench_speculative.py --alphas ... --ks 1 4
         saves/unlearn/sud/results/summary.csv               main grid (alphas 0.5 0.7 0.8 0.9 0.95)
         plots/results/**/TOFU_SUMMARY.json                  pass c evals (alphas the grid lacks)
         saves/eval/... and saves/unlearn/baselines/...      greedy reference evals (target / draft / retain)
Outputs  plots/figs/fig_alpha_tradeoff.{pdf,png}, plots/fig_alpha_data.csv, plots/README_alpha.md

Run in the `unlearning` env:  conda run -n unlearning python plots/make_alpha_figure.py
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import make_figure as kf   # styling (rcParams applied on import), loaders, theory helpers

REPO, PLOTS, FIGS = kf.REPO, kf.PLOTS, kf.FIGS
DEFAULT_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0]
K_BASE, K_FAST = 1, 4
FQ_PASS = 0.01                       # TOFU: forget_quality (KS-test p-value) >= 0.01 counts as unlearned
CELL = dict(target="Llama-3.2-1B-Instruct", draft="Llama-3.2-1B-Instruct_NPO", split="forget10")  # draft set per --method
MAIN_SUMMARY = REPO / "saves/unlearn/sud/results/summary.csv"
REFS = {   # greedy standard evals of the fixed models of this cell (draft path set per --method)
    "target": dict(label="target $p$ (greedy)", alpha=0.0, marker="s",
                   path=REPO / "saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget10/TOFU_SUMMARY.json"),
    "draft": dict(label="draft $q$ (greedy)", alpha=1.0, marker="^",
                  path=REPO / "saves/unlearn/baselines/tofu/NPO/Llama-3.2-1B-Instruct/forget10/evals/TOFU_SUMMARY.json"),
    "retain": dict(label="retain model", alpha=None,
                   path=REPO / "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_SUMMARY.json"),
}
METRICS = ["model_utility", "forget_quality", "privleak"]
SUD = kf.SERIES_STYLES[0]["color"]            # one entity (SUD on this cell) -> one colour everywhere
REF_INK = "#4d4d4d"


# ── loading ─────────────────────────────────────────────────────────────────
def load_bench(path: Path):
    d = kf.read_json(path) or {}
    recs = {(float(r["alpha"]), int(r["k"])): r for r in d.get("records", []) if r.get("tokens_per_sec")}
    return d.get("meta", {}), recs


def load_metrics(alphas, results_root: Path):
    """{alpha: {metric..., source}} for SUD at k = 1, seed 0: main grid first, then plots/results."""
    out = {}
    if MAIN_SUMMARY.exists():
        with MAIN_SUMMARY.open() as fh:
            for r in csv.DictReader(fh):
                if (r["target"], r["draft"], r["split"]) != (CELL["target"], CELL["draft"], CELL["split"]):
                    continue
                if int(r["k"]) != K_BASE or int(r["seed"]) != 0 or r.get("complete", "True") != "True":
                    continue
                a = float(r["alpha"])
                out[a] = {m: float(r[m]) for m in METRICS if r.get(m) not in (None, "")}
                out[a]["source"] = "main grid"
    for run in kf.load_runs(results_root):
        if run["k"] != K_BASE or run["seed"] != 0 or not run["complete"] or run["alpha"] in out:
            continue
        summ = kf.read_json(Path(run["dir"]) / "TOFU_SUMMARY.json") or {}
        if all(summ.get(m) is not None for m in METRICS):
            out[run["alpha"]] = {m: float(summ[m]) for m in METRICS}
            out[run["alpha"]]["source"] = "plots/results (pass c)"
    return {a: out[a] for a in sorted(out) if a in set(alphas)}


def load_refs():
    out = {}
    for name, spec in REFS.items():
        summ = kf.read_json(spec["path"])
        if summ:
            out[name] = dict(spec, **{m: summ.get(m) for m in METRICS})
    return out


# ── statistics ──────────────────────────────────────────────────────────────
def paired_speedup_ci(c_fast, c_base, n_boot=4000, seed=0):
    """Bootstrap the tokens/s ratio over the SAME prompt resample for both cells."""
    b = {p["idx"]: p for p in c_base.get("per_prompt", [])}
    f_ = {p["idx"]: p for p in c_fast.get("per_prompt", [])}
    idx = sorted(set(b) & set(f_))
    if len(idx) < 2:
        return None
    tb = np.array([b[i]["new_tokens"] for i in idx], float); sb = np.array([b[i]["seconds"] for i in idx], float)
    tf = np.array([f_[i]["new_tokens"] for i in idx], float); sf = np.array([f_[i]["seconds"] for i in idx], float)
    rng = np.random.default_rng(seed)
    ii = rng.integers(0, len(idx), size=(n_boot, len(idx)))
    r = (tf[ii].sum(1) / sf[ii].sum(1)) / (tb[ii].sum(1) / sb[ii].sum(1))
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def alpha_stats(alphas, recs, metrics, refs):
    rows = []
    for a in alphas:
        st = dict(alpha=a)
        cells = {k: recs[(a, k)] for k in (K_BASE, K_FAST) if (a, k) in recs}
        acc_n = sum(c["accepted"] for c in cells.values()); ver_n = sum(c["verified"] for c in cells.values())
        if ver_n:
            st["acceptance"] = acc_n / ver_n
            st["a_se"] = math.sqrt(max(st["acceptance"] * (1 - st["acceptance"]), 1e-12) / ver_n)
            st["acceptance_by_k"] = {k: c["acceptance"] for k, c in cells.items()}
        for k, c in cells.items():
            st[f"tps_k{k}"] = c["tokens_per_sec"]
            st[f"fwd_per_tok_k{k}"] = (c["proposed"] + c["rounds"]) / c["committed"]
            st[f"kbar_k{k}"] = c["proposed"] / c["rounds"]
        if len(cells) == 2:
            st["speedup"] = cells[K_FAST]["tokens_per_sec"] / cells[K_BASE]["tokens_per_sec"]
            st["speedup_ci"] = paired_speedup_ci(cells[K_FAST], cells[K_BASE])
            st["algo_speedup"] = st[f"fwd_per_tok_k{K_BASE}"] / st[f"fwd_per_tok_k{K_FAST}"]   # exact, timing-free
            fpt, src = kf.theory_fwd_per_tok(cells[K_FAST], st["acceptance"])
            st["theory_speedup"], st["theory_src"] = 2.0 / fpt, src                  # k = 1 costs exactly 2 forwards/token
        if a in metrics:
            st.update({m: metrics[a].get(m) for m in METRICS}, metrics_source=metrics[a]["source"])
            if st.get("forget_quality") is not None:
                st["fq_pass"] = st["forget_quality"] >= FQ_PASS
        rows.append(st)
    return rows


def theory_speedup_curve(rows, grid):
    """Theory speed-up as a smooth function of alpha: a(alpha) interpolated between the
    measured acceptances, F(4) from the k = 4 histogram of the nearest measured alpha."""
    pts = [(r["alpha"], r["acceptance"]) for r in rows if "acceptance" in r]
    hist_rows = [r for r in rows if "theory_speedup" in r]
    if len(pts) < 2 or not hist_rows:
        return None
    xs, ys = zip(*pts)
    out = []
    for g in grid:
        if g < min(xs) - 1e-9 or g > max(xs) + 1e-9:   # never extrapolate beyond the measured alphas
            out.append(np.nan)
            continue
        a = float(np.interp(g, xs, ys))
        near = min(hist_rows, key=lambda r: abs(r["alpha"] - g))
        out.append(2.0 / kf.theory_fwd_per_tok(_cell_of(near), a)[0])
    return np.array(out)


_CELLS = {}


def _cell_of(row):
    return _CELLS[(row["alpha"], K_FAST)]


# ── figure ──────────────────────────────────────────────────────────────────
def alpha_axis(ax, label=True):
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if label:
        ax.set_xlabel(r"Blend strength $\alpha$")
    ax.tick_params(axis="both", which="major", pad=2)


def sud_kw(lw=1.1, ms=3.6):
    return dict(color=SUD, ls="-", lw=lw, marker="o", ms=ms, mec="white", mew=0.6,
                solid_capstyle="round", solid_joinstyle="round")


def ref_marker(ax, ref, metric, zorder=4):
    v = ref.get(metric)
    if v is None or ref.get("alpha") is None:
        return None
    ax.plot([ref["alpha"]], [v], ls="none", marker=ref["marker"], ms=5.0, mfc="white", mec=REF_INK, mew=0.9, zorder=zorder)
    return v


def fq_band(ax, rows):
    """Shade the alphas whose forget_quality passes (contiguous run only)."""
    ok = sorted(r["alpha"] for r in rows if r.get("fq_pass"))
    if not ok:
        return None
    alphas_with_fq = sorted(r["alpha"] for r in rows if r.get("fq_pass") is not None)
    lo_i = alphas_with_fq.index(ok[0])
    if alphas_with_fq[lo_i:] != ok:      # not a contiguous tail: draw nothing rather than mislead
        return None
    # start the band halfway to the last failing alpha (the threshold lies in between)
    lo = ok[0] if lo_i == 0 else 0.5 * (ok[0] + alphas_with_fq[lo_i - 1])
    ax.axvspan(lo, 1.04, color="#000000", alpha=0.045, lw=0, zorder=0.5)
    return lo


def draw(rows, refs, out_stem: Path, pdf_dir=None):
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.4), sharex=True,
                             gridspec_kw=dict(hspace=0.42, wspace=0.3))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.865, bottom=0.11)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # (a) acceptance
    pts = [(r["alpha"], r["acceptance"], r["a_se"]) for r in rows if "acceptance" in r]
    if pts:
        x, y, se = map(np.array, zip(*pts))
        ax_a.errorbar(x, y, yerr=1.96 * se, fmt="none", ecolor=SUD, elinewidth=0.7, capsize=1.4, zorder=2.5)
        ax_a.plot(x, y, zorder=3, **sud_kw())
        lo = min(y.min(), 0.5)
        ax_a.set_ylim(lo - 0.03, 1.02)
    else:
        ax_a.text(0.5, 0.5, "benchmark not run yet", ha="center", va="center", transform=ax_a.transAxes, color=kf.MUTED)
    kf.panel_title(ax_a, "(a)", "Acceptance rate")
    ax_a.set_ylabel(r"Acceptance $a$")

    # (b) speed-up k = 4 over k = 1
    sp = [r for r in rows if "speedup" in r]
    if sp:
        grid = np.linspace(0, 1, 201)
        th = theory_speedup_curve(rows, grid)
        if th is not None:
            ax_b.plot(grid, th, color=SUD, zorder=2, **kf.THEORY_STYLE)
        x = np.array([r["alpha"] for r in sp]); y = np.array([r["speedup"] for r in sp])
        cis = [r["speedup_ci"] for r in sp]
        if all(cis):
            ax_b.errorbar(x, y, yerr=[y - np.array([c[0] for c in cis]), np.array([c[1] for c in cis]) - y],
                          fmt="none", ecolor=SUD, elinewidth=0.7, capsize=1.4, zorder=2.5)
        ax_b.plot(x, y, zorder=3, **sud_kw())
        ax_b.axhline(1.0, color=kf.AXIS, lw=0.6, ls=(0, (1.0, 1.8)), zorder=1)
        ax_b.text(0.02, 1.0, "no gain", fontsize=6, color=kf.INK2, ha="left", va="bottom")
        ymax = max(y.max(), (np.nanmax(th) if th is not None else 0), 1.0)
        ax_b.set_ylim(min(0.9, y.min() - 0.05), ymax * 1.08)
        ax_b.legend(handles=[Line2D([], [], color=kf.INK2, label="theory: $2/F(4)$ from $a$", **kf.THEORY_STYLE)],
                    loc="upper left", fontsize=6.5, handlelength=2.4, borderaxespad=0.3, handletextpad=0.5,
                    bbox_to_anchor=(0.0, 0.93))
    else:
        ax_b.text(0.5, 0.5, "benchmark not run yet", ha="center", va="center", transform=ax_b.transAxes, color=kf.MUTED)
    kf.panel_title(ax_b, "(b)", f"Speed-up, $k={K_FAST}$ over $k={K_BASE}$")
    ax_b.set_ylabel("Tokens/s ratio")

    # (c) model utility, (d) privleak
    for ax, metric, tag, title, ylabel in [(ax_c, "model_utility", "(c)", "Model utility", "Model utility"),
                                           (ax_d, "privleak", "(d)", "Privacy leakage", "privleak")]:
        band_lo = fq_band(ax, rows)
        mp = [(r["alpha"], r[metric]) for r in rows if r.get(metric) is not None]
        ys = []
        if mp:
            x, y = map(np.array, zip(*mp))
            ax.plot(x, y, zorder=3, **sud_kw())
            ys += list(y)
        for name in ("target", "draft"):
            if name in refs:
                v = ref_marker(ax, refs[name], metric)
                if v is not None:
                    ys.append(v)
        if metric == "model_utility" and refs.get("retain", {}).get(metric) is not None:
            v = refs["retain"][metric]
            ax.axhline(v, color=REF_INK, lw=0.8, ls=(0, (6.0, 2.0)), zorder=1.5)
            ys.append(v)
        if metric == "privleak":
            ax.axhline(0.0, color=kf.INK2, lw=0.6, ls=(0, (1.0, 1.8)), zorder=1)
            ax.text(0.02, 0.0, "calibrated (0)", fontsize=6, color=kf.INK2, ha="left", va="bottom")
            ys.append(0.0)
        if ys:
            lo, hi = min(ys), max(ys)
            pad = max(0.1 * (hi - lo), 0.004)
            ax.set_ylim(lo - pad, hi + pad * 1.6)
        if band_lo is not None and metric == "model_utility":   # label the band once; (d) shares it
            ax.text(min(band_lo + 0.01, 0.9), ax.get_ylim()[1], f"forget quality ≥ {FQ_PASS:g}",
                    fontsize=6, color=kf.INK2, ha="left", va="top")
        kf.panel_title(ax, tag, title)
        ax.set_ylabel(ylabel)
        alpha_axis(ax, label=True)
    alpha_axis(ax_a, label=False); alpha_axis(ax_b, label=False)

    handles = [Line2D([], [], label=f"SUD, $k={K_BASE}$, sampled (seed 0)", **sud_kw())]
    for name in ("target", "draft"):
        if name in refs:
            handles.append(Line2D([], [], ls="none", marker=refs[name]["marker"], ms=5.0, mfc="white", mec=REF_INK, mew=0.9,
                                  label=refs[name]["label"]))
    if "retain" in refs:
        handles.append(Line2D([], [], color=REF_INK, lw=0.8, ls=(0, (6.0, 2.0)), label="retain model (greedy)"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 0.995),
               handlelength=2.4, columnspacing=1.2, handletextpad=0.5)
    kf.save(fig, out_stem, pdf_dir)


# ── CSV + README ─────────────────────────────────────────────────────────────
def fmt(x, nd=4):
    return kf.f(x, nd)


def write_csv(path: Path, rows):
    cols = ["alpha", "acceptance", "a_se", f"tps_k{K_BASE}", f"tps_k{K_FAST}", "speedup", "speedup_ci_lo", "speedup_ci_hi",
            "algo_speedup", "theory_speedup", "theory_src", f"kbar_k{K_FAST}", "model_utility", "forget_quality", "privleak",
            "fq_pass", "metrics_source"]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in rows:
            ci = r.get("speedup_ci") or (None, None)
            w.writerow([r.get(c, "") if c not in ("speedup_ci_lo", "speedup_ci_hi") else (ci[0] if c.endswith("lo") else ci[1])
                        for c in cols])


def write_readme(path: Path, rows, refs, meta, args, alphas, P, fig_stem: Path):
    n_bench = sum(1 for r in rows if "speedup" in r); n_acc = sum(1 for r in rows if "acceptance" in r)
    n_met = sum(1 for r in rows if r.get("model_utility") is not None)
    best = max((r for r in rows if "speedup" in r), key=lambda r: r["speedup"], default=None)
    method = P["method"]
    png_rel = kf.rel(fig_stem.with_suffix(".png"), path.parent)
    rebuild = (f"conda run -n unlearning python plots/make_alpha_figure.py --method {method}"
               + (f" --out-dir {kf.rel(args.out_dir, REPO)}" if Path(args.out_dir) != FIGS else "")
               + (f" --pdf-dir {kf.rel(args.pdf_dir, REPO)}" if Path(args.pdf_dir) != Path(args.out_dir) else "")
               + (f" --doc-dir {kf.rel(args.doc_dir, REPO)}" if Path(args.doc_dir) != PLOTS else ""))
    L = [f"# alpha-tradeoff figure for SUD ({method} draft) — speed and unlearning vs the blend strength α",
         f"\n_Generated {datetime.now():%Y-%m-%dT%H:%M} by `plots/make_alpha_figure.py --method {method}`._\n",
         f"![alpha tradeoff]({png_rel})\n",
         "## Suggested caption\n",
         "**Figure. Stronger unlearning is faster unlearning.** TOFU forget10, target p = Llama-3.2-1B-Instruct fine-tuned on TOFU, "
         f"draft q = its {method}-unlearned checkpoint. (a) Per-token acceptance a = P(accept) of draft-verify sampling from "
         "π ∝ p^(1−α) q^α, pooled over the k = 1 and k = 4 benchmark cells (100 forget prompts, batch 1, no KV cache; bars: 95% binomial). "
         f"(b) Measured throughput of k = {K_FAST} relative to k = {K_BASE} (paired bootstrap 95% CI over prompts); dotted: 2/F(4) with F the "
         "predicted forwards per committed token from a and the round-length histogram, so speed is a function of a alone. "
         "(c) Model utility and (d) privacy leakage of the same α at k = 1 (seed 0, sampled at T = 1); the grey band marks α with "
         f"forget quality ≥ {FQ_PASS:g}. Hollow grey markers: the standard greedy evals of the target (α = 0) and draft (α = 1); "
         "dashed: the retain model. Pushing π toward q raises a, so the α that calibrates leakage also decodes fastest"
         + (f" (best {best['speedup']:.2f}× at α = {best['alpha']:g})." if best else "."),
         "\n## Status\n",
         f"- Benchmark cells: {n_bench}/{len(alphas)} α with both k = {K_BASE} and k = {K_FAST} ({n_acc} with acceptance) from `{args.bench.relative_to(REPO) if args.bench.is_relative_to(REPO) else args.bench}`"
         + (f" (GPU: {meta.get('gpu')}, n = {meta.get('n_prompts')} prompts, max_new_tokens = {meta.get('max_new_tokens')})" if meta else
            f" — not recorded for this draft yet; run `plots/bench_speculative.py --draft {P['draft']} --alphas {' '.join(f'{a:g}' for a in alphas)} "
            f"--ks {K_BASE} {K_FAST} --results-root {kf.rel(P['results_root'], REPO)} --out {kf.rel(args.bench, REPO)} --resume`"),
         f"- Unlearning metrics: {n_met}/{len(alphas)} α at k = {K_BASE}, seed 0 — sources: "
         + ", ".join(f"α={r['alpha']:g}: {r['metrics_source']}" for r in rows if r.get("metrics_source")),
         "- Greedy reference evals: " + ", ".join(f"{n} ({fmt(v.get('model_utility'))} utility, {fmt(v.get('privleak'), 1)} privleak)"
                                                    for n, v in refs.items()),
         "\n## Per-α table\n"]
    hdr = ["α", "a", f"tok/s k={K_BASE}", f"tok/s k={K_FAST}", "speed-up [95% CI]", "algo. speed-up", "theory", "utility", "forget quality", "privleak", "metrics from"]
    body = []
    for r in rows:
        ci = r.get("speedup_ci")
        body.append([f"{r['alpha']:g}", fmt(r.get("acceptance"), 3), fmt(r.get(f"tps_k{K_BASE}"), 1), fmt(r.get(f"tps_k{K_FAST}"), 1),
                     (f"{r['speedup']:.2f}× [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else fmt(r.get("speedup"), 2)),
                     (f"{r['algo_speedup']:.2f}×" if "algo_speedup" in r else "—"),
                     (f"{r['theory_speedup']:.2f}× ({r['theory_src']})" if "theory_speedup" in r else "—"),
                     fmt(r.get("model_utility")), (f"{r['forget_quality']:.3g}" if r.get("forget_quality") is not None else "—"),
                     fmt(r.get("privleak"), 2), r.get("metrics_source", "—")])
    L.append(kf.md_table(hdr, body))
    L += ["\n## Notes\n",
          "- Acceptance is a per-token quantity and does not depend on k; the k = 1 and k = 4 cells are pooled (their separate values are in the CSV).",
          "- \"algo. speed-up\" is the exact, timing-free ratio of forwards per committed token (k = 1 costs 2 by construction); "
          "\"theory\" predicts it from a alone plus the k = 4 round-length histogram (`rounds` = exact histogram, `prompts` = per-prompt means for older cells).",
          "- The SUD metrics are sampled at T = 1 while the grey references are greedy; sampling alone costs ≈ 0.015 model_utility on this grid, "
          "so compare SUD's α = 0 / 1 points (which also sample) with the hollow markers to read the handicap directly.",
          f"- forget_quality is the TOFU KS-test p-value (≥ {FQ_PASS:g} = indistinguishable from the retain model); privleak 0 = calibrated to the retain model, "
          "negative = under-unlearned (leaks), positive = overshoot.",
          f"\nRebuild: `{rebuild}`"]
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default=kf.DEFAULT_METHOD,
                    help="weight-unlearning method of the draft q; sets the cell, the default bench / results root "
                         "and the draft's greedy reference eval, and suffixes the output names")
    ap.add_argument("--bench", type=Path, default=None, help="default: plots/bench/alpha_bench[_<method>].json")
    ap.add_argument("--results-root", type=Path, default=None, help="default: this method's own sub-tree of plots/results")
    ap.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    ap.add_argument("--out-dir", type=Path, default=FIGS, help="where the PNG goes (default: plots/figs)")
    ap.add_argument("--pdf-dir", type=Path, default=None, help="where the PDF goes (default: --out-dir)")
    ap.add_argument("--doc-dir", type=Path, default=PLOTS, help="where the README and CSV go")
    args = ap.parse_args()
    alphas = sorted(args.alphas)

    P = kf.method_paths(args.method)
    args.bench = args.bench or P["alpha_bench"]
    args.results_root = args.results_root or P["results_root"]
    args.pdf_dir = args.pdf_dir or args.out_dir
    P.update(results_root=args.results_root)
    CELL["draft"] = P["draft_tag"]
    REFS["draft"]["path"] = REPO / P["draft"] / "evals" / "TOFU_SUMMARY.json"
    sfx = P["suffix"]

    if kf.bench_for_method(args.bench, args.method) is None and args.bench.exists():
        print(f"note: {args.bench.name} was recorded with another draft — ignored for method {args.method}")
        meta, recs = {}, {}
    else:
        meta, recs = load_bench(args.bench)
    _CELLS.update(recs)
    metrics = load_metrics(alphas, args.results_root)
    refs = load_refs()
    rows = alpha_stats(alphas, recs, metrics, refs)

    for d in (args.out_dir, args.pdf_dir, args.doc_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
    fig_stem = args.out_dir / f"fig_alpha_tradeoff{sfx}"
    draw(rows, refs, fig_stem, args.pdf_dir)
    write_csv(args.doc_dir / f"fig_alpha_data{sfx}.csv", rows)
    write_readme(args.doc_dir / f"README_alpha{sfx}.md", rows, refs, meta, args, alphas, P, fig_stem)
    for r in rows:
        print(f"α={r['alpha']:<5g} a={fmt(r.get('acceptance'), 3):>6} speedup={fmt(r.get('speedup'), 2):>5} "
              f"theory={fmt(r.get('theory_speedup'), 2):>5} mu={fmt(r.get('model_utility')):>7} fq={fmt(r.get('forget_quality'), 3):>7} "
              f"pl={fmt(r.get('privleak'), 1):>6} [{r.get('metrics_source', '—')}]")
    print(f"wrote {fig_stem.name}.png -> {args.out_dir}, .pdf -> {args.pdf_dir}; fig_alpha_data{sfx}.csv + README_alpha{sfx}.md -> {args.doc_dir}")


if __name__ == "__main__":
    main()
