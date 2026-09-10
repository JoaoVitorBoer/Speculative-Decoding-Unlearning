"""SUD results against the weight baselines they decode from.

One experiment: an ORIGINAL, memorised TOFU model p blended at decode time with
an UNLEARNED draft q at strength α,

    log π = (1 − α) · log p + α · log q

and every run here is that `original-unlearned` combination. The question is only
ever the same one: how does the blend score against the weight-unlearned
checkpoint it decodes from? That checkpoint IS the run's draft — same model, same
split, same method — so it is the only comparison drawn. The original model is
not shown.

Three filters (model size, forget split, method), one comparison chart, one
α breakdown, one table. Nothing else.

Run it from the project root:

    conda run -n unlearning streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import sud_results_data as D  # noqa: E402

st.set_page_config(page_title="SUD vs. baselines", page_icon="📊", layout="wide")

# Two colours, the same two everywhere: blue is the blend, orange the weight
# baseline it is measured against.
SUD, BASE = "#2a78d6", "#eb6834"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"


@st.cache_data(show_spinner="Reading results…")
def load(stamp: tuple[float, float]) -> D.Dataset:
    """Cached on `stamp`, which must NOT be underscore-prefixed — Streamlit
    excludes underscore-prefixed arguments from a cache key, which would pin the
    page to whatever the CSVs said on first load."""
    return D.load_dataset()


def stamp() -> tuple[float, float]:
    def mtime(p: Path) -> float:
        return p.stat().st_mtime if p.exists() else 0.0
    return (mtime(D.DEFAULT_RESULTS / "summary.csv"), mtime(D.DEFAULT_BASELINES))


ds = load(stamp())

# ── filters ──────────────────────────────────────────────────────────────────
st.title("SUD vs. the weight baselines")
st.caption(
    "Blue = SUD at some blend strength α. Orange = the weight-unlearned "
    "checkpoint it decodes from, evaluated on its own. α = 1 would be that "
    "checkpoint exactly, so the interesting runs are the ones below it."
)

f1, f2, f3 = st.columns(3)
sizes = sorted(ds.runs.target.unique(), key=len)
target = f1.selectbox("Model size", sizes, format_func=D.pretty_model)
avail = ds.slices[ds.slices.target == target]
split = f2.selectbox("Forget split", sorted(avail.split.unique()))
avail = avail[avail.split == split]
method = f3.selectbox("Method", sorted(avail.method.unique()),
                      format_func=D.pretty_method)

sweep = ds.sweep(target, split, method)
if sweep.empty:
    st.warning("No runs for that combination.")
    st.stop()
base = ds.baseline(target, split, method)

(st.warning if ds.freshness.stale else st.caption)(ds.freshness.message)
if base is None:
    st.warning(
        "This draft's weight baseline is not on disk, so there is nothing to "
        "compare against — the SUD numbers are shown alone.")


# ── chart helpers ────────────────────────────────────────────────────────────

def style(fig: go.Figure, height: int) -> go.Figure:
    """Plotly's default template ships its own palette, so it is switched off and
    every colour set here."""
    fig.update_layout(
        template="none", height=height, bargap=0.35,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12),
        margin=dict(l=8, r=24, t=52, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)"),
    )
    # automargin grows the OUTER margin to fit tick labels. Safe here because
    # these are single plots — on a subplot grid it shrinks each panel's domain
    # instead and strands the panels.
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7", automargin=True,
                     tickangle=-25)
    fig.update_yaxes(gridcolor="#e1e0d9", zeroline=False, linecolor="#c3c2b7",
                     automargin=True)
    return fig


def bar(name: str, x, y, colour: str, texts=None) -> go.Bar:
    return go.Bar(name=name, x=x, y=y, marker_color=colour,
                  marker_cornerradius=4, text=texts, textposition="outside",
                  textfont=dict(size=10), cliponaxis=False,
                  hovertemplate="%{x}<br><b>%{y:.4f}</b><extra>" + name + "</extra>")


# ── 1. every metric: p / q / best SUD ────────────────────────────────────────
st.subheader("Metrics against the baseline")
best_by = st.selectbox(
    "Pick the best SUD run by", D.METRIC_KEYS,
    index=D.METRIC_KEYS.index("aggregate"),
    format_func=lambda k: D.METRICS[k].label,
    help="One α is shown per bar group. This chooses which one.")
best = D.best_row(sweep, best_by)

labels = [D.METRICS[k].label + (" ↑" if D.METRICS[k].higher_is_better else " ↓")
          for k in D.BAR_METRICS]
fig = go.Figure([
    bar("weight baseline", labels,
        [D.get(base, k) for k in D.BAR_METRICS], BASE),
    bar(f"SUD · α={best.alpha:g}" if best is not None else "SUD", labels,
        [D.get(best, k) for k in D.BAR_METRICS], SUD),
])
st.plotly_chart(style(fig, 420))
st.caption(
    f"Best SUD run by **{D.METRICS[best_by].label}** is "
    f"**α = {best.alpha:g}**. `↑` / `↓` marks which direction is better, so a "
    f"taller blue bar is a win on the `↑` metrics and a loss on the `↓` ones. "
    f"A missing bar means no value for that metric."
)


# ── 2. one metric across α ───────────────────────────────────────────────────
st.subheader("How the metric moves with α")
metric = st.selectbox(
    "Metric", D.METRIC_KEYS, index=D.METRIC_KEYS.index("aggregate"),
    format_func=lambda k: f"{D.METRICS[k].label}  ·  {D.METRICS[k].group}")
M = D.METRICS[metric]

d = sweep.dropna(subset=[metric]) if metric in sweep.columns else sweep.iloc[:0]
fig2 = go.Figure([bar("SUD", [f"α={a:g}" for a in d.alpha], list(d[metric]), SUD,
                      texts=[f"{v:.3g}" for v in d[metric]])])
q_value = D.get(base, metric)
if q_value is not None:
    fig2.add_hline(y=q_value, line=dict(color=BASE, width=2),
                   annotation_text="baseline", annotation_position="right",
                   annotation_font=dict(size=11))
    fig2.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                              name="weight baseline",
                              line=dict(color=BASE, width=2), hoverinfo="skip"))
fig2.update_yaxes(title_text=f"{M.label} — {M.note}",
                  type="log" if M.log_scale else "linear")
st.plotly_chart(style(fig2, 400))
st.caption(M.help + ".")


# ── 3. the numbers ───────────────────────────────────────────────────────────
st.subheader("Every value")
rows = [{"row": "weight baseline", **{D.METRICS[k].label: D.get(base, k)
                                      for k in D.METRIC_KEYS}}]
for _, r in sweep.iterrows():
    rows.append({"row": f"SUD α={r.alpha:g}",
                 **{D.METRICS[k].label: D.get(r, k) for k in D.METRIC_KEYS}})
table = pd.DataFrame(rows)

# One delta column, for the metric the reader is already looking at. Signed so
# positive always means SUD beat the baseline, whichever way the metric points --
# and a ratio in decades for forget quality, which spans 200 of them.
table[f"Δ {M.label} vs baseline"] = [
    D.improvement(metric, D.get(r, metric), q_value) if r["row"].startswith("SUD")
    else None
    for r in rows]

st.dataframe(
    table, hide_index=True,
    column_config={c: st.column_config.NumberColumn(c, format="%.4f")
                   for c in table.columns if table[c].dtype.kind == "f"})
st.download_button(
    "Download as CSV", table.to_csv(index=False),
    file_name=f"sud_{D.pretty_model(target)}_{split}_{method}.csv",
    mime="text/csv")
if M.log_scale:
    st.caption(f"Δ for {M.label} is a ratio in **decades** (log₁₀ of SUD / q), "
               f"not a difference: the metric spans ~200 orders of magnitude.")

for note in ds.notes:
    st.caption(note)
