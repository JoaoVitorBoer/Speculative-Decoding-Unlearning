"""Interactive metrics comparison dashboard for SUD unlearning experiments.

Run from the project root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH  = ROOT / "saves" / "unlearn" / "sud" / "results" / "summary.csv"
BASELINE_PATH = ROOT / "saves" / "unlearn" / "sud" / "results" / "summary_baselines.csv"

FILTER_COLS     = ["target", "draft", "alpha", "split"]
DEFAULT_METRICS = ["model_utility", "memorization", "agg_memorization", "privacy_score"]

_METRIC_PALETTE = px.colors.qualitative.Plotly
_DRAFT_PREFIXES = ["draft-meta-llama_", "draft-", "meta-llama_", "tofu_"]


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    main = _read_csv(SUMMARY_PATH)
    if BASELINE_PATH.exists():
        return main, _read_csv(BASELINE_PATH), True
    return main, pd.DataFrame(), False


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if col in FILTER_COLS:
            df[col] = df[col].fillna("").astype(str).str.strip()
        else:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
    return df


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in FILTER_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


# ── Label helpers ──────────────────────────────────────────────────────────────
def _short_draft(draft: str) -> str:
    s = draft
    for p in _DRAFT_PREFIXES:
        s = s.replace(p, "")
    return s


def _short_target(target: str) -> str:
    s = target.replace("unlearn_tofu_", "").replace("tofu_", "")
    return s if len(s) <= 55 else s[:52] + "…"


def _baseline_label(row: pd.Series) -> str:
    tgt = row.get("target", "baseline")
    for p in ("unlearn_", "tofu_"):
        tgt = tgt.replace(p, "", 1)
    label = tgt[:55].rstrip("_- ")
    split = row.get("split", "")
    if split:
        label += f"  [{split}]"
    return label


# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SUD Metrics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] { min-width: 420px !important; max-width: 520px !important; }
        [data-testid="stSidebar"] [data-baseweb="tag"] {
            max-width: 390px !important; white-space: normal !important;
            height: auto !important; padding: 4px 8px !important;
        }
        [data-testid="stSidebar"] [role="option"] { white-space: normal !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Theme detection ────────────────────────────────────────────────────────────
try:
    IS_DARK = st.get_option("theme.base") == "dark"
except Exception:
    IS_DARK = False

PLOT_BG     = "rgba(0,0,0,0)"
PAPER_BG    = "rgba(0,0,0,0)"
GRID_COLOR  = "rgba(255,255,255,0.10)" if IS_DARK else "rgba(0,0,0,0.07)"
AXIS_COLOR  = "rgba(255,255,255,0.25)" if IS_DARK else "rgba(0,0,0,0.20)"
FONT_COLOR  = "#e8e8e8"                if IS_DARK else "#1a1a1a"
LEGEND_BG   = "rgba(30,30,30,0.80)"   if IS_DARK else "rgba(255,255,255,0.92)"
LEGEND_BORD = "rgba(255,255,255,0.20)" if IS_DARK else "#cccccc"
ANNOT_BG    = "rgba(20,20,20,0.88)"   if IS_DARK else "rgba(255,255,255,0.90)"


# ── Load data ─────────────────────────────────────────────────────────────────
st.title("SUD — Speculative Unlearning Decoding · Metrics Dashboard")

main_df, base_df, has_baselines = load_data()

if not has_baselines:
    st.warning("`summary_baselines.csv` not found — showing main results only.")

all_metrics = _numeric_cols(main_df)

# Stable metric → colour (indexed over all_metrics so colours never shift)
metric_color: dict[str, str] = {
    m: _METRIC_PALETTE[i % len(_METRIC_PALETTE)]
    for i, m in enumerate(all_metrics)
}

# ── Initialise baseline state (always defined before sidebar) ─────────────────
sel_baselines: pd.DataFrame = pd.DataFrame()
baseline_line_visible: dict[str, bool] = {}   # metric → show dashed line?

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    targets = sorted(t for t in main_df["target"].unique() if t)
    sel_target = st.selectbox("Target", targets, format_func=_short_target)
    st.caption(f"**Full name:** `{sel_target}`")

    tgt_df = main_df[main_df["target"] == sel_target]

    drafts = sorted(d for d in tgt_df["draft"].unique() if d)
    sel_drafts = st.multiselect(
        "Draft", drafts,
        default=drafts[:1] if drafts else [],
        format_func=_short_draft,
    )
    if sel_drafts:
        with st.expander("Full draft names"):
            for d in sel_drafts:
                st.caption(f"`{d}`")

    d_df = tgt_df[tgt_df["draft"].isin(sel_drafts)] if sel_drafts else tgt_df
    alphas = sorted(a for a in d_df["alpha"].unique() if a)
    sel_alphas = st.multiselect("Alpha (chart columns)", alphas, default=alphas)

    a_df = d_df[d_df["alpha"].isin(sel_alphas)] if sel_alphas else d_df
    splits = sorted(s for s in a_df["split"].unique() if s)
    sel_splits = st.multiselect("Split", splits, default=splits)

    st.divider()
    st.header("Metrics")
    default_visible = [m for m in DEFAULT_METRICS if m in all_metrics] or all_metrics[:4]
    sel_metrics = st.multiselect("Visible metrics", all_metrics, default=default_visible)
    hidden = [m for m in all_metrics if m not in sel_metrics]
    if hidden:
        st.caption(f"Hidden: {', '.join(hidden)}")

    # ── Baseline controls ──────────────────────────────────────────────────────
    if has_baselines and not base_df.empty:
        st.divider()
        st.header("Baselines")

        base_df_copy = base_df.copy()
        base_df_copy["_label"] = base_df_copy.apply(_baseline_label, axis=1)
        baseline_opts = base_df_copy["_label"].tolist()

        sel_bl_labels = st.multiselect(
            "Show baselines",
            baseline_opts,
            default=baseline_opts,
            help="Select which baseline runs to overlay on the chart.",
        )
        sel_baselines = base_df_copy[base_df_copy["_label"].isin(sel_bl_labels)].copy()

        if sel_bl_labels:
            with st.expander("Full baseline target names"):
                for _, brow in sel_baselines.iterrows():
                    st.caption(f"`{brow['target']}`")

        if not sel_baselines.empty and sel_metrics:
            st.markdown("**Show baseline line per metric:**")
            for m in sel_metrics:
                baseline_line_visible[m] = st.checkbox(
                    m,
                    value=True,
                    key=f"bl_vis_{m}",
                )


# ── Filtered data ──────────────────────────────────────────────────────────────
def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["target"] == sel_target
    if sel_drafts:
        mask &= df["draft"].isin(sel_drafts)
    if sel_alphas:
        mask &= df["alpha"].isin(sel_alphas)
    if sel_splits:
        mask &= df["split"].isin(sel_splits)
    return df[mask].copy()


filtered = _apply_filters(main_df)

_chart_drafts = sel_drafts if sel_drafts else sorted(d for d in tgt_df["draft"].unique() if d)
_chart_splits = sel_splits if sel_splits else splits or ["(all)"]
_chart_alphas = sorted(
    sel_alphas or alphas,
    key=lambda a: float(a) if a.replace(".", "").isdigit() else a,
)

# X-axis items: (label, draft_val, split_val)
if len(_chart_drafts) <= 1:
    _x_items = [(s, _chart_drafts[0] if _chart_drafts else "", s) for s in _chart_splits]
else:
    _x_items = [
        (f"{_short_draft(d)} | {s}", d, s)
        for d in _chart_drafts
        for s in _chart_splits
    ]
_x_labels = [xi[0] for xi in _x_items]


# ── Metric summary cards ───────────────────────────────────────────────────────
if sel_metrics and not filtered.empty:
    st.subheader("Summary")
    n_cards = min(len(sel_metrics), 4)
    card_cols = st.columns(n_cards)
    ref_row = sel_baselines.iloc[0] if not sel_baselines.empty else None
    for i, m in enumerate(sel_metrics[:4]):
        if m not in filtered.columns:
            continue
        avg = filtered[m].mean()
        delta_str = None
        if ref_row is not None and m in ref_row.index:
            bval = pd.to_numeric(ref_row[m], errors="coerce")
            if pd.notna(bval) and pd.notna(avg):
                delta_str = f"{avg - bval:+.4f} vs baseline"
        with card_cols[i % n_cards]:
            st.metric(
                label=m,
                value=f"{avg:.4f}" if pd.notna(avg) else "N/A",
                delta=delta_str,
            )


# ── Main chart ─────────────────────────────────────────────────────────────────
if not sel_metrics:
    st.info("Select at least one metric from the sidebar.")
elif filtered.empty:
    st.warning("No rows match the current filters — adjust the sidebar.")
else:
    st.subheader("Metric Comparison")
    st.caption(
        "Each column = one **α** value.  "
        "Bar colour = metric.  "
        "Dashed lines = baseline values (toggle per metric in the sidebar)."
    )

    n_panels = max(len(_chart_alphas), 1)
    fig = make_subplots(
        rows=1,
        cols=n_panels,
        subplot_titles=[f"<b>α = {a}</b>" for a in _chart_alphas],
        shared_yaxes=True,
        horizontal_spacing=0.05,
    )

    ymin, ymax = float("inf"), float("-inf")

    for col_idx, alpha_str in enumerate(_chart_alphas, start=1):
        panel = filtered[filtered["alpha"] == alpha_str]

        # ── Metric bars ────────────────────────────────────────────────────────
        for m_idx, metric in enumerate(sel_metrics):
            if metric not in panel.columns:
                continue

            xs, ys = [], []
            for xlab, draft_val, split_val in _x_items:
                mask = panel["split"] == split_val
                if len(_chart_drafts) > 1:
                    mask &= panel["draft"] == draft_val
                rows = panel[mask]
                val = (
                    float(rows[metric].iloc[0])
                    if not rows.empty and pd.notna(rows[metric].iloc[0])
                    else None
                )
                xs.append(xlab)
                ys.append(val)
                if val is not None:
                    ymin = min(ymin, val)
                    ymax = max(ymax, val)

            color = metric_color.get(metric, "#888")
            fig.add_trace(
                go.Bar(
                    x=xs,
                    y=ys,
                    name=metric,
                    legendgroup=f"metric|{metric}",
                    legendgrouptitle_text="Metrics" if (col_idx == 1 and m_idx == 0) else None,
                    showlegend=(col_idx == 1),
                    marker=dict(color=color, opacity=0.85, line=dict(width=0)),
                    text=[f"{y:.2f}" if y is not None else "" for y in ys],
                    textposition="outside",
                    textfont=dict(size=9, color=FONT_COLOR),
                    cliponaxis=False,
                    constraintext="none",
                    hovertemplate=(
                        f"<b>{metric}</b><br>%{{x}}<br>%{{y:.4f}}<extra></extra>"
                    ),
                ),
                row=1, col=col_idx,
            )

        # ── Baseline horizontal lines via add_hline (full-width, no annotation) ──
        for _, brow in (sel_baselines.iterrows() if not sel_baselines.empty else []):
            for metric in sel_metrics:
                if not baseline_line_visible.get(metric, True):
                    continue
                if metric not in brow.index:
                    continue
                bval = brow[metric]
                if not pd.notna(bval):
                    continue
                bval = float(bval)
                ymin = min(ymin, bval)
                ymax = max(ymax, bval)

                color = metric_color.get(metric, "#888")
                fig.add_hline(
                    y=bval,
                    line_dash="dash",
                    line_color=color,
                    line_width=2,
                    row=1,
                    col=col_idx,
                )

    # ── Layout ─────────────────────────────────────────────────────────────────
    if ymin < float("inf"):
        pad    = max((ymax - ymin) * 0.10, 0.02)
        yrange = [max(0.0, ymin - pad), ymax + pad * 6]
    else:
        yrange = None

    fig.update_layout(
        height=520,
        barmode="group",
        bargap=0.18,
        bargroupgap=0.06,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR),
        margin=dict(t=60, b=80, l=70, r=200),
        legend=dict(
            groupclick="togglegroup",
            orientation="v",
            xanchor="left",
            x=1.02,
            yanchor="top",
            y=1.0,
            font=dict(size=11, color=FONT_COLOR),
            bgcolor=LEGEND_BG,
            bordercolor=LEGEND_BORD,
            borderwidth=1,
            tracegroupgap=10,
        ),
        hovermode="x unified",
    )

    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        showline=True,
        linecolor=AXIS_COLOR,
        zeroline=False,
        tickfont=dict(size=11, color=FONT_COLOR),
        tickangle=-30,
        automargin=True,
        tickmode="array",
        tickvals=_x_labels,
        ticktext=[f"<b>{xl}</b>" for xl in _x_labels],
    )

    fig.update_yaxes(
        range=yrange,
        gridcolor=GRID_COLOR,
        showline=True,
        linecolor=AXIS_COLOR,
        zeroline=False,
        tickfont=dict(size=11, color=FONT_COLOR),
    )

    for ann in fig.layout.annotations:
        ann.font.color = FONT_COLOR

    st.plotly_chart(fig, use_container_width=True)

    # ── Baseline reference values (replaces chart annotations) ─────────────────
    if not sel_baselines.empty and sel_metrics and any(baseline_line_visible.values()):
        parts: list[str] = []
        for m in sel_metrics:
            if not baseline_line_visible.get(m, True):
                continue
            vals: list[str] = []
            for _, brow in sel_baselines.iterrows():
                if m in brow.index and pd.notna(brow[m]):
                    vals.append(f"{float(brow[m]):.4f}")
            if vals:
                c = metric_color.get(m, "#888")
                parts.append(
                    f'<span style="color:{c}"><b>{m}</b>: {" / ".join(vals)}</span>'
                )
        if parts:
            st.markdown(
                "**Baseline →** " + " &nbsp;&nbsp;·&nbsp;&nbsp; ".join(parts),
                unsafe_allow_html=True,
            )


# ── Data tables ────────────────────────────────────────────────────────────────
st.subheader("Data Table")

_disp_filter = [c for c in FILTER_COLS if not filtered.empty and c in filtered.columns]
display_cols = _disp_filter + sel_metrics

tab_sud, tab_base, tab_combined = st.tabs(["SUD Results", "Baselines", "Combined"])

with tab_sud:
    if filtered.empty:
        st.info("No rows match the current filters.")
    else:
        show = (
            filtered[[c for c in display_cols if c in filtered.columns]]
            .sort_values(["split", "draft", "alpha"])
            .reset_index(drop=True)
        )
        st.dataframe(show, use_container_width=True)

with tab_base:
    if sel_baselines.empty:
        msg = "`summary_baselines.csv` not found." if not has_baselines else "No baselines selected."
        st.info(msg)
    else:
        show_b = (
            sel_baselines[[c for c in display_cols if c in sel_baselines.columns]]
            .reset_index(drop=True)
        )
        st.dataframe(show_b, use_container_width=True)

with tab_combined:
    frames: list[pd.DataFrame] = []
    if not filtered.empty:
        f = (
            filtered[[c for c in display_cols if c in filtered.columns]]
            .sort_values(["split", "draft", "alpha"])
            .copy()
        )
        f.insert(0, "source", "SUD")
        frames.append(f)
    if not sel_baselines.empty:
        b = sel_baselines[[c for c in display_cols if c in sel_baselines.columns]].copy()
        b.insert(0, "source", "BASELINE")
        frames.append(b)
    if not frames:
        st.info("No data to display.")
    else:
        combined = pd.concat(frames, ignore_index=True)
        st.dataframe(combined, use_container_width=True)
        if not sel_baselines.empty:
            st.caption("BASELINE rows appear at the bottom.")
