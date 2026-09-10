#!/usr/bin/env python
"""normalizer_truncation.py — full-vocabulary SUD normalizer Z vs top-k truncated Zhat.

Scope: this script measures *one* thing. SUD's ideal unlearned next-token
distribution is the geometric interpolation

    pi(x) = (1/Z) * p(x)^(1-alpha) * q(x)^alpha,
    Z     = sum_{x in V} p(x)^(1-alpha) * q(x)^alpha,

whose normalizer Z ranges over the whole vocabulary V. A practical decoder
cannot afford that sum at every step, so it truncates to a set S and uses

    Zhat = sum_{x in S} p(x)^(1-alpha) * q(x)^alpha   (<= Z, since every term > 0).

S is the set Algorithm 1 forms: the UNION of the top-k tokens under the target p
and the top-k tokens under the draft q, so |S| lands in [k, 2k]. That choice is
the measurement -- truncating to top-k(p) alone leaves q's head outside S and
reports an error about the missing set rather than about truncation. TOPK_BASIS
can switch to that variant, or to top-k of the blend, for comparison.

This script computes both, exactly, at real TOFU forget-set decoding positions,
and reports how the relative normalizer error

    (Z - Zhat) / Z    equivalently    1 - Zhat/Z

shrinks as k grows.

That quantity is not a proxy for anything — it is *identically* the total
variation distance between the ideal sampling distribution pi and the truncated
one pi_hat:

    TV(pi, pi_hat) = (Z - Zhat) / Z          (an exact identity, not a bound)

because 2*TV splits into the mass pi puts outside S, which is (Z-Zhat)/Z, plus
the renormalization error inside S, which is
sum_{x in S} pi_hat(x)*(Z-Zhat)/Z = (Z-Zhat)/Z as well. So the reported number is
the exact error in the distribution the sampler draws from — the very quantity
Theorem 4.4 bounds — and `verify_tv_identity()` re-checks it numerically on every
run. Nothing else: no bound evaluation, no downstream unlearning metrics, no
timing, no free generation. The decoder itself is never run — every number comes
from teacher-forced forward passes, so the script is fully deterministic.

Method
------
For every evaluated position the script needs log Zhat for many k at once. It
gets them with one sort plus two `logcumsumexp` passes over the vocabulary axis:

  * order tokens by ascending min(rank_p, rank_q), the k at which each token
    enters the union S_k = top-k(p) u top-k(q). The family is nested in k, so
    every S_k is a prefix of that one order and the scans below still see all
    k at once; `union_order_and_counts` also returns |S_k|, the column each k
    must be read at (`TOPK_BASIS` can switch to top-k(p) or to top-k of the
    blend, whose S_k is simply the first k entries),
  * a forward `logcumsumexp` over the reordered log-scores u = (1-a)log p +
    a log q gives log Zhat_k at index k-1, for every k simultaneously,
  * a reverse `logcumsumexp` gives log(Z - Zhat_k) at index k directly.

Taking the tail from the reverse pass rather than subtracting two nearly equal
normalizers is what keeps the small-error regime meaningful: `(Z - Zhat)/Z` is
recovered as `exp(log_tail - log_Z)` with no catastrophic cancellation, so
float32 resolves errors far below 1e-7. Everything above the raw bf16 logits is
promoted to float32, matching `src/model/sud.py`.

Every k, not a grid
-------------------
The reverse pass above yields log(Z - Zhat_k) for EVERY k in one sweep, so the
curve is measured at every integer k in [1, |V|] rather than at a handful of
octave samples. That costs one extra full-width exp per alpha and no extra scan.
What it cannot do is keep the per-position array at full width — that would be
n_positions x n_alphas x |V| floats, 53 GB in float32 for forget10 alone — so
the position-mean and position-max are reduced on the GPU as chunks stream past
(`FullCurveAccumulator`), leaving an (n_alphas, |V|) pair per run.

Quantiles cannot be streamed that way, so median/p90/p99 stay on the sparse
K_VALUES grid, where the per-position array is small enough to hold. The two
paths share the scan but not the reduction, and `check_full_curve_matches_grid`
compares them at the shared k on every run.

Reduction
---------
Every evaluated position yields one TV, and the reported number is the mean over
positions: concatenate a split's answers into one vector of decoding positions
(the same token string at two positions counts twice — different context, so its
own p, q, Z and its own TV), sum the TVs, divide by the number of positions.
That per-position mean is `tv_mean`, and it is what the figure plots; the median,
p90, p99 and max of the same vector are in the CSV beside it.

Usage
-----
No CLI flags. Every setting is a constant in the CONFIG block below.

    /home/joaoabitante/miniconda3/envs/unlearning/bin/python scripts/normalizer_truncation.py

or, on the cluster, `sbatch scripts/run_normalizer_truncation.sh`.

To redraw the figure from a finished run without touching a GPU, set
`REPLOT_FROM_CSV = True` (or call `replot_from_csv()`); it reads the full-curve
`.npz` and `run_config.json` from `OUTPUT_DIR`, and the aggregate CSV too when
present. Iterating on the drawing costs nothing, so do it there rather than
re-running the measurement.

The drafts are the local GA+GDR baseline checkpoints under `saves/unlearn/`; only
the shared target is fetched from the HF hub (and is normally already cached).
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

# ============================================================================
# CONFIG — every setting lives here. No argparse, no environment overrides.
# ============================================================================

# Target / draft checkpoints per TOFU forget split. 1B only.
#   target = the fully fine-tuned ("full") TOFU checkpoint; it has seen both the
#            retain and the forget split, so it is the memorizing model p.
#   draft  = the weight-unlearned checkpoint for that same split, i.e. the draft
#            this project actually decodes against.
#
# On the naming: GA+GDR is gradient ascent on the forget set plus a gradient-
# descent (NLL) retain term, which is exactly GradDiff with retain_loss_type=NLL
# (see configs/trainer/GradDiff.yaml and the GDR retain mode in
# scripts/run_tofu_weight_baselines.sh). Pure GA (configs/trainer/GradAscent.yaml,
# no retain term) is a different method and is commented out of the baseline sweep.
#
# All three runs below carry `trainer.method_args.retain_loss_type=NLL` in their
# .hydra/overrides.yaml, so all three are the GDR variant; the `_GDR` suffix is
# kept as this script's method TAG (it is what the figure colours and labels key
# off) while DRAFT_RUNS maps each tag to the directory the baseline scripts
# actually write. Those directories are now named by bare method name (one per
# method, retunes overwrite in place), so the mapping is just the suffix strip —
# it stays explicit so a re-point after a retune has one place to happen.
# Every DRAFT comes from here and nowhere else; `main` enforces it rather than
# trusting the MODELS comprehension below.
BASELINES_ROOT: str = "saves/unlearn/baselines/tofu"
DRAFT_FAMILY: str = "Llama-3.2-1B-Instruct"

# The TARGET cannot come from BASELINES_ROOT, and that is structural rather than
# an oversight: that tree holds only UNLEARNED checkpoints, while p is by
# definition the model that still memorizes the forget set. Pointing p at a
# baseline would make both sides of p^(1-a) q^a unlearned and the measured
# quantity would no longer be SUD's blend. So p stays the hub's full TOFU
# checkpoint -- the same weights every baseline in BASELINES_ROOT was unlearned
# FROM, which is what makes them comparable.
#
# It resolves from the local HF cache and needs no network. To pin it to a
# fixed local snapshot instead, set the path printed by
#   huggingface-cli download open-unlearning/tofu_Llama-3.2-1B-Instruct_full
TARGET_MODEL: str = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

# Draft methods to sweep, each crossed with every split in SPLITS: every
# unlearned checkpoint that exists under BASELINES_ROOT, in alphabetical order
# (which is the order the legend and the tables read in). GradDiff_GDR is GA+GDR;
# NPO/SimNPO replace the unbounded ascent term with a saturating one, so they
# are the drafts that should not collapse. AltPO has a training script but no
# checkpoint on disk, so it is absent rather than excluded.
#
# On the _GDR suffix: six of the seven carry `retain_loss_type=NLL` in their
# .hydra/overrides.yaml, i.e. they are the GDR variant, and the suffix records
# that. RMU is the exception -- its retain term is EMBED_DIFF, a different
# mechanism -- so it is tagged bare, and `_method_label` prints it as-is.
DRAFT_METHODS: Tuple[str, ...] = (
    "GradDiff_GDR",
    "IdkDPO_GDR",
    "IdkNLL_GDR",
    "NPO_GDR",
    "RMU",
    "SimNPO_GDR",
    "UNDIAL_GDR",
)

# Method tag -> the tuned run directory under BASELINES_ROOT. Re-point these
# after a retune; nothing else in the file needs to know the hyperparameters.
DRAFT_RUNS: Dict[str, str] = {
    "GradDiff_GDR": "GradDiff",
    "IdkDPO_GDR": "IdkDPO",
    "IdkNLL_GDR": "IdkNLL",
    "NPO_GDR": "NPO",
    "RMU": "RMU",
    "SimNPO_GDR": "SimNPO",
    "UNDIAL_GDR": "UNDIAL",
}

# (method, split) -> {target, draft}. Edit DRAFT_RUNS to point at other runs.
MODELS: Dict[Tuple[str, str], Dict[str, str]] = {
    (method, split): {
        "target": TARGET_MODEL,
        "draft": f"{BASELINES_ROOT}/{DRAFT_RUNS[method]}/{DRAFT_FAMILY}/{split}",
    }
    for method in DRAFT_METHODS
    for split in ("forget01", "forget05", "forget10")
}

# Which splits to run, in order. Keys must exist in MODELS.
SPLITS: List[str] = ["forget01", "forget05", "forget10"]

# Tilt strengths to sweep. alpha=0 -> pi == p and alpha=1 -> pi == q; both give
# Z == 1 exactly and are kept as sanity anchors.
# alpha=0 and alpha=1 used to be carried as sanity anchors (Z == 1 exactly at
# both, and they are eps(k) and eta(k) of Theorem 4.4). The reported comparison
# is the single operating point, so they are dropped; the checks that used them
# are all guarded by `if alpha in ALPHAS` and simply go quiet. Put them back if
# a run should re-establish those anchors -- it costs one extra scan each.
ALPHAS: List[float] = [0.5]

# --- the two k grids ------------------------------------------------------
#
# The figure and the threshold table are drawn from the FULL curve: every
# integer k in [1, |V|]. That is not an approximation of the sweep below, it is
# the exact thing -- see FULL_K_CURVE.
#
# K_VALUES is the separate, sparse grid on which the per-position array is kept
# so that QUANTILES (median / p90 / p99) can be computed. Quantiles need every
# position's value at a given k held at once, which at full width would be
# n_positions x n_alphas x |V| floats -- 53 GB in float32 for forget10 alone --
# so they stay on this grid, where the same array is 66 MB. Powers of two from
# 2^0 to 2^15; must satisfy 1 <= k <= vocab_size.
K_VALUES: List[int] = [2**e for e in range(16)]  # 1 ... 32768

# Append k = vocab_size to K_VALUES, where Zhat == Z by construction (a
# zero-error check). It is the exact-normalizer anchor for the sanity checks,
# never a plotted point (its error is identically 0, which a log axis cannot
# show).
INCLUDE_FULL_VOCAB_K: bool = True

# Accumulate the position-mean and position-max of TV at EVERY k in [1, |V|].
#
# This is close to free. The sort and the two logcumsumexp passes in
# `truncation_errors` already produce log(Z - Zhat_k) for every k at once; the
# K_VALUES loop just throws all but a handful of those columns away. Turning
# this on costs one extra full-width exp per alpha per chunk and no extra scan.
# What it must NOT do is keep the per-position array (see above), so the
# reduction runs on the GPU as chunks stream past and only an (n_alphas, |V|)
# sum and max survive -- 7.2 MB, whatever the split's position count.
#
# Mean and max are exactly the summaries a streaming reduction can produce.
# Quantiles cannot be streamed and stay on K_VALUES.
FULL_K_CURVE: bool = True

# Which rule defines the truncation set S.
#   "union"  -> top-k(p) UNION top-k(q), which is the set Algorithm 1 forms and
#               therefore the only one whose numbers describe the paper's
#               sampler. |S_k| lands in [k, 2k]; see union_order_and_counts.
#   "target" -> top-k under the target p alone. A valid S for Theorem 4.4
#               (it contains top-k(p)) but NOT the algorithm's: it leaves q's
#               head outside S, so eta = q(S^c) is far larger and a draft whose
#               argmax p ranks deep looks catastrophic for reasons that are
#               about the set, not about truncation.
#   "blend"  -> top-k under the blended score p^(1-a) q^a (alpha-dependent).
#               Does NOT contain top-k(p), so Theorem 4.4 does not cover it;
#               the TV identity still holds. Diagnostic only.
TOPK_BASIS: str = "union"

# Number of forget-set QA pairs per split; None uses the WHOLE split
# (forget01=40, forget05=200, forget10=400 QA pairs). Set to an int to subsample
# with a seeded RNG.
N_PROMPTS_PER_SPLIT: Optional[int] = None

# Cap on evaluated positions per prompt (the first N answer positions); None
# uses every answer position. Positions are the ground-truth answer tokens plus
# the closing <|eot_id|>, i.e. exactly the positions SUD would decode.
MAX_POSITIONS_PER_PROMPT: Optional[int] = None

# Prompt construction — mirrors configs/model/sud.yaml `template_args` and
# src/data/utils.py so the evaluated contexts match the repo's TOFU eval.
APPLY_CHAT_TEMPLATE: bool = True
SYSTEM_PROMPT: str = "You are a helpful assistant."
DATE_STRING: str = "10 Apr 2025"
QUESTION_KEY: str = "question"
ANSWER_KEY: str = "answer"
MAX_LENGTH: int = 512  # matches configs/data/datasets/TOFU_QA_forget.yaml

# Execution.
DEVICE: str = "cuda"  # "cuda" or "cpu"
TORCH_DTYPE: str = "bfloat16"  # dtype the two LMs are loaded in
# Attention kernels tried in order; the first one that loads wins.
ATTN_IMPLEMENTATIONS: Tuple[str, ...] = ("sdpa", "eager")
BATCH_SIZE: int = 8  # prompts per forward pass
POSITION_CHUNK: int = 64  # positions per sort / logcumsumexp block
SEED: int = 0

# Reporting.
OUTPUT_DIR: str = "saves/analysis/normalizer_truncation"
# Long-format CSV of the raw per-position errors; off by default because on the
# full splits it is ~2.5M rows. The aggregate CSV (K_VALUES grid, with
# quantiles) is always written, and so is the full-curve .npz, which is what the
# figure and the threshold table are drawn from.
WRITE_PER_POSITION_CSV: bool = False
MAKE_PLOT: bool = True  # figures; skipped if matplotlib is missing
PLOT_FORMATS: Tuple[str, ...] = ("png", "pdf")
# For each (split, alpha), report the smallest k whose relative normalizer error
# falls at or below each of these. With FULL_K_CURVE on, that k is EXACT -- the
# true crossing, not the first sampled grid point above it.
REL_ERROR_THRESHOLDS: Tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6)

# Re-draw the figure from an existing run instead of measuring. Lets the plot be
# iterated on without a GPU; `replot_from_csv()` is also callable directly.
# Everything else in this block still applies.
REPLOT_FROM_CSV: bool = False

# Draft-health probe. A truncation result is only interpretable if the draft is a
# sane next-token distribution: a gradient-ascent draft that has collapsed puts
# its mass on tokens the target rules out. Under TOPK_BASIS="target" that moves
# the whole blend outside S and drives the measured error to ~1 for reasons that
# have nothing to do with truncation; the default union basis pulls q's head
# back into S, so a collapsed draft no longer masquerades as a truncation
# failure -- but it is still a broken draft, which is what this reports.
# This is q's mass inside p's top-this-many.
DIAG_TOPK_PROBE: int = 1024

# --- figure ----------------------------------------------------------------
# There is one figure: mean per-token TV against k, one panel per split and one
# line per draft method, at a single alpha. Everything else the sweep measures
# lives in the printed tables and the aggregate CSV.
#
# x runs over EVERY k in [1, |V|), so each curve is 128,255 points rather than a
# handful of octave samples. That changes three things about how it is drawn:
# ticks become decades instead of one-per-sample, markers move to fixed decade
# anchors instead of a sample stride, and the panel gets enough width that five
# decades of k do not collide.

# Which drafts the MAIN figure draws. The full DRAFT_METHODS sweep is still
# measured and written to the CSV -- the others stay out of this figure because
# it is the one the paper carries at \textwidth, where two curves per panel is
# what leaves the shape of the decay legible. GradDiff_GDR in particular is
# collapsed on forget10 at 1B, so its curve there is a picture of that collapse
# rather than of truncation, and RMU is collapsed on forget05/forget10 the other
# way (it flattens toward uniform rather than spiking). The three drawn here are
# healthy on all three splits.
PLOT_METHODS: Tuple[str, ...] = ("NPO_GDR", "SimNPO_GDR", "UNDIAL_GDR")

# A SECOND figure, same axes, drawing every measured draft: the survey version,
# for the appendix or for reviewing the sweep. It is deliberately a separate
# file rather than a replacement -- seven curves over three shared panels is a
# different claim ("no draft escapes the truncation cost") from the two-curve
# figure's ("these two drafts differ, and here is by how much"), and at seven
# the hue channel is saturated, so identity leans on dash and marker.
ALL_METHOD_FIGURE: bool = True
ALL_METHOD_FIGURE_STEM: str = "normalizer_truncation_all_methods"
# Methods for that figure; empty tuple means "everything in DRAFT_METHODS".
ALL_METHOD_FIGURE_METHODS: Tuple[str, ...] = ()
# Legend columns there. Seven entries in one row is wider than the figure, so
# they wrap to two rows of four, which costs ~0.2in of height above the panels.
ALL_METHOD_LEGEND_NCOL: int = 4

# The alpha the figure is drawn at. The rest of ALPHAS is still measured; 0 and
# 1 in particular are the eps(k) and eta(k) of Theorem 4.4 and the anchors the
# sanity checks lean on (Z == 1 exactly at both).
COMPARISON_ALPHA: float = 0.5

# Seaborn's "colorblind" palette, in its own order, never cycled. These are the
# literal values of `seaborn.color_palette("colorblind")` (a Wong-derived set),
# hardcoded rather than imported so that `replot_from_csv` keeps working on a
# machine with only matplotlib.
#
# Hue is NOT the only identity channel here -- see DASH_PATTERNS and
# MARKER_SPECS. That redundancy is what carries the figure for a reader with
# CVD or a greyscale printout, and it is also what lets the palette be used as
# published: slot 1 (orange) is 2.61:1 on white, below the 3:1 line a
# hue-only encoding would need.
#   contrast on #ffffff: slot 0 5.13:1, slot 1 2.61:1, slot 2 3.42:1,
#   slot 3 3.87:1, slot 4 2.98:1, slot 5 3.13:1, slot 6 3.02:1.
#
# Slots 4-6 exist for the all-methods survey figure. They are the honest limit
# of the channel: seven categorical hues cannot all be separable, the added
# three sit near 3:1, and slot 6 is a neutral grey that is only not chrome
# because it is 1.6pt wide and marked. At seven series the dash and the marker
# stop being redundancy and become the primary identity, which is why both are
# extended in step below. Yellow (#ECE133) and light pink (#FBAFE4) from the
# same palette are skipped: at ~1.3:1 on white they are not lines, they are
# suggestions.
CATEGORICAL: Tuple[str, ...] = (
    "#0173B2",
    "#DE8F05",
    "#029E73",
    "#D55E00",
    "#CC78BC",
    "#CA9161",
    "#949494",
)

# Dash pattern per categorical slot, parallel to CATEGORICAL. This is the THIRD
# identity channel and the one that survives everything: it is legible in
# greyscale, under any CVD, and at the line widths a two-column figure forces.
# Slot 0 keeps a solid stroke because solid-vs-broken is the single most
# separable pair; the rest take patterns chosen to differ in period as well as
# in duty cycle, so they stay apart when the curve is steep and the dashes
# foreshorten.
DASH_PATTERNS: Tuple[Tuple[int, Tuple[float, ...]], ...] = (
    (0, ()),                              # solid
    (0, (5.0, 1.8)),                      # dashed
    (0, (1.3, 1.5)),                      # dotted
    (0, (6.0, 1.6, 1.2, 1.6)),            # dash-dot
    (0, (9.0, 2.2)),                      # long dash
    (0, (5.0, 1.5, 1.2, 1.5, 1.2, 1.5)),  # dash-dot-dot
    (0, (2.6, 1.4)),                      # short dash
)

# Categorical slot per draft method, bound to the METHOD rather than to its row
# in PLOT_METHODS: colour has to follow the entity, so dropping a method from the
# figure must not repaint the ones that stay, and a reader who learned "NPO is
# blue" here is not contradicted by the next figure. Slots 0-1 (blue solid,
# orange dashed) go to the two drafts the main figure draws, so those two keep
# their identity between the two figures; GradDiff_GDR keeps slot 2 (green
# dotted) as before, and the four drafts that only ever appear in the survey
# figure take the weaker slots 3-6.
# Slots 0-2 are the three separable ones -- blue solid, orange dashed, green
# dotted -- so they go to the three drafts the paper figure draws, in the order
# it draws them. UNDIAL holds slot 2 rather than GradDiff because it is in that
# figure and GradDiff is not: tan (slot 5) beside orange is the palette's
# weakest pair, and spending it on a series that only ever appears in the
# seven-curve survey, where dash and marker already carry identity, costs least.
#
# Slot 6 is the grey one, so it goes to a draft that sits inside the tight
# cluster of near-identical curves (IdkDPO), where a recessive line costs
# nothing to read. RMU takes a real hue instead: it is the collapsed outlier at
# the top of two panels, i.e. the one series the eye is meant to land on, and
# grey there would read as chrome rather than as data.
METHOD_SLOTS: Dict[str, int] = {
    "NPO_GDR": 0,
    "SimNPO_GDR": 1,
    "UNDIAL_GDR": 2,
    "IdkNLL_GDR": 3,
    "RMU": 4,
    "GradDiff_GDR": 5,
    "IdkDPO_GDR": 6,
}

# Marker (shape, size in points) per categorical slot, parallel to CATEGORICAL.
# Shape is the SECOND identity channel beside hue: the palette's blue and orange
# convert to similar mid-greys, so a reviewer printing in greyscale — or reading
# with full-severity CVD — separates the series by mark and by dash. Sizes are
# per-shape because a square reads larger than a circle at equal `ms`.
MARKER_SPECS: Tuple[Tuple[str, float], ...] = (
    ("o", 3.6),
    ("s", 3.2),
    ("^", 3.8),
    ("D", 3.2),
    ("v", 3.8),
    ("P", 3.9),
    ("X", 3.6),
)

# Where the identity markers go, now that the curve is continuous. A marker per
# sample would be 128k marks; a fixed stride would put them at arbitrary k that
# differ between panels. Decade anchors instead: every series carries a mark at
# the same handful of round k, so the shape channel survives and the marks
# double as a reading aid ("what does k=1000 cost?"). Anchors above |V| are
# dropped, and the curve's last point always takes one so each series is marked
# where it leaves the panel.
# Decade spacing, matching the x ticks: each mark sits on a labelled gridline,
# so it doubles as a reading aid ("what does k = 1000 cost?") instead of landing
# at an arbitrary k. Six marks per curve is enough for the shape channel to
# register as identity, and few enough that at 2.2in per panel the marks stay
# punctuation rather than becoming the series -- which at half-decade spacing,
# with two near-coincident curves, they did.
MARKER_ANCHORS: Tuple[int, ...] = (1, 10, 100, 1000, 10_000, 100_000)

# The same idea for a LINEAR k axis, where the decade anchors above would stack
# every mark inside the first 0.8% of the panel and read as one blob on the
# y axis. Quarters of the vocabulary are the linear equivalent of decades: round,
# evenly spaced, and the same stops the axis is ticked at.
MARKER_ANCHORS_LINEAR: Tuple[int, ...] = (1, 32_000, 64_000, 96_000, 128_000)

# Vertices emitted per curve. Purely a rendering budget: the full 128,255-point
# path is ~50x the panel's pixel columns, so it costs file size and draw time
# and buys nothing. `_decimate_log` thins log-uniformly, which keeps every k
# below ~1e3 and only merges k that already share a pixel. Reported numbers
# never come from the decimated array.
PLOT_MAX_POINTS: int = 6000

# Chart chrome: recessive hairlines, text in ink tokens (never a series color).
INK_PRIMARY: str = "#0b0b0b"
INK_SECONDARY: str = "#52514e"
INK_MUTED: str = "#898781"
GRIDLINE: str = "#e1e0d9"
AXIS_LINE: str = "#c3c2b7"

# Figure size in inches, at PUBLICATION scale. The figure is now drawn at the
# size it will print at, so every point size below is the size a reviewer
# actually reads -- no scale-down step to reason about. 7.0in is the ACL/EMNLP
# full text width, i.e. a `figure*` spanning both columns.
#
# The panels are STACKED (three rows, one column), so each one gets the whole
# 7.0in for its 5.1 decades of k instead of a third of it. That is the trade the
# layout makes: side by side, the three splits were directly comparable by eye
# but each panel was 2.2in wide and 1.4in tall, which is a strip -- the shallow
# power law that is the figure's actual finding reads as a flat line at that
# aspect. Stacked, the comparison costs a saccade instead of being free, and in
# exchange each curve is drawn at an aspect where its slope is legible. 6.6in of
# height gives three ~1.9in panels plus the legend, titles and shared x label.
FIG_WIDTH_IN: float = 7.0
FIG_HEIGHT_IN: float = 6.6

# Type. BASE_FONT_PT is the figure's body size and everything else is a ratio of
# it, so the whole figure re-scales from one number without losing its
# hierarchy. At 8.5 the rendered sizes are panel titles 9.6, axis labels 8.5,
# tick labels 8.0, legend 8.5, annotations 7.7 -- the range an ACL figure at
# \textwidth wants. Nothing in the figure is bold: at these sizes weight reads
# as emphasis, and no single element here is the one being emphasised.
BASE_FONT_PT: float = 8.5
TITLE_FONT_SCALE: float = 1.13
TICK_FONT_SCALE: float = 0.94
ANNOTATION_FONT_SCALE: float = 0.90

# Decades of TV shown, measured down from the largest plotted value.
#
# The full range is ~13.5 decades, and roughly eight of them are the terminal
# collapse: once k passes ~5e4 the vocabulary is running out, the remaining tail
# mass vanishes superexponentially, and TV falls from 1e-5 to 1e-15 inside the
# last third of a decade of k. Drawn in full that is a vertical line occupying
# most of the panel height, and it flattens the 4.5 decades of k that anyone
# actually chooses between into a strip at the top.
#
# So the axis floors after this many decades. The clip costs almost no x-extent
# -- at 7 decades the curves cross the floor past k = 1.2e5, i.e. beyond 99.8%
# of the log-x span -- so they still visibly plunge into the bottom-right
# corner, which is what they do. Where they end up is one short note in panel
# (a) and the rest of the story is the caption's.
# 5 decades, not 7. The curves live between ~2e-1 and ~1e-4, which holds the
# power-law regime that dominates every k a decoder would actually use, and the
# start of the exponential one above it. At 7 decades the small-TV end took 60%
# of each panel and squeezed everything else into the top third.
#
# What is NOT true, and was written here first: that the plunge at the right
# edge is only the vocabulary running out and so is safe to clip. It is a real
# second regime -- past k~8K the decay is exponential in k, ~0.024 decades per
# 1000 tokens, R2 > 0.99 (see the semilog figure, which is the view that shows
# it as the long straight run it is rather than a cliff). Clipping at 5 decades
# keeps it visible to k~105K of 128K; curves exit at 98.3%..99.2% of the x span,
# which is what the run print checks. Raise this to 7 to see it end.
PLOT_Y_DECADES: float = 5.0

# How many y decades get a LABELLED gridline. The stacked layout gives each
# panel ~1.9in for PLOT_Y_DECADES decades, i.e. ~19pt per decade at 8.5pt type,
# which fits a label per decade with air to spare -- so every decade is drawn.
# This is not cosmetic: when the y ladder is thinned, the labelled gridlines sit
# 100x apart while the x gridlines sit 10x apart, and a reader estimating the
# power law off the picture gets an exponent twice the real one. The measured
# slope is ~ -0.5; at stride 2 it looks like -1.
PLOT_Y_LABEL_MAX: int = 8

# A small heading inside the figure. OFF by default: in the paper the LaTeX
# caption sits immediately under the panels and carries all of it, including the
# setup line (drafts and target Llama-3.2-1B-Instruct, alpha = 0.5) that used to
# be a subtitle. Turn it on only when a PNG has to stand on its own.
PLOT_TITLES: bool = False
PLOT_TITLE_TEXT: str = "Top-$k$ truncation error of the SUD normalizer"

# Basename of the figure written into OUTPUT_DIR. Set to
# "topk_truncation_error" to get the filename the paper's \includegraphics
# expects; the shell wrapper's docs name the default.
FIGURE_STEM: str = "normalizer_truncation"

# The linear-axis companion. Same data, same panels, same series -- both axes
# drawn linearly instead of log-log. It is written as a SEPARATE file rather
# than replacing the log figure because the three views show different things
# and none substitutes for the others: log-log resolves the k^-0.5 power law of
# the head, semilog-y (linear k, log TV) resolves the EXPONENTIAL tail above
# k~8K that log-log compresses into an apparent cliff, and fully linear shows
# the raw magnitude of the error plus the fact that TV really does reach 0 at
# k=|V| -- a point neither log-y view can draw, since log 0 is minus infinity,
# and which they have to state in words in a corner note instead.
LINEAR_FIGURE_STEM: str = "normalizer_truncation_linear"

# The semilog companion: linear k, log TV. It is the third of the three ways to
# draw this and it isolates one claim the other two blur -- on a linear k axis a
# log TV axis turns "how many decades of error are left" into a height you read
# straight off the panel at the k you would actually ship, without the log k
# axis's compression making every small k look equally far from the origin.
# What it inherits from the linear figure is the crowding: k is still linear, so
# the head is still squeezed against the left spine.
SEMILOG_FIGURE_STEM: str = "normalizer_truncation_semilogy"
SEMILOG_FIGURE: bool = True

# Upper end of the semilog figure's k axis. None = the whole vocabulary. Unlike
# the fully linear figure this stays readable at full range, because the log y
# axis keeps the flat tail legible instead of pressing it onto the zero rule.
SEMILOG_X_MAX: Optional[int] = None
LINEAR_FIGURE: bool = True

# Upper end of the linear figure's k axis. None means the whole vocabulary,
# which is the honest full-range answer and also a nearly empty figure: TV falls
# below 1% of its k=1 value by k~90, so at full range 99.93% of the panel width
# holds a flat line on the zero rule and the entire sweep is one vertical spike
# against the y axis. That is not a drawing flaw, it is what "the error decays
# fast in absolute terms but slowly in relative terms" LOOKS like on linear
# axes -- and it is exactly the reason the main figure is log-log.
#
# A number here zooms to the head, where a linear axis does have something to
# say that the log one flattens: the actual SHAPE of the drop, and how quickly
# it stops paying. 256 keeps the knee and the start of the flat run.
LINEAR_X_MAX: Optional[int] = 256

# ============================================================================
# END CONFIG
# ============================================================================

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed every RNG this script can reach.

    The measurement itself is deterministic (teacher-forced forwards, no
    sampling), so this only pins prompt subsampling and any library-internal
    randomness — but it makes the whole run bit-reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - older torch
        pass


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_examples(
    tokenizer,
    split: str,
    n_prompts: Optional[int],
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    """Tokenize a TOFU split into (full_ids, prompt_len) records.

    `full_ids` is the chat-templated question+answer sequence and `prompt_len`
    is where the ground-truth answer starts, so the evaluated positions are the
    answer tokens (including the closing turn token) and nothing else.
    """
    ds = load_dataset("locuslab/TOFU", name=split, split="train")
    n_total = len(ds)
    if n_prompts is not None and n_prompts < n_total:
        idx = np.sort(rng.choice(n_total, size=n_prompts, replace=False))
        ds = ds.select(idx.tolist())

    examples: List[Dict[str, object]] = []
    for row in ds:
        question, answer = row[QUESTION_KEY], row[ANSWER_KEY]

        if APPLY_CHAT_TEMPLATE:
            chat = []
            if SYSTEM_PROMPT:
                chat.append({"role": "system", "content": SYSTEM_PROMPT})
            chat.append({"role": "user", "content": question})
            chat.append({"role": "assistant", "content": answer})
            date_info = {"date_string": DATE_STRING} if DATE_STRING else {}
            full_ids = tokenizer.apply_chat_template(
                chat, tokenize=True, add_generation_prompt=False, **date_info
            )
            prompt_ids = tokenizer.apply_chat_template(
                chat[:-1], tokenize=True, add_generation_prompt=True, **date_info
            )
        else:
            prompt_text = f"{question}\n"
            prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
            full_ids = tokenizer(
                prompt_text + answer, add_special_tokens=True
            )["input_ids"]
            if full_ids[-1] != tokenizer.eos_token_id:
                full_ids = full_ids + [tokenizer.eos_token_id]

        prompt_len = len(prompt_ids)
        # Truncating inside the prompt would silently change the context, so
        # drop over-long examples rather than corrupt them.
        if len(full_ids) > MAX_LENGTH:
            full_ids = full_ids[:MAX_LENGTH]
        if len(full_ids) <= prompt_len:
            continue
        examples.append({"full_ids": full_ids, "prompt_len": prompt_len})
    return examples


def make_batches(
    examples: Sequence[Dict[str, object]], batch_size: int
) -> List[List[Dict[str, object]]]:
    """Group examples of similar length together to limit padding waste."""
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["full_ids"]))
    return [
        [examples[i] for i in order[s : s + batch_size]]
        for s in range(0, len(order), batch_size)
    ]


def collate(
    batch: Sequence[Dict[str, object]], pad_id: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int, int]]]:
    """Right-pad a batch and list the logit positions to evaluate.

    Right padding is safe for a teacher-forced causal forward: the pads sit
    after every real token, so causal masking means no real position can attend
    to them, and their own logits are never read.

    Returns (input_ids, attention_mask, selections), where each selection is
    (row, logit_position, answer_index). Logit position t predicts token t+1, so
    the distribution that governs answer token j (at sequence index
    prompt_len + j) lives at logit position prompt_len + j - 1.
    """
    max_len = max(len(ex["full_ids"]) for ex in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    selections: List[Tuple[int, int, int]] = []

    for row, ex in enumerate(batch):
        ids, prompt_len = ex["full_ids"], ex["prompt_len"]
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1
        n_answer = len(ids) - prompt_len
        if MAX_POSITIONS_PER_PROMPT is not None:
            n_answer = min(n_answer, MAX_POSITIONS_PER_PROMPT)
        for j in range(n_answer):
            selections.append((row, prompt_len + j - 1, j))

    return input_ids.to(device), attention_mask.to(device), selections


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_lm(name: str, device: torch.device, dtype: torch.dtype):
    """Load a causal LM, trying each configured attention implementation."""
    last_error: Optional[Exception] = None
    for attn in ATTN_IMPLEMENTATIONS:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=dtype, attn_implementation=attn
            )
            print(f"  loaded {name}  (attn={attn})", flush=True)
            return model.to(device).eval()
        except Exception as exc:  # pragma: no cover - depends on local install
            last_error = exc
            print(f"  attn={attn} failed for {name}: {type(exc).__name__}", flush=True)
    raise RuntimeError(f"could not load {name}") from last_error


def assert_shared_vocab(target_name: str, draft_name: str) -> None:
    """The geometric blend is only meaningful under one shared token mapping."""
    tok_t = AutoTokenizer.from_pretrained(target_name)
    tok_d = AutoTokenizer.from_pretrained(draft_name)
    if tok_t.get_vocab() != tok_d.get_vocab():
        raise AssertionError(
            f"tokenizer vocabularies differ: {target_name} vs {draft_name}"
        )


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------


class FullCurveAccumulator:
    """Streaming position-mean and position-max of TV, at every k in [1, |V|].

    The whole curve is already computed -- `truncation_errors` builds
    log(Z - Zhat_k) for every k in one reverse scan and then keeps only the
    K_VALUES columns. What cannot be afforded is KEEPING the per-position array:
    at full width it is n_positions x n_alphas x |V| floats, which is 53 GB in
    float32 for forget10 alone and 252 GB across the nine runs.

    So the reduction happens here, on the GPU, as chunks stream past. Only the
    running sum and max survive, each (n_alphas, |V|): 11 MB together, whatever
    the split's position count. The mean is the same number the sparse path
    reports as `tv_mean`, computed the same way (sum of per-position TVs divided
    by the position count) -- just at every k instead of sixteen of them.

    The sum is float64 because it runs over up to ~15k positions; per-chunk
    reductions are float64 too (`sum(dtype=...)` reduces in that dtype without
    materializing a float64 copy of the chunk).
    """

    def __init__(self, n_alphas: int, vocab: int, device: torch.device) -> None:
        self.sum = torch.zeros((n_alphas, vocab), dtype=torch.float64, device=device)
        self.max = torch.zeros((n_alphas, vocab), dtype=torch.float32, device=device)
        self.n = 0

    def add(self, a_i: int, rel_full: torch.Tensor) -> None:
        """Fold one (n_positions, |V|) block of TVs into alpha row `a_i`."""
        self.sum[a_i] += rel_full.sum(0, dtype=torch.float64)
        torch.maximum(self.max[a_i], rel_full.amax(0), out=self.max[a_i])

    def note_positions(self, n: int) -> None:
        """Count a chunk once, not once per alpha."""
        self.n += n

    def mean(self) -> np.ndarray:
        return (self.sum / max(self.n, 1)).float().cpu().numpy()

    def maximum(self) -> np.ndarray:
        return self.max.cpu().numpy()


def union_order_and_counts(
    log_p: torch.Tensor, log_q: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nested ordering and set sizes for S_k = top-k(p) UNION top-k(q).

    Algorithm 1 truncates to the union of the two top-k sets, not to top-k(p)
    alone, and the union is what this returns. The point that makes it as cheap
    as the single-model case is that the family is still NESTED: token x enters
    the union at

        key(x) = min(rank_p(x), rank_q(x)),      S_k = {x : key(x) < k},

    so sorting by `key` ascending makes every S_k a PREFIX of one order, and the
    same forward/reverse `logcumsumexp` pair yields log Zhat_k and
    log(Z - Zhat_k) for every k at once -- exactly as for the top-k(p) basis.

    What differs is the column to read: |S_k| is not k but sits somewhere in
    [k, 2k] depending on how much the two heads overlap, and it varies from
    position to position. `counts[:, k-1] = |S_k|` is that width, obtained with
    one batched `searchsorted` over the sorted keys, and it is the index the
    scans are gathered at.

    Returns (order, counts): `order` is (n, V) ascending in `key`; `counts` is
    (n, V) with column k-1 holding |S_k| in [1, V].
    """
    n, vocab = log_p.shape
    device = log_p.device
    ar = torch.arange(vocab, device=device).expand(n, vocab)

    # rank_*[x] = position of x in that model's descending order (0 = argmax).
    rank_p = torch.empty((n, vocab), dtype=torch.long, device=device)
    rank_p.scatter_(-1, log_p.argsort(dim=-1, descending=True), ar)
    rank_q = torch.empty((n, vocab), dtype=torch.long, device=device)
    rank_q.scatter_(-1, log_q.argsort(dim=-1, descending=True), ar)

    key = torch.minimum(rank_p, rank_q)
    del rank_p, rank_q
    order = key.argsort(dim=-1)
    sorted_key = key.gather(-1, order).contiguous()
    del key

    # |S_k| = #{x : key(x) < k}, which for ascending keys is the insertion point
    # of k on the left. Column k-1 holds k = 1..V.
    ks = torch.arange(1, vocab + 1, device=device).expand(n, vocab).contiguous()
    counts = torch.searchsorted(sorted_key, ks)
    del sorted_key
    return order, counts


def truncation_errors(
    log_p: torch.Tensor,
    log_q: torch.Tensor,
    alphas: Sequence[float],
    k_values: Sequence[int],
    topk_basis: str,
    acc: Optional["FullCurveAccumulator"] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Relative normalizer error for a block of positions, every (alpha, k).

    Args:
        log_p: (n, V) normalized target log-probabilities.
        log_q: (n, V) normalized draft log-probabilities.
        acc: if given, also fold this block's TVs at EVERY k into the running
            full-curve reduction. Costs one full-width exp per alpha; adds no
            sort and no extra scan, because the scan is already full width.

    Returns:
        rel_error: (n, len(alphas), len(k_values)) array of (Z - Zhat_k) / Z.
        log_z: (n, len(alphas)) array of the exact full-vocabulary log Z.
    """
    n, vocab = log_p.shape
    neg_inf = torch.full((n, 1), float("-inf"), device=log_p.device, dtype=log_p.dtype)

    rel_error = np.empty((n, len(alphas), len(k_values)), dtype=np.float64)
    log_z_out = np.empty((n, len(alphas)), dtype=np.float64)

    # For "target" and "union" the ordering depends only on p and q, not on
    # alpha, so it is built once. `counts` is None for the bases whose S_k is
    # the first k entries of the order; for "union" it is the per-position |S_k|
    # the scans must be gathered at.
    counts: Optional[torch.Tensor] = None
    if topk_basis == "target":
        shared_order = log_p.argsort(dim=-1, descending=True)
    elif topk_basis == "union":
        shared_order, counts = union_order_and_counts(log_p, log_q)
    else:
        shared_order = None

    for a_i, alpha in enumerate(alphas):
        # Unnormalized log-score of the blend: log(p^(1-a) q^a).
        u = (1.0 - alpha) * log_p + alpha * log_q
        order = shared_order if shared_order is not None else u.argsort(
            dim=-1, descending=True
        )
        u_sorted = torch.gather(u, -1, order)

        # prefix[:, j]   = log sum_{i<=j} exp(u_sorted[i]) = log Zhat_{k=j+1}
        # suffix[:, j]   = log sum_{i>=j} exp(u_sorted[i]) = log (Z - Zhat_{k=j})
        prefix = torch.logcumsumexp(u_sorted, dim=-1)
        suffix = torch.logcumsumexp(u_sorted.flip(-1), dim=-1).flip(-1)
        # Pad so k == vocab (an empty tail) is addressable.
        suffix = torch.cat([suffix, neg_inf], dim=-1)

        log_z = prefix[:, -1]
        log_z_out[:, a_i] = log_z.double().cpu().numpy()

        if acc is not None:
            # suffix[:, k] is log(Z - Zhat_k), so columns 1..V are exactly
            # k = 1..V -- the entire curve, already sitting in memory. Reduce it
            # away immediately; the (n, V) block is the only thing this costs.
            # Column j of `tails` is k = j+1. For a prefix basis that is
            # suffix[:, j+1]; for the union it is suffix at |S_k|, which is
            # where that k's set actually ends.
            tails = (
                suffix.gather(-1, counts) if counts is not None else suffix[:, 1:]
            )
            acc.add(
                a_i,
                torch.exp(tails - log_z[:, None]).clamp_(0.0, 1.0),
            )
            del tails

        for k_i, k in enumerate(k_values):
            # Read the tail off the reverse pass instead of computing
            # Z - Zhat: no cancellation, so tiny errors stay resolvable.
            log_tail = (
                suffix.gather(-1, counts[:, k - 1 : k]).squeeze(-1)
                if counts is not None
                else suffix[:, k]
            )
            rel = torch.exp(log_tail - log_z)
            # (Z - Zhat)/Z is a total variation distance, so it lies in [0, 1]
            # exactly. float32 logsumexp over a 128k vocabulary overshoots by
            # ~2e-6 when the tail carries essentially all the mass; clamping
            # keeps that dust out of the reported numbers and the axis limits.
            rel_error[:, a_i, k_i] = rel.clamp(0.0, 1.0).double().cpu().numpy()

    if acc is not None:
        acc.note_positions(n)
    return rel_error, log_z_out


def draft_diagnostics(
    log_p: torch.Tensor, log_q: torch.Tensor, k_probe: int
) -> Dict[str, np.ndarray]:
    """Per-position signals that say whether the draft q is a sane distribution.

    A collapsed gradient-ascent draft shows up here as near-zero entropy, a max
    probability at 1.0, an argmax that p ranks tens of thousands of tokens deep,
    and essentially no q mass inside p's head.
    """
    ent_p = -(log_p.exp() * log_p).sum(-1)
    ent_q = -(log_q.exp() * log_q).sum(-1)
    max_q = log_q.max(-1).values.exp()

    # Rank of argmax(q) under p, without materializing a full rank tensor:
    # count how many tokens p scores strictly above it.
    arg_q = log_q.argmax(-1, keepdim=True)
    rank_q = (log_p > log_p.gather(-1, arg_q)).sum(-1)

    top = log_p.argsort(dim=-1, descending=True)[:, :k_probe]
    q_in_head = log_q.exp().gather(-1, top).sum(-1)

    return {
        "entropy_p": ent_p.double().cpu().numpy(),
        "entropy_q": ent_q.double().cpu().numpy(),
        "max_prob_q": max_q.double().cpu().numpy(),
        "rank_argmax_q_under_p": rank_q.double().cpu().numpy(),
        "q_mass_in_p_head": q_in_head.double().cpu().numpy(),
    }


def verify_tv_identity(seed: int = 0, tol: float = 1e-6) -> float:
    """Check TV(pi, pi_hat) == (Z - Zhat)/Z, the identity the report relies on.

    Runs on small random distributions (no model needed, milliseconds) and
    compares `truncation_errors` against a brute-force TV built from the
    definition, for EVERY truncation basis -- the identity holds for any S, but
    the union basis reaches its columns through `counts` rather than by slicing
    a prefix, and that indexing is exactly what a wrong answer would come from.
    Raises if the identity ever stops holding.
    """
    g = torch.Generator().manual_seed(seed)
    vocab, worst = 512, 0.0
    log_p = (torch.randn(6, vocab, generator=g) * 3).log_softmax(-1)
    log_q = (torch.randn(6, vocab, generator=g) * 3).log_softmax(-1)
    alphas, ks = [0.0, 0.25, 0.5, 0.75, 1.0], [1, 7, 64, vocab]

    order_p = log_p.argsort(dim=-1, descending=True)
    order_q = log_q.argsort(dim=-1, descending=True)

    for basis in ("union", "target", "blend"):
        rel, _ = truncation_errors(log_p, log_q, alphas, ks, basis)
        for a_i, alpha in enumerate(alphas):
            u = ((1.0 - alpha) * log_p + alpha * log_q).exp()
            pi = u / u.sum(-1, keepdim=True)
            order_b = ((1.0 - alpha) * log_p + alpha * log_q).argsort(
                dim=-1, descending=True
            )
            for k_i, k in enumerate(ks):
                # Membership as a mask, so a variable-width S needs no padding.
                keep = torch.zeros_like(u, dtype=torch.bool)
                if basis == "blend":
                    keep.scatter_(-1, order_b[:, :k], True)
                else:
                    keep.scatter_(-1, order_p[:, :k], True)
                    if basis == "union":
                        keep.scatter_(-1, order_q[:, :k], True)
                masked = u * keep
                pi_hat = masked / masked.sum(-1, keepdim=True)
                tv = 0.5 * (pi - pi_hat).abs().sum(-1)
                dev = (tv - torch.tensor(rel[:, a_i, k_i])).abs().max()
                worst = max(worst, float(dev))
    if worst > tol:
        raise AssertionError(
            f"TV identity broken: max deviation {worst:.3e} > tol {tol:.0e}"
        )
    return worst


def run_split(
    method: str,
    split: str,
    target,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    k_values: Sequence[int],
    vocab_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, float], Optional[Dict[str, np.ndarray]]]:
    """Measure one split end to end.

    Returns (rel_error, log_z, n_prompts, draft_health, full_curve). `rel_error`
    has shape (n_positions, len(ALPHAS), len(k_values)) and `log_z` shape
    (n_positions, len(ALPHAS)); `full_curve` is None unless FULL_K_CURVE, and
    otherwise holds "mean" and "max" of shape (len(ALPHAS), vocab_size), indexed
    so that column j is k = j + 1.
    """
    target_name = MODELS[(method, split)]["target"]
    draft_name = MODELS[(method, split)]["draft"]

    print(f"\n=== {method} | {split} ===", flush=True)
    print(f"  target = {target_name}", flush=True)
    print(f"  draft  = {draft_name}", flush=True)
    assert_shared_vocab(target_name, draft_name)

    examples = build_examples(tokenizer, split, N_PROMPTS_PER_SPLIT, rng)
    batches = make_batches(examples, BATCH_SIZE)
    n_positions = sum(
        len(collate(b, tokenizer.pad_token_id, torch.device("cpu"))[2]) for b in batches
    )
    print(f"  {len(examples)} prompts, {n_positions} decoding positions", flush=True)

    draft = load_lm(draft_name, device, dtype)

    acc = (
        FullCurveAccumulator(len(ALPHAS), vocab_size, device)
        if FULL_K_CURVE
        else None
    )
    rel_chunks: List[np.ndarray] = []
    logz_chunks: List[np.ndarray] = []
    diag_chunks: List[Dict[str, np.ndarray]] = []

    for b_i, batch in enumerate(batches):
        input_ids, attention_mask, selections = collate(
            batch, tokenizer.pad_token_id, device
        )
        with torch.no_grad():
            t_logits = target(input_ids=input_ids, attention_mask=attention_mask).logits
            d_logits = draft(input_ids=input_ids, attention_mask=attention_mask).logits

        rows = torch.tensor([s[0] for s in selections], device=device)
        cols = torch.tensor([s[1] for s in selections], device=device)
        # Promote to float32 before normalizing, as src/model/sud.py does.
        log_p_all = F.log_softmax(t_logits[rows, cols].float(), dim=-1)
        log_q_all = F.log_softmax(d_logits[rows, cols].float(), dim=-1)
        del t_logits, d_logits

        for start in range(0, log_p_all.shape[0], POSITION_CHUNK):
            stop = start + POSITION_CHUNK
            rel, logz = truncation_errors(
                log_p_all[start:stop],
                log_q_all[start:stop],
                ALPHAS,
                k_values,
                TOPK_BASIS,
                acc,
            )
            rel_chunks.append(rel)
            logz_chunks.append(logz)
            diag_chunks.append(
                draft_diagnostics(
                    log_p_all[start:stop], log_q_all[start:stop], DIAG_TOPK_PROBE
                )
            )

        del log_p_all, log_q_all
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  batch {b_i + 1}/{len(batches)} done", flush=True)

    del draft
    full_curve = (
        {"mean": acc.mean(), "max": acc.maximum()} if acc is not None else None
    )
    del acc
    if device.type == "cuda":
        torch.cuda.empty_cache()

    diag = {k: np.concatenate([c[k] for c in diag_chunks]) for k in diag_chunks[0]}
    health = {
        "entropy_p_mean": float(diag["entropy_p"].mean()),
        "entropy_q_mean": float(diag["entropy_q"].mean()),
        "max_prob_q_mean": float(diag["max_prob_q"].mean()),
        "rank_argmax_q_under_p_median": float(
            np.median(diag["rank_argmax_q_under_p"])
        ),
        "q_mass_in_p_head_mean": float(diag["q_mass_in_p_head"].mean()),
        "frac_positions_argmax_q_outside_p_head": float(
            (diag["rank_argmax_q_under_p"] >= DIAG_TOPK_PROBE).mean()
        ),
    }
    return (
        np.concatenate(rel_chunks),
        np.concatenate(logz_chunks),
        len(examples),
        health,
        full_curve,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def aggregate_rows(
    method: str,
    split: str,
    rel: np.ndarray,
    log_z: np.ndarray,
    k_values: Sequence[int],
) -> List[Dict[str, object]]:
    """Per-(split, alpha, k) summary of TV(pi, pi_hat) = (Z - Zhat)/Z.

    Every `tv_*` reduces over decoding positions, one token each.
    """
    rows: List[Dict[str, object]] = []
    for a_i, alpha in enumerate(ALPHAS):
        for k_i, k in enumerate(k_values):
            e = rel[:, a_i, k_i]
            rows.append(
                {
                    "draft_method": method,
                    "split": split,
                    "alpha": alpha,
                    "k": k,
                    "n_positions": int(e.size),
                    # (Z - Zhat)/Z, which IS TV(pi, pi_hat) exactly.
                    "tv_mean": float(e.mean()),
                    "tv_median": float(np.median(e)),
                    "tv_p90": float(np.quantile(e, 0.90)),
                    "tv_p99": float(np.quantile(e, 0.99)),
                    "tv_max": float(e.max()),
                    # Zhat / Z == 1 - rel_err
                    "zhat_over_z_mean": float(1.0 - e.mean()),
                    "zhat_over_z_min": float(1.0 - e.max()),
                    "log_z_mean": float(log_z[:, a_i].mean()),
                }
            )
    return rows


def is_collapsed(health: Dict[str, float]) -> bool:
    """A healthy draft disagrees with the target about *which* head token, not
    about whether the head exists at all."""
    return health["q_mass_in_p_head_mean"] < 0.5 or health["entropy_q_mean"] < 0.01


def print_draft_health(meta: Dict[Tuple[str, str], object]) -> None:
    """Flag collapsed drafts, which make a truncation result uninterpretable."""
    print("\nDraft health (q as a next-token distribution)")
    print(
        f"  {'draft method':<14s} {'split':<10s} {'H(q)':>8s} {'H(p)':>8s} "
        f"{'max q':>8s} {'rank argmax(q)|p':>18s} "
        f"{'q in p top-' + str(DIAG_TOPK_PROBE):>16s}"
    )
    print("  " + "-" * 89)
    suspect: List[str] = []
    for (method, split), info in meta.items():
        h = info["draft_health"]
        print(
            f"  {method:<14s} {split:<10s} {h['entropy_q_mean']:>8.3f} "
            f"{h['entropy_p_mean']:>8.3f} {h['max_prob_q_mean']:>8.4f} "
            f"{h['rank_argmax_q_under_p_median']:>18.0f} "
            f"{h['q_mass_in_p_head_mean']:>16.4f}"
        )
        if is_collapsed(h):
            suspect.append(f"{method}/{split}")
    if suspect:
        print(
            "\n  WARNING: draft appears COLLAPSED for: "
            + ", ".join(suspect)
            + "\n  Its mass sits outside the target's head, so the measured "
            "truncation error\n  reflects that collapse, not the cost of "
            "truncating a sane normalizer."
        )


def print_table(rows: Sequence[Dict[str, object]], k_values: Sequence[int]) -> None:
    """Print mean and p99 TV(pi, pi_hat) as an alpha x k grid per split."""
    by_split: Dict[str, Dict[Tuple[float, int], Dict[str, object]]] = {}
    for r in rows:
        key = f"{r['draft_method']} | {r['split']}"
        by_split.setdefault(key, {})[(r["alpha"], r["k"])] = r

    for split, cells in by_split.items():
        for stat, label in (("tv_mean", "mean"), ("tv_p99", "p99")):
            print(f"\n{split} — {label} TV(pi, pi_hat) = (Z - Zhat)/Z")
            header = "  alpha |" + "".join(f"{k:>11d}" for k in k_values)
            print(header)
            print("  " + "-" * (len(header) - 2))
            for alpha in ALPHAS:
                line = f"  {alpha:>5.2f} |"
                for k in k_values:
                    v = cells[(alpha, k)][stat]
                    line += "      0.0  " if v == 0.0 else f"{v:>11.2e}"
                print(line)


def print_threshold_summary(
    rows: Sequence[Dict[str, object]],
    k_values: Sequence[int],
    stat: str = "tv_p99",
    label: str = "p99 TV(pi, pi_hat)",
) -> None:
    """Smallest swept k whose `stat` meets each threshold.

    Which summary you pick moves the answer by orders of magnitude, so the
    caller runs this for more than one of them.
    """
    if not all(stat in r for r in rows):
        return
    lookup = {
        (r["draft_method"], r["split"], r["alpha"], r["k"]): r[stat] for r in rows
    }
    methods = sorted({r["draft_method"] for r in rows})
    print(f"\nSmallest swept k with {label} <= threshold")
    head = "  draft method   split      alpha |" + "".join(
        f"{t:>12.0e}" for t in REL_ERROR_THRESHOLDS
    )
    print(head)
    print("  " + "-" * (len(head) - 2))
    for method in methods:
      for split in SPLITS:
        for alpha in ALPHAS:
            # `.get` rather than `[]`: a replot may be reading a CSV from an
            # older sweep whose (method, split, alpha) set does not match the
            # current CONFIG block, and a stale side table must not raise.
            cells = [
                (k, lookup.get((method, split, alpha, k))) for k in k_values
            ]
            if all(v is None for _, v in cells):
                continue
            line = f"  {method:<14s} {split:<10s} {alpha:>5.2f} |"
            for thr in REL_ERROR_THRESHOLDS:
                hit = next(
                    (k for k, v in cells if v is not None and v <= thr), None
                )
                line += f"{'none':>12s}" if hit is None else f"{hit:>12d}"
            print(line)


def exact_threshold_k(curve: np.ndarray, tau: float) -> Optional[int]:
    """Smallest k with curve[k-1] <= tau, or None if the curve never gets there.

    `curve` is TV against k for k = 1..|V| and is monotone non-increasing (S only
    ever grows), so the first True in `curve <= tau` is the crossing and argmax
    finds it in one pass. This is the number the sparse grid could only bracket:
    it reports the first SAMPLED k below tau, which overstates the true crossing
    by up to the grid spacing -- a factor of two on an octave grid.
    """
    below = curve <= tau
    if not below.any():
        return None
    return int(np.argmax(below)) + 1


def print_exact_threshold_summary(
    curves: Dict[str, Dict[str, np.ndarray]],
    alphas: Sequence[float],
    stat: str = "mean",
    label: str = "mean TV(pi, pi_hat)",
) -> None:
    """Exact smallest k meeting each threshold, from the full curve.

    `alphas` must be the alphas the CURVE was measured at, not the CONFIG block's
    ALPHAS. On a replot of an older file the two differ, and indexing row a_i by
    the wrong list silently reports a different alpha's curve under the right
    alpha's label -- which looks like every draft agreeing to the digit, because
    row 0 of an old sweep is alpha=0 and alpha=0 does not depend on the draft.
    """
    if not curves:
        return
    print(f"\nSmallest k with {label} <= threshold  (EXACT, from the full curve)")
    head = "  draft method   split      alpha |" + "".join(
        f"{t:>12.0e}" for t in REL_ERROR_THRESHOLDS
    )
    print(head)
    print("  " + "-" * (len(head) - 2))
    for method in DRAFT_METHODS:
        for split in SPLITS:
            entry = curves.get(f"{method}__{split}")
            if entry is None:
                continue
            for a_i, alpha in enumerate(alphas):
                if a_i >= entry[stat].shape[0]:
                    continue
                line = f"  {method:<14s} {split:<10s} {alpha:>5.2f} |"
                for thr in REL_ERROR_THRESHOLDS:
                    hit = exact_threshold_k(entry[stat][a_i], thr)
                    line += f"{'none':>12s}" if hit is None else f"{hit:>12d}"
                print(line)


def check_full_curve_matches_grid(
    curves: Dict[str, Dict[str, np.ndarray]],
    rows: Sequence[Dict[str, object]],
    vocab_size: int,
    alphas: Sequence[float],
) -> None:
    """Cross-check the streamed curve against the independently-kept grid.

    `alphas` is the curve's own alpha list, for the reason in
    `print_exact_threshold_summary`: indexed by a stale ALPHAS this check
    compares two different alphas and reports a deviation of 1.0.

    The two paths share the scan but not the reduction: the grid path gathers a
    column, ships it to the CPU and means it in numpy over the concatenated
    positions, while the curve path reduces on the GPU chunk by chunk into a
    float64 running sum. Agreement at the sixteen shared k is therefore a real
    check that the streaming arithmetic is sound, not a tautology.
    """
    if not curves:
        return
    grid = {
        (r["draft_method"], r["split"], r["alpha"], r["k"]): r["tv_mean"] for r in rows
    }
    worst, worst_at, compared = 0.0, None, 0
    for key, entry in curves.items():
        method, split = key.split("__", 1)
        for a_i, alpha in enumerate(alphas):
            for k in sorted({r["k"] for r in rows}):
                b = grid.get((method, split, alpha, k))
                # A stale CSV simply has fewer cells to check against.
                if b is None or k > vocab_size or a_i >= entry["mean"].shape[0]:
                    continue
                a = float(entry["mean"][a_i, k - 1])
                # Relative on a quantity spanning many decades; float32 storage
                # of the curve is the floor, so ~1e-6 is the expected agreement.
                dev = abs(a - b) / max(b, 1e-12)
                compared += 1
                if dev > worst:
                    worst, worst_at = dev, (method, split, alpha, k)
    if not compared:
        print("  full-curve vs grid tv_mean: no overlapping cells to compare")
        return
    print(
        f"  full-curve vs grid tv_mean: max relative deviation {worst:.2e} "
        f"at {worst_at} over {compared} cells "
        "(float32 curve storage, expect <= 1e-5)"
    )


def write_full_curve_npz(
    path: str, curves: Dict[str, Dict[str, np.ndarray]], vocab_size: int
) -> None:
    """Store every full curve in one compressed .npz.

    Entries are `<method>__<split>__mean` and `__max`, each (len(ALPHAS), |V|)
    float32 with column j holding k = j + 1. float32 is deliberate: these are
    means of a quantity in [0, 1] that get read on a log axis, so seven
    significant digits is far past what any use needs, and it halves a file that
    is otherwise 130 MB across nine runs.
    """
    arrays: Dict[str, np.ndarray] = {
        "alphas": np.asarray(ALPHAS, dtype=np.float64),
        "vocab_size": np.asarray(vocab_size, dtype=np.int64),
    }
    for key, entry in curves.items():
        for stat, arr in entry.items():
            arrays[f"{key}__{stat}"] = np.asarray(arr, dtype=np.float32)
    np.savez_compressed(path, **arrays)
    print(f"  wrote {path}  ({len(curves)} runs x {len(ALPHAS)} alphas x {vocab_size} k)")


def read_full_curve_npz(
    path: str,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], int, List[float]]:
    """Read back `write_full_curve_npz`, so the figure can be redrawn on a CPU.

    The alphas come from the file rather than from the CONFIG block: a replot
    must index the stored rows by the alphas that were MEASURED, not by whatever
    ALPHAS happens to say now.
    """
    with np.load(path) as npz:
        vocab_size = int(npz["vocab_size"])
        alphas = [float(a) for a in npz["alphas"]]
        curves: Dict[str, Dict[str, np.ndarray]] = {}
        for name in npz.files:
            if name in ("alphas", "vocab_size"):
                continue
            key, stat = name.rsplit("__", 1)
            curves.setdefault(key, {})[stat] = npz[name]
    return curves, vocab_size, alphas


def print_sanity(rows: Sequence[Dict[str, object]], vocab_size: int) -> None:
    """Check the identities this measurement must satisfy."""
    print("\nSanity checks")
    lookup = {(r["draft_method"], r["split"], r["alpha"], r["k"]): r for r in rows}
    methods = sorted({r["draft_method"] for r in rows})

    # Z = sum p^(1-a) q^a <= 1 by Hoelder, with equality at alpha in {0, 1}.
    # Tolerances are float32 logsumexp noise over a 128k-wide vocabulary.
    tol = 1e-5
    worst_z = max(r["log_z_mean"] for r in rows)
    print(
        f"  max mean log Z over all (split, alpha): {worst_z:+.3e}  "
        f"(must be <= 0 up to float32 noise, tol {tol:.0e})"
    )
    for alpha in (0.0, 1.0):
        if alpha in ALPHAS:
            vals = [abs(r["log_z_mean"]) for r in rows if r["alpha"] == alpha]
            print(
                f"  |log Z| at alpha={alpha:.0f} (pi collapses to a single model): "
                f"max {max(vals):.3e}  (must be 0 up to tol {tol:.0e})"
            )

    # Truncating to the whole vocabulary cannot lose any mass.
    if vocab_size in {r["k"] for r in rows}:
        vals = [r["tv_max"] for r in rows if r["k"] == vocab_size]
        print(
            f"  max TV at k=vocab_size={vocab_size}: {max(vals):.3e}"
            "  (must be 0)"
        )

    # The error must be monotone non-increasing in k: S only ever grows.
    ks = sorted({r["k"] for r in rows})
    violations = 0
    for method in methods:
      for split in SPLITS:
        for alpha in ALPHAS:
            seq = [lookup[(method, split, alpha, k)]["tv_mean"] for k in ks]
            violations += sum(
                1 for a, b in zip(seq, seq[1:]) if b > a + 1e-12
            )
    print(f"  monotonicity violations in k: {violations}  (must be 0)")


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    import csv

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)", flush=True)


def read_aggregate_csv(path: str) -> List[Dict[str, object]]:
    """Read back the aggregate CSV so the figures can be redrawn without a GPU."""
    import csv

    int_cols = {"k", "n_positions"}
    rows: List[Dict[str, object]] = []
    with open(path, newline="") as fh:
        for raw in csv.DictReader(fh):
            row: Dict[str, object] = {}
            for key, val in raw.items():
                if key in ("draft_method", "split"):
                    row[key] = val
                elif key in int_cols:
                    row[key] = int(val)
                elif key.startswith("tv_"):
                    # A TV lies in [0, 1] exactly. CSVs written before the
                    # measurement clamped it carry up to ~1e-5 of float32
                    # logsumexp dust above 1, which would otherwise buy an
                    # empty top decade on every axis.
                    row[key] = min(max(float(val), 0.0), 1.0)
                else:
                    row[key] = float(val)
            rows.append(row)
    return rows


def write_per_position_csv(
    path: str,
    per_split: Dict[str, np.ndarray],
    k_values: Sequence[int],
) -> None:
    import csv

    n = 0
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "position_index", "alpha", "k", "rel_err"])
        for split, rel in per_split.items():
            for a_i, alpha in enumerate(ALPHAS):
                for k_i, k in enumerate(k_values):
                    col = rel[:, a_i, k_i]
                    for pos_i, val in enumerate(col):
                        writer.writerow([split, pos_i, alpha, k, f"{val:.12e}"])
                        n += 1
    print(f"  wrote {path}  ({n} rows)", flush=True)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
#
# One figure, `normalizer_truncation.png`: mean per-token TV against k, one
# panel per split and one line per draft method, at a single alpha. It reads
# left to right as "spend more k, lose less of the normalizer".
#
# x is EVERY integer k from 1 to |V| - 1, so each curve is 128,255 points and
# the shape is the measurement rather than an interpolation through octave
# samples. k = |V| is the one k that cannot be drawn: there S is the whole
# vocabulary, so TV is identically 0, which is minus infinity on a log axis.
#
# The decay is close to a power law, so log-log is the only scale the shape is
# legible on. Draft method is a nominal identity, so it takes categorical slots
# in fixed order. Text stays in ink tokens, never a series color.
#
# The figure is a single row drawn at print size: three panels, shared y, one
# shared legend, one shared label per axis. Everything a caption can say has
# been moved to the caption -- the setup line, the definition of the plotted
# quantity, and the fact that the curves fall past the floor into TV = 0 at
# k = |V|. What is left inside the axes is data.


def _mpl():
    """Import matplotlib headless-safe. Returns pyplot, or None if unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Every size is a ratio of BASE_FONT_PT, and the figure is drawn at the
        # size it prints at, so these ARE the rendered point sizes. Rules are
        # hairlines: at 7in wide a 1pt spine is heavier than the curves it
        # frames. fonttype 42 embeds TrueType rather than Type 3, which is what
        # the ACL submission checker asks for.
        b = BASE_FONT_PT
        plt.rcParams.update(
            {
                "font.size": b,
                "axes.labelsize": b,
                "axes.titlesize": b * TITLE_FONT_SCALE,
                "legend.fontsize": b,
                "xtick.labelsize": b * TICK_FONT_SCALE,
                "ytick.labelsize": b * TICK_FONT_SCALE,
                "axes.linewidth": 0.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "grid.linewidth": 0.4,
                "lines.solid_capstyle": "round",
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
        return plt
    except Exception as exc:  # pragma: no cover - depends on local install
        print(f"  skipping figure ({type(exc).__name__}: matplotlib unavailable)")
        return None


def _method_label(method: str) -> str:
    """Short display name; the shared '+GDR' retain term is said once, in the title."""
    return method.replace("_GDR", "").replace("_", "+")


def _split_label(split: str) -> str:
    """`forget05` -> `5%`.

    The panels are a proportion of TOFU forgotten, so they are labelled as one.
    Repeating "forget" over three adjacent titles spends the reader's attention
    on the word the panels have in common instead of the number that separates
    them. "TOFU" and "forget-set" are common to all three panels, so they are
    said once, in the caption; the panel title carries the number and the
    (a)/(b)/(c) label the text cites it by.
    """
    digits = "".join(ch for ch in split if ch.isdigit())
    return f"{int(digits)}%" if digits else split


def _slot(method: str) -> int:
    """Categorical slot for a draft method -- see METHOD_SLOTS.

    Methods with no explicit slot fall in after the mapped ones in the fixed
    DRAFT_METHODS order, so they are stable under filtering too.
    """
    if method in METHOD_SLOTS:
        return METHOD_SLOTS[method]
    rest = [m for m in DRAFT_METHODS if m not in METHOD_SLOTS]
    return len(METHOD_SLOTS) + (rest.index(method) if method in rest else 0)


def _series_style(slot: int, broken: bool) -> Dict[str, object]:
    """Line and marker style for categorical `slot`, shared by plot and legend.

    Identity is carried THREE times over -- hue, dash pattern, and marker shape
    -- all keyed to the same slot. Any one of them alone identifies the series,
    which is the point: the palette's blue and orange convert to near-identical
    mid-greys, so on a greyscale printout or under full-severity CVD the dash and
    the mark are what still separate the curves. Hue is then a convenience for
    the reader who can use it, never the thing the figure depends on.

    `broken` (the draft collapsed, so the curve pictures that rather than
    truncation) can no longer be a dash, because dash is identity now. It
    inverts the marker fill instead: hollow marks, coloured edge. That reads as
    "same series, flagged" rather than as a fourth identity, and it stays one
    mark rather than adding a stroke.

    The white marker edge is the surface ring: it keeps overlapping marks legible
    where two curves coincide, which on forget01 they very nearly do.
    """
    marker, size = MARKER_SPECS[slot % len(MARKER_SPECS)]
    color = CATEGORICAL[slot % len(CATEGORICAL)]
    return {
        "color": color,
        "lw": 1.6,
        "ls": DASH_PATTERNS[slot % len(DASH_PATTERNS)],
        "marker": marker,
        "ms": size,
        "markerfacecolor": "white" if broken else color,
        "markeredgecolor": color if broken else "white",
        "markeredgewidth": 0.9 if broken else 0.5,
    }


def _plot_ks(vocab_size: int) -> np.ndarray:
    """The k that get drawn: every integer in [1, |V| - 1].

    |V| itself is excluded because its TV is identically 0 (S is then the whole
    vocabulary), and a log axis has no room for zero. It is named in the corner
    annotation instead.
    """
    return np.arange(1, vocab_size, dtype=np.int64)


def _fmt_k(v: int) -> str:
    """Tick label for a position in the vocabulary: round, plain, same form.

    The axis spans 1 to 128,256, so the ladder has to be decades -- any other
    spacing either crowds the low end or leaves the high end unmarked, and
    non-round stops (32, 64,000) read as measurements rather than as a ruler.
    What decades still get wrong is NOTATION: mixing "1,000" with "$10^4$" makes
    two neighbouring ticks look like two different kinds of quantity. So every
    label is the plain number, with a K above a thousand, and the reader can see
    at a glance which part of the vocabulary a point sits in.
    """
    if v < 1000:
        return f"{v:,d}"
    if v % 1_000_000 == 0:
        return f"{v // 1_000_000}M"
    return f"{v // 1000}K"


def _decade_ticks(vocab_size: int) -> Tuple[List[int], List[str], List[int]]:
    """Major decade ticks with labels, plus unlabelled 2..9 minors.

    With 128k samples a tick per sample is meaningless, so the ladder switches
    from "one per swept k" to plain decades. |V| gets a labelled tick of its own
    at the right edge -- it is the ceiling the whole question is posed against,
    and leaving the axis to trail off at an unmarked 10^5 would hide it.
    """
    majors, labels = [], []
    e = 0
    while 10**e < vocab_size:
        # |V| = 128,256 sits only 0.11 decades above 10^5, so that decade's
        # label would print on top of it. The ceiling wins: it is the quantity
        # with meaning, the decade is just a ruler mark.
        if np.log10(vocab_size / 10**e) >= 0.35:
            majors.append(10**e)
            labels.append(_fmt_k(10**e))
        e += 1
    majors.append(vocab_size)
    # Just the symbol. The number it stands for is a five-digit string that
    # would either stack a second line under every panel or collide with the
    # 10K tick; it is said once, in the shared x-axis label.
    labels.append("$|V|$")
    minors = [
        m * 10**p
        for p in range(0, e)
        for m in range(2, 10)
        if 1 < m * 10**p < vocab_size
    ]
    return majors, labels, minors


def _linear_x_ticks(x_max: int, vocab_size: int) -> Tuple[List[int], List[str]]:
    """Quarter-of-vocabulary ticks for a LINEAR k axis.

    On a linear axis the decade ladder is useless -- 1, 10, 100 and 1K all land
    inside the first 0.8% of the panel and print on top of each other. What a
    linear axis is good for is reading POSITION IN THE VOCABULARY, so it is
    ticked at the fractions of |V| a reader actually wants to convert to: a
    quarter, a half, three quarters, all of it. `_fmt_k` supplies the same K
    notation the log axis uses, so the two figures are ticked in one language,
    and |V| gets its own labelled stop at the right edge exactly as before.
    """
    if x_max >= vocab_size:
        majors = [int(round(f * vocab_size)) for f in (0.0, 0.25, 0.5, 0.75)]
        labels = [_fmt_k(v) for v in majors]
        majors.append(vocab_size)
        labels.append("$|V|$")
        return majors, labels
    # Zoomed to the head: quarters of |V| are all off-panel, so the ladder is
    # quarters of the ZOOM instead. Same principle, different ceiling -- and no
    # $|V|$ stop, because the panel no longer reaches it.
    majors = [int(round(f * x_max)) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    return majors, [_fmt_k(v) for v in majors]


def _decimate_linear(ks: np.ndarray, max_points: int) -> np.ndarray:
    """Indices of a linear-uniform subsample of `ks`, for DRAWING only.

    The counterpart of `_decimate_log`, and it has to be a different function
    rather than the same one reused: log-uniform sampling puts thousands of
    vertices in the first 1% of a linear panel and leaves gaps of ~200 k out
    near |V|, which is where a linear axis spends nearly all of its pixels.
    Uniform sampling matches the axis -- at `max_points` = 6000 over 128k the
    gap is ~21 k, against ~71 k per pixel column at print resolution, so the
    drawn path is again sub-pixel faithful.
    """
    if len(ks) <= max_points:
        return np.arange(len(ks))
    idx = np.unique(np.linspace(0, len(ks) - 1, max_points).round().astype(np.int64))
    return np.unique(np.concatenate([idx, [0, len(ks) - 1]]))


def _y_decade_ticks(lo: float, hi: float, max_labels: int = 5) -> List[float]:
    """Labelled decades inside [lo, hi], thinned from the TOP down.

    Seven decades in a 1.4in panel is a label every ~15pt, which at 8pt type is
    a column of digits rather than a scale, so the ladder thins to every second
    decade. Counting down from the top is what decides the PHASE: it guarantees
    a label at the decade the curves start in, which is the one a reader lands
    on first, where an automatic locator is as likely to leave the top rung bare
    and label the empty floor instead.
    """
    top, bottom = int(np.floor(np.log10(hi))), int(np.ceil(np.log10(lo)))
    n = top - bottom + 1
    if n <= 0:
        return []
    stride = int(np.ceil(n / max_labels))
    return [10.0**e for e in range(top, bottom - 1, -stride)][::-1]


def _style_axes(
    ax,
    vocab_size: int,
    show_xlabels: bool,
    ylog: bool = True,
    xlog: bool = True,
    x_max: Optional[int] = None,
) -> None:
    """Frame: hairline horizontal grid, round ticks, no box.

    `xlog`/`ylog` pick the scales. They are separate flags rather than one
    "linear figure" switch because the axes fail differently: a linear x hides
    the small-k behaviour, a linear y hides the small-TV behaviour, and which
    of those is acceptable depends on what the figure is for.
    """
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    if xlog:
        majors, labels, minors = _decade_ticks(vocab_size)
        ax.set_xscale("log")
        # Left edge at 1, not 0: log 0 is minus infinity, so k=0 cannot be drawn.
        ax.set_xlim(1, vocab_size)
        ax.xaxis.set_minor_locator(FixedLocator(minors))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        majors, labels = _linear_x_ticks(x_max or vocab_size, vocab_size)
        ax.set_xlim(0, x_max or vocab_size)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(FixedLocator(majors))
    ax.set_xticklabels(labels if show_xlabels else [""] * len(majors),
                       color=INK_SECONDARY)

    if ylog:
        from matplotlib.ticker import LogLocator

        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        # Minor decade ticks over ten-plus decades are a ladder of noise; the
        # decade grid already says where you are.
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())

    # Horizontal grid only. A vertical rule per decade competes with the curves,
    # which here are long and shallow and have no vertical structure to read.
    ax.grid(True, axis="y", which="major", color=GRIDLINE, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_LINE)
    ax.tick_params(colors=INK_MUTED, width=0.5, length=2.4, pad=2.0)
    ax.tick_params(which="minor", length=1.3, width=0.4)
    for lbl in ax.get_yticklabels():
        lbl.set_color(INK_SECONDARY)


def _marker_indices(ks: np.ndarray, logx: bool = True) -> List[int]:
    """Indices into `ks` of the decade anchors, plus the last point.

    Fixed anchors rather than a stride: every series then carries a mark at the
    same round k -- the same round k the axis is ticked at -- which is what lets
    the shape channel work as identity across panels while the marks double as a
    reading aid. The final point is included so each series is marked where it
    leaves the panel.
    """
    if logx:
        anchors: Sequence[int] = MARKER_ANCHORS
    elif ks[-1] >= MARKER_ANCHORS_LINEAR[-1]:
        anchors = MARKER_ANCHORS_LINEAR
    else:
        # Zoomed: the vocabulary-scale anchors are all past the right edge, so
        # the marks fall back to quarters of whatever range is drawn.
        anchors = [int(round(f * ks[-1])) or 1 for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    idx = [int(np.searchsorted(ks, a)) for a in anchors if a <= ks[-1]]
    idx.append(len(ks) - 1)
    return sorted(set(i for i in idx if 0 <= i < len(ks)))


def _log_limits(
    values: Sequence[float], max_decades: Optional[float] = None
) -> Tuple[float, float]:
    """Tightest y range covering `values`, snapped to a 1-3-per-decade ladder.

    Rounding out to whole decades buys empty panel: a maximum of 0.22 pulls the
    top rung to 1.0 and hands two thirds of a decade — a sixth of the panel
    height — to blank space, which on an ACL page is height the curves should
    have. Allowing the 3x rung halves that waste while leaving every *decade*
    rung intact, so panels sharing the range still get labelled gridlines in the
    same places (the reason the old version aligned to decades at all).
    """
    arr = np.asarray(values, dtype=np.float64)
    pos = arr[arr > 0]
    if not pos.size:
        return 1e-6, 1.0
    lo_raw, hi_raw = float(pos.min()), float(pos.max())
    # `max_decades` floors the axis rather than letting one collapsing tail set
    # a range in which nothing else can be read. See PLOT_Y_DECADES.
    if max_decades is not None:
        lo_raw = max(lo_raw, hi_raw * 10.0**-max_decades)
    pos = np.array([lo_raw, hi_raw])

    def snap(v: float, up: bool) -> float:
        # Rungs bracketing v's decade, so the answer is always in the list.
        e = int(np.floor(np.log10(v)))
        rungs = [f * 10.0**p for p in (e - 1, e, e + 1) for f in (1.0, 3.0)]
        if up:
            return float(next(r for r in rungs if r >= v * (1 - 1e-12)))
        return float(next(r for r in reversed(rungs) if r <= v * (1 + 1e-12)))

    return snap(min(pos), up=False), snap(max(pos), up=True)


def _save(fig, path_stem: str) -> None:
    """Write every PLOT_FORMATS copy. The PDF is the one the paper includes.

    `bbox_inches="tight"` trims the constrained-layout margin that LaTeX would
    otherwise have to absorb, so the panels really do span \textwidth; the tiny
    pad keeps the outermost tick label from being shaved. dpi only affects the
    PNG, which exists for eyeballing.
    """
    for fmt in PLOT_FORMATS:
        out = f"{path_stem}.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.01)
        print(f"  wrote {out}", flush=True)


def _decimate_log(ks: np.ndarray, max_points: int) -> np.ndarray:
    """Indices of a log-uniform subsample of `ks`, for DRAWING only.

    A 128k-vertex path per series is far more geometry than the panel has pixel
    columns, and it bloats the vector output for nothing. Sampling log-uniformly
    keeps every integer k where the axis is expanded -- below ~1e3 the implied
    spacing is under 1, so nothing is dropped there -- and thins only where
    hundreds of k already share a pixel: at `max_points` = 6000 over 5.1 decades
    the coarsest gap is ~200 k near |V|, against ~470 k per pixel column at print
    resolution. The drawn path is therefore sub-pixel faithful to the full curve.

    Every number the text reports -- thresholds, slopes -- is computed from the
    full array. This only decides which vertices get emitted.
    """
    if len(ks) <= max_points:
        return np.arange(len(ks))
    want = np.unique(
        np.round(np.logspace(0, np.log10(float(ks[-1])), max_points)).astype(np.int64)
    )
    idx = np.searchsorted(ks, want).clip(0, len(ks) - 1)
    return np.unique(np.concatenate([idx, [0, len(ks) - 1]]))


def make_method_comparison(
    path_stem: str,
    curves: Dict[str, Dict[str, np.ndarray]],
    vocab_size: int,
    alphas: Sequence[float],
    collapsed: Sequence[Tuple[str, str]] = (),
    alpha: float = COMPARISON_ALPHA,
    plot_methods: Sequence[str] = PLOT_METHODS,
    legend_ncol: Optional[int] = None,
    logx: bool = True,
    logy: bool = True,
    x_max: Optional[int] = None,
) -> None:
    """The figure: one panel per split, one line per draft method, stacked.

    The plotted value is the mean over decoding positions of
    TV(pi, pi_hat_k) = (Z - Zhat_k)/Z -- every position in the split's answers
    contributes its own TV, and they are summed and divided by the number of
    positions. Everything except the draft is held fixed (same target, same
    positions, same alpha), so the vertical gap between the lines is the draft's
    doing and nothing else.

    Drawn at print size for a two-column `figure*`: 7.0 x 6.6in, three rows of
    one column, shared x and y so the splits are compared by reading across
    panels rather than by reading axes, shared legend, one label per axis.

    The stack is the deliberate choice. Side by side, each panel had 2.2in for
    5.1 decades of k and 1.4in for 7 decades of TV -- an aspect at which the
    k^-0.5 power law renders as a flat line, and at which the exponential regime
    above k~8K is compressed into the last 2% of the width and reads as a
    cliff rather than as the regime change it is. Full-width panels put the
    slope back at an angle a reader can see, and the k axis gets three times the
    room, which is where the whole question lives. What it costs is the free side-by-side
    comparison of the three splits; that comparison survives because x and y are
    shared, so a horizontal position means the same k in all three.

    The reading order the layout is built for is (1) error falls with k but only
    as a shallow power law, (2) three conditions, (3) NPO and SimNPO coincide at
    5% and 10%, (4) they separate at 1%, (5) everything collapses at the
    vocabulary -- and none of those needs an annotation to land.

    `plot_methods` selects the series, so the same layout serves both the paper
    figure (two drafts) and the all-methods survey (every measured draft, with
    `legend_ncol` wrapping the key to two rows). Slots come from METHOD_SLOTS,
    which is keyed by method rather than by position, so a draft keeps its
    colour, dash and marker across both figures.

    `logx`/`logy` draw the same panels on linear axes instead. What that costs
    is not cosmetic and is worth stating: linearly, 99% of the panel width goes
    to k > 1,300, where every curve is already flat, so the entire regime the
    figure is about is compressed into the left edge -- and a linear y puts
    every value below ~1e-3 on the zero line, which is most of the sweep. What
    it buys is the two things a log axis genuinely cannot show: the raw size of
    the error, and k=|V| itself, where TV is exactly 0 and log is undefined.
    Drawn with `_decimate_linear` and MARKER_ANCHORS_LINEAR, because the log
    versions of both sample the wrong part of a linear axis.
    """
    plt = _mpl()
    if plt is None:
        return
    from matplotlib.ticker import FixedLocator

    if alpha not in list(alphas):
        print(f"  skipping figure: alpha={alpha} not among measured {list(alphas)}")
        return
    a_i = list(alphas).index(alpha)

    splits = [s for s in SPLITS if any(f"{m}__{s}" in curves for m in plot_methods)]
    methods = [m for m in plot_methods if any(f"{m}__{s}" in curves for s in SPLITS)]
    if not (splits and methods):
        return
    missing = [m for m in plot_methods if m not in methods]
    if missing:
        # Silently drawing fewer series than asked for is how a figure ends up
        # quietly under-reporting a sweep, so the gap is named.
        print(f"  figure {os.path.basename(path_stem)}: no curves for {missing}")

    # On a linear axis k=|V| is an ordinary point and TV=0 an ordinary value, so
    # the sweep is drawn to its true end. `_plot_ks` stops one short because a
    # log axis cannot place either.
    # k=|V| is drawn only when the y axis is linear: TV is exactly 0 there, which
    # is an ordinary point on a linear y and minus infinity on a log one. The x
    # scale has nothing to do with it -- the semilog figure has a linear k axis
    # that reaches |V| and still has to stop one short.
    if logx:
        ks = _plot_ks(vocab_size)
    else:
        end = min(x_max or vocab_size, vocab_size)
        if logy:
            end = min(end, vocab_size - 1)
        ks = np.arange(1, end + 1, dtype=np.int64)
    # Column j of the stored curve is k = j + 1, so k = 1..|V|-1 is [:len(ks)].
    tv = {
        (m, s): np.asarray(curves[f"{m}__{s}"]["mean"][a_i][: len(ks)], dtype=np.float64)
        for m in methods
        for s in splits
        if f"{m}__{s}" in curves
    }
    draw = (_decimate_log if logx else _decimate_linear)(ks, PLOT_MAX_POINTS)
    collapsed = set(collapsed)

    stacked = np.concatenate([arr for arr in tv.values()])
    if not logy:
        # Linear y is anchored at 0 -- it is a distance, 0 is meaningful, and it
        # is reached. The top is padded 6% so the k=1 marks are not clipped by
        # the spine. No decade snapping: there are no decades on this axis.
        hi = float(np.max(stacked))
        ylim = (0.0, hi * 1.06)
        print(
            f"  figure: {len(ks):,d} k per series, drawn with {len(draw):,d} "
            f"vertices; linear y over 0..{ylim[1]:.3f}, linear x over 0..{ks[-1]:,d}"
            + (f" (zoomed from |V|={vocab_size:,d})" if ks[-1] < vocab_size else "")
        )
    else:
        ylim = _log_limits(stacked, max_decades=PLOT_Y_DECADES)
        full = _log_limits(stacked)
        # Where each series leaves the panel, as a fraction of the log-x span.
        # If this is not ~1.0 the clip is hiding real x-extent and
        # PLOT_Y_DECADES needs raising, so it is printed rather than assumed.
        exits = [
            float(np.log10(ks[np.argmax(arr < ylim[0])]) / np.log10(float(ks[-1])))
            for arr in tv.values()
            if (arr < ylim[0]).any()
        ]
        print(
            f"  figure: {len(ks):,d} k per series, drawn with {len(draw):,d} "
            f"vertices; y clipped to {ylim[0]:.0e}..{ylim[1]:.0e} "
            f"({np.log10(ylim[1] / ylim[0]):.1f} of "
            f"{np.log10(full[1] / full[0]):.1f} decades)"
            + (
                f"; curves exit at {min(exits):.3%}..{max(exits):.3%} of the x span"
                if exits
                else "; no curve reaches the floor"
            )
        )

    fig, axes = plt.subplots(
        len(splits),
        1,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        sharey=True,
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    # Stacked, the panels share both a y range and an x range, so the only gap
    # that has to exist is the one each panel title needs. `sharex` is what pays
    # for it: only the bottom panel spends height on tick labels, so the two
    # above it give their full band to the curves. hspace is set to that title
    # gap and no more -- wider and the three splits stop reading as one figure.
    fig.get_layout_engine().set(w_pad=0.015, h_pad=0.015, wspace=0.0, hspace=0.055)

    for row, split in enumerate(splits):
        ax = axes[row][0]
        # Only the bottom panel carries x tick labels. This matches what
        # `sharex` does to the inner panels anyway, so passing True here would
        # set label TEXT on ticks matplotlib has already hidden -- an invisible
        # disagreement between the two mechanisms rather than a second row of
        # numbers.
        _style_axes(
            ax,
            vocab_size,
            show_xlabels=(row == len(splits) - 1),
            ylog=logy,
            xlog=logx,
            x_max=x_max,
        )
        ax.set_ylim(*ylim)
        # After set_ylim, so the ladder is phased against the range the panels
        # actually share rather than against the locator's guess. A ~1.9in
        # panel has room for every decade, so the ladder no longer thins to
        # every second one: at the old 1.4in height the labelled gridlines were
        # 100x apart while the x gridlines were 10x apart, which silently
        # doubled every slope read off the figure by eye.
        if logy:
            ax.yaxis.set_major_locator(
                FixedLocator(_y_decade_ticks(*ylim, max_labels=PLOT_Y_LABEL_MAX))
            )
        else:
            # Round steps chosen by the locator, with 0 pinned by the ylim. A
            # linear TV axis wants plain fractions, not a snapped ladder.
            from matplotlib.ticker import MaxNLocator

            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))

        for method in methods:
            if (method, split) not in tv:
                continue
            style = _series_style(_slot(method), (method, split) in collapsed)
            ax.plot(
                ks[draw],
                tv[(method, split)][draw],
                markevery=_marker_indices(ks[draw], logx=logx),
                label=_method_label(method),
                zorder=3,
                **style,
            )

        # Corner notes, stacked bottom-left. Every curve here falls left to
        # right, so that corner is the one region guaranteed empty in every
        # panel -- which the old mid-panel placement was not: with seven series
        # it printed straight through the collapsed draft it was describing.
        notes: List[str] = []
        # Gated on logy alone: the note exists because a LOG Y AXIS cannot
        # place 0, which is true of the semilog figure too. It was briefly
        # gated on both, which would have dropped it from the one figure whose
        # x axis reaches |V| and whose y axis still cannot draw the value there.
        if row == 0 and logy:
            # The one thing the axes genuinely cannot show: k = |V| is the only
            # k with TV identically 0 (there S IS the vocabulary), which is
            # minus infinity on a log scale, so the curves run off the bottom
            # rather than landing anywhere. Everything else the old annotations
            # said now lives in the caption.
            notes.append(r"$\mathrm{TV}=0$ at $k=|V|$")
        broken = [_method_label(m) for m in methods if (m, split) in collapsed]
        if broken:
            # Names them. "collapsed draft" alone was readable when the figure
            # held two series and only one could be meant; at seven the reader
            # has to match a hollow marker against seven keys to find out who,
            # so the note says who. Collapse is per (method, split), which is
            # why it is listed per panel rather than in the shared legend.
            notes.append("collapsed: " + ", ".join(broken))
        if notes:
            ax.annotate(
                "\n".join(notes),
                xy=(0.035, 0.05),
                xycoords="axes fraction",
                ha="left",
                va="bottom",
                fontsize=BASE_FONT_PT * ANNOTATION_FONT_SCALE,
                color=INK_MUTED,
                linespacing=1.45,
            )
        # "TOFU" and "forget-set size" are common to all three panels, so they
        # are the caption's job; the title carries only what separates them,
        # plus the (a)/(b)/(c) the text cites individual panels by.
        ax.set_title(
            f"({chr(ord('a') + row)}) Forget {_split_label(split)}",
            color=INK_PRIMARY,
            pad=3.5,
        )

    # One x label under the stack, one y label beside it. Both are figure-level:
    # stacked, `axes[0][0]` is the TOP panel, so an Axes ylabel there would name
    # the quantity for one third of the figure and leave the other two panels
    # captioned by nothing. supylabel centres it against all three, which is the
    # scope it actually has. The vocabulary size rides along on the x label
    # rather than under the final tick: it is said once, where it cannot crowd
    # a panel.
    fig.supxlabel(
        r"Truncation parameter $k$"
        + f"   ($|V| = {vocab_size:,d}$)".replace(",", "{,}"),
        color=INK_SECONDARY,
        fontsize=BASE_FONT_PT,
    )
    fig.supylabel(
        r"mean $\mathrm{TV}(\pi,\hat{\pi}_k)$",
        color=INK_SECONDARY,
        fontsize=BASE_FONT_PT,
    )

    # One legend for the figure, above the panels, frameless, in ink rather than
    # in either series colour. Handles are built from `_series_style` rather than
    # read off the axes, so the key shows the marker shape regardless of where
    # the plotted marks happen to fall. Identity is carried three times over --
    # hue, dash, mark -- so the key still works in greyscale.
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            label=_method_label(method),
            **_series_style(
                _slot(method), any((method, s) in collapsed for s in splits)
            ),
        )
        for method in methods
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=legend_ncol or len(handles),
        frameon=False,
        handlelength=2.4,
        handletextpad=0.5,
        columnspacing=2.0,
        borderpad=0.0,
        borderaxespad=0.0,
        labelcolor=INK_SECONDARY,
    )
    if PLOT_TITLES:
        # Deliberately no larger than a panel title, and no subtitle: the setup
        # (checkpoints, alpha) belongs to the caption. See PLOT_TITLES.
        fig.suptitle(PLOT_TITLE_TEXT, fontsize=BASE_FONT_PT, color=INK_PRIMARY)
    _save(fig, path_stem)
    plt.close(fig)


def render_figures(
    output_dir: str,
    curves: Dict[str, Dict[str, np.ndarray]],
    vocab_size: int,
    alphas: Sequence[float],
    collapsed: Sequence[Tuple[str, str]] = (),
) -> None:
    """Draw the figure from the full-k curves."""
    if not curves:
        print("  no full-k curves to plot (FULL_K_CURVE off, or no .npz found)")
        return
    make_method_comparison(
        os.path.join(output_dir, FIGURE_STEM),
        curves,
        vocab_size,
        alphas,
        set(collapsed),
    )
    if LINEAR_FIGURE:
        # The linear-axis companion to the paper figure. See LINEAR_FIGURE_STEM
        # for why it is written alongside the log figure rather than instead of
        # it -- the two answer questions neither axis can answer alone.
        make_method_comparison(
            os.path.join(output_dir, LINEAR_FIGURE_STEM),
            curves,
            vocab_size,
            alphas,
            set(collapsed),
            logx=False,
            logy=False,
            x_max=LINEAR_X_MAX,
        )
    if SEMILOG_FIGURE:
        # Linear k, log TV. See SEMILOG_FIGURE_STEM.
        make_method_comparison(
            os.path.join(output_dir, SEMILOG_FIGURE_STEM),
            curves,
            vocab_size,
            alphas,
            set(collapsed),
            logx=False,
            logy=True,
            x_max=SEMILOG_X_MAX,
        )
    if ALL_METHOD_FIGURE:
        # The survey figure. Same axes, same slots, every measured draft --
        # written alongside rather than instead of the paper figure, because the
        # two answer different questions. See ALL_METHOD_FIGURE.
        make_method_comparison(
            os.path.join(output_dir, ALL_METHOD_FIGURE_STEM),
            curves,
            vocab_size,
            alphas,
            set(collapsed),
            plot_methods=ALL_METHOD_FIGURE_METHODS or DRAFT_METHODS,
            legend_ncol=ALL_METHOD_LEGEND_NCOL,
        )


def replot_from_csv(output_dir: str = OUTPUT_DIR) -> None:
    """Redraw the figure from an existing run, without loading any model.

    Needs `normalizer_truncation_full_curve.npz` and `run_config.json` in
    `output_dir`; the latter carries the draft-health numbers that decide which
    panels are flagged as collapsed. The aggregate CSV is read too when present,
    but only for the printed grid tables -- the figure comes from the .npz.

    Keeping this cheap is the point: the measurement needs a GPU and ~an hour,
    the figure needs neither, so every iteration on the drawing happens here.
    """
    npz_path = os.path.join(output_dir, "normalizer_truncation_full_curve.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"{npz_path} not found -- the figure is drawn from the full-k curve. "
            "Re-run the measurement with FULL_K_CURVE = True."
        )
    curves, vocab_size, alphas = read_full_curve_npz(npz_path)
    with open(os.path.join(output_dir, "run_config.json")) as fh:
        cfg = json.load(fh)
    collapsed = {
        tuple(key.split("__", 1))
        for key, info in cfg.get("runs", {}).items()
        if is_collapsed(info["draft_health"])
    }
    print(
        f"replotting {npz_path}  ({len(curves)} runs, vocab_size={vocab_size}, "
        f"alphas={list(alphas)})"
    )

    # The quantiles only exist on the K_VALUES grid, so the p99 table comes from
    # the CSV when it is there. It is beside the point of a replot, not part of
    # it -- a missing or stale CSV must not stop the figure being drawn.
    csv_path = os.path.join(output_dir, "normalizer_truncation_aggregate.csv")
    if os.path.exists(csv_path):
        rows = read_aggregate_csv(csv_path)
        print_threshold_summary(rows, sorted({int(r["k"]) for r in rows}))
        check_full_curve_matches_grid(curves, rows, vocab_size, alphas)

    print_exact_threshold_summary(curves, alphas)
    render_figures(output_dir, curves, vocab_size, alphas, collapsed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    seed_everything(SEED)

    if REPLOT_FROM_CSV:
        replot_from_csv()
        return

    # Every draft must be a local checkpoint under BASELINES_ROOT -- no hub ids,
    # no other save tree. Checked here rather than trusted from the MODELS
    # comprehension so that a hand-edited entry cannot quietly pull weights from
    # somewhere else, and so the run dies now instead of after the target has
    # been loaded and the first split tokenized.
    root = os.path.normpath(BASELINES_ROOT) + os.sep
    for method in DRAFT_METHODS:
        for split in SPLITS:
            if (method, split) not in MODELS:
                raise KeyError(f"({method!r}, {split!r}) has no entry in MODELS")
            draft = MODELS[(method, split)]["draft"]
            if not os.path.normpath(draft).startswith(root):
                raise ValueError(
                    f"draft {draft!r} is outside {BASELINES_ROOT!r}; drafts must "
                    "be local baseline checkpoints"
                )
            if not os.path.isdir(draft):
                raise FileNotFoundError(f"no draft checkpoint at {draft}")
            if not any(
                f.endswith((".safetensors", ".bin"))
                for f in os.listdir(draft)
                if f != "training_args.bin"
            ):
                raise FileNotFoundError(f"no model weights in {draft}")
    if TOPK_BASIS not in {"union", "target", "blend"}:
        raise ValueError(
            f"TOPK_BASIS must be 'union', 'target' or 'blend', got {TOPK_BASIS!r}"
        )
    if not all(0.0 <= a <= 1.0 for a in ALPHAS):
        raise ValueError("every alpha must lie in [0, 1]")

    dev_tv = verify_tv_identity()
    print(
        f"TV identity check: max |TV(pi,pi_hat) - (Z-Zhat)/Z| = {dev_tv:.2e}  OK",
        flush=True,
    )

    device = torch.device(DEVICE if torch.cuda.is_available() or DEVICE == "cpu" else "cpu")
    dtype = getattr(torch, TORCH_DTYPE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Every (method, split) shares the same "full" target, so load it once.
    target_names = {MODELS[(m, s)]["target"] for m in DRAFT_METHODS for s in SPLITS}
    if len(target_names) != 1:
        raise ValueError(
            "this script loads one shared target; MODELS lists several: "
            f"{sorted(target_names)}"
        )
    target_name = target_names.pop()

    tokenizer = AutoTokenizer.from_pretrained(target_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"device={device}  dtype={dtype}  seed={SEED}  topk_basis={TOPK_BASIS}")
    print(f"loading shared target once: {target_name}", flush=True)
    target = load_lm(target_name, device, dtype)

    vocab_size = int(target.config.vocab_size)
    k_values = sorted({k for k in K_VALUES if 1 <= k <= vocab_size})
    dropped = [k for k in K_VALUES if k > vocab_size]
    if dropped:
        print(f"  dropped k > vocab_size={vocab_size}: {dropped}")
    if INCLUDE_FULL_VOCAB_K and vocab_size not in k_values:
        k_values.append(vocab_size)
    print(f"  vocab_size={vocab_size}  k values: {k_values}")

    per_split_rel: Dict[str, np.ndarray] = {}
    full_curves: Dict[str, Dict[str, np.ndarray]] = {}
    agg_rows: List[Dict[str, object]] = []
    meta: Dict[Tuple[str, str], object] = {}

    for method in DRAFT_METHODS:
        for split in SPLITS:
            # Re-derive the prompt RNG per (method, split) so the evaluated
            # positions are identical across methods -- that is what makes the
            # draft comparison apples-to-apples.
            rng = np.random.default_rng(SEED)
            rel, log_z, n_prompts, health, curve = run_split(
                method, split, target, tokenizer, device, dtype, k_values,
                vocab_size, rng,
            )
            per_split_rel[f"{method}__{split}"] = rel
            if curve is not None:
                full_curves[f"{method}__{split}"] = curve
            agg_rows.extend(aggregate_rows(method, split, rel, log_z, k_values))
            meta[(method, split)] = {
                "target": MODELS[(method, split)]["target"],
                "draft": MODELS[(method, split)]["draft"],
                "n_prompts": n_prompts,
                "n_positions": int(rel.shape[0]),
                "draft_health": health,
            }

    print_draft_health(meta)
    print_table(agg_rows, k_values)
    print_threshold_summary(agg_rows, k_values)
    print_exact_threshold_summary(full_curves, ALPHAS)
    print_exact_threshold_summary(
        full_curves, ALPHAS, "max", "worst-position TV(pi, pi_hat)"
    )
    print_sanity(agg_rows, vocab_size)
    check_full_curve_matches_grid(full_curves, agg_rows, vocab_size, ALPHAS)

    print("\nOutputs")
    write_csv(os.path.join(OUTPUT_DIR, "normalizer_truncation_aggregate.csv"), agg_rows)
    if full_curves:
        write_full_curve_npz(
            os.path.join(OUTPUT_DIR, "normalizer_truncation_full_curve.npz"),
            full_curves,
            vocab_size,
        )
    if WRITE_PER_POSITION_CSV:
        write_per_position_csv(
            os.path.join(OUTPUT_DIR, "normalizer_truncation_per_position.csv"),
            per_split_rel,
            k_values,
        )
    collapsed = {
        (m, s)
        for m in DRAFT_METHODS
        for s in SPLITS
        if is_collapsed(meta[(m, s)]["draft_health"])
    }
    if MAKE_PLOT:
        render_figures(OUTPUT_DIR, full_curves, vocab_size, ALPHAS, collapsed)

    run_meta = {
        "seed": SEED,
        "device": str(device),
        "torch_dtype": TORCH_DTYPE,
        "topk_basis": TOPK_BASIS,
        "draft_methods": list(DRAFT_METHODS),
        "draft_runs": dict(DRAFT_RUNS),
        "draft_family": DRAFT_FAMILY,
        "alphas": ALPHAS,
        "k_values": k_values,
        "full_k_curve": bool(full_curves),
        "vocab_size": vocab_size,
        "n_prompts_per_split": N_PROMPTS_PER_SPLIT,
        "max_positions_per_prompt": MAX_POSITIONS_PER_PROMPT,
        "apply_chat_template": APPLY_CHAT_TEMPLATE,
        "system_prompt": SYSTEM_PROMPT,
        "date_string": DATE_STRING,
        "max_length": MAX_LENGTH,
        "runs": {f"{m}__{s}": v for (m, s), v in meta.items()},
    }
    meta_path = os.path.join(OUTPUT_DIR, "run_config.json")
    with open(meta_path, "w") as fh:
        json.dump(run_meta, fh, indent=2)
    print(f"  wrote {meta_path}", flush=True)


if __name__ == "__main__":
    main()
