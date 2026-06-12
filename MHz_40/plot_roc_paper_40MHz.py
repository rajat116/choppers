#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_roc_paper_40MHz.py
=======================
Professional ROC-curve figures for the 40 MHz dataset (paper-quality).

Paper plot strategy
-------------------
Primary signal : A -> 4l  (widest method-to-method spread -> most illustrative)
Individual PDFs, one story per file:

  fig_A_chopping_cost.pdf      - Full VAE (KL, MSE) vs encoder-only (mu)
  fig_B_kd_students.pdf        - Teacher (MSE) vs NN students (large/small/medium)
  fig_C_bdt_vs_students.pdf    - Small student vs BDT regression targets
  fig_D_bottom_line.pdf        - One representative from each compression class
  fig_E_efficiency_heatmap.pdf - All methods x all 4 signals at FPR working point
  fig_F_summary_dotplot.pdf    - Signal efficiency at WP and AUC
                                  for all methods, 4 signals as coloured dots
  fig_full_<signal>.pdf        - Full comparison per signal (supplementary / appendix)

FPR working-point line: kept as a subtle gray dashed vertical - gives the reader
the trigger-relevant operating context without competing visually with the curves.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc
# gaussian_kde removed — fig_F uses strip plot instead of violin

# ── Output directory (override via --tag on CLI) ───────────────────────────────
OUT_DIR = "paper_plots_40MHz"
BDT_REG_SUFFIX = ""   # e.g. "_T200" → regressed_KL_T200.csv
os.makedirs(OUT_DIR, exist_ok=True)

# ── Global style  (matches the JetImages collaborator style) ─────────────────
matplotlib.rcParams.update({
    "text.usetex":          False,
    "font.family":          "serif",
    "font.size":            12,
    "axes.labelsize":       13,
    "axes.titlesize":       13,
    "axes.titlepad":        9,
    "xtick.labelsize":      11,
    "ytick.labelsize":      11,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.top":            True,
    "ytick.right":          True,
    "xtick.minor.visible":  False,
    "ytick.minor.visible":  False,
    "legend.fontsize":      9.5,
    "legend.framealpha":    0.92,
    "legend.edgecolor":     "0.75",
    "legend.handlelength":  2.5,
    "lines.linewidth":      2.0,
    "axes.linewidth":       0.9,
    "axes.grid":            True,
    "grid.color":           "0.82",
    "grid.linestyle":       "--",
    "grid.linewidth":       0.55,
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.05,
})

# ── Physics config ────────────────────────────────────────────────────────────
signal_tags   = ["Ato4l", "hChToTauNu", "leptoquark", "hToTauTau"]
signal_labels = {
    "Ato4l":       r"$A \to 4\ell$",
    "hChToTauNu":  r"$H^{\pm} \to \tau\nu$",
    "leptoquark":  r"Leptoquark",
    "hToTauTau":   r"$h \to \tau\tau$",
}

PRIMARY_SIGNAL = "Ato4l"   # widest method spread → best for paper ROC plots
FPR_WP         = 1e-3      # L1 trigger working point

# ── Method catalogue & visual style ──────────────────────────────────────────
# Grouped into three classes used to build plot subsets.
STYLE = {
    # ── Teacher (full / chopped VAE) ──
    "MSE": {
        "label": r"$D_{\mathrm{MSE}}$",
        "color": "#1f77b4",   # blue
        "ls": "-", "lw": 2.5, "zorder": 5,
    },
    "KL": {
        "label": r"$D_{\mathrm{KL}}$",
        "color": "#ff7f0e",   # orange
        "ls": "-", "lw": 2.5, "zorder": 5,
    },
    "mus": {
        "label": r"$D_{\mu}$",   # chopped encoder score (CSV tag: clipped_KL)
        "color": "#2ca02c",   # green
        "ls": (0, (5, 2)),    # long dash
        "lw": 2.2, "zorder": 4,
    },
    # ── NN students (KL teacher) ──
    "large_student": {
        "label": r"$\hat{D}_{\mathrm{KL}}^{\mathrm{NN,large}}$",
        "color": "#2ca02c",   # green (matches JetImages collaborator)
        "ls": "--", "lw": 1.8, "zorder": 3,
    },
    "small_student": {
        "label": r"$\hat{D}_{\mathrm{KL}}^{\mathrm{NN,small}}$",
        "color": "#ff7f0e",   # orange
        "ls": "--", "lw": 1.8, "zorder": 3,
    },
    "medium_student": {
        "label": r"$\hat{D}_{\mathrm{KL}}^{\mathrm{NN,medium}}$",
        "color": "#d62728",   # red
        "ls": "--", "lw": 1.8, "zorder": 3,
    },
    # ── BDT students ──
    "regressed_KL": {
        "label": r"$\hat{D}_{\mathrm{KL}}^{\mathrm{BDT}}$",
        "color": "#e377c2",   # pink  (matches JetImages student_BDT colour)
        "ls": "-.", "lw": 1.8, "zorder": 3,
    },
    "regressed_MSE": {
        "label": r"$\hat{D}_{\mathrm{MSE}}^{\mathrm{BDT}}$",
        "color": "#7f7f7f",   # grey
        "ls": "-.", "lw": 1.8, "zorder": 3,
    },
    "regressed_mus": {
        "label": r"$\hat{D}_{\mu}^{\mathrm{BDT}}$",
        "color": "#bcbd22",   # yellow-green
        "ls": "-.", "lw": 1.8, "zorder": 3,
    },
}

# Plot keys vs on-disk CSV suffix (40 MHz teacher/BDT files use clipped_KL)
_AD_CSV_TAG = {"mus": "clipped_KL"}


def _ad_csv_tag(method: str) -> str:
    """Map plot method key -> filename AD suffix."""
    if method in _TEACHER_KEYS + _STUDENT_KEYS:
        return _AD_CSV_TAG.get(method, method)
    base = method.replace("regressed_", "")
    return _AD_CSV_TAG.get(base, base)


# Convenience lists
_TEACHER_KEYS  = ["MSE", "KL", "mus"]
_STUDENT_KEYS  = ["small_student", "medium_student", "large_student"]
_BDT_KEYS      = ["regressed_KL", "regressed_MSE", "regressed_mus"]
ALL_METHODS    = _TEACHER_KEYS + _STUDENT_KEYS + _BDT_KEYS


# ── Data loading (precomputed, cached) ───────────────────────────────────────
# Background scores are the same regardless of signal → load each file once.
_BKG_CACHE = {}   # {method: np.array of background scores}
_ROC_CACHE = {}   # {signal_tag: {method: (fpr, tpr, auc_val, tpr_wp)}}


def _bkg_scores(method):
    """Return background scores for `method`, using cache.
    Only the score column is read — avoids loading all 57 feature columns."""
    if method not in _BKG_CACHE:
        if method in _TEACHER_KEYS + _STUDENT_KEYS:
            tag = _ad_csv_tag(method)
            df = pd.read_csv(f"vae_tmva_input__Background__{tag}_AD_scores.csv",
                             usecols=["target"])
            _BKG_CACHE[method] = df["target"].values
        else:
            base = _ad_csv_tag(method)
            df = pd.read_csv(
                f"vae_tmva_input__Background__regressed_{base}{BDT_REG_SUFFIX}.csv",
                usecols=["score"],
            )
            _BKG_CACHE[method] = df["score"].values
    return _BKG_CACHE[method]


def _sig_scores(signal_tag, method):
    """Return signal scores for `method` and `signal_tag`."""
    if method in _TEACHER_KEYS + _STUDENT_KEYS:
        tag = _ad_csv_tag(method)
        df = pd.read_csv(f"vae_tmva_input__{signal_tag}__{tag}_AD_scores.csv",
                         usecols=["target"])
        return df["target"].values
    else:
        base = _ad_csv_tag(method)
        df = pd.read_csv(
            f"vae_tmva_input__{signal_tag}__regressed_{base}{BDT_REG_SUFFIX}.csv",
            usecols=["score"],
        )
        return df["score"].values


def precompute_all():
    """Load every CSV once and cache all ROC curves."""
    print("  Pre-loading scores and computing ROC curves...")
    for sig in signal_tags:
        _ROC_CACHE[sig] = {}
        for m in ALL_METHODS:
            bkg = _bkg_scores(m)          # cached after first call
            sig_s = _sig_scores(sig, m)
            y_score = np.concatenate([bkg, sig_s])
            y_true  = np.concatenate([np.zeros(len(bkg)), np.ones(len(sig_s))])
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_val     = auc(fpr, tpr)
            tpr_wp      = tpr[np.argmin(np.abs(fpr - FPR_WP))]
            _ROC_CACHE[sig][m] = (fpr, tpr, auc_val, tpr_wp)
        print(f"    ROC done: {signal_labels[sig]}")


def load_roc(signal_tag):
    """Return cached dict of {method: (fpr, tpr, auc_val, tpr_wp)}."""
    return _ROC_CACHE[signal_tag]


# ── Drawing helpers ───────────────────────────────────────────────────────────
def _new_fig(width=6.5, height=5.0):
    """Return (fig, ax) with standard size."""
    return plt.subplots(figsize=(width, height))


def _decorate_ax(ax, legend_handles=None, legend_loc="lower right",
                 xlim=(1e-5, 1.0), ylim=(1e-5, 1.0)):
    """Apply common axis cosmetics."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel(r"Signal efficiency $\varepsilon_S$")

    # Major gridlines only — matches JetImages plot style
    ax.grid(True, which="major", color="0.82", linestyle="--", linewidth=0.55)
    ax.grid(False, which="minor")

    # Diagonal random-guess line
    xd = np.logspace(-5, 0, 300)
    ax.plot(xd, xd, color="0.55", ls="--", lw=0.9, zorder=1)

    # Working-point marker — subtle grey, no legend entry
    ax.axvline(FPR_WP, color="0.50", ls=":", lw=1.1, zorder=2)
    ax.text(FPR_WP * 1.18, ylim[0] * 3,
            fr"FPR $= 10^{{-3}}$",
            fontsize=8.5, color="0.45", va="bottom", rotation=90)

    if legend_handles is not None:
        ax.legend(handles=legend_handles, loc=legend_loc,
                  framealpha=0.92, edgecolor="0.75",
                  handlelength=2.5, handleheight=0.9,
                  borderpad=0.7, labelspacing=0.45)


def _draw_curve(ax, roc_data, method):
    """Plot one ROC curve and return its Line2D for legend."""
    fpr, tpr, auc_val, wp = roc_data[method]
    s   = STYLE[method]
    lbl = (f'{s["label"]}'
           f'  (AUC={auc_val:.3f},'
           fr' $\varepsilon_S$={wp:.3f})')
    line = ax.plot(fpr, tpr,
                   color=s["color"], ls=s["ls"], lw=s["lw"],
                   zorder=s["zorder"], label=lbl)[0]
    return line


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved  {path}")


def _figF_xtick(base: str, sup: str = "") -> str:
    """Compact one-line x tick: e.g. D_MSE or D-hat_KL^{NN,small}."""
    if not sup:
        return rf"${base}$"
    return rf"${base}^{{\mathrm{{{sup}}}}}$"


# ─────────────────────────────────────────────────────────────────────────────
# FIG A — Chopping: decoder removal cost
#   Full VAE (MSE, KL)  vs  encoder-only (μ)
#   Message: chopping the decoder costs little; latent-space scores work
# ─────────────────────────────────────────────────────────────────────────────
def fig_A_chopping_cost():
    print("\n[A] Chopping cost")
    roc = load_roc(PRIMARY_SIGNAL)
    methods = ["MSE", "KL", "mus"]

    fig, ax = _new_fig()
    handles = [_draw_curve(ax, roc, m) for m in methods]
    _decorate_ax(ax, handles)
    _save(fig, "fig_A_chopping_cost.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# FIG B — Knowledge distillation: teacher vs NN students
#   Teacher (MSE)  vs  large / small / medium NN students
#   Message: students track teacher well; medium student notably weaker on A→4ℓ
# ─────────────────────────────────────────────────────────────────────────────
def fig_B_kd_students():
    print("\n[B] KD: teacher vs NN students")
    roc = load_roc(PRIMARY_SIGNAL)
    # KL is the actual training target for all NN students; MSE shown for context
    methods = ["KL", "small_student", "medium_student", "large_student", "MSE"]

    fig, ax = _new_fig()
    handles = [_draw_curve(ax, roc, m) for m in methods]
    _decorate_ax(ax, handles)
    _save(fig, "fig_B_kd_students.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# FIG C — BDT students vs small NN student
#   Teacher (MSE) as anchor, small student, all three BDT targets
#   Message: BDT(MSE) matches or beats small NN student at similar latency
# ─────────────────────────────────────────────────────────────────────────────
def fig_C_bdt_vs_students():
    print("\n[C] BDT vs NN students")
    roc = load_roc(PRIMARY_SIGNAL)
    methods = ["MSE", "small_student",
               "regressed_KL", "regressed_MSE", "regressed_mus"]

    fig, ax = _new_fig()
    handles = [_draw_curve(ax, roc, m) for m in methods]
    _decorate_ax(ax, handles)
    _save(fig, "fig_C_bdt_vs_students.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# FIG D — Bottom line: one representative per compression class
#   Full VAE (MSE) | encoder-only | small NN student | BDT (MSE)
#   Message: compressed deployable models stay near full-VAE performance
# ─────────────────────────────────────────────────────────────────────────────
def fig_D_bottom_line():
    print("\n[D] Bottom line")
    roc = load_roc(PRIMARY_SIGNAL)
    methods = ["MSE", "mus", "small_student", "regressed_MSE"]

    fig, ax = _new_fig()
    handles = [_draw_curve(ax, roc, m) for m in methods]
    _decorate_ax(ax, handles)
    _save(fig, "fig_D_bottom_line.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# FIG E — Signal efficiency heatmap (all methods × all 4 signals)
#   One number per cell: signal efficiency at FPR = 10^-3
#   Message: full landscape at a glance; works as a summary table replacement
# ─────────────────────────────────────────────────────────────────────────────
def fig_E_efficiency_heatmap():
    print("\n[E] Signal efficiency heatmap")

    eff = np.zeros((len(ALL_METHODS), len(signal_tags)))
    for j, sig in enumerate(signal_tags):
        roc = load_roc(sig)
        for i, m in enumerate(ALL_METHODS):
            eff[i, j] = roc[m][3]   # tpr_wp

    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    im = ax.imshow(eff, aspect="auto", cmap="YlOrRd", vmin=0, vmax=eff.max())

    for i in range(len(ALL_METHODS)):
        for j in range(len(signal_tags)):
            val = eff[i, j]
            tc  = "white" if val > 0.60 * eff.max() else "black"
            ax.text(j, i, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=10, color=tc, fontweight="bold")

    # Horizontal separator lines between groups
    for sep in [len(_TEACHER_KEYS) - 0.5,
                len(_TEACHER_KEYS) + len(_STUDENT_KEYS) - 0.5]:
        ax.axhline(sep, color="white", lw=2.0)

    ax.set_xticks(range(len(signal_tags)))
    ax.set_xticklabels([signal_labels[s] for s in signal_tags], fontsize=12)
    ax.set_yticks(range(len(ALL_METHODS)))
    ax.set_yticklabels([STYLE[m]["label"] for m in ALL_METHODS], fontsize=10.5)

    # Group labels on the left margin
    group_info = [
        (_TEACHER_KEYS,  "VAE teacher"),
        (_STUDENT_KEYS,  "NN students"),
        (_BDT_KEYS,      "BDT students"),
    ]
    offset = 0
    for keys, gname in group_info:
        mid = offset + (len(keys) - 1) / 2
        ax.annotate(gname,
                    xy=(-0.52, mid), xycoords=("axes fraction", "data"),
                    ha="center", va="center", fontsize=10.5,
                    fontweight="bold", rotation=90,
                    annotation_clip=False)
        offset += len(keys)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(fr"$\varepsilon_S$  at  FPR $= 10^{{-3}}$", fontsize=12)

    _save(fig, "fig_E_efficiency_heatmap.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# FIG F — Summary dot-plot: analog of JetImages Fig 5
#
#   Two panels side-by-side:
#     Left  — Signal efficiency @ FPR = 10^-3  (trigger working point)
#     Right — AUC
#   X-axis  — 9 methods, grouped into Teacher / NN students / BDT students
#   Dots    — one per BSM signal (4 colours), shown individually + range bar
#   Red box — highlights the "Chopped score" methods (encoder-only teacher
#             + BDT trained on the encoder-only target)
# ─────────────────────────────────────────────────────────────────────────────
def fig_F_summary_dotplot():
    """
    Standard two-panel bar chart: mean across 4 BSM signals per method,
    with min-max whiskers.  Bar colour encodes method class.
    KL and μ bars are outlined in red (chopped / encoder-only).
    """
    print("\n[F] Summary bar chart (signal efficiency & AUC across all methods)")

    method_order = [
        "MSE",
        "KL", "mus",
        "small_student", "medium_student", "large_student",
        "regressed_KL", "regressed_MSE", "regressed_mus",
    ]
    n_m = len(method_order)
    n_s = len(signal_tags)

    # ── Build data matrices ───────────────────────────────────────────────────
    tpr_mat = np.zeros((n_m, n_s))
    auc_mat = np.zeros((n_m, n_s))
    for j, sig in enumerate(signal_tags):
        roc = load_roc(sig)
        for i, m in enumerate(method_order):
            _, _, auc_val, tpr_wp = roc[m]
            tpr_mat[i, j] = tpr_wp
            auc_mat[i, j] = auc_val

    # ── Bar style: colour by method class ────────────────────────────────────
    # Teacher (0-2): steel blue | NN students (3-5): green | BDT (6-8): orange
    bar_colors = (["#5b9bd5"] * 3) + (["#70ad47"] * 3) + (["#ed7d31"] * 3)
    # Chopped bars (KL=1, μ=2) get a red edge
    bar_edge   = ["none"] * n_m
    bar_edge[1] = "#cc0000"
    bar_edge[2] = "#cc0000"
    bar_lw     = [0.0] * n_m
    bar_lw[1]  = 2.0
    bar_lw[2]  = 2.0

    # One-line superscript labels (same notation as ROC legends; avoids stacked-line overlap)
    xlabels = [
        _figF_xtick(r"D_{\mathrm{MSE}}"),
        _figF_xtick(r"D_{\mathrm{KL}}"),
        _figF_xtick(r"D_{\mu}"),
        _figF_xtick(r"\hat{D}_{\mathrm{KL}}", "NN,small"),
        _figF_xtick(r"\hat{D}_{\mathrm{KL}}", "NN,medium"),
        _figF_xtick(r"\hat{D}_{\mathrm{KL}}", "NN,large"),
        _figF_xtick(r"\hat{D}_{\mathrm{KL}}", "BDT"),
        _figF_xtick(r"\hat{D}_{\mathrm{MSE}}", "BDT"),
        _figF_xtick(r"\hat{D}_{\mu}", "BDT"),
    ]

    x       = np.arange(n_m)
    bar_w   = 0.58

    panel_specs = [
        (tpr_mat, r"Signal efficiency $\varepsilon_S$ @ FPR $= 10^{-3}$"),
        (auc_mat, "AUC"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6))

    for ax, (mat, ylabel) in zip(axes, panel_specs):
        means = mat.mean(axis=1)
        lo    = mat.min(axis=1)
        hi    = mat.max(axis=1)

        # ── Bars ──────────────────────────────────────────────────────────────
        for i in range(n_m):
            ax.bar(x[i], means[i], width=bar_w,
                   color=bar_colors[i],
                   edgecolor=bar_edge[i], linewidth=bar_lw[i],
                   zorder=2, alpha=0.88)

        # ── Min–max whiskers ──────────────────────────────────────────────────
        ax.errorbar(x, means,
                    yerr=[means - lo, hi - means],
                    fmt="none", ecolor="0.25", elinewidth=1.4,
                    capsize=4, capthick=1.4, zorder=3)

        # ── Individual signal dots on each bar (jitter x so markers don't stack) ─
        sig_colors  = ["#1a1a2e", "#e94560", "#0f3460", "#533483"]
        sig_markers = ["o", "s", "^", "D"]
        jitter = np.linspace(-0.14, 0.14, n_s)
        for j in range(n_s):
            ax.scatter(x + jitter[j], mat[:, j],
                       color=sig_colors[j], marker=sig_markers[j],
                       s=38, zorder=4, edgecolors="white", linewidths=0.5)

        # ── Group separators and labels ───────────────────────────────────────
        ymax  = mat.max()
        yspan = ymax - mat.min()
        ylim_top = ymax + yspan * 0.55
        ax.set_ylim(0, ylim_top)

        for boundary in [2.5, 5.5]:
            ax.axvline(boundary, color="0.60", lw=0.9, ls="--", zorder=1)

        label_y = ymax + yspan * 0.38
        for cx, gname in [(1.0, "Teacher"), (4.0, "NN students"), (7.0, "BDT students")]:
            ax.text(cx, label_y, gname,
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="0.20")

        # ── "Chopped score" bracket over KL and μ bars (cols 1-2) ─────────────
        brk_y  = ymax + yspan * 0.08
        brk_y2 = ymax + yspan * 0.20
        # horizontal line over the two bars
        ax.annotate("", xy=(x[1] - bar_w/2 - 0.05, brk_y2),
                    xytext=(x[2] + bar_w/2 + 0.05, brk_y2),
                    arrowprops=dict(arrowstyle="-", color="#cc0000", lw=1.5))
        # left serif
        ax.plot([x[1] - bar_w/2 - 0.05]*2, [brk_y, brk_y2],
                color="#cc0000", lw=1.5)
        # right serif
        ax.plot([x[2] + bar_w/2 + 0.05]*2, [brk_y, brk_y2],
                color="#cc0000", lw=1.5)
        ax.text((x[1] + x[2]) / 2, brk_y2 + yspan * 0.02,
                "Chopped score",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#cc0000")

        # ── Grid and axis labels ──────────────────────────────────────────────
        ax.grid(True, axis="y", color="0.85", ls="--", lw=0.55, zorder=0)
        ax.grid(False, axis="x")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=8.5, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(-0.6, n_m - 0.4)
        ax.tick_params(axis="x", pad=4)

    # ── Shared legend (method classes + chopped indicator) ────────────────────
    legend_handles = [
        matplotlib.patches.Patch(facecolor="#5b9bd5", label="Teacher"),
        matplotlib.patches.Patch(facecolor="#70ad47", label="NN students"),
        matplotlib.patches.Patch(facecolor="#ed7d31", label="BDT students"),
        matplotlib.patches.Patch(facecolor="none",
                                  edgecolor="#cc0000", linewidth=2.0,
                                  label="Encoder-only (chopped)"),
    ]
    sig_display = [
        r"$A \to 4\ell$", r"$H^{\pm} \to \tau\nu$",
        r"Leptoquark",    r"$h \to \tau\tau$",
    ]
    for j in range(n_s):
        legend_handles.append(
            Line2D([0], [0], marker=sig_markers[j], color=sig_colors[j],
                   linestyle="none", markersize=7,
                   markeredgecolor="white", markeredgewidth=0.5,
                   label=sig_display[j])
        )
    fig.legend(handles=legend_handles,
               loc="lower center",
               bbox_to_anchor=(0.5, -0.08),
               framealpha=0.93, edgecolor="0.75",
               fontsize=9.5, borderpad=0.7, labelspacing=0.45,
               ncol=4)

    fig.tight_layout(w_pad=3.0)
    fig.subplots_adjust(bottom=0.32)
    path = os.path.join(OUT_DIR, "fig_F_summary_dotplot.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SUPPLEMENTARY — Full comparison per signal (all methods, individual PDFs)
# ─────────────────────────────────────────────────────────────────────────────
def fig_supp_full_per_signal():
    """
    9 curves → legend placed outside the axes to the right so it never
    overlaps the ROC lines.  Figure is widened to give it room.
    """
    print("\n[Supp] Full comparison — one PDF per signal")
    for sig in signal_tags:
        roc  = load_roc(sig)
        # Wider figure: left panel holds the plot, right margin holds legend
        fig, ax = _new_fig(width=9.5, height=5.5)
        handles = [_draw_curve(ax, roc, m) for m in ALL_METHODS]
        _decorate_ax(ax, legend_handles=None)   # no legend inside

        # Legend anchored outside to the right of the axes
        ax.legend(handles=handles,
                  loc="upper left",
                  bbox_to_anchor=(1.02, 1.0),
                  borderaxespad=0,
                  framealpha=0.92, edgecolor="0.75",
                  handlelength=2.5, handleheight=0.9,
                  fontsize=9.5,
                  borderpad=0.7, labelspacing=0.5)

        fig.tight_layout()
        path = os.path.join(OUT_DIR, f"fig_supp_{sig}.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved  {path}")


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="", help="BDT run tag, e.g. T200")
    args = ap.parse_args()
    if args.tag:
        BDT_REG_SUFFIX = f"_{args.tag}"
        OUT_DIR = f"paper_plots_40MHz_{args.tag}"
        os.makedirs(OUT_DIR, exist_ok=True)
        _BKG_CACHE.clear()
        _ROC_CACHE.clear()

    print("=" * 60)
    print(f"  Output dir  : {OUT_DIR}/")
    if BDT_REG_SUFFIX:
        print(f"  BDT scores  : regressed_*{BDT_REG_SUFFIX}.csv")
    print(f"  Primary sig : {signal_labels[PRIMARY_SIGNAL]}")
    print(f"  FPR WP      : {FPR_WP:.0e}")
    print("=" * 60)

    precompute_all()   # load all CSVs and ROC curves once up front

    fig_A_chopping_cost()
    fig_B_kd_students()
    fig_C_bdt_vs_students()
    fig_D_bottom_line()
    fig_E_efficiency_heatmap()
    fig_F_summary_dotplot()
    fig_supp_full_per_signal()

    print("\n" + "=" * 60)
    print(f"  Done. All PDFs in  {OUT_DIR}/")
    print("=" * 60)
