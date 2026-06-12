#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
40 MHz BDT: teacher vs student validation plots.

Uses subsampled scatter (readable diagonal) + optional 2D density.
MSE often looks poor until retrained on log(1+MSE); KL is typically r~0.94.

Run:
    python3 plot_40MHz_bdt_distributions.py
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "axes.facecolor": "white",
})

HERE = Path(__file__).resolve().parent
BDT_REG_SUFFIX = ""
RNG = np.random.default_rng(42)
SUBSAMPLE = 12_000
ZOOM_PERCENTILE = 99.5

# Plot key "mus" — on-disk CSVs still use clipped_KL
_AD_CSV_TAG = {"mus": "clipped_KL"}

AD_TYPES = {
    "KL": (r"$D_{\mathrm{KL}}$", r"$\hat{D}_{\mathrm{KL}}^{\mathrm{BDT}}$"),
    "MSE": (r"$D_{\mathrm{MSE}}$", r"$\hat{D}_{\mathrm{MSE}}^{\mathrm{BDT}}$"),
    "mus": (r"$D_{\mu}$", r"$\hat{D}_{\mu}^{\mathrm{BDT}}$"),
}

_AD_TITLE = {"KL": r"$D_{\mathrm{KL}}$", "MSE": r"$D_{\mathrm{MSE}}$", "mus": r"$D_{\mu}$"}


def _csv_tag(ad: str) -> str:
    return _AD_CSV_TAG.get(ad, ad)


def load_scores(ad: str):
    tag = _csv_tag(ad)
    t = pd.read_csv(HERE / f"vae_tmva_input__Background__{tag}_AD_scores.csv")["target"].to_numpy()
    p = pd.read_csv(
        HERE / f"vae_tmva_input__Background__regressed_{tag}{BDT_REG_SUFFIX}.csv"
    )["score"].to_numpy()
    return t, p


def metrics(t, p):
    coef = np.polyfit(t, p, 1)
    resid = p - t
    rp, _ = pearsonr(t, p)
    rs, _ = spearmanr(t, p)
    return dict(
        r=rp, rs=rs,
        rmse=float(np.sqrt(np.mean(resid ** 2))),
        mae=float(np.mean(np.abs(resid))),
        a=float(coef[0]), b=float(coef[1]),
    )


def zoom_mask(t, p):
    lim = np.percentile(p, ZOOM_PERCENTILE) * 1.02
    return (t >= 0) & (p >= 0) & (t <= lim) & (p <= lim), lim


def subsample_xy(t, p, n=SUBSAMPLE):
    if len(t) <= n:
        return t, p
    idx = RNG.choice(len(t), size=n, replace=False)
    return t[idx], p[idx]


def plot_scatter_correlation(ax, t, p, xlabel, ylabel, lim):
    """Subsampled scatter — easiest way to see correlation."""
    msk = (t >= 0) & (p >= 0) & (t <= lim) & (p <= lim)
    t, p = t[msk], p[msk]
    m = metrics(t, p)
    ts, ps = subsample_xy(t, p)

    ax.scatter(ts, ps, s=3, alpha=0.25, c="#1a5276", edgecolors="none", rasterized=True)
    ax.plot([0, lim], [0, lim], "r-", lw=2, label="$y = x$", zorder=5)
    xs = np.array([0.0, lim])
    ax.plot(xs, m["a"] * xs + m["b"], color="#e67e22", lw=1.5, ls="--",
            label="linear fit", zorder=4)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2, ls="-", lw=0.5)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    txt = (
        f"Pearson r = {m['r']:.3f}\n"
        f"Spearman r = {m['rs']:.3f}\n"
        f"RMSE = {m['rmse']:.3f}\n"
        f"N = {msk.sum():,}"
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.75", alpha=0.95))
    return m


def plot_rank_correlation(ax, t, p, lim):
    """Rank vs rank — shows ordering agreement (Spearman) clearly."""
    msk = (t >= 0) & (p >= 0) & (t <= lim) & (p <= lim)
    t, p = t[msk], p[msk]
    m = metrics(t, p)
    rt = np.argsort(np.argsort(t))
    rp = np.argsort(np.argsort(p))
    ts, rs = subsample_xy(rt.astype(float), rp.astype(float))

    ax.scatter(ts, rs, s=3, alpha=0.2, c="#1a5276", edgecolors="none", rasterized=True)
    ax.plot([0, len(t)], [0, len(t)], "r-", lw=2, label="perfect rank match")

    ax.set_xlim(0, len(t))
    ax.set_ylim(0, len(t))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Teacher rank")
    ax.set_ylabel("BDT rank")
    ax.grid(True, alpha=0.2)
    ax.set_title(f"Rank correlation (Spearman r = {m['rs']:.3f})", fontsize=11)
    return m


def make_figure(ad: str, out_name: str):
    """One row: scatter | rank — per AD type, for paper."""
    t, p = load_scores(ad)
    _, lim = zoom_mask(t, p)
    xlab, ylab = AD_TYPES[ad]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plot_scatter_correlation(axes[0], t, p, xlab, ylab, lim)
    axes[0].set_title(f"{_AD_TITLE[ad]}: score vs score", fontsize=12)
    plot_rank_correlation(axes[1], t, p, lim)

    fig.suptitle(
        f"40 MHz BDT student — {_AD_TITLE[ad]} (background, zoomed to BDT {ZOOM_PERCENTILE}%)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    path = HERE / out_name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


def make_summary_row():
    """3 targets, scatter only."""
    out_tag = BDT_REG_SUFFIX
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, ad in zip(axes, AD_TYPES):
        t, p = load_scores(ad)
        _, lim = zoom_mask(t, p)
        m = plot_scatter_correlation(ax, t, p, AD_TYPES[ad][0], AD_TYPES[ad][1], lim)
        ax.set_title(f"{_AD_TITLE[ad]}  (r = {m['r']:.3f})", fontsize=12)
    fig.suptitle("40 MHz: teacher vs BDT (scatter, same range as BDT)", fontsize=13, y=1.02)
    plt.tight_layout()
    path = HERE / f"40MHz_bdt_correlation_scatter{out_tag}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


def run_plots():
    out_tag = BDT_REG_SUFFIX

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, ad in zip(axes, AD_TYPES):
        t, p = load_scores(ad)
        m = metrics(t, p)
        hi = np.percentile(t, 99.5)
        bins = np.linspace(0, hi, 50)
        ax.hist(t, bins=bins, histtype="step", lw=2, color="black", density=True,
                label=AD_TYPES[ad][0])
        ax.hist(np.clip(p, 0, hi), bins=bins, histtype="step", lw=2, color="#1a5276",
                density=True, label=AD_TYPES[ad][1])
        ax.set_xlabel(AD_TYPES[ad][0])
        ax.set_ylabel("Density")
        ax.set_title(f"r = {m['r']:.3f}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(HERE / f"40MHz_bdt_score_distributions{out_tag}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: 40MHz_bdt_score_distributions{out_tag}.pdf")

    make_summary_row()
    for ad in AD_TYPES:
        make_figure(ad, f"40MHz_bdt_validation_{ad}{out_tag}.pdf")

    print(f"\nDone (BDT suffix={BDT_REG_SUFFIX or 'default'}).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="", help="Match train tag, e.g. T200")
    args = ap.parse_args()
    if args.tag:
        BDT_REG_SUFFIX = f"_{args.tag}"  # module-level assignment (no global needed here)
    run_plots()
