#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_efficiency_uncertainties.py
====================================
Bootstrap statistical uncertainties on signal efficiency (epsilon_S)
at FPR = 10^{-3} for all methods in Table 1.

Method:
  For each method x signal pair:
    1. Load background scores (b) and signal scores (s).
    2. Run N_BOOT bootstrap iterations:
         - Resample background with replacement -> b*
         - Find threshold T* = quantile of b* at (1 - FPR_WP)
         - Compute eps* = fraction of signal scores above T*
    3. Report mean, std (=statistical uncertainty), and 68% CI.

The binomial uncertainty on TPR at a fixed threshold is also reported
as a cross-check.

Usage:
    python3 compute_efficiency_uncertainties.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).resolve().parent
FPR_WP     = 1e-3
N_BOOT     = 2000
RNG_SEED   = 42
BDT_SUFFIX = "_T200"   # set "" for default (500-tree) models

signal_tags = ["Ato4l", "hChToTauNu", "leptoquark", "hToTauTau"]
signal_labels = {
    "Ato4l":       "A->4l",
    "hChToTauNu":  "H+->tau nu",
    "leptoquark":  "Leptoquark",
    "hToTauTau":   "h->tau tau",
}

# Map method key -> (bkg_file, sig_file_template, score_col)
# sig_file_template uses {sig} as placeholder
def _bkg_file(ad_tag):
    return HERE / f"vae_tmva_input__Background__{ad_tag}_AD_scores.csv"

def _sig_file(ad_tag, sig):
    return HERE / f"vae_tmva_input__{sig}__{ad_tag}_AD_scores.csv"

def _bkg_bdt(ad_tag):
    return HERE / f"vae_tmva_input__Background__regressed_{ad_tag}{BDT_SUFFIX}.csv"

def _sig_bdt(ad_tag, sig):
    return HERE / f"vae_tmva_input__{sig}__regressed_{ad_tag}{BDT_SUFFIX}.csv"

def _bkg_nn(size):
    return HERE / f"vae_tmva_input__Background__{size}_student_AD_scores.csv"

def _sig_nn(size, sig):
    return HERE / f"vae_tmva_input__{sig}__{size}_student_AD_scores.csv"

# (display_name, bkg_loader, sig_loader)
METHODS = {
    "D_MSE":            ("D_MSE  [Teacher]",
                          lambda: pd.read_csv(_bkg_file("MSE"))["target"].values,
                          lambda s: pd.read_csv(_sig_file("MSE", s))["target"].values),
    "D_KL":             ("D_KL   [Teacher]",
                          lambda: pd.read_csv(_bkg_file("KL"))["target"].values,
                          lambda s: pd.read_csv(_sig_file("KL", s))["target"].values),
    "D_mu":             ("D_mu   [Teacher]",
                          lambda: pd.read_csv(_bkg_file("clipped_KL"))["target"].values,
                          lambda s: pd.read_csv(_sig_file("clipped_KL", s))["target"].values),
    "Dhat_KL_NN_large": ("D-hat_KL^NN-large  [Student]",
                          lambda: pd.read_csv(_bkg_nn("large"))["target"].values,
                          lambda s: pd.read_csv(_sig_nn("large", s))["target"].values),
    "Dhat_KL_NN_med":   ("D-hat_KL^NN-medium [Student]",
                          lambda: pd.read_csv(_bkg_nn("medium"))["target"].values,
                          lambda s: pd.read_csv(_sig_nn("medium", s))["target"].values),
    "Dhat_KL_NN_small": ("D-hat_KL^NN-small  [Student]",
                          lambda: pd.read_csv(_bkg_nn("small"))["target"].values,
                          lambda s: pd.read_csv(_sig_nn("small", s))["target"].values),
    "Dhat_KL_BDT":      ("D-hat_KL^BDT (mu target) [Student]",
                          lambda: pd.read_csv(_bkg_bdt("clipped_KL"))["score"].values,
                          lambda s: pd.read_csv(_sig_bdt("clipped_KL", s))["score"].values),
    "Dhat_MSE_BDT":     ("D-hat_MSE^BDT [Student]",
                          lambda: pd.read_csv(_bkg_bdt("MSE"))["score"].values,
                          lambda s: pd.read_csv(_sig_bdt("MSE", s))["score"].values),
    "Dhat_KL_BDT_KL":   ("D-hat_KL^BDT (KL target) [Student]",
                          lambda: pd.read_csv(_bkg_bdt("KL"))["score"].values,
                          lambda s: pd.read_csv(_sig_bdt("KL", s))["score"].values),
}

# ── helpers ───────────────────────────────────────────────────────────────────
def bootstrap_eps(bkg: np.ndarray, sig: np.ndarray,
                  fpr_wp: float, n_boot: int, rng: np.random.Generator):
    """
    Bootstrap uncertainty on signal efficiency at fixed FPR working point.
    Returns (mean_eps, std_eps, lo_68, hi_68) all in [0,1].
    """
    n_bkg = len(bkg)
    # nominal threshold and eps
    threshold_nom = np.quantile(bkg, 1.0 - fpr_wp)
    eps_nom       = float((sig > threshold_nom).mean())

    # bootstrap
    eps_boot = np.empty(n_boot)
    for i in range(n_boot):
        b_star = rng.choice(bkg, size=n_bkg, replace=True)
        T_star = np.quantile(b_star, 1.0 - fpr_wp)
        eps_boot[i] = float((sig > T_star).mean())

    lo, hi = np.percentile(eps_boot, [16.0, 84.0])
    return eps_nom, eps_boot.std(ddof=1), lo, hi


def binomial_sigma(eps: float, n_sig: int) -> float:
    """Simple binomial sigma on fraction eps with sample size n_sig."""
    return float(np.sqrt(eps * (1.0 - eps) / n_sig))


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(RNG_SEED)

    # Pre-load background scores (cached per method)
    print(f"\nFPR working point : {FPR_WP:.0e}")
    print(f"Bootstrap samples : {N_BOOT}")
    print(f"BDT suffix        : '{BDT_SUFFIX or '(default)'}'")
    print()

    all_rows = []

    for method_key, (display, bkg_fn, sig_fn) in METHODS.items():
        print(f"=== {display} ===")
        try:
            bkg = bkg_fn()
        except FileNotFoundError as e:
            print(f"  SKIP (missing file): {e}")
            continue

        for sig_tag in signal_tags:
            label = signal_labels[sig_tag]
            try:
                sig = sig_fn(sig_tag)
            except FileNotFoundError as e:
                print(f"  {label}: SKIP ({e})")
                continue

            eps_nom, boot_std, lo68, hi68 = bootstrap_eps(
                bkg, sig, FPR_WP, N_BOOT, rng)
            binom_sig = binomial_sigma(eps_nom, len(sig))

            print(f"  {label:20s}  "
                  f"eps={eps_nom*100:5.2f}%  "
                  f"boot_std={boot_std*100:.3f}%  "
                  f"68%CI=[{lo68*100:.2f},{hi68*100:.2f}]%  "
                  f"binom_sigma={binom_sig*100:.3f}%  "
                  f"(N_sig={len(sig):,})")

            all_rows.append(dict(
                method=display,
                signal=label,
                eps_pct=round(eps_nom * 100, 2),
                boot_std_pct=round(boot_std * 100, 4),
                lo68_pct=round(lo68 * 100, 2),
                hi68_pct=round(hi68 * 100, 2),
                binom_std_pct=round(binom_sig * 100, 4),
                N_sig=len(sig),
                N_bkg=len(bkg),
            ))
        print()

    # ── summary ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    out_csv = HERE / "efficiency_uncertainties.csv"
    df.to_csv(out_csv, index=False)
    print(f"Results saved to: {out_csv}\n")

    print("=" * 60)
    print("SUMMARY (all method x signal pairs)")
    print("=" * 60)
    print(f"  Bootstrap std  : "
          f"mean={df.boot_std_pct.mean():.3f}%  "
          f"max={df.boot_std_pct.max():.3f}%  "
          f"min={df.boot_std_pct.min():.3f}%")
    print(f"  Binomial sigma : "
          f"mean={df.binom_std_pct.mean():.3f}%  "
          f"max={df.binom_std_pct.max():.3f}%  "
          f"min={df.binom_std_pct.min():.3f}%")
    print()

    # dominant uncertainty source per signal
    print("Dominant uncertainty source: bootstrap std > binomial sigma?")
    df["boot_dominates"] = df.boot_std_pct > df.binom_std_pct
    print(df.groupby("signal")[["boot_std_pct", "binom_std_pct"]].mean().round(4))

    print()
    print("Suggested paper quote:")
    max_u = df.boot_std_pct.max()
    typ_u = df.boot_std_pct.median()
    print(f"  'Statistical uncertainties on signal efficiency "
          f"(from finite sample size) are at most {max_u:.2f}% "
          f"and typically {typ_u:.2f}% absolute.'")


if __name__ == "__main__":
    main()
