#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path

# ----------------------------
# Config
# ----------------------------
tmva_dir = Path("tmva_inputs")

# only student (regressed) curves
regressed_ad_types = ["MSE", "KL", "MSE_KL"]    # available: regressed_{AD}.csv
signals = ["t", "w", "z"]                       # combined = t + w + z

# ----------------------------
# Helpers
# ----------------------------
def load_regressed_scores(ad_type: str, tag: str) -> np.ndarray:
    f = tmva_dir / f"vae_tmva_input__{tag}__regressed_{ad_type}.csv"
    df = pd.read_csv(f, usecols=["score"])
    return df["score"].to_numpy()

def concat_signal_scores(loader_fn, ad_type: str, sig_list):
    return np.concatenate([loader_fn(ad_type, f"signal_{s}") for s in sig_list])

def plot_curve(y_true, y_score, label, lw=2):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    idx = np.argmin(np.abs(fpr - 1e-5))
    eps_tpr = tpr[idx] if len(tpr) else np.nan
    plt.plot(fpr, tpr, linewidth=lw, label=f"{label} (AUC={roc_auc:.3f}, ε@1e-5={eps_tpr:.2e})")

# ----------------------------
# Main
# ----------------------------
def main():
    curves = 0
    plt.figure(figsize=(12, 8))

    for ad in regressed_ad_types:
        try:
            bkg = load_regressed_scores(ad, "Background")
            sig = concat_signal_scores(load_regressed_scores, ad, signals)
        except FileNotFoundError as e:
            print(f"⚠️  Skipping {ad}: {e}")
            continue

        y_score = np.concatenate([bkg, sig])
        y_true  = np.concatenate([np.zeros_like(bkg), np.ones_like(sig)])
        plot_curve(y_true, y_score, label=f"{ad} (BDT student)")
        curves += 1

    # cosmetics
    plt.plot([1e-6, 1], [1e-6, 1], "k--", label="Random")  # diagonal
    plt.axvline(x=1e-5, color="red", linestyle="--", label="FPR = 1e-5")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC: Combined Signal (t+w+z) vs Background — BDT Student Only")
    plt.grid(True, which="both", ls=":")
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    out_pdf = "ROC_combined_student_only.pdf"
    out_png = "ROC_combined_student_only.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=200)
    print(f"✅ Saved {out_pdf} and {out_png}")

    if curves == 0:
        print("❗ No curves were plotted — check that regressed CSVs exist in tmva_inputs/.")

    plt.show()

if __name__ == "__main__":
    main()
