#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path

# ---- CONFIG ----
signal_tags = ["signal_t", "signal_w", "signal_z"]
ad_types = ["MSE", "KL", "MSE_KL"]              # the AD types we trained BDTs for
tmva_dir = Path("tmva_inputs")                  # folder with CSVs

def safe_eps_tpr(fpr, tpr, target_fpr=1e-5):
    """Return TPR at closest FPR to target_fpr (guarding edge cases)."""
    if len(fpr) == 0:
        return np.nan
    idx = np.argmin(np.abs(fpr - target_fpr))
    return float(tpr[idx])

def plot_roc_for_signal(signal_tag: str):
    print(f"\n📈 Plotting ROC (BDT regressed only) for: {signal_tag}")
    curves = []  # (label, fpr, tpr, auc, eps)

    for ad in ad_types:
        bkg_csv = tmva_dir / f"vae_tmva_input__Background__regressed_{ad}.csv"
        sig_csv = tmva_dir / f"vae_tmva_input__{signal_tag}__regressed_{ad}.csv"

        # Load scores
        df_bkg = pd.read_csv(bkg_csv)
        df_sig = pd.read_csv(sig_csv)

        y_score = np.concatenate([df_bkg["score"].values, df_sig["score"].values])
        y_true  = np.concatenate([np.zeros(len(df_bkg), dtype=int),
                                  np.ones(len(df_sig),  dtype=int)])

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        eps_tpr = safe_eps_tpr(fpr, tpr, 1e-5)

        curves.append((f"regressed_{ad}", fpr, tpr, roc_auc, eps_tpr))

    # ---- Plot ----
    plt.figure(figsize=(12, 8))
    # sort legend by AUC (best first)
    curves.sort(key=lambda x: x[3], reverse=True)

    for label, fpr, tpr, roc_auc, eps in curves:
        plt.plot(fpr, tpr, linewidth=2,
                 label=f"{label} (AUC={roc_auc:.3f}, ε@1e-5={eps:.2e})")

    plt.plot([1e-6, 1], [1e-6, 1], 'k--', label='Random')
    plt.axvline(x=1e-5, color='red', linestyle='--', label='FPR = 1e-5')

    plt.xscale('log'); plt.yscale('log')
    plt.xlim(1e-6, 1);  plt.ylim(1e-6, 1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC (BDT regressed scores): {signal_tag} vs Background")
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, which='both', ls=':')
    plt.tight_layout()
    plt.savefig(f"ROC_{signal_tag}_BDT.pdf")
    plt.show()

if __name__ == "__main__":
    for sig in signal_tags:
        plot_roc_for_signal(sig)
