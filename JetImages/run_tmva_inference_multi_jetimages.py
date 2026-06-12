#!/usr/bin/env python3
import os
from pathlib import Path
import ROOT
import pandas as pd
from array import array

# ---------- CONFIG ----------
AD_TYPES   = ["MSE", "KL", "MSE_KL"]
SAMPLES    = ["Background", "signal_t", "signal_w", "signal_z"]
N_FEATS    = 24*24  # 576

TMVA_INPUT_DIR = Path("tmva_inputs")  # where Step-1 wrote CSVs

def run_inference_for_ad(ad_type: str):
    weights_path = Path(f"dataset_{ad_type}/weights/TMVARegression_{ad_type}_BDT.weights.xml")
    if not weights_path.exists():
        raise FileNotFoundError(f"❌ Missing weights for {ad_type}: {weights_path}")

    print(f"\n🚀 Inference for AD: {ad_type}")
    print(f"   Using weights: {weights_path}")

    # Setup TMVA Reader
    reader = ROOT.TMVA.Reader("!Color:!Silent")
    var_arrays = {f"feature_{i}": array('f', [0.0]) for i in range(N_FEATS)}
    for name, arr in var_arrays.items():
        reader.AddVariable(name, arr)
    reader.BookMVA("BDT", str(weights_path))

    # Loop over samples
    for sample in SAMPLES:
        in_csv  = TMVA_INPUT_DIR / f"vae_tmva_input__{sample}__{ad_type}_AD_scores.csv"
        out_csv = TMVA_INPUT_DIR / f"vae_tmva_input__{sample}__regressed_{ad_type}.csv"

        if not in_csv.exists():
            print(f"⚠️  Skipping missing: {in_csv}")
            continue

        df = pd.read_csv(in_csv)
        # Ensure expected features are present
        feat_cols = [f"feature_{i}" for i in range(N_FEATS)]
        if not all(c in df.columns for c in feat_cols):
            raise ValueError(f"CSV {in_csv} does not contain full feature set 0..{N_FEATS-1}")

        scores = []
        # Event loop
        it = df.itertuples(index=False, name=None)
        for row in it:
            # row layout: feature_0..feature_575, target
            # we only need features for inference
            for i in range(N_FEATS):
                var_arrays[f"feature_{i}"][0] = float(row[i])
            s = reader.EvaluateMVA("BDT")
            scores.append(s)

        # Save scores only (like before)
        pd.DataFrame({"score": scores}).to_csv(out_csv, index=False)
        print(f"✅ Saved: {out_csv}  (N={len(scores)})")

def main():
    for ad in AD_TYPES:
        run_inference_for_ad(ad)

if __name__ == "__main__":
    main()
