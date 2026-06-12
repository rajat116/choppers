#!/usr/bin/env python3
"""Run TMVA BDT regression on all 40 MHz samples."""
import argparse
import json
import os
from array import array
from pathlib import Path

import numpy as np
import pandas as pd
import ROOT

AD_TYPES = ["MSE", "KL", "clipped_KL"]
SAMPLES = ["Background", "Ato4l", "hChToTauNu", "leptoquark", "hToTauTau"]
INPUT_VARS = [f"feature_{i}" for i in range(57)]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tag", type=str, default="",
        help="Must match train tag (e.g. T200). Empty = default dataset_<AD>/",
    )
    return ap.parse_args()


def paths(ad_type: str, tag: str):
    sfx = f"_{tag}" if tag else ""
    dataset = f"dataset_{ad_type}{sfx}"
    factory = f"TMVARegression_{ad_type}{sfx}"
    regressed_suffix = sfx  # regressed_KL_T200.csv when tag=T200
    return dataset, factory, regressed_suffix


def load_meta(dataset_dir: Path) -> dict:
    meta_path = dataset_dir / "bdt_train_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {"target_transform": "none", "inference_inverse": None}


def apply_inverse(score: float, meta: dict) -> float:
    inv = meta.get("inference_inverse") or meta.get("target_transform")
    if inv in (None, "none"):
        return score
    if inv == "expm1":
        return float(np.expm1(score))
    raise ValueError(f"Unknown inference inverse: {inv}")


def main():
    args = parse_args()
    for ad_type in AD_TYPES:
        dataset, factory, reg_sfx = paths(ad_type, args.tag)
        print(f"\nInference: {ad_type}  (tag={args.tag or 'default'})")

        weights_path = f"{dataset}/weights/{factory}_BDT.weights.xml"
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing weights: {weights_path}")

        meta = load_meta(Path(dataset))
        reader = ROOT.TMVA.Reader("!Color:!Silent")
        var_arrays = {var: array("f", [0.0]) for var in INPUT_VARS}
        for var in INPUT_VARS:
            reader.AddVariable(var, var_arrays[var])
        reader.BookMVA("BDT", weights_path)

        for sample in SAMPLES:
            input_csv = f"vae_tmva_input__{sample}__{ad_type}_AD_scores.csv"
            out_name = f"regressed_{ad_type}{reg_sfx}.csv"
            output_csv = f"vae_tmva_input__{sample}__{out_name}"
            if not os.path.exists(input_csv):
                print(f"  Skip missing {input_csv}")
                continue

            df = pd.read_csv(input_csv)
            scores = []
            for _, row in df.iterrows():
                for var in INPUT_VARS:
                    var_arrays[var][0] = row[var]
                raw = reader.EvaluateRegression("BDT")[0]
                scores.append(apply_inverse(raw, meta))

            pd.DataFrame({"score": scores}).to_csv(output_csv, index=False)
            print(f"  Saved {output_csv}")


if __name__ == "__main__":
    main()
