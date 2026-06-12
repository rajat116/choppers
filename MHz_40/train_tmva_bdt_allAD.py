#!/usr/bin/env python3
"""
Train TMVA BDT students for 40 MHz (paper-correct, background-only + tail weights).

Default: 500 trees → dataset_<AD>/
Use --ntrees 200 --tag T200 for ablation (separate dataset_<AD>_T200/, no overwrite).
"""
import argparse
import json
import ROOT
import numpy as np
import pandas as pd
from array import array
from pathlib import Path

AD_TYPES = ["MSE", "KL", "clipped_KL"]
TREE_NAME = "tree"
WEIGHT_LEVELS = [(0.95, 5.0), (0.99, 20.0), (0.999, 80.0)]

# Set in main() from CLI
RUN_TAG = ""       # e.g. "T200" → dataset_KL_T200
FILE_SUFFIX = ""   # e.g. "_T200" → tmva_train__KL_T200.root
BDT_OPTS = ""


def bdt_opts(ntrees: int) -> str:
    return (
        f"!H:!V:NTrees={ntrees}:MinNodeSize=0.5%:MaxDepth=5:BoostType=Grad:Shrinkage=0.05:"
        "UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=40"
    )


def tail_weights(y: np.ndarray) -> np.ndarray:
    w = np.ones(len(y), dtype=np.float32)
    for q, factor in WEIGHT_LEVELS:
        w[y >= np.quantile(y, q)] = factor
    return w


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["weight"] = tail_weights(out["target"].to_numpy(dtype=np.float64))
    return out


def df_to_root(df: pd.DataFrame, output_file: str, tree_name: str) -> None:
    root_file = ROOT.TFile(output_file, "RECREATE")
    tree = ROOT.TTree(tree_name, tree_name)
    branches = {col: array("f", [0.0]) for col in df.columns}
    for col in df.columns:
        tree.Branch(col, branches[col], f"{col}/F")
    for _, row in df.iterrows():
        for col in df.columns:
            branches[col][0] = float(row[col])
        tree.Fill()
    tree.Write()
    root_file.Close()
    print(f"  Wrote {output_file}  ({len(df):,} rows)")


def write_meta(dataloader_dir: Path, meta: dict) -> None:
    dataloader_dir.mkdir(parents=True, exist_ok=True)
    path = dataloader_dir / "bdt_train_meta.json"
    path.write_text(json.dumps(meta, indent=2))
    print(f"  Meta: {path}")


def _train_one(ad_type: str, ntrees: int):
    print(f"\n{'=' * 60}")
    print(f"  TMVA BDT  {ad_type}  (NTrees={ntrees}, tag={RUN_TAG or 'default'})")
    print(f"{'=' * 60}")

    sfx = FILE_SUFFIX
    train_csv = f"vae_tmva_input__train__{ad_type}_AD_scores.csv"
    val_csv = f"vae_tmva_input__val__{ad_type}_AD_scores.csv"
    train_root = f"tmva_train__{ad_type}{sfx}.root"
    test_root = f"tmva_test__{ad_type}{sfx}.root"
    output_file = f"tmva_output__{ad_type}{sfx}.root"
    factory_name = f"TMVARegression_{ad_type}{sfx}"
    dataloader_name = f"dataset_{ad_type}{sfx}" if sfx else f"dataset_{ad_type}"

    df_train = prepare_frame(pd.read_csv(train_csv))
    df_val = prepare_frame(pd.read_csv(val_csv))
    print(f"  Train rows: {len(df_train):,}  (weight max={df_train['weight'].max():.0f})")
    print(f"  Val rows:   {len(df_val):,}")

    df_to_root(df_train, train_root, TREE_NAME)
    df_to_root(df_val, test_root, TREE_NAME)

    ROOT.TMVA.Tools.Instance()
    output = ROOT.TFile(output_file, "RECREATE")
    factory = ROOT.TMVA.Factory(
        factory_name, output,
        "!V:!Silent:Color:DrawProgressBar:AnalysisType=Regression",
    )
    dataloader = ROOT.TMVA.DataLoader(dataloader_name)

    feat_cols = [c for c in df_train.columns if c.startswith("feature_")]
    for var in feat_cols:
        dataloader.AddVariable(var, "F")
    dataloader.AddTarget("target")
    dataloader.SetWeightExpression("weight", "Regression")

    f_train = ROOT.TFile(train_root)
    f_test = ROOT.TFile(test_root)
    dataloader.AddRegressionTree(f_train.Get(TREE_NAME), 1.0, ROOT.TMVA.Types.kTraining)
    dataloader.AddRegressionTree(f_test.Get(TREE_NAME), 1.0, ROOT.TMVA.Types.kTesting)

    factory.BookMethod(dataloader, ROOT.TMVA.Types.kBDT, "BDT", BDT_OPTS)
    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()

    write_meta(
        Path(dataloader_name),
        {
            "ad_type": ad_type,
            "run_tag": RUN_TAG,
            "ntrees": ntrees,
            "train_csv": train_csv,
            "val_csv": val_csv,
            "target_transform": "none",
            "inference_inverse": None,
            "regressed_csv_suffix": sfx,
            "tail_weights": WEIGHT_LEVELS,
            "bdt_opts": BDT_OPTS,
        },
    )
    print(f"  Weights: {dataloader_name}/weights/{factory_name}_BDT.weights.xml")


def main():
    global RUN_TAG, FILE_SUFFIX, BDT_OPTS
    ap = argparse.ArgumentParser(description="Train 40 MHz TMVA BDT students")
    ap.add_argument("--ntrees", type=int, default=500, help="Number of BDT trees (default 500)")
    ap.add_argument(
        "--tag", type=str, default="",
        help="Run label for separate outputs (default: T<ntrees>, e.g. T200)",
    )
    args = ap.parse_args()
    RUN_TAG = args.tag or f"T{args.ntrees}"
    if args.ntrees == 500 and not args.tag:
        RUN_TAG = ""
        FILE_SUFFIX = ""
    else:
        FILE_SUFFIX = f"_{RUN_TAG}"
    BDT_OPTS = bdt_opts(args.ntrees)

    print(f"NTrees={args.ntrees}  tag={RUN_TAG or '(default)'}  suffix={FILE_SUFFIX or '(none)'}")
    for ad_type in AD_TYPES:
        _train_one(ad_type, args.ntrees)
    print("\nDone → run_tmva_inference_multi.py", end="")
    if FILE_SUFFIX:
        print(f" --tag {RUN_TAG}", end="")
    print(" → plot scripts with same --tag")


if __name__ == "__main__":
    main()
