#!/usr/bin/env python3
import os
from pathlib import Path
import ROOT
import pandas as pd
from array import array

# ---------- CONFIG ----------
AD_TYPES   = ["MSE", "KL", "MSE_KL"]     # train one model per target
TREE_NAME  = "tree"
N_FEATS    = 24*24                       # 576 features: feature_0..feature_575

# where prepare_tmva_inputs_jetimages.py wrote the CSVs
TMVA_INPUT_DIR = Path("tmva_inputs")     # change if you wrote elsewhere

# ---------- helpers ----------
def csv_to_root(csv_path: Path, root_path: Path, tree_name: str = TREE_NAME):
    df = pd.read_csv(csv_path)
    # sanity
    feat_cols = [f"feature_{i}" for i in range(N_FEATS)]
    if not all(c in df.columns for c in feat_cols) or "target" not in df.columns:
        raise ValueError(f"CSV {csv_path} must contain feature_0..feature_{N_FEATS-1} and 'target'")

    f = ROOT.TFile(str(root_path), "RECREATE")
    t = ROOT.TTree(tree_name, tree_name)

    # create branches
    branches = {c: array('f', [0.0]) for c in feat_cols + ["target"]}
    for c in feat_cols + ["target"]:
        t.Branch(c, branches[c], f"{c}/F")

    for _, row in df.iterrows():
        for c in feat_cols:
            branches[c][0] = float(row[c])
        branches["target"][0] = float(row["target"])
        t.Fill()

    t.Write()
    f.Close()
    print(f"✅ Created ROOT: {root_path}  (N={len(df)})")

# ---------- main ----------
def main():
    ROOT.TMVA.Tools.Instance()

    # We’ll use Background as training set, and Test as TMVA testing set.
    # (Signals will be used later for inference & ROC.)
    for ad in AD_TYPES:
        print(f"\n🚀 Training TMVA BDT for target: {ad}")

        train_csv = TMVA_INPUT_DIR / f"vae_tmva_input__Background__{ad}_AD_scores.csv"
        test_csv  = TMVA_INPUT_DIR / f"vae_tmva_input__Test__{ad}_AD_scores.csv"

        if not train_csv.exists():
            raise FileNotFoundError(f"Missing: {train_csv}")
        if not test_csv.exists():
            raise FileNotFoundError(f"Missing: {test_csv}")

        # Paths per target (avoid overwriting)
        train_root   = Path(f"tmva_train__{ad}.root")
        test_root    = Path(f"tmva_test__{ad}.root")
        output_file  = Path(f"tmva_output__{ad}.root")
        factory_name = f"TMVARegression_{ad}"
        dloader_name = f"dataset_{ad}"

        # CSV -> ROOT
        csv_to_root(train_csv, train_root)
        csv_to_root(test_csv,  test_root)

        # TMVA setup
        output = ROOT.TFile(str(output_file), "RECREATE")
        factory = ROOT.TMVA.Factory(
            factory_name, output,
            "!V:!Silent:Color:DrawProgressBar:AnalysisType=Regression"
        )
        dataloader = ROOT.TMVA.DataLoader(dloader_name)

        # Variables
        for i in range(N_FEATS):
            dataloader.AddVariable(f"feature_{i}", "F")
        dataloader.AddTarget("target")

        # Trees
        ftr = ROOT.TFile(str(train_root))
        fte = ROOT.TFile(str(test_root))
        ttr = ftr.Get(TREE_NAME)
        tte = fte.Get(TREE_NAME)

        dataloader.AddRegressionTree(ttr, 1.0, ROOT.TMVA.Types.kTraining)
        dataloader.AddRegressionTree(tte, 1.0, ROOT.TMVA.Types.kTesting)

        # Book BDT (same settings you used before)
        factory.BookMethod(
            dataloader, ROOT.TMVA.Types.kBDT, "BDT",
            "!H:!V:"
            "NTrees=100:"
            "MinNodeSize=5%:"
            "MaxDepth=3:"
            "BoostType=Grad:"
            "Shrinkage=0.1:"
            "UseBaggedBoost:"
            "BaggedSampleFraction=0.5:"
            "nCuts=20"
        )

        # Train/Test/Eval
        factory.TrainAllMethods()
        factory.TestAllMethods()
        factory.EvaluateAllMethods()

        print(f"✅ Done: {ad}")
        print(f"   Weights: {dloader_name}/weights/{factory_name}_BDT.weights.xml")

if __name__ == "__main__":
    main()
