#!/usr/bin/env python3
"""
export_fwX_configs_40MHz.py
Generate fwX JSON + testpoint files for 40 MHz TMVA BDT regression students.

TREE_PATTERN (fwX-only, NOT TMVA NTrees)
----------------------------------------
  Integer passed to get_set(tree_pattern=...):

  - If TREE_PATTERN >= NTrees in the weights XML:
      no merging → one SingleRegressionTree per TMVA tree (required for EVAL_METHOD=2).

  - If TREE_PATTERN < NTrees:
      fwX merges trees into groups (fewer firmware trees, faster, but breaks EVAL_METHOD=2).

  This script sets TREE_PATTERN = NTrees automatically from each weights file.

Usage
-----
  python export_fwX_configs_40MHz.py              # dataset_KL/ (500 trees)
  python export_fwX_configs_40MHz.py --tag T200  # dataset_KL_T200/ (200 trees)
"""
import argparse
import re
import pandas as pd
from pathlib import Path
from fwX import get_regression_BDT_from_TMVA

AD_TYPES = ["KL", "MSE", "clipped_KL"]

N_BITS      = 8
EVAL_METHOD = 2       # recursive SingleRegressionTree evaluation
BIN_ENGINE  = "LUBE"
P_HIGH      = 0.999
EPS_ABS     = 1e-6
EPS_REL     = 1e-3

HERE = Path(__file__).parent.resolve()


def read_ntrees(tmva_xml: Path) -> int:
    text = tmva_xml.read_text()
    m = re.search(r'<Weights\s+NTrees="(\d+)"', text)
    if not m:
        raise RuntimeError(f"Could not parse NTrees from {tmva_xml}")
    return int(m.group(1))


def widen_span(lo: float, hi: float) -> tuple[float, float]:
    if hi > lo:
        return lo, hi
    bump = max(EPS_ABS, EPS_REL * max(abs(lo), abs(hi), 1.0))
    return lo, lo + bump


def paths(ad_type: str, tag: str):
    sfx = f"_{tag}" if tag else ""
    return {
        "tmva_xml": HERE / f"dataset_{ad_type}{sfx}" / "weights" / f"TMVARegression_{ad_type}{sfx}_BDT.weights.xml",
        "tmva_root": HERE / f"tmva_output__{ad_type}{sfx}.root",
        "dataset": f"dataset_{ad_type}{sfx}",
        "out_tag": f"_{tag}" if tag else "",
    }


def export_one(ad_type: str, tag: str) -> None:
    p = paths(ad_type, tag)
    tmva_xml, tmva_root, dataset_name = p["tmva_xml"], p["tmva_root"], p["dataset"]
    bkg_csv = HERE / f"vae_tmva_input__Background__{ad_type}_AD_scores.csv"

    for path in (tmva_xml, tmva_root, bkg_csv):
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")

    ntrees = read_ntrees(tmva_xml)
    tree_pattern = ntrees   # >= NTrees → no merging (required for EVAL_METHOD=2)

    df = pd.read_csv(bkg_csv)
    feats = [c for c in df.columns if c.startswith("feature_")]
    if "target" not in df.columns:
        raise RuntimeError(f"'target' missing in {bkg_csv}")

    mins = df[feats].min()
    qhi = df[feats].quantile(P_HIGH)
    maxi = df[feats].max()

    cut_variable_ranges = {}
    n_fixed = 0
    for v in feats:
        lo, hi = float(mins[v]), float(qhi[v])
        if hi <= lo:
            hi = float(maxi[v])
        lo, hi = widen_span(lo, hi)
        if hi - lo <= EPS_ABS * 10:
            n_fixed += 1
        cut_variable_ranges[v] = (lo, hi)

    t_lo = float(df["target"].min())
    t_hi = float(df["target"].quantile(P_HIGH))
    if t_hi <= t_lo:
        t_hi = float(df["target"].max())
    t_lo, t_hi = widen_span(t_lo, t_hi)

    print(f"\n  TMVA NTrees     : {ntrees}")
    print(f"  TREE_PATTERN    : {tree_pattern}  (no merge — matches NTrees)")
    print(f"  features        : {len(feats)}  |  zero-span fixed : {n_fixed}")
    print(f"  target range    : [{t_lo:.4g}, {t_hi:.4g}]")

    bdt_object = get_regression_BDT_from_TMVA(
        xml_filepath=str(tmva_xml),
        root_filepath=str(tmva_root),
        tmva_dataset_name=dataset_name,
        tmva_bdt_name="BDT",
    )

    set_ = bdt_object.get_set(
        tree_pattern=tree_pattern,
        target_variable_precision=N_BITS,
        cut_precisions=N_BITS,
        bin_engine=BIN_ENGINE,
        cut_variable_ranges=cut_variable_ranges,
        target_variable_range=[t_lo, t_hi],
        evaluation_method=EVAL_METHOD,
    )

    out = p["out_tag"]
    cfg_file = HERE / f"fwX-config_40MHz_{ad_type}{out}_prec{N_BITS}.json"
    tpt_file = HERE / f"fwX-testpoints_40MHz_{ad_type}{out}_prec{N_BITS}.txt"
    set_.save_config(str(cfg_file))
    set_.save_testpoints(str(tpt_file))
    print(f"  Saved  {cfg_file.name}")
    print(f"  Saved  {tpt_file.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="", help="e.g. T200 for dataset_KL_T200/")
    args = ap.parse_args()

    print("=" * 60)
    print("  fwX export — 40 MHz BDT")
    print(f"  tag          : {args.tag or '(default 500-tree models)'}")
    print(f"  EVAL_METHOD  : {EVAL_METHOD}  (needs unmerged SingleRegressionTrees)")
    print(f"  N_BITS       : {N_BITS}")
    print("=" * 60)

    for ad_type in AD_TYPES:
        print(f"\n>>> {ad_type}")
        export_one(ad_type, args.tag)

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
