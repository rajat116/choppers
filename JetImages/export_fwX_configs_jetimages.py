#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from fwX import get_regression_BDT_from_TMVA

# ---- CONFIG ----
AD_TYPE   = "KL"
BASE_DIR  = Path("/Users/rajat/Desktop/HPE_Work/fwX-dev/examples/chopper/JetImages_for_choppers/All_targets/BDT_students_out_all/vae_3layer_latent6_clipped/")
TMVA_XML  = Path(".") / f"dataset_{AD_TYPE}/weights/TMVARegression_{AD_TYPE}_BDT.weights.xml"
TMVA_ROOT = BASE_DIR / f"tmva_output__{AD_TYPE}.root"
BKG_CSV   = BASE_DIR / f"tmva_inputs"/ f"vae_tmva_input__Background__{AD_TYPE}_AD_scores.csv"

# fwX settings
N_BITS        = 8           # fixed-point bits
TREE_PATTERN  = 80           # merge 100 trees -> 10 merged trees
EVAL_METHOD   = 2
BIN_ENGINE    = "LUBE"
P_HIGH        = 0.999        # clip upper tail
EPS_ABS       = 1e-6         # minimum span if flat
EPS_REL       = 1e-3         # relative span if values not tiny

def widen_span(lo: float, hi: float) -> tuple[float, float]:
    """Ensure hi>lo: if flat, bump hi by a tiny epsilon."""
    if hi > lo:
        return lo, hi
    # if completely flat, create a tiny positive span
    bump = max(EPS_ABS, EPS_REL * max(abs(lo), abs(hi), 1.0))
    return lo, lo + bump

def main():
    # sanity checks
    for p in (TMVA_XML, TMVA_ROOT, BKG_CSV):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    # derive ranges from background
    df = pd.read_csv(BKG_CSV)
    feats = [c for c in df.columns if c.startswith("feature_")]
    if "target" not in df.columns:
        raise RuntimeError(f"'target' column missing in {BKG_CSV}")

    # robust per-feature ranges
    mins = df[feats].min()
    qhi  = df[feats].quantile(P_HIGH)
    maxi = df[feats].max()

    cut_variable_ranges = {}
    n_fixed = 0
    for v in feats:
        lo = float(mins[v])
        hi = float(qhi[v])
        if hi <= lo:
            # fall back to absolute max; if still flat, widen
            hi = float(maxi[v])
        lo, hi = widen_span(lo, hi)
        if hi - lo <= EPS_ABS * 10:
            n_fixed += 1
        cut_variable_ranges[v] = (lo, hi)

    # target range (also robust + widened if flat)
    t_lo = float(df["target"].min())
    t_hi = float(df["target"].quantile(P_HIGH))
    if t_hi <= t_lo:
        t_hi = float(df["target"].max())
    t_lo, t_hi = widen_span(t_lo, t_hi)

    print(f"\n🚀 Exporting fwX config for: {AD_TYPE}")
    print(f"   features = {len(feats)}  | zero-span fixed = {n_fixed}")
    print(f"   target range (clipped) = [{t_lo}, {t_hi}]")

    # load TMVA model (dataset_{AD_TYPE} matches your ROOT keys)
    bdt_object = get_regression_BDT_from_TMVA(
        xml_filepath=str(TMVA_XML),
        root_filepath=str(TMVA_ROOT),
        tmva_dataset_name=f"dataset_{AD_TYPE}",
        tmva_bdt_name="BDT",
    )

    # build fwX set (no n_testpoints here)
    set_ = bdt_object.get_set(
        tree_pattern=TREE_PATTERN,
        target_variable_precision=N_BITS,
        cut_precisions=N_BITS,
        bin_engine=BIN_ENGINE,
        cut_variable_ranges=cut_variable_ranges,
        target_variable_range=[t_lo, t_hi],
        evaluation_method=EVAL_METHOD,
    )

    # save artifacts
    cfg = f"fwX-config_{AD_TYPE}_prec{N_BITS}.json"
    tpt = f"fwX-testpoints_{AD_TYPE}_prec{N_BITS}.txt"
    set_.save_config(cfg)
    set_.save_testpoints(tpt)

    print(f"✅ Saved {cfg}")
    print(f"✅ Saved {tpt}")

if __name__ == "__main__":
    main()
