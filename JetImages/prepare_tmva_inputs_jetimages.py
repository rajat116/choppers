#!/usr/bin/env python3
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd

AD_TYPES = ["MSE", "KL", "MSE_KL"]       # what we will produce
IMG_SHAPE = (24, 24)                     # H, W
N_FEATS = IMG_SHAPE[0] * IMG_SHAPE[1]    # 576

# ---------- utilities ----------
def log1p_transform(x: np.ndarray) -> np.ndarray:
    # x expected shape: (N, 24, 24) or (N, 24, 24, 1)
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 3 or x.shape[1:] != IMG_SHAPE:
        raise ValueError(f"Unexpected image shape {x.shape}, expected (*,24,24[,1])")
    return np.log1p(x)

def flatten_images(x: np.ndarray) -> np.ndarray:
    # x expected shape: (N, 24, 24)
    return x.reshape(x.shape[0], -1)

def write_tmva_csv(features: np.ndarray, target: np.ndarray, out_path: Path):
    cols = [f"feature_{i}" for i in range(N_FEATS)] + ["target"]
    df = pd.DataFrame(
        np.column_stack([features, target.astype(np.float32)]),
        columns=cols
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Wrote {out_path}  (N={len(df)})")

def make_outname(outdir: Path, tag: str, ad_type: str) -> Path:
    return outdir / f"vae_tmva_input__{tag}__{ad_type}_AD_scores.csv"

# ---------- main routine ----------
def main():
    ap = argparse.ArgumentParser(
        description="Prepare TMVA-ready CSVs from jet image npy files + VAE targets."
    )
    ap.add_argument("--in-dir", required=True, type=Path,
                    help="Directory containing npy + target CSVs")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Where to write TMVA input CSVs")
    ap.add_argument("--transform", default="log1p", choices=["log1p","none"],
                    help="Pixel transform to apply (default: log1p)")
    ap.add_argument("--signals", nargs="*", default=["t","w","z"],
                    help="Signal suffixes present (default: t w z)")
    args = ap.parse_args()

    in_dir  = args.in_dir
    out_dir = args.out_dir

    # input files (expected names per your message)
    f_train = in_dir / "jetImages_train_data.npy"
    f_test  = in_dir / "jetImages_test_data.npy"
    f_val   = in_dir / "jetImages_val_data.npy"

    f_tar_train = in_dir / "vae_5layer_latent6_log_train_targets.csv"
    f_tar_test  = in_dir / "vae_5layer_latent6_log_test_targets.csv"
    f_tar_val   = in_dir / "vae_5layer_latent6_log_val_targets.csv"

    # sanity checks
    for p in [f_train, f_test, f_val, f_tar_train, f_tar_test, f_tar_val]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    # load background (q/g) splits
    X_train = np.load(f_train)  # (N,24,24,1)
    X_test  = np.load(f_test)
    X_val   = np.load(f_val)

    # transform
    if args.transform == "log1p":
        X_train = log1p_transform(X_train)
        X_test  = log1p_transform(X_test)
        X_val   = log1p_transform(X_val)
    elif args.transform == "none":
        # still squeeze the last channel if present
        if X_train.ndim == 4 and X_train.shape[-1] == 1:
            X_train = X_train[...,0]
            X_test  = X_test[...,0]
            X_val   = X_val[...,0]

    # flatten
    F_train = flatten_images(X_train)  # (N,576)
    F_test  = flatten_images(X_test)
    F_val   = flatten_images(X_val)

    # targets
    T_train = pd.read_csv(f_tar_train)[AD_TYPES]
    T_test  = pd.read_csv(f_tar_test)[AD_TYPES]
    T_val   = pd.read_csv(f_tar_val)[AD_TYPES]

    # align lengths
    if not (len(F_train) == len(T_train) and len(F_test) == len(T_test) and len(F_val) == len(T_val)):
        raise RuntimeError(
            f"Length mismatch: train({len(F_train)} vs {len(T_train)}), "
            f"test({len(F_test)} vs {len(T_test)}), val({len(F_val)} vs {len(T_val)})"
        )

    # write CSVs for Background/Test/Val for each AD type
    for ad in AD_TYPES:
        write_tmva_csv(F_train, T_train[ad].to_numpy(), make_outname(out_dir, "Background", ad))
        write_tmva_csv(F_test,  T_test[ad].to_numpy(),  make_outname(out_dir, "Test", ad))
        write_tmva_csv(F_val,   T_val[ad].to_numpy(),   make_outname(out_dir, "Val", ad))

    # signals
    for sfx in args.signals:
        f_sig = in_dir / f"jetImages_signal_{sfx}.npy"
        if not f_sig.exists():
            print(f"⚠️  Missing signal npy: {f_sig} — skipping")
            continue

        X_sig = np.load(f_sig)
        if args.transform == "log1p":
            X_sig = log1p_transform(X_sig)
        elif args.transform == "none" and X_sig.ndim == 4 and X_sig.shape[-1] == 1:
            X_sig = X_sig[...,0]
        F_sig = flatten_images(X_sig)

        # For signals we do not have teacher targets; fill a dummy target (0.0).
        dummy = np.zeros((len(F_sig),), dtype=np.float32)
        for ad in AD_TYPES:
            write_tmva_csv(F_sig, dummy, make_outname(out_dir, f"signal_{sfx}", ad))

    print("\n✅ Done. CSVs are ready for TMVA training/inference.")

if __name__ == "__main__":
    main()
