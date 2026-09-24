#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:15:00 2026

@author: nibio
"""






#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HSIwheat — unchanged measured40-MLP transfer to reconstructed HSI
=================================================================

No new yield model is trained in this script.

The measured40 yield model produced by 02_run_representative_triplets.py and
its saved measured40 training scaler are kept fixed and applied directly to:
  1. genuine measured40 test features
  2. the 10 representative direct-MLP reconstructed40 feature sets
  3. the T12 direct-MLP reconstructed40 feature set

Every representation is transformed with the SAME measured40 checkpoint scaler:
    Z = (X - measured40_train_mean) / measured40_train_std

There is no fine-tuning, no reconstructed-data normalization fitting, and no
model selection using reconstructed inputs or test labels.

This release script intentionally excludes the older Gram/SparseHSR diagnostic
branches because they are not part of the final manuscript.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =============================================================================
# SETTINGS
# =============================================================================

ROOT = Path("/home/nibio/HSIwheat_40")

STAGE1_ROOT = ROOT / "original190_paper_reproduction"
META_PATH = STAGE1_ROOT / "subplot_metadata.csv"
SPLIT_PATH = STAGE1_ROOT / "split_assignments.csv"

EXP2_OUT = ROOT / "experiment2_direct_mlp_3to40"
MEASURED40_FEATURE_CACHE = EXP2_OUT / "cache" / "measured40_features_frozen.npy"
MEASURED40_CHECKPOINT_DIR = EXP2_OUT / "measured40_yield" / "checkpoints"

OUT = ROOT / "frozen_measured40_model_transfer"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PRED_BATCH = 4096

EVALUATE_ALL_MEASURED40_SEEDS = True


# =============================================================================
# EXACT YIELD NETWORK
# =============================================================================

class YieldMLP(nn.Module):
    def __init__(self, input_dim=81, hidden_units=(10, 10, 10, 10)):
        super(YieldMLP, self).__init__()

        layers = []
        previous = int(input_dim)

        for width in hidden_units:
            layers.append(nn.Linear(previous, int(width)))
            layers.append(nn.ReLU())
            previous = int(width)

        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).reshape(-1)


# =============================================================================
# HELPERS
# =============================================================================

def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_frozen_meta():
    if not META_PATH.exists() or not SPLIT_PATH.exists():
        raise FileNotFoundError(
            "Could not find frozen metadata/split.\n"
            f"  {META_PATH}\n"
            f"  {SPLIT_PATH}"
        )

    meta = pd.read_csv(META_PATH)
    split = pd.read_csv(SPLIT_PATH)

    key_cols = ["field", "plot", "subplot_index"]

    for c in key_cols:
        if c not in meta.columns or c not in split.columns:
            raise ValueError(f"Missing key column: {c}")

    if "split" not in split.columns:
        raise ValueError(f"{SPLIT_PATH} has no split column")

    meta = meta.merge(
        split[key_cols + ["split"]],
        on=key_cols,
        how="left",
        validate="one_to_one",
    ).reset_index(drop=True)

    if meta["split"].isna().any():
        raise RuntimeError("Some rows have no frozen split.")

    return meta


def get_test_indices(meta):
    idx = np.where(meta["split"].astype(str).to_numpy() == "test")[0]
    if len(idx) == 0:
        raise RuntimeError("Frozen test split is empty.")
    return idx


def calc_metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def load_model_checkpoint(checkpoint_path, device):
    ckpt = safe_torch_load(checkpoint_path, device)

    required = [
        "model_state_dict",
        "feature_mean",
        "feature_std",
        "input_dim",
        "model_seed",
    ]

    for key in required:
        if key not in ckpt:
            raise KeyError(f"{checkpoint_path} missing {key}")

    if int(ckpt["input_dim"]) != 81:
        raise RuntimeError(
            f"Measured40 checkpoint input_dim must be 81, got {ckpt['input_dim']}"
        )

    hidden_units = tuple(
        int(x) for x in ckpt.get("hidden_units", [10, 10, 10, 10])
    )

    model = YieldMLP(
        input_dim=81,
        hidden_units=hidden_units,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt


def standardize_with_measured40_scaler(X, ckpt):
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 2 or X.shape[1] != 81:
        raise ValueError(f"Expected N x 81 features, got {X.shape}")

    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32).reshape(-1)

    if mean.shape != (81,) or std.shape != (81,):
        raise RuntimeError(
            f"Measured40 scaler has wrong shape: mean={mean.shape}, std={std.shape}"
        )

    std = std.copy()
    std[std < 1e-12] = 1.0

    return ((X - mean[None, :]) / std[None, :]).astype(np.float32)


def predict(model, Z, idx, device):
    outputs = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(idx), PRED_BATCH):
            ii = idx[start:start + PRED_BATCH]
            xb = torch.from_numpy(Z[ii]).float().to(device)
            outputs.append(model(xb).cpu().numpy())

    return np.concatenate(outputs, axis=0)


def infer_primary_measured40_seed():
    # Select the reference seed by measured40 validation RMSE only.
    candidates = sorted(
        MEASURED40_CHECKPOINT_DIR.glob("seed_*_bestval.pth")
    )

    if not candidates:
        raise FileNotFoundError(
            f"No measured40 checkpoints in {MEASURED40_CHECKPOINT_DIR}. "
            "Run 02_run_representative_triplets.py first."
        )

    rows = []

    for checkpoint_path in candidates:
        ckpt = safe_torch_load(checkpoint_path, "cpu")
        rows.append(
            (
                float(ckpt["best_validation_RMSE"]),
                int(ckpt["model_seed"]),
            )
        )

    rows.sort()
    return rows[0][1]

def label_from_path(path, prefix):
    try:
        rel = path.parent.relative_to(EXP2_OUT)
        text = str(rel)
    except ValueError:
        text = str(path.parent)

    return prefix + "::" + text.replace("\\", "/")


def verify_row_alignment(name, X, X_measured40):
    """
    Every 81-D feature vector is:
        40 means + 40 standard deviations + SL area.

    The final area feature must be identical row-by-row if cache ordering is
    compatible with the frozen metadata.
    """
    if X.shape != X_measured40.shape:
        raise RuntimeError(
            f"{name}: shape mismatch {X.shape} vs {X_measured40.shape}"
        )

    max_area_diff = float(
        np.max(
            np.abs(
                X[:, -1].astype(np.float64)
                - X_measured40[:, -1].astype(np.float64)
            )
        )
    )

    if max_area_diff > 1e-6:
        raise RuntimeError(
            f"{name}: row alignment check failed; "
            f"max area difference={max_area_diff}"
        )


# =============================================================================
# LOAD FROZEN DATA + MEASURED40
# =============================================================================

OUT.mkdir(parents=True, exist_ok=True)
device = torch.device(DEVICE)

print("=" * 110)
print("FROZEN MEASURED40 YIELD MODEL -> RECONSTRUCTED40 TRANSFER")
print("=" * 110)
print("Device:", device)

meta = load_frozen_meta()

if "subplot_yield" not in meta.columns:
    raise ValueError("Frozen metadata has no subplot_yield column.")

y = meta["subplot_yield"].to_numpy(dtype=np.float32)
test_idx = get_test_indices(meta)

if not MEASURED40_FEATURE_CACHE.exists():
    raise FileNotFoundError(
        f"Missing measured40 feature cache: {MEASURED40_FEATURE_CACHE}. "
        "Run 02_run_representative_triplets.py first."
    )

X_measured40 = np.load(
    MEASURED40_FEATURE_CACHE
).astype(
    np.float32,
    copy=False,
)

if X_measured40.shape != (len(meta), 81):
    raise RuntimeError(
        f"Measured40 features have shape {X_measured40.shape}; "
        f"expected {(len(meta), 81)}"
    )

print("\nFrozen dataset:")
print("  all subplots =", f"{len(meta):,}")
print("  test         =", f"{len(test_idx):,}")
print("  measured40   =", X_measured40.shape)


# =============================================================================
# DISCOVER READY RECONSTRUCTED REPRESENTATIONS
# =============================================================================

representations = [
    {
        "name": "measured40",
        "kind": "measured40_reference",
        "path": str(MEASURED40_FEATURE_CACHE),
        "key": "measured40",
        "X": X_measured40,
    }
]


# ---- Direct MLP: automatically find all 10 representative triplets + T12 ----

direct_files = sorted(EXP2_OUT.rglob("yield_features.npz"))

print("\nDirect-MLP yield_features.npz files found:", len(direct_files))

for path in direct_files:
    data = np.load(path)

    if "reconstructed40" not in data.files:
        continue

    X = data["reconstructed40"].astype(np.float32, copy=False)

    if X.ndim != 2 or X.shape[1] != 81:
        print("  SKIP wrong shape:", path, X.shape)
        continue

    name = label_from_path(path, "direct_mlp")
    verify_row_alignment(name, X, X_measured40)

    representations.append(
        {
            "name": name,
            "kind": "direct_mlp_reconstructed40",
            "path": str(path),
            "key": "reconstructed40",
            "X": X,
        }
    )

    print("  +", name)


n_direct = sum(
    rep["kind"] == "direct_mlp_reconstructed40"
    for rep in representations
)

if n_direct != 11:
    raise RuntimeError(
        f"Expected exactly 11 direct-MLP reconstructed40 feature sets, "
        f"but found {n_direct}. Run scripts 02 and 03 and remove stale/duplicate "
        "yield_features.npz files under experiment2_direct_mlp_3to40."
    )


# The final manuscript evaluates the 11 direct-MLP reconstructions only.
# Legacy Gram/SparseHSR feature discovery is intentionally omitted here.


# =============================================================================
# PRIMARY MODEL — PRE-SELECTED USING MEASURED40 VALIDATION ONLY
# =============================================================================

primary_seed = infer_primary_measured40_seed()

primary_checkpoint = (
    MEASURED40_CHECKPOINT_DIR
    / f"seed_{primary_seed}_bestval.pth"
)

if not primary_checkpoint.exists():
    raise FileNotFoundError(
        f"Primary measured40 checkpoint missing: {primary_checkpoint}"
    )

model, ckpt = load_model_checkpoint(
    primary_checkpoint,
    device,
)

print("\n" + "=" * 110)
print("PRIMARY MEASURED40 MODEL")
print("=" * 110)
print("checkpoint =", primary_checkpoint)
print("seed       =", int(ckpt["model_seed"]))
print("best epoch =", int(ckpt.get("best_epoch", -1)))
print(
    "measured40 validation RMSE =",
    float(ckpt.get("best_validation_RMSE", np.nan)),
)
print("NO RETRAINING. SAME MODEL + SAME MEASURED40 SCALER FOR EVERY INPUT.")


# =============================================================================
# PRIMARY TRANSFER RESULTS
# =============================================================================

primary_rows = []
prediction_rows = []

for rep in representations:
    Z = standardize_with_measured40_scaler(
        rep["X"],
        ckpt,
    )

    pred = predict(
        model,
        Z,
        test_idx,
        device,
    )

    yt = y[test_idx]
    m = calc_metrics(yt, pred)

    primary_rows.append(
        {
            "representation": rep["name"],
            "kind": rep["kind"],
            "source_file": rep["path"],
            "source_key": rep["key"],
            "measured40_model_seed": int(ckpt["model_seed"]),
            "R2": m["R2"],
            "RMSE_g": m["RMSE"],
            "MAE_g": m["MAE"],
        }
    )

    for j, row_idx in enumerate(test_idx):
        row = meta.iloc[row_idx]

        prediction_rows.append(
            {
                "representation": rep["name"],
                "row_index": int(row_idx),
                "field": row.get("field", ""),
                "plot": row.get("plot", ""),
                "subplot_index": row.get("subplot_index", ""),
                "observed_yield_g": float(yt[j]),
                "predicted_yield_g": float(pred[j]),
            }
        )

    print(
        f"{rep['name']:<80s} "
        f"R2={m['R2']:.4f} | "
        f"RMSE={m['RMSE']:.4f} g | "
        f"MAE={m['MAE']:.4f} g"
    )


primary_df = pd.DataFrame(primary_rows)

baseline = primary_df.loc[
    primary_df["representation"] == "measured40"
].iloc[0]

primary_df["R2_change_vs_measured40"] = (
    primary_df["R2"] - float(baseline["R2"])
)

primary_df["delta_R2_degradation"] = (
    float(baseline["R2"]) - primary_df["R2"]
)

primary_df["delta_RMSE_g_vs_measured40"] = (
    primary_df["RMSE_g"] - float(baseline["RMSE_g"])
)

primary_df["delta_MAE_g_vs_measured40"] = (
    primary_df["MAE_g"] - float(baseline["MAE_g"])
)

primary_path = OUT / "frozen_measured40_primary_transfer.csv"
prediction_path = OUT / "frozen_measured40_primary_predictions.csv"

primary_df.to_csv(primary_path, index=False)
pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)

print("\n" + "=" * 110)
print("PRIMARY FROZEN-MODEL TRANSFER")
print("=" * 110)

print(
    primary_df[
        [
            "representation",
            "R2",
            "RMSE_g",
            "MAE_g",
            "delta_R2_degradation",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# =============================================================================
# SECONDARY: ALL EXISTING MEASURED40 MODEL SEEDS, STILL ZERO TRAINING
# =============================================================================

if EVALUATE_ALL_MEASURED40_SEEDS:
    checkpoint_files = sorted(
        MEASURED40_CHECKPOINT_DIR.glob("seed_*_bestval.pth")
    )

    if not checkpoint_files:
        raise FileNotFoundError(
            f"No measured40 checkpoints in {MEASURED40_CHECKPOINT_DIR}"
        )

    all_rows = []

    print("\n" + "=" * 110)
    print("ALL PRE-EXISTING MEASURED40 SEED MODELS")
    print("=" * 110)
    print("n checkpoints =", len(checkpoint_files))

    for checkpoint_path in checkpoint_files:
        model_i, ckpt_i = load_model_checkpoint(
            checkpoint_path,
            device,
        )

        seed_i = int(ckpt_i["model_seed"])

        for rep in representations:
            Z = standardize_with_measured40_scaler(
                rep["X"],
                ckpt_i,
            )

            pred = predict(
                model_i,
                Z,
                test_idx,
                device,
            )

            m = calc_metrics(
                y[test_idx],
                pred,
            )

            all_rows.append(
                {
                    "measured40_model_seed": seed_i,
                    "measured40_best_validation_RMSE": float(
                        ckpt_i.get("best_validation_RMSE", np.nan)
                    ),
                    "representation": rep["name"],
                    "kind": rep["kind"],
                    "R2": m["R2"],
                    "RMSE_g": m["RMSE"],
                    "MAE_g": m["MAE"],
                }
            )

        print("  finished measured40 seed", seed_i)

    all_df = pd.DataFrame(all_rows)

    all_path = OUT / "frozen_measured40_all_models_transfer.csv"
    all_df.to_csv(all_path, index=False)

    summary_df = (
        all_df
        .groupby(
            ["representation", "kind"],
            as_index=False,
        )
        .agg(
            R2_mean=("R2", "mean"),
            R2_sd=("R2", lambda x: np.std(x, ddof=0)),
            RMSE_g_mean=("RMSE_g", "mean"),
            RMSE_g_sd=("RMSE_g", lambda x: np.std(x, ddof=0)),
            MAE_g_mean=("MAE_g", "mean"),
            MAE_g_sd=("MAE_g", lambda x: np.std(x, ddof=0)),
            n_models=("measured40_model_seed", "count"),
        )
    )

    summary_path = OUT / "frozen_measured40_all_models_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nMean +/- SD across existing measured40 model seeds:")
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


# =============================================================================
# AUDIT MANIFEST
# =============================================================================

manifest = pd.DataFrame(
    [
        {
            "representation": rep["name"],
            "kind": rep["kind"],
            "source_file": rep["path"],
            "source_key": rep["key"],
            "n_rows": int(rep["X"].shape[0]),
            "n_features": int(rep["X"].shape[1]),
            "primary_measured40_model_seed": int(primary_seed),
            "primary_checkpoint": str(primary_checkpoint),
            "scaler": "measured40 checkpoint scaler only",
            "yield_model_retrained": False,
            "scaler_refit_on_reconstruction": False,
        }
        for rep in representations
    ]
)

manifest_path = OUT / "transfer_run_manifest.csv"
manifest.to_csv(manifest_path, index=False)

print("\n" + "=" * 110)
print("DONE — ZERO NEW TRAINING")
print("=" * 110)
print("Primary results:")
print(" ", primary_path)
print("Primary test predictions:")
print(" ", prediction_path)

if EVALUATE_ALL_MEASURED40_SEEDS:
    print("All measured40 seed-model results:")
    print(" ", OUT / "frozen_measured40_all_models_transfer.csv")
    print("Mean +/- SD:")
    print(" ", OUT / "frozen_measured40_all_models_summary.csv")

print("Manifest:")
print(" ", manifest_path)









