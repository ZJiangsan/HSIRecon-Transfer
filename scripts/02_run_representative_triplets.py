#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HSIwheat Experiment 2
=====================

QUESTION
--------
Does the information content of the three physically observed bands still
matter after a simple supervised 3-band -> 40-band reconstruction?

THIS SCRIPT DELIBERATELY DOES NOT USE:
    - Gram matrices
    - decomposition / encoder / decoder
    - residual learning
    - the previous SparseHSR mapper

RECONSTRUCTION MODEL
--------------------
A single fixed direct MLP is used for every triplet:

    3 -> 128 -> 128 -> 40

Training objective:
    MSE(predicted 40-band spectrum, measured 40-band spectrum)

Stopping rule:
    stop when validation MSE has not improved for 40 consecutive epochs
    (maximum 500 epochs)

IMPORTANT EXPERIMENTAL CONTROL
------------------------------
ONLY the three input wavelengths change between triplets.
Everything else stays fixed:
    - same MLP architecture
    - same reconstruction train/validation plots
    - same optimizer and loss
    - same early-stopping rule
    - same frozen HSIwheat subplot masks
    - same frozen train/validation/test yield split
    - same yield feature extraction
    - same yield DNN
    - same yield seeds

RECONSTRUCTION SPLIT
--------------------
The 50 frozen downstream TEST plots are never used to fit the reconstruction
MLP. The remaining development plots are divided once into reconstruction
train/validation plots. The same split is used for all triplets.

DOWNSTREAM YIELD PROTOCOL
-------------------------
This script reuses the frozen HSIwheat subplot metadata/masks/split.
For each subplot:

    sparse3:
        3 means + 3 std + SL pixel area = 7 features

    reconstructed40:
        40 means + 40 std + SL pixel area = 81 features

    measured40:
        40 means + 40 std + SL pixel area = 81 features

Yield DNN:
    input -> 10 -> 10 -> 10 -> 10 -> 1
    ReLU
    Adam
    MSE
    100 epochs
    batch size 32
    best validation-RMSE checkpoint
    seeds 0..9

OUTPUT
------
/home/nibio/HSIwheat_40/experiment2_direct_mlp_3to40/

Main file:
    experiment2_results.csv

The measured40 baseline is also saved as ten validation-selected checkpoints
under:
    experiment2_direct_mlp_3to40/measured40_yield/checkpoints/

These checkpoints are reused unchanged by the transfer analyses.

Columns include:
    - direct sparse3 yield R2
    - reconstruction test RMSE
    - reconstruction test SAM
    - reconstructed40 yield R2
    - measured40 baseline R2

The first run builds a reusable all-valid-pixel cache directly from the
measured HSI40 cubes. No old decomposition output is required.
"""

from __future__ import print_function

import copy
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

ROOT = Path("/home/nibio/HSIwheat_40")
FIELDS = ("C3", "C4", "C9")

# Measured 40-band cubes:
MEASURED40_ROOTS = {
    field: ROOT / field / "hsi40"
    for field in FIELDS
}

WAVELENGTHS40_PATH = ROOT / "wavelengths_40.npy"

# Frozen downstream setup from the existing successful yield experiment:
STAGE1_ROOT = ROOT / "original190_paper_reproduction"
META_PATH = STAGE1_ROOT / "subplot_metadata.csv"
SPLIT_PATH = STAGE1_ROOT / "split_assignments.csv"
SL_MASK_ROOT = STAGE1_ROOT / "sl_masks"

# New experiment output. Nothing from the old Gram/decomposition mapper is used.
OUT = ROOT / "experiment2_direct_mlp_3to40"
CACHE_DIR = OUT / "cache"

PIXEL_MATRIX_PATH = CACHE_DIR / "all_valid_pixels_40.npy"
CUBE_RANGES_PATH = CACHE_DIR / "cube_ranges.csv"
RECON_SPLIT_PATH = CACHE_DIR / "reconstruction_plot_split.csv"

# -------------------------------------------------------------------------
# Representative triplets.
# These are NOT selected by yield performance.
# Each nominal wavelength is snapped to the nearest available HSIwheat band.
# -------------------------------------------------------------------------
NOMINAL_TRIPLETS = [
    ("current_461_551_802",       (461.0, 551.0, 802.0)),
    ("rgb_like_461_551_651",      (461.0, 551.0, 651.0)),
    ("visible_spread_431_551_681",(431.0, 551.0, 681.0)),
    ("visible_cluster_461_501_551",(461.0, 501.0, 551.0)),
    ("green_rededge_nir",         (551.0, 711.0, 802.0)),
    ("red_rededge_nir",           (651.0, 711.0, 802.0)),
    ("blue_rededge_nir",          (461.0, 711.0, 852.0)),
    ("rededge_cluster",           (681.0, 711.0, 741.0)),
    ("nir_cluster",               (802.0, 852.0, 902.0)),
    ("wide_spectral_span",        (431.0, 711.0, 952.0)),
]

# -------------------------------------------------------------------------
# Reconstruction split/training
# -------------------------------------------------------------------------
RECON_SPLIT_SEED = 2026
RECON_VAL_FRACTION = 0.10

RECON_SEED = 42
RECON_MAX_EPOCHS = 500
RECON_PATIENCE = 40
RECON_MIN_DELTA = 1e-8

RECON_BATCH_SIZE = 16384
RECON_SAMPLES_PER_EPOCH = 1000000
RECON_VAL_MAX_PIXELS = 300000

RECON_LR = 1e-3
RECON_WEIGHT_DECAY = 1e-5

# Inference chunk size for full cubes.
INFERENCE_CHUNK = 131072

# Loading the all-pixel matrix into RAM is much faster for random training
# batches. The matrix is around 1.4 GB for ~8.9 million x 40 float32 pixels.
LOAD_PIXEL_MATRIX_IN_RAM = True

# -------------------------------------------------------------------------
# Frozen yield-prediction protocol
# -------------------------------------------------------------------------
YIELD_HIDDEN_UNITS = (10, 10, 10, 10)
YIELD_EPOCHS = 100
YIELD_BATCH_SIZE = 32
YIELD_LR = 1e-3
YIELD_MODEL_SEEDS = tuple(range(10))

# Set to e.g. (0,) for a fast pilot. For the manuscript use range(10).
# YIELD_MODEL_SEEDS = (0,)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

EPS = 1e-12


# =============================================================================
# 2. REPRODUCIBILITY
# =============================================================================

def seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# =============================================================================
# 3. SMALL HELPERS
# =============================================================================

def normalize_plot_id(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def key_of(field, plot):
    return (str(field).strip(), normalize_plot_id(plot))


def nearest_band_index(wavelengths, target_nm):
    wavelengths = np.asarray(wavelengths).reshape(-1)
    return int(np.argmin(np.abs(wavelengths - float(target_nm))))


def snap_triplet(wavelengths, nominal):
    idx = tuple(
        nearest_band_index(wavelengths, w)
        for w in nominal
    )

    if len(set(idx)) != 3:
        raise RuntimeError(
            "Triplet {} maps to duplicate measured bands: {}".format(
                nominal, idx
            )
        )

    actual = tuple(float(wavelengths[i]) for i in idx)

    # Keep spectral order.
    order = np.argsort(np.asarray(actual))
    idx = tuple(int(np.asarray(idx)[order][i]) for i in range(3))
    actual = tuple(float(np.asarray(actual)[order][i]) for i in range(3))

    return idx, actual


def map_npy_by_stem(root):
    files = sorted(Path(root).glob("*.npy"))
    if not files:
        files = sorted(Path(root).rglob("*.npy"))

    if not files:
        raise RuntimeError("No .npy files found under {}".format(root))

    mapping = {}
    for path in files:
        stem = normalize_plot_id(path.stem)
        if stem in mapping:
            raise RuntimeError(
                "Duplicate .npy stem under {}: {}".format(root, stem)
            )
        mapping[stem] = path

    return mapping


# =============================================================================
# 4. LOAD THE FROZEN YIELD METADATA/SPLIT
# =============================================================================

def load_frozen_metadata():
    if not META_PATH.exists() or not SPLIT_PATH.exists():
        raise FileNotFoundError(
            "Could not find frozen Stage-1 metadata/split.\n"
            "Run 01_reproduce_frozen_hsiwheat_benchmark.py first.\n"
            "Expected:\n"
            "  {}\n  {}".format(META_PATH, SPLIT_PATH)
        )

    print("\nLoading frozen Stage-1 metadata/split:")
    print(" ", META_PATH)
    print(" ", SPLIT_PATH)

    meta = pd.read_csv(META_PATH)
    split = pd.read_csv(SPLIT_PATH)

    key_cols = ["field", "plot", "subplot_index"]

    for c in key_cols:
        if c not in meta.columns:
            raise ValueError("{} missing column {}".format(META_PATH, c))
        if c not in split.columns:
            raise ValueError("{} missing column {}".format(SPLIT_PATH, c))

    if "split" not in split.columns:
        raise ValueError("{} missing column split".format(SPLIT_PATH))

    meta = meta.merge(
        split[key_cols + ["split"]],
        on=key_cols,
        how="left",
        validate="one_to_one",
    )

    if meta["split"].isna().any():
        raise RuntimeError("Some subplot rows have no frozen split.")

    required = [
        "field",
        "plot",
        "subplot_index",
        "subplot_yield",
        "row0",
        "row1",
        "col0",
        "col1",
        "n_SL_pixels",
        "split",
    ]

    for c in required:
        if c not in meta.columns:
            raise ValueError("Frozen metadata missing required column: {}".format(c))

    meta = meta.reset_index(drop=True)
    meta["field"] = meta["field"].astype(str)
    meta["plot"] = meta["plot"].map(normalize_plot_id)

    if not SL_MASK_ROOT.exists():
        raise FileNotFoundError(
            "Frozen SL-mask directory not found:\n{}".format(SL_MASK_ROOT)
        )

    print("\nFrozen downstream dataset:")
    print("  subplots =", "{:,}".format(len(meta)))
    print(meta["split"].value_counts().to_string())

    return meta


def get_yield_indices(meta):
    split_arr = meta["split"].astype(str).to_numpy()

    train_idx = np.where(split_arr == "train")[0]
    val_idx = np.where(split_arr == "validation")[0]
    test_idx = np.where(split_arr == "test")[0]

    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise RuntimeError("Frozen train/validation/test split is incomplete.")

    return train_idx, val_idx, test_idx


# =============================================================================
# 5. BUILD A CLEAN ALL-VALID-PIXEL CACHE DIRECTLY FROM HSI40
# =============================================================================

def discover_measured40_files():
    maps = {}
    records = []

    for field in FIELDS:
        maps[field] = map_npy_by_stem(MEASURED40_ROOTS[field])

        for plot, path in sorted(maps[field].items()):
            records.append((field, plot, path))

        print(
            "{}: {} measured40 cubes".format(
                field, len(maps[field])
            )
        )

    return maps, records


def build_or_load_pixel_cache(records):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if PIXEL_MATRIX_PATH.exists() and CUBE_RANGES_PATH.exists():
        print("\nUsing existing direct-MLP pixel cache:")
        print(" ", PIXEL_MATRIX_PATH)
        print(" ", CUBE_RANGES_PATH)

        matrix = np.load(
            PIXEL_MATRIX_PATH,
            mmap_mode=None if LOAD_PIXEL_MATRIX_IN_RAM else "r",
        )

        if matrix.ndim != 2 or matrix.shape[1] != 40:
            raise RuntimeError(
                "Invalid cached pixel matrix shape: {}".format(matrix.shape)
            )

        rows = pd.read_csv(CUBE_RANGES_PATH)
        rows["field"] = rows["field"].astype(str)
        rows["plot"] = rows["plot"].map(normalize_plot_id)

        print("  pixel matrix shape =", matrix.shape)
        print("  cube ranges        =", len(rows))

        return matrix, rows

    print("\nBuilding all-valid-pixel cache directly from measured HSI40...")
    print("PASS 1/2: count valid pixels")

    audit = []
    total = 0

    for i, (field, plot, path) in enumerate(records, start=1):
        cube = np.load(path, mmap_mode="r")

        if cube.ndim != 3 or cube.shape[2] != 40:
            raise ValueError(
                "{}: expected HxWx40, got {}".format(path, cube.shape)
            )

        flat = cube.reshape(-1, 40)
        valid = np.any(flat != 0, axis=1)
        n_valid = int(valid.sum())

        start = total
        end = total + n_valid

        audit.append(
            {
                "cube_index": i - 1,
                "field": field,
                "plot": plot,
                "path": str(path),
                "height": int(cube.shape[0]),
                "width": int(cube.shape[1]),
                "n_valid": n_valid,
                "start": start,
                "end": end,
            }
        )

        total = end

        if i == 1 or i % 100 == 0 or i == len(records):
            print(
                "  {}/{} | {}/{} | valid={:,} | total={:,}".format(
                    i, len(records), field, plot, n_valid, total
                )
            )

    print("\nPASS 2/2: write {:,} x 40 float32 matrix".format(total))

    matrix_out = np.lib.format.open_memmap(
        PIXEL_MATRIX_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(total, 40),
    )

    cursor = 0

    for i, row in enumerate(audit, start=1):
        cube = np.load(row["path"], mmap_mode="r")
        flat = cube.reshape(-1, 40)
        valid = np.any(flat != 0, axis=1)
        px = np.asarray(flat[valid], dtype=np.float32)

        n = px.shape[0]
        matrix_out[cursor:cursor + n] = px
        cursor += n

        if i == 1 or i % 100 == 0 or i == len(audit):
            print("  wrote {}/{} cubes".format(i, len(audit)))

    matrix_out.flush()
    del matrix_out

    pd.DataFrame(audit).to_csv(CUBE_RANGES_PATH, index=False)

    matrix = np.load(
        PIXEL_MATRIX_PATH,
        mmap_mode=None if LOAD_PIXEL_MATRIX_IN_RAM else "r",
    )

    rows = pd.read_csv(CUBE_RANGES_PATH)
    rows["field"] = rows["field"].astype(str)
    rows["plot"] = rows["plot"].map(normalize_plot_id)

    print("\nCache complete:")
    print("  matrix =", PIXEL_MATRIX_PATH)
    print("  ranges =", CUBE_RANGES_PATH)
    print("  shape  =", matrix.shape)

    return matrix, rows


# =============================================================================
# 6. RECONSTRUCTION PLOT SPLIT
# =============================================================================

def create_or_load_reconstruction_split(meta, cube_ranges):
    if RECON_SPLIT_PATH.exists():
        split_df = pd.read_csv(RECON_SPLIT_PATH)
        split_df["field"] = split_df["field"].astype(str)
        split_df["plot"] = split_df["plot"].map(normalize_plot_id)

        print("\nUsing existing reconstruction plot split:")
        print(" ", RECON_SPLIT_PATH)
        print(split_df["recon_split"].value_counts().to_string())
        return split_df

    all_keys = [
        key_of(row.field, row.plot)
        for row in cube_ranges.itertuples(index=False)
    ]

    test_keys = set(
        key_of(row.field, row.plot)
        for row in meta.loc[meta["split"] == "test", ["field", "plot"]]
        .drop_duplicates()
        .itertuples(index=False)
    )

    development_keys = [k for k in all_keys if k not in test_keys]

    rng = np.random.RandomState(RECON_SPLIT_SEED)

    recon_val = set()

    for field in FIELDS:
        field_keys = sorted([k for k in development_keys if k[0] == field])

        if not field_keys:
            continue

        n_val = max(
            1,
            int(round(len(field_keys) * RECON_VAL_FRACTION))
        )

        chosen = rng.choice(
            len(field_keys),
            size=min(n_val, len(field_keys)),
            replace=False,
        )

        for j in chosen:
            recon_val.add(field_keys[int(j)])

    rows = []

    for field, plot in all_keys:
        k = (field, plot)

        if k in test_keys:
            s = "test"
        elif k in recon_val:
            s = "validation"
        else:
            s = "train"

        rows.append(
            {
                "field": field,
                "plot": plot,
                "recon_split": s,
            }
        )

    split_df = pd.DataFrame(rows)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(RECON_SPLIT_PATH, index=False)

    print("\nCreated fixed reconstruction plot split:")
    print(split_df["recon_split"].value_counts().to_string())
    print("saved:", RECON_SPLIT_PATH)

    return split_df


def ranges_for_split(cube_ranges, recon_split_df, split_name):
    merged = cube_ranges.merge(
        recon_split_df,
        on=["field", "plot"],
        how="left",
        validate="one_to_one",
    )

    if merged["recon_split"].isna().any():
        raise RuntimeError("Some cubes have no reconstruction split.")

    subset = merged[merged["recon_split"] == split_name].copy()

    return [
        (int(r.start), int(r.end))
        for r in subset.itertuples(index=False)
    ]


def concatenate_range_indices(ranges):
    if not ranges:
        return np.empty((0,), dtype=np.int64)

    parts = [
        np.arange(start, end, dtype=np.int64)
        for start, end in ranges
    ]

    return np.concatenate(parts)


# =============================================================================
# 7. EXACT TRAINING MEAN/STD FROM RECONSTRUCTION TRAIN PLOTS
# =============================================================================

def compute_train_spectral_stats(pixel_matrix, train_ranges):
    total_n = 0
    sum40 = np.zeros((40,), dtype=np.float64)
    sumsq40 = np.zeros((40,), dtype=np.float64)

    for start, end in train_ranges:
        x = np.asarray(pixel_matrix[start:end], dtype=np.float64)

        sum40 += x.sum(axis=0)
        sumsq40 += np.square(x).sum(axis=0)
        total_n += x.shape[0]

    if total_n == 0:
        raise RuntimeError("Zero reconstruction-training pixels.")

    mean40 = sum40 / float(total_n)

    variance40 = (
        sumsq40 / float(total_n)
        - np.square(mean40)
    )

    variance40 = np.maximum(variance40, 1e-12)
    std40 = np.sqrt(variance40)

    return (
        mean40.astype(np.float32),
        std40.astype(np.float32),
        int(total_n),
    )


# =============================================================================
# 8. FIXED DIRECT 3 -> 40 MLP
# =============================================================================

class DirectMLP3to40(nn.Module):
    def __init__(self):
        super(DirectMLP3to40, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 40),
        )

    def forward(self, x):
        return self.net(x)


def validation_mse(
    model,
    pixel_matrix,
    val_indices,
    selected_idx,
    mean40,
    std40,
    device,
):
    model.eval()

    sse = 0.0
    n_values = 0

    input_mean = mean40[list(selected_idx)]
    input_std = std40[list(selected_idx)]

    with torch.no_grad():
        for start in range(0, len(val_indices), INFERENCE_CHUNK):
            ii = val_indices[start:start + INFERENCE_CHUNK]

            y_np = np.asarray(pixel_matrix[ii], dtype=np.float32)
            x_np = y_np[:, list(selected_idx)]

            x_np = (x_np - input_mean.reshape(1, 3)) / input_std.reshape(1, 3)
            y_scaled = (y_np - mean40.reshape(1, 40)) / std40.reshape(1, 40)

            xb = torch.from_numpy(x_np).to(device)
            yb = torch.from_numpy(y_scaled).to(device)

            pred = model(xb)

            diff = pred - yb

            sse += float(torch.sum(diff * diff).item())
            n_values += int(diff.numel())

    return sse / max(n_values, 1)


def train_direct_mlp(
    pixel_matrix,
    train_indices,
    val_indices,
    selected_idx,
    mean40,
    std40,
    device,
    out_dir,
):
    seed_everything(RECON_SEED)

    model = DirectMLP3to40().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=RECON_LR,
        weight_decay=RECON_WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    input_mean = mean40[list(selected_idx)]
    input_std = std40[list(selected_idx)]

    best_val = np.inf
    best_epoch = -1
    best_state = None
    bad_epochs = 0

    history = []

    # Fixed validation subset for every epoch and every triplet.
    if len(val_indices) > RECON_VAL_MAX_PIXELS:
        rng_val = np.random.RandomState(RECON_SEED + 999)
        choose = rng_val.choice(
            len(val_indices),
            size=RECON_VAL_MAX_PIXELS,
            replace=False,
        )
        val_eval_idx = val_indices[choose]
    else:
        val_eval_idx = val_indices

    print(
        "  reconstruction training pixels = {:,}".format(len(train_indices))
    )
    print(
        "  reconstruction validation pixels used = {:,}".format(
            len(val_eval_idx)
        )
    )

    rng = np.random.RandomState(RECON_SEED)

    for epoch in range(1, RECON_MAX_EPOCHS + 1):
        model.train()

        # Same number of random training pixels each epoch.
        n_epoch = min(
            int(RECON_SAMPLES_PER_EPOCH),
            max(1, len(train_indices)),
        )

        # Sampling with replacement keeps every epoch fixed-size and avoids
        # creating an enormous 7-8 million-element permutation.
        positions = rng.randint(
            0,
            len(train_indices),
            size=n_epoch,
        )

        epoch_indices = train_indices[positions]

        train_sse = 0.0
        train_values = 0

        for start in range(0, n_epoch, RECON_BATCH_SIZE):
            ii = epoch_indices[start:start + RECON_BATCH_SIZE]

            y_np = np.asarray(pixel_matrix[ii], dtype=np.float32)
            x_np = y_np[:, list(selected_idx)]

            x_np = (
                x_np - input_mean.reshape(1, 3)
            ) / input_std.reshape(1, 3)

            y_scaled = (
                y_np - mean40.reshape(1, 40)
            ) / std40.reshape(1, 40)

            xb = torch.from_numpy(x_np).to(device)
            yb = torch.from_numpy(y_scaled).to(device)

            optimizer.zero_grad(set_to_none=True)

            pred = model(xb)
            loss = criterion(pred, yb)

            loss.backward()
            optimizer.step()

            diff = pred.detach() - yb
            train_sse += float(torch.sum(diff * diff).item())
            train_values += int(diff.numel())

        train_mse = train_sse / max(train_values, 1)

        val_mse = validation_mse(
            model=model,
            pixel_matrix=pixel_matrix,
            val_indices=val_eval_idx,
            selected_idx=selected_idx,
            mean40=mean40,
            std40=std40,
            device=device,
        )

        improved = val_mse < (best_val - RECON_MIN_DELTA)

        if improved:
            best_val = float(val_mse)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "train_mse_scaled": float(train_mse),
                "validation_mse_scaled": float(val_mse),
                "best_validation_mse_scaled": float(best_val),
                "best_epoch": int(best_epoch),
                "bad_epochs": int(bad_epochs),
            }
        )

        if (
            epoch == 1
            or epoch % 10 == 0
            or improved
            or bad_epochs >= RECON_PATIENCE
        ):
            print(
                "    epoch {:3d} | train MSE {:.8f} | "
                "val MSE {:.8f} | best {:.8f}@{} | patience {}/{}".format(
                    epoch,
                    train_mse,
                    val_mse,
                    best_val,
                    best_epoch,
                    bad_epochs,
                    RECON_PATIENCE,
                )
            )

        if bad_epochs >= RECON_PATIENCE:
            print(
                "  EARLY STOP: validation MSE did not improve for "
                "{} consecutive epochs.".format(RECON_PATIENCE)
            )
            break

    if best_state is None:
        raise RuntimeError("No best reconstruction checkpoint was created.")

    model.load_state_dict(best_state)

    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(history).to_csv(
        out_dir / "reconstruction_training_history.csv",
        index=False,
    )

    torch.save(
        {
            "selected_band_indices_40": list(map(int, selected_idx)),
            "input_mean": input_mean.astype(np.float32),
            "input_std": input_std.astype(np.float32),
            "output_mean40": mean40.astype(np.float32),
            "output_std40": std40.astype(np.float32),
            "best_epoch": int(best_epoch),
            "best_validation_mse_scaled": float(best_val),
            "model_state_dict": best_state,
            "architecture": "3-128-128-40_ReLU",
        },
        out_dir / "best_direct_mlp_3to40.pth",
    )

    return model, int(best_epoch), float(best_val)


def reconstruct_pixels(
    model,
    measured_pixels,
    selected_idx,
    mean40,
    std40,
    device,
):
    measured_pixels = np.asarray(measured_pixels, dtype=np.float32)

    input_mean = mean40[list(selected_idx)]
    input_std = std40[list(selected_idx)]

    outputs = []

    model.eval()

    with torch.no_grad():
        for start in range(0, len(measured_pixels), INFERENCE_CHUNK):
            y_np = measured_pixels[start:start + INFERENCE_CHUNK]
            x_np = y_np[:, list(selected_idx)]

            x_scaled = (
                x_np - input_mean.reshape(1, 3)
            ) / input_std.reshape(1, 3)

            xb = torch.from_numpy(x_scaled).to(device)

            pred_scaled = model(xb).cpu().numpy()

            pred = (
                pred_scaled * std40.reshape(1, 40)
                + mean40.reshape(1, 40)
            )

            outputs.append(pred.astype(np.float32, copy=False))

    if not outputs:
        return np.empty((0, 40), dtype=np.float32)

    return np.concatenate(outputs, axis=0)


# =============================================================================
# 9. RECONSTRUCTION METRICS
# =============================================================================

def accumulate_reconstruction_metrics(gt, pred, state):
    gt64 = np.asarray(gt, dtype=np.float64)
    pred64 = np.asarray(pred, dtype=np.float64)

    error = pred64 - gt64

    state["sq_sum"] += float(np.sum(error * error))
    state["abs_sum"] += float(np.sum(np.abs(error)))
    state["value_count"] += int(error.size)

    numerator = np.sum(gt64 * pred64, axis=1)
    denominator = (
        np.linalg.norm(gt64, axis=1)
        * np.linalg.norm(pred64, axis=1)
    )

    cosine = numerator / np.maximum(denominator, EPS)
    cosine = np.clip(cosine, -1.0, 1.0)

    sam = np.degrees(np.arccos(cosine))

    state["sam_sum"] += float(np.sum(sam))
    state["pixel_count"] += int(len(sam))


def finalize_reconstruction_metrics(state):
    return {
        "rmse": math.sqrt(
            state["sq_sum"] / max(state["value_count"], 1)
        ),
        "mae": state["abs_sum"] / max(state["value_count"], 1),
        "sam_deg": state["sam_sum"] / max(state["pixel_count"], 1),
        "test_pixels": int(state["pixel_count"]),
    }


# =============================================================================
# 10. MEASURED40 FEATURE EXTRACTION (ONCE)
# =============================================================================

def frozen_mask_path(field, plot):
    return SL_MASK_ROOT / str(field) / "{}_sl_mask.npy".format(plot)


def extract_measured40_features(meta, measured_maps):
    cache = CACHE_DIR / "measured40_features_frozen.npy"

    if cache.exists():
        X = np.load(cache)
        if X.shape == (len(meta), 81):
            print("\nUsing cached measured40 yield features:")
            print(" ", cache)
            return X

    print("\nExtracting measured40 features once...")

    X = np.zeros((len(meta), 81), dtype=np.float32)

    groups = meta.groupby(["field", "plot"], sort=False)

    for counter, ((field, plot), group) in enumerate(groups, start=1):
        field = str(field)
        plot = normalize_plot_id(plot)

        cube_path = measured_maps[field][plot]
        cube = np.load(cube_path).astype(np.float32, copy=False)

        mask_path = frozen_mask_path(field, plot)
        if not mask_path.exists():
            raise FileNotFoundError("Missing frozen mask: {}".format(mask_path))

        sl_mask = np.load(mask_path).astype(bool, copy=False)

        if cube.shape[:2] != sl_mask.shape or cube.shape[2] != 40:
            raise ValueError(
                "{}/{} cube/mask shape mismatch: {} vs {}".format(
                    field, plot, cube.shape, sl_mask.shape
                )
            )

        for row_idx, row in group.iterrows():
            r0 = int(row["row0"])
            r1 = int(row["row1"])
            c0 = int(row["col0"])
            c1 = int(row["col1"])

            m = sl_mask[r0:r1, c0:c1]

            n_sl = int(m.sum())
            expected = int(row["n_SL_pixels"])

            if n_sl != expected:
                raise RuntimeError(
                    "Frozen SL count changed for {}/{} subplot {}: "
                    "{} vs {}".format(
                        field, plot, row["subplot_index"], n_sl, expected
                    )
                )

            px = cube[r0:r1, c0:c1, :][m]

            X[row_idx] = np.concatenate(
                [
                    px.mean(axis=0),
                    px.std(axis=0, ddof=0),
                    np.array([float(n_sl)], dtype=np.float32),
                ]
            ).astype(np.float32)

        if counter == 1 or counter % 100 == 0 or counter == groups.ngroups:
            print(
                "  measured40 features: {}/{} plots".format(
                    counter, groups.ngroups
                )
            )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, X)

    return X


# =============================================================================
# 11. TRIPLET FEATURES + HELD-OUT RECONSTRUCTION METRICS
# =============================================================================

def extract_triplet_features_and_metrics(
    meta,
    measured_maps,
    recon_plot_split,
    selected_idx,
    model,
    mean40,
    std40,
    device,
    triplet_out,
):
    feature_cache = triplet_out / "yield_features.npz"
    metrics_cache = triplet_out / "reconstruction_test_metrics.json"

    if feature_cache.exists() and metrics_cache.exists():
        data = np.load(feature_cache)
        X3 = data["sparse3"].astype(np.float32, copy=False)
        Xrec = data["reconstructed40"].astype(np.float32, copy=False)

        with open(metrics_cache, "r") as f:
            metrics = json.load(f)

        print("  using cached triplet features + reconstruction metrics")
        return X3, Xrec, metrics

    X3 = np.zeros((len(meta), 7), dtype=np.float32)
    Xrec = np.zeros((len(meta), 81), dtype=np.float32)

    split_lookup = {
        key_of(r.field, r.plot): str(r.recon_split)
        for r in recon_plot_split.itertuples(index=False)
    }

    metric_state = {
        "sq_sum": 0.0,
        "abs_sum": 0.0,
        "value_count": 0,
        "sam_sum": 0.0,
        "pixel_count": 0,
    }

    groups = meta.groupby(["field", "plot"], sort=False)

    for counter, ((field, plot), group) in enumerate(groups, start=1):
        field = str(field)
        plot = normalize_plot_id(plot)

        cube_path = measured_maps[field][plot]
        cube = np.load(cube_path).astype(np.float32, copy=False)

        mask_path = frozen_mask_path(field, plot)
        sl_mask = np.load(mask_path).astype(bool, copy=False)

        flat = cube.reshape(-1, 40)
        valid = np.any(flat != 0, axis=1)
        measured_valid = flat[valid]

        recon_valid = reconstruct_pixels(
            model=model,
            measured_pixels=measured_valid,
            selected_idx=selected_idx,
            mean40=mean40,
            std40=std40,
            device=device,
        )

        recon_flat = np.zeros_like(flat, dtype=np.float32)
        recon_flat[valid] = recon_valid
        recon_cube = recon_flat.reshape(cube.shape)

        # Reconstruction metrics are ONLY on frozen held-out reconstruction
        # test plots (the same plot set as the downstream frozen test plots).
        if split_lookup[key_of(field, plot)] == "test":
            accumulate_reconstruction_metrics(
                measured_valid,
                recon_valid,
                metric_state,
            )

        for row_idx, row in group.iterrows():
            r0 = int(row["row0"])
            r1 = int(row["row1"])
            c0 = int(row["col0"])
            c1 = int(row["col1"])

            m = sl_mask[r0:r1, c0:c1]
            n_sl = int(m.sum())

            expected = int(row["n_SL_pixels"])
            if n_sl != expected:
                raise RuntimeError(
                    "Frozen SL count changed for {}/{} subplot {}".format(
                        field, plot, row["subplot_index"]
                    )
                )

            px40 = cube[r0:r1, c0:c1, :][m]
            px3 = px40[:, list(selected_idx)]
            pxrec = recon_cube[r0:r1, c0:c1, :][m]

            area = np.array([float(n_sl)], dtype=np.float32)

            X3[row_idx] = np.concatenate(
                [
                    px3.mean(axis=0),
                    px3.std(axis=0, ddof=0),
                    area,
                ]
            ).astype(np.float32)

            Xrec[row_idx] = np.concatenate(
                [
                    pxrec.mean(axis=0),
                    pxrec.std(axis=0, ddof=0),
                    area,
                ]
            ).astype(np.float32)

        if counter == 1 or counter % 100 == 0 or counter == groups.ngroups:
            print(
                "  feature/reconstruction pass: {}/{} plots".format(
                    counter, groups.ngroups
                )
            )

    metrics = finalize_reconstruction_metrics(metric_state)

    triplet_out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        feature_cache,
        sparse3=X3,
        reconstructed40=Xrec,
    )

    with open(metrics_cache, "w") as f:
        json.dump(metrics, f, indent=2)

    return X3, Xrec, metrics


# =============================================================================
# 12. FROZEN YIELD DNN
# =============================================================================

class YieldMLP(nn.Module):
    def __init__(self, input_dim):
        super(YieldMLP, self).__init__()

        layers = []
        previous = int(input_dim)

        for width in YIELD_HIDDEN_UNITS:
            dense = nn.Linear(previous, int(width))

            nn.init.xavier_uniform_(dense.weight)
            nn.init.zeros_(dense.bias)

            layers.extend([dense, nn.ReLU()])
            previous = int(width)

        output = nn.Linear(previous, 1)

        nn.init.xavier_uniform_(output.weight)
        nn.init.zeros_(output.bias)

        layers.append(output)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).reshape(-1)


def standardize_from_train(X, train_idx):
    mean = X[train_idx].mean(axis=0, dtype=np.float64)

    std = X[train_idx].std(
        axis=0,
        dtype=np.float64,
        ddof=0,
    )

    std[std < 1e-12] = 1.0

    Z = (
        (X.astype(np.float64) - mean) / std
    ).astype(np.float32)

    return (
        Z,
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def make_yield_loader(X, y, idx, shuffle):
    ds = TensorDataset(
        torch.from_numpy(X[idx]).float(),
        torch.from_numpy(y[idx]).float(),
    )

    return DataLoader(
        ds,
        batch_size=YIELD_BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False,
    )


def loader_mse(model, dl, device):
    model.eval()

    sse = 0.0
    n = 0

    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)

            sse += float(
                torch.sum((pred - yb) ** 2).item()
            )
            n += int(yb.numel())

    return sse / max(n, 1)


def predict_yield_indices(model, X, idx, device):
    outputs = []

    model.eval()

    with torch.no_grad():
        for start in range(0, len(idx), 4096):
            ii = idx[start:start + 4096]

            xb = torch.from_numpy(X[ii]).float().to(device)

            outputs.append(
                model(xb).cpu().numpy()
            )

    return np.concatenate(outputs)


def evaluate_yield(y, pred, test_idx):
    yt = y[test_idx]

    return {
        "R2": float(r2_score(yt, pred)),
        "RMSE": float(
            math.sqrt(mean_squared_error(yt, pred))
        ),
        "MAE": float(mean_absolute_error(yt, pred)),
    }


def train_yield_representation(
    representation,
    X,
    y,
    train_idx,
    val_idx,
    test_idx,
    device,
    out_dir,
):
    Z, feature_mean, feature_std = standardize_from_train(
        X, train_idx
    )

    rows = []

    out_dir.mkdir(parents=True, exist_ok=True)

    for model_seed in YIELD_MODEL_SEEDS:
        seed_everything(model_seed)

        model = YieldMLP(Z.shape[1]).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=YIELD_LR,
        )

        criterion = nn.MSELoss()

        train_loader = make_yield_loader(
            Z, y, train_idx, True
        )

        val_loader = make_yield_loader(
            Z, y, val_idx, False
        )

        best_val_mse = np.inf
        best_epoch = -1
        best_state = None

        for epoch in range(1, YIELD_EPOCHS + 1):
            model.train()

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad(set_to_none=True)

                pred = model(xb)
                loss = criterion(pred, yb)

                loss.backward()
                optimizer.step()

            val_mse = loader_mse(
                model,
                val_loader,
                device,
            )

            if val_mse < best_val_mse:
                best_val_mse = float(val_mse)
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)

        pred = predict_yield_indices(
            model,
            Z,
            test_idx,
            device,
        )

        metrics = evaluate_yield(
            y,
            pred,
            test_idx,
        )

        row = {
            "representation": representation,
            "model_seed": int(model_seed),
            "input_dim": int(X.shape[1]),
            "best_epoch": int(best_epoch),
            "best_validation_RMSE": float(math.sqrt(best_val_mse)),
            "test_R2": metrics["R2"],
            "test_RMSE": metrics["RMSE"],
            "test_MAE": metrics["MAE"],
        }

        rows.append(row)

        checkpoint_dir = out_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "representation": representation,
                "model_seed": int(model_seed),
                "input_dim": int(X.shape[1]),
                "hidden_units": list(YIELD_HIDDEN_UNITS),
                "best_epoch": int(best_epoch),
                "best_validation_RMSE": float(math.sqrt(best_val_mse)),
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "model_state_dict": best_state,
            },
            checkpoint_dir / "seed_{}_bestval.pth".format(model_seed),
        )

        print(
            "    yield {:>20s} | seed {:2d} | "
            "best val {:.4f}@{} | test R2 {:.4f} | RMSE {:.4f}".format(
                representation,
                int(model_seed),
                row["best_validation_RMSE"],
                best_epoch,
                row["test_R2"],
                row["test_RMSE"],
            )
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        out_dir / "{}_all_seeds.csv".format(representation),
        index=False,
    )

    # Manuscript-primary rule from the existing pipeline:
    # choose the seed ONLY by lowest validation RMSE.
    primary_idx = int(df["best_validation_RMSE"].idxmin())
    primary = df.loc[primary_idx].to_dict()

    summary = {
        "primary_seed": int(primary["model_seed"]),
        "primary_best_validation_RMSE": float(
            primary["best_validation_RMSE"]
        ),
        "primary_test_R2": float(primary["test_R2"]),
        "primary_test_RMSE": float(primary["test_RMSE"]),
        "primary_test_MAE": float(primary["test_MAE"]),
        "mean_test_R2": float(df["test_R2"].mean()),
        "sd_test_R2": float(df["test_R2"].std(ddof=0)),
        "mean_test_RMSE": float(df["test_RMSE"].mean()),
        "sd_test_RMSE": float(df["test_RMSE"].std(ddof=0)),
    }

    with open(
        out_dir / "{}_summary.json".format(representation),
        "w",
    ) as f:
        json.dump(summary, f, indent=2)

    return summary


# =============================================================================
# 13. MAIN
# =============================================================================

def main():
    t0 = time.time()

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    seed_everything(RECON_SEED)

    device = torch.device(DEVICE)

    print("=" * 88)
    print("HSIwheat EXPERIMENT 2 — FIXED DIRECT MLP 3 -> 40")
    print("=" * 88)
    print("Device:", device)
    print("Root:", ROOT)
    print("Reconstruction network: 3 -> 128 -> 128 -> 40")
    print("Reconstruction loss: measured40 MSE")
    print(
        "Stopping rule: validation MSE no improvement for {} epochs; "
        "max {} epochs".format(
            RECON_PATIENCE,
            RECON_MAX_EPOCHS,
        )
    )
    print("NO Gram matrix. NO decomposition. NO residual learning.")

    if not WAVELENGTHS40_PATH.exists():
        raise FileNotFoundError(
            "Missing {}".format(WAVELENGTHS40_PATH)
        )

    wavelengths40 = np.load(WAVELENGTHS40_PATH).reshape(-1)

    if wavelengths40.shape != (40,):
        raise ValueError(
            "Expected wavelengths_40.npy to contain 40 values, got {}".format(
                wavelengths40.shape
            )
        )

    print("\n40 measured wavelengths:")
    print(np.round(wavelengths40, 3))

    # ---------------------------------------------------------------------
    # Frozen downstream metadata and measured40 cubes
    # ---------------------------------------------------------------------
    meta = load_frozen_metadata()
    y = meta["subplot_yield"].to_numpy(dtype=np.float32)

    yield_train_idx, yield_val_idx, yield_test_idx = get_yield_indices(meta)

    measured_maps, records = discover_measured40_files()

    # Check every frozen plot exists.
    for field, plot in (
        meta[["field", "plot"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        field = str(field)
        plot = normalize_plot_id(plot)

        if field not in measured_maps or plot not in measured_maps[field]:
            raise FileNotFoundError(
                "Frozen plot has no measured40 cube: {}/{}".format(
                    field, plot
                )
            )

    # ---------------------------------------------------------------------
    # Direct all-pixel cache
    # ---------------------------------------------------------------------
    pixel_matrix, cube_ranges = build_or_load_pixel_cache(records)

    # ---------------------------------------------------------------------
    # Fixed reconstruction plot split
    # ---------------------------------------------------------------------
    recon_split_df = create_or_load_reconstruction_split(
        meta,
        cube_ranges,
    )

    train_ranges = ranges_for_split(
        cube_ranges,
        recon_split_df,
        "train",
    )

    val_ranges = ranges_for_split(
        cube_ranges,
        recon_split_df,
        "validation",
    )

    test_ranges = ranges_for_split(
        cube_ranges,
        recon_split_df,
        "test",
    )

    train_indices = concatenate_range_indices(train_ranges)
    val_indices = concatenate_range_indices(val_ranges)

    print("\nReconstruction pixel counts:")
    print("  train      =", "{:,}".format(len(train_indices)))
    print("  validation =", "{:,}".format(len(val_indices)))
    print(
        "  test       =",
        "{:,}".format(
            sum(end - start for start, end in test_ranges)
        )
    )

    # Exact training-only spectral normalization.
    mean40, std40, stats_n = compute_train_spectral_stats(
        pixel_matrix,
        train_ranges,
    )

    np.save(CACHE_DIR / "reconstruction_train_mean40.npy", mean40)
    np.save(CACHE_DIR / "reconstruction_train_std40.npy", std40)

    print("\nReconstruction normalization:")
    print("  pixels used =", "{:,}".format(stats_n))

    # ---------------------------------------------------------------------
    # Measured40 downstream baseline ONCE.
    # This should reproduce the existing measured40 result (~0.8633 primary R2)
    # if the frozen metadata/masks are unchanged.
    # ---------------------------------------------------------------------
    X_measured40 = extract_measured40_features(
        meta,
        measured_maps,
    )

    print("\n" + "=" * 88)
    print("MEASURED40 YIELD BASELINE")
    print("=" * 88)

    measured40_summary = train_yield_representation(
        representation="measured40",
        X=X_measured40,
        y=y,
        train_idx=yield_train_idx,
        val_idx=yield_val_idx,
        test_idx=yield_test_idx,
        device=device,
        out_dir=OUT / "measured40_yield",
    )

    print(
        "\nMeasured40 primary R2 = {:.4f} | RMSE = {:.4f}".format(
            measured40_summary["primary_test_R2"],
            measured40_summary["primary_test_RMSE"],
        )
    )

    # ---------------------------------------------------------------------
    # Snap representative nominal triplets to actual available channels.
    # ---------------------------------------------------------------------
    snapped_triplets = []
    seen = set()

    print("\nRepresentative triplets:")

    for name, nominal in NOMINAL_TRIPLETS:
        idx, actual = snap_triplet(
            wavelengths40,
            nominal,
        )

        if idx in seen:
            print(
                "  SKIP {:>24s}: duplicates actual indices {}".format(
                    name, idx
                )
            )
            continue

        seen.add(idx)

        snapped_triplets.append(
            (name, nominal, idx, actual)
        )

        print(
            "  {:>24s}: idx={} | actual nm={}".format(
                name,
                idx,
                tuple(round(x, 3) for x in actual),
            )
        )

    pd.DataFrame(
        [
            {
                "triplet_name": name,
                "nominal_wl1": nominal[0],
                "nominal_wl2": nominal[1],
                "nominal_wl3": nominal[2],
                "band_idx1": idx[0],
                "band_idx2": idx[1],
                "band_idx3": idx[2],
                "actual_wl1": actual[0],
                "actual_wl2": actual[1],
                "actual_wl3": actual[2],
            }
            for name, nominal, idx, actual in snapped_triplets
        ]
    ).to_csv(
        OUT / "representative_triplets.csv",
        index=False,
    )

    # ---------------------------------------------------------------------
    # Main Experiment 2 loop
    # ---------------------------------------------------------------------
    result_rows = []

    for run_i, (name, nominal, selected_idx, actual) in enumerate(
        snapped_triplets,
        start=1,
    ):
        print("\n" + "#" * 88)
        print(
            "TRIPLET {}/{}: {}".format(
                run_i, len(snapped_triplets), name
            )
        )
        print("#" * 88)
        print("Nominal:", nominal)
        print("Actual :", tuple(round(x, 3) for x in actual))
        print("Indices:", selected_idx)

        tag = "{}_{:.0f}_{:.0f}_{:.0f}".format(
            name,
            actual[0],
            actual[1],
            actual[2],
        ).replace(".", "p")

        triplet_out = OUT / tag
        triplet_out.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------
        # A. Train the SAME fixed direct MLP
        # -------------------------------------------------------------
        print("\n[A] Train direct 3 -> 40 MLP")

        model, best_epoch, best_val_mse = train_direct_mlp(
            pixel_matrix=pixel_matrix,
            train_indices=train_indices,
            val_indices=val_indices,
            selected_idx=selected_idx,
            mean40=mean40,
            std40=std40,
            device=device,
            out_dir=triplet_out,
        )

        # -------------------------------------------------------------
        # B. Extract direct sparse3 and reconstructed40 frozen yield
        #    features, and reconstruction metrics on held-out TEST plots.
        # -------------------------------------------------------------
        print("\n[B] Extract frozen downstream features")

        X3, Xrec, recon_metrics = extract_triplet_features_and_metrics(
            meta=meta,
            measured_maps=measured_maps,
            recon_plot_split=recon_split_df,
            selected_idx=selected_idx,
            model=model,
            mean40=mean40,
            std40=std40,
            device=device,
            triplet_out=triplet_out,
        )

        print(
            "  held-out reconstruction: SAM={:.4f} deg | "
            "RMSE={:.6f} | MAE={:.6f} | pixels={:,}".format(
                recon_metrics["sam_deg"],
                recon_metrics["rmse"],
                recon_metrics["mae"],
                recon_metrics["test_pixels"],
            )
        )

        # -------------------------------------------------------------
        # C. Direct 3-band -> yield
        # -------------------------------------------------------------
        print("\n[C] Direct 3-band -> yield")

        direct_summary = train_yield_representation(
            representation="direct_sparse3",
            X=X3,
            y=y,
            train_idx=yield_train_idx,
            val_idx=yield_val_idx,
            test_idx=yield_test_idx,
            device=device,
            out_dir=triplet_out / "yield_direct_sparse3",
        )

        # -------------------------------------------------------------
        # D. Reconstructed40 -> yield
        # -------------------------------------------------------------
        print("\n[D] Reconstructed40 -> yield")

        recon_yield_summary = train_yield_representation(
            representation="reconstructed40",
            X=Xrec,
            y=y,
            train_idx=yield_train_idx,
            val_idx=yield_val_idx,
            test_idx=yield_test_idx,
            device=device,
            out_dir=triplet_out / "yield_reconstructed40",
        )

        row = {
            "triplet_name": name,
            "nominal_wl1": nominal[0],
            "nominal_wl2": nominal[1],
            "nominal_wl3": nominal[2],
            "band_idx1": selected_idx[0],
            "band_idx2": selected_idx[1],
            "band_idx3": selected_idx[2],
            "actual_wl1": actual[0],
            "actual_wl2": actual[1],
            "actual_wl3": actual[2],

            "recon_best_epoch": best_epoch,
            "recon_best_val_mse_scaled": best_val_mse,
            "recon_test_sam_deg": recon_metrics["sam_deg"],
            "recon_test_rmse": recon_metrics["rmse"],
            "recon_test_mae": recon_metrics["mae"],
            "recon_test_pixels": recon_metrics["test_pixels"],

            "direct_primary_seed": direct_summary["primary_seed"],
            "direct_yield_r2": direct_summary["primary_test_R2"],
            "direct_yield_rmse": direct_summary["primary_test_RMSE"],
            "direct_yield_mae": direct_summary["primary_test_MAE"],
            "direct_yield_r2_mean10": direct_summary["mean_test_R2"],
            "direct_yield_r2_sd10": direct_summary["sd_test_R2"],

            "reconstructed_primary_seed": recon_yield_summary["primary_seed"],
            "reconstructed_yield_r2": recon_yield_summary["primary_test_R2"],
            "reconstructed_yield_rmse": recon_yield_summary["primary_test_RMSE"],
            "reconstructed_yield_mae": recon_yield_summary["primary_test_MAE"],
            "reconstructed_yield_r2_mean10": recon_yield_summary["mean_test_R2"],
            "reconstructed_yield_r2_sd10": recon_yield_summary["sd_test_R2"],

            "measured40_yield_r2": measured40_summary["primary_test_R2"],
            "measured40_yield_rmse": measured40_summary["primary_test_RMSE"],
        }

        result_rows.append(row)

        pd.DataFrame(result_rows).to_csv(
            OUT / "experiment2_results_partial.csv",
            index=False,
        )

        print("\nTRIPLET RESULT:")
        print(
            "  Direct sparse3 yield R2      = {:.4f}".format(
                row["direct_yield_r2"]
            )
        )
        print(
            "  Reconstruction test SAM     = {:.4f} deg".format(
                row["recon_test_sam_deg"]
            )
        )
        print(
            "  Reconstruction test RMSE    = {:.6f}".format(
                row["recon_test_rmse"]
            )
        )
        print(
            "  Reconstructed40 yield R2    = {:.4f}".format(
                row["reconstructed_yield_r2"]
            )
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # Final Experiment-2 table
    # ---------------------------------------------------------------------
    results = pd.DataFrame(result_rows)

    results.to_csv(
        OUT / "experiment2_results.csv",
        index=False,
    )

    # Experiment 3 preview: relationships among the three quantities.
    relation_cols = [
        "direct_yield_r2",
        "recon_test_sam_deg",
        "recon_test_rmse",
        "reconstructed_yield_r2",
    ]

    pearson = results[relation_cols].corr(method="pearson")
    spearman = results[relation_cols].corr(method="spearman")

    pearson.to_csv(
        OUT / "experiment3_preview_pearson.csv"
    )

    spearman.to_csv(
        OUT / "experiment3_preview_spearman.csv"
    )

    print("\n" + "=" * 110)
    print("FINAL EXPERIMENT-2 RESULTS")
    print("=" * 110)

    show_cols = [
        "triplet_name",
        "actual_wl1",
        "actual_wl2",
        "actual_wl3",
        "direct_yield_r2",
        "recon_test_sam_deg",
        "recon_test_rmse",
        "reconstructed_yield_r2",
    ]

    print(
        results[show_cols].to_string(
            index=False,
            float_format=lambda x: "{:.4f}".format(x),
        )
    )

    print("\nMeasured40 primary yield baseline:")
    print(
        "  R2={:.4f}, RMSE={:.4f}".format(
            measured40_summary["primary_test_R2"],
            measured40_summary["primary_test_RMSE"],
        )
    )

    print("\nPearson relationship matrix:")
    print(pearson.to_string(float_format=lambda x: "{:.3f}".format(x)))

    elapsed = time.time() - t0

    print("\nSaved:")
    print(" ", OUT / "experiment2_results.csv")
    print(" ", OUT / "representative_triplets.csv")
    print(" ", OUT / "experiment3_preview_pearson.csv")
    print(" ", OUT / "experiment3_preview_spearman.csv")

    print(
        "\nTotal elapsed: {:.2f} h".format(
            elapsed / 3600.0
        )
    )


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()













