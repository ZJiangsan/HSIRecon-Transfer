#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HSIwheat — model-class sensitivity of prediction-model transfer degradation
============================================================================

PURPOSE
-------
Test whether prediction-model transfer degradation depends on the downstream
model class.

Three measured-HSI reference models are compared:

    1) PLSR
       Linear / latent-variable spectral regression baseline.

    2) Random Forest
       Nonlinear tree-based baseline.

    3) Existing measured40 MLP
       The already trained neural model used in the manuscript:
       81 -> 10 -> 10 -> 10 -> 10 -> 1.

IMPORTANT DESIGN RULE
---------------------
Every model is developed ONLY on genuine measured40 data.

For PLSR and Random Forest:
    - fit on the frozen measured40 TRAIN split
    - select hyperparameters on the frozen measured40 VALIDATION split
    - freeze the complete selected pipeline
    - test first on genuine measured40 TEST data
    - then apply the SAME frozen pipeline to each reconstructed40 TEST input

For the MLP:
    - reuse the already trained measured40 checkpoint selected by measured40
      validation RMSE
    - reuse its saved measured40 training scaler
    - do not retrain it

There is NO:
    - training on reconstructed HSI
    - fine-tuning on reconstructed HSI
    - reconstruction-specific scaling
    - hyperparameter selection using reconstructed data
    - test-label-based model selection

PRIMARY QUANTITY
----------------
Prediction-model transfer degradation:

    delta_R2   = R2_measured40 - R2_reconstructed
    delta_RMSE = RMSE_reconstructed - RMSE_measured40
    delta_MAE  = MAE_reconstructed - MAE_measured40

A positive delta_R2 therefore means that performance degraded after replacing
genuine measured40 with reconstructed40.

The experiment asks whether this degradation is similar across a linear
spectral model, a nonlinear tree model, and the existing neural model.

NOTES
-----
PLSR, Random Forest and MLP should NOT be described as a strict "capacity
ladder". They differ in both flexibility and inductive bias.

The script evaluates the 11 direct-MLP reconstructed40 representations used
in the final manuscript (ten representative triplets plus T12). Legacy
Gram/SparseHSR diagnostic outputs are intentionally excluded from this release.

No main() function is used.
"""

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path("/home/nibio/HSIwheat_40")

STAGE1_ROOT = ROOT / "original190_paper_reproduction"
META_PATH = STAGE1_ROOT / "subplot_metadata.csv"
SPLIT_PATH = STAGE1_ROOT / "split_assignments.csv"

EXP2_OUT = ROOT / "experiment2_direct_mlp_3to40"
MEASURED40_FEATURE_CACHE = EXP2_OUT / "cache" / "measured40_features_frozen.npy"
MEASURED40_CHECKPOINT_DIR = EXP2_OUT / "measured40_yield" / "checkpoints"

OUT = ROOT / "prediction_model_transfer_model_classes"


# =============================================================================
# 2. EXPERIMENT SETTINGS
# =============================================================================

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MLP_PRED_BATCH = 4096

# Manuscript experiment = ten representative triplets + T12.
EXPECTED_N_DIRECT_RECONSTRUCTIONS = 11
REQUIRE_EXACTLY_11_DIRECT_RECONSTRUCTIONS = True

# ---------------- PLSR ----------------
# Selected ONLY by measured40 validation RMSE.
PLSR_COMPONENT_GRID = list(range(1, 21)) + [25, 30, 35, 40]
PLSR_MAX_ITER = 1000
PLSR_TOL = 1e-06

# ---------------- RANDOM FOREST ----------------
# Compact, prespecified grid. Selected ONLY by measured40 validation RMSE.
#
# n_estimators is fixed so the experiment focuses on tree complexity /
# regularization rather than an unnecessarily large tuning exercise.
RF_N_ESTIMATORS = 300
RF_RANDOM_STATE = 42
RF_N_JOBS = -1

RF_GRID = [
    {"max_depth": None, "min_samples_leaf": 1,  "max_features": 1.0},
    {"max_depth": None, "min_samples_leaf": 3,  "max_features": 1.0},
    {"max_depth": None, "min_samples_leaf": 10, "max_features": 1.0},

    {"max_depth": None, "min_samples_leaf": 1,  "max_features": 0.5},
    {"max_depth": None, "min_samples_leaf": 3,  "max_features": 0.5},
    {"max_depth": None, "min_samples_leaf": 10, "max_features": 0.5},

    {"max_depth": 20,   "min_samples_leaf": 1,  "max_features": 1.0},
    {"max_depth": 20,   "min_samples_leaf": 3,  "max_features": 1.0},
    {"max_depth": 20,   "min_samples_leaf": 10, "max_features": 1.0},

    {"max_depth": 20,   "min_samples_leaf": 1,  "max_features": 0.5},
    {"max_depth": 20,   "min_samples_leaf": 3,  "max_features": 0.5},
    {"max_depth": 20,   "min_samples_leaf": 10, "max_features": 0.5},
]


# =============================================================================
# 3. MLP ARCHITECTURE — EXACT EXISTING MEASURED40 MODEL
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
# 4. GENERAL HELPERS
# =============================================================================

def rmse(y_true, y_pred):
    return float(
        math.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


def metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE_g": rmse(y_true, y_pred),
        "MAE_g": float(mean_absolute_error(y_true, y_pred)),
    }


def safe_torch_load(path, device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def load_frozen_meta():
    if not META_PATH.exists() or not SPLIT_PATH.exists():
        raise FileNotFoundError(
            "Could not find the frozen metadata/split.\n"
            f"  {META_PATH}\n"
            f"  {SPLIT_PATH}"
        )

    meta = pd.read_csv(META_PATH)
    split = pd.read_csv(SPLIT_PATH)

    key_cols = [
        "field",
        "plot",
        "subplot_index",
    ]

    for column in key_cols:
        if column not in meta.columns:
            raise ValueError(
                f"{META_PATH} missing {column}"
            )

        if column not in split.columns:
            raise ValueError(
                f"{SPLIT_PATH} missing {column}"
            )

    if "split" not in split.columns:
        raise ValueError(
            f"{SPLIT_PATH} missing split column"
        )

    meta = meta.merge(
        split[key_cols + ["split"]],
        on=key_cols,
        how="left",
        validate="one_to_one",
    ).reset_index(drop=True)

    if meta["split"].isna().any():
        raise RuntimeError(
            "Some frozen metadata rows have no split assignment."
        )

    return meta


def get_split_indices(meta):
    split = (
        meta["split"]
        .astype(str)
        .to_numpy()
    )

    train_idx = np.where(
        split == "train"
    )[0]

    val_idx = np.where(
        split == "validation"
    )[0]

    test_idx = np.where(
        split == "test"
    )[0]

    if (
        len(train_idx) == 0
        or len(val_idx) == 0
        or len(test_idx) == 0
    ):
        raise RuntimeError(
            "Frozen train/validation/test split is incomplete."
        )

    return (
        train_idx,
        val_idx,
        test_idx,
    )


def label_from_path(path, prefix):
    try:
        relative = path.parent.relative_to(
            EXP2_OUT
        )

        text = str(relative)

    except ValueError:
        text = str(path.parent)

    text = text.replace(
        "\\",
        "/",
    )

    return (
        prefix
        + "::"
        + text
    )


def verify_row_alignment(
    name,
    X,
    X_measured40,
):
    """
    Every 40-band yield feature vector is:
        40 means + 40 standard deviations + SL area

    Therefore the final feature (area) should be exactly the same row-by-row.
    This gives us a simple check that cached reconstructed features are aligned
    with the frozen metadata and measured40 feature matrix.
    """
    if X.shape != X_measured40.shape:
        raise RuntimeError(
            f"{name}: feature shape {X.shape} does not match "
            f"measured40 {X_measured40.shape}"
        )

    max_area_difference = float(
        np.max(
            np.abs(
                X[:, -1].astype(np.float64)
                - X_measured40[:, -1].astype(np.float64)
            )
        )
    )

    if max_area_difference > 1e-6:
        raise RuntimeError(
            f"{name}: row-alignment check failed. "
            f"Max area-feature difference = {max_area_difference}"
        )


# =============================================================================
# 5. LOAD DATA
# =============================================================================

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 118)
print("MODEL-CLASS SENSITIVITY OF PREDICTION-MODEL TRANSFER DEGRADATION")
print("=" * 118)

meta = load_frozen_meta()

if "subplot_yield" not in meta.columns:
    raise ValueError(
        "Frozen metadata has no subplot_yield column."
    )

y = meta[
    "subplot_yield"
].to_numpy(
    dtype=np.float32
)

(
    train_idx,
    val_idx,
    test_idx,
) = get_split_indices(
    meta
)

print("\nFrozen downstream split:")
print("  train      =", f"{len(train_idx):,}")
print("  validation =", f"{len(val_idx):,}")
print("  test       =", f"{len(test_idx):,}")

if not MEASURED40_FEATURE_CACHE.exists():
    raise FileNotFoundError(
        f"Missing measured40 feature cache:\n{MEASURED40_FEATURE_CACHE}\n"
        "Run 02_run_representative_triplets.py first."
    )

X_measured40 = np.load(
    MEASURED40_FEATURE_CACHE
).astype(
    np.float32,
    copy=False,
)

expected_shape = (
    len(meta),
    81,
)

if X_measured40.shape != expected_shape:
    raise RuntimeError(
        f"measured40 shape = {X_measured40.shape}; "
        f"expected {expected_shape}"
    )

print("  measured40 features =", X_measured40.shape)


# =============================================================================
# 6. DISCOVER THE 11 DIRECT-MLP RECONSTRUCTED40 FEATURE SETS
# =============================================================================

representations = [
    {
        "representation": "measured40",
        "kind": "genuine_measured40",
        "source_file": str(
            MEASURED40_FEATURE_CACHE
        ),
        "source_key": "measured40",
        "X": X_measured40,
    }
]

direct_files = sorted(
    EXP2_OUT.rglob(
        "yield_features.npz"
    )
)

direct_representations = []

for path in direct_files:
    data = np.load(path)

    if "reconstructed40" not in data.files:
        continue

    X = data[
        "reconstructed40"
    ].astype(
        np.float32,
        copy=False,
    )

    if (
        X.ndim != 2
        or X.shape[1] != 81
    ):
        continue

    name = label_from_path(
        path,
        "direct_mlp",
    )

    verify_row_alignment(
        name,
        X,
        X_measured40,
    )

    direct_representations.append(
        {
            "representation": name,
            "kind": "direct_mlp_reconstructed40",
            "source_file": str(path),
            "source_key": "reconstructed40",
            "X": X,
        }
    )

print(
    "\nDirect-MLP reconstructed40 inputs found:",
    len(direct_representations),
)

for rep in direct_representations:
    print(
        "  +",
        rep["representation"],
    )

if (
    REQUIRE_EXACTLY_11_DIRECT_RECONSTRUCTIONS
    and len(direct_representations)
    != EXPECTED_N_DIRECT_RECONSTRUCTIONS
):
    raise RuntimeError(
        "\nExpected exactly "
        f"{EXPECTED_N_DIRECT_RECONSTRUCTIONS} direct-MLP reconstructed40 "
        f"feature sets, but found {len(direct_representations)}.\n"
        "Do not continue silently. Check the paths printed above for a missing "
        "triplet or duplicate output directory."
    )

representations.extend(
    direct_representations
)


# =============================================================================
# 7. FINAL-MANUSCRIPT INPUT SET
# =============================================================================

# The public release intentionally evaluates only the 11 direct-MLP
# reconstructions used in the final manuscript. Legacy Gram/SparseHSR
# diagnostic branches are not part of this workflow.


# =============================================================================
# 8. PLSR — SELECT COMPONENT COUNT USING MEASURED40 VALIDATION ONLY
# =============================================================================

print("\n" + "=" * 118)
print("PLSR — MEASURED40 TRAINING / VALIDATION ONLY")
print("=" * 118)

# One measured40 scaler is fitted on measured40 TRAIN only.
plsr_scaler = StandardScaler(
    with_mean=True,
    with_std=True,
)

plsr_scaler.fit(
    X_measured40[
        train_idx
    ]
)

Z40_plsr = plsr_scaler.transform(
    X_measured40
).astype(
    np.float64,
    copy=False,
)

plsr_selection_rows = []

best_plsr_key = None
best_plsr_val_rmse = np.inf
best_plsr = None

for n_components in PLSR_COMPONENT_GRID:
    if n_components > X_measured40.shape[1]:
        continue

    start = time.time()

    model = PLSRegression(
        n_components=int(
            n_components
        ),
        scale=False,
        max_iter=PLSR_MAX_ITER,
        tol=PLSR_TOL,
    )

    model.fit(
        Z40_plsr[
            train_idx
        ],
        y[
            train_idx
        ],
    )

    val_pred = model.predict(
        Z40_plsr[
            val_idx
        ]
    ).reshape(-1)

    val_metrics = metrics(
        y[
            val_idx
        ],
        val_pred,
    )

    elapsed = time.time() - start

    row = {
        "n_components": int(
            n_components
        ),
        "validation_R2": val_metrics["R2"],
        "validation_RMSE_g": val_metrics["RMSE_g"],
        "validation_MAE_g": val_metrics["MAE_g"],
        "seconds": float(elapsed),
    }

    plsr_selection_rows.append(
        row
    )

    print(
        f"  components={n_components:2d} | "
        f"val R2={val_metrics['R2']:.4f} | "
        f"RMSE={val_metrics['RMSE_g']:.4f} g"
    )

    if (
        val_metrics["RMSE_g"]
        < best_plsr_val_rmse
    ):
        best_plsr_val_rmse = (
            val_metrics[
                "RMSE_g"
            ]
        )

        best_plsr_key = int(
            n_components
        )

        best_plsr = model

plsr_selection_df = pd.DataFrame(
    plsr_selection_rows
)

plsr_selection_df.to_csv(
    OUT
    / "plsr_measured40_validation_selection.csv",
    index=False,
)

print(
    "\nSelected PLSR components:",
    best_plsr_key,
)
print(
    "Selected PLSR validation RMSE:",
    f"{best_plsr_val_rmse:.6f} g",
)


# =============================================================================
# 9. RANDOM FOREST — SELECT HYPERPARAMETERS USING MEASURED40 VALIDATION ONLY
# =============================================================================

print("\n" + "=" * 118)
print("RANDOM FOREST — MEASURED40 TRAINING / VALIDATION ONLY")
print("=" * 118)

rf_selection_rows = []

best_rf_key = None
best_rf_val_rmse = np.inf
best_rf = None

for grid_index, params in enumerate(
    RF_GRID
):
    start = time.time()

    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=params[
            "max_depth"
        ],
        min_samples_leaf=params[
            "min_samples_leaf"
        ],
        max_features=params[
            "max_features"
        ],
        random_state=RF_RANDOM_STATE,
        n_jobs=RF_N_JOBS,
        criterion="squared_error",
    )

    model.fit(
        X_measured40[
            train_idx
        ],
        y[
            train_idx
        ],
    )

    val_pred = model.predict(
        X_measured40[
            val_idx
        ]
    )

    val_metrics = metrics(
        y[
            val_idx
        ],
        val_pred,
    )

    elapsed = time.time() - start

    row = {
        "grid_index": int(
            grid_index
        ),
        "n_estimators": RF_N_ESTIMATORS,
        "max_depth": (
            "None"
            if params["max_depth"] is None
            else int(
                params[
                    "max_depth"
                ]
            )
        ),
        "min_samples_leaf": int(
            params[
                "min_samples_leaf"
            ]
        ),
        "max_features": params[
            "max_features"
        ],
        "random_state": RF_RANDOM_STATE,
        "validation_R2": val_metrics["R2"],
        "validation_RMSE_g": val_metrics["RMSE_g"],
        "validation_MAE_g": val_metrics["MAE_g"],
        "seconds": float(elapsed),
    }

    rf_selection_rows.append(
        row
    )

    print(
        f"  grid={grid_index:02d} | "
        f"depth={params['max_depth']} | "
        f"leaf={params['min_samples_leaf']} | "
        f"features={params['max_features']} | "
        f"val R2={val_metrics['R2']:.4f} | "
        f"RMSE={val_metrics['RMSE_g']:.4f} g"
    )

    if (
        val_metrics["RMSE_g"]
        < best_rf_val_rmse
    ):
        best_rf_val_rmse = (
            val_metrics[
                "RMSE_g"
            ]
        )

        best_rf_key = int(
            grid_index
        )

        best_rf = model

best_rf_params = RF_GRID[
    best_rf_key
]

rf_selection_df = pd.DataFrame(
    rf_selection_rows
)

rf_selection_df.to_csv(
    OUT
    / "random_forest_measured40_validation_selection.csv",
    index=False,
)

print(
    "\nSelected RF grid index:",
    best_rf_key,
)
print(
    "Selected RF parameters:",
    best_rf_params,
)
print(
    "Selected RF validation RMSE:",
    f"{best_rf_val_rmse:.6f} g",
)


# =============================================================================
# 10. LOAD THE EXISTING MEASURED40 MLP — NO NEW MLP TRAINING
# =============================================================================

print("\n" + "=" * 118)
print("EXISTING MEASURED40 MLP — NO RETRAINING")
print("=" * 118)

checkpoint_files = sorted(
    MEASURED40_CHECKPOINT_DIR.glob(
        "seed_*_bestval.pth"
    )
)

if not checkpoint_files:
    raise FileNotFoundError(
        f"No MLP checkpoints under {MEASURED40_CHECKPOINT_DIR}. "
        "Run 02_run_representative_triplets.py first."
    )

checkpoint_rows = []

for path in checkpoint_files:
    checkpoint = safe_torch_load(
        path,
        "cpu",
    )

    checkpoint_rows.append(
        (
            float(
                checkpoint[
                    "best_validation_RMSE"
                ]
            ),
            int(
                checkpoint[
                    "model_seed"
                ]
            ),
        )
    )

checkpoint_rows.sort(
    key=lambda item: item[0]
)

mlp_seed = checkpoint_rows[
    0
][1]

mlp_checkpoint_path = (
    MEASURED40_CHECKPOINT_DIR
    / f"seed_{mlp_seed}_bestval.pth"
)

if not mlp_checkpoint_path.exists():
    raise FileNotFoundError(
        f"Missing MLP checkpoint: {mlp_checkpoint_path}"
    )

device = torch.device(
    DEVICE
)

mlp_checkpoint = safe_torch_load(
    mlp_checkpoint_path,
    device,
)

if int(
    mlp_checkpoint[
        "input_dim"
    ]
) != 81:
    raise RuntimeError(
        "Measured40 MLP checkpoint is not 81-dimensional."
    )

mlp_hidden_units = tuple(
    int(x)
    for x in mlp_checkpoint.get(
        "hidden_units",
        [10, 10, 10, 10],
    )
)

mlp = YieldMLP(
    input_dim=81,
    hidden_units=mlp_hidden_units,
).to(
    device
)

mlp.load_state_dict(
    mlp_checkpoint[
        "model_state_dict"
    ]
)

mlp.eval()

mlp_mean = np.asarray(
    mlp_checkpoint[
        "feature_mean"
    ],
    dtype=np.float32,
).reshape(-1)

mlp_std = np.asarray(
    mlp_checkpoint[
        "feature_std"
    ],
    dtype=np.float32,
).reshape(-1)

mlp_std = mlp_std.copy()
mlp_std[
    mlp_std < 1e-12
] = 1.0

print(
    "MLP checkpoint:",
    mlp_checkpoint_path,
)
print(
    "MLP seed:",
    mlp_seed,
)
print(
    "MLP best measured40 validation RMSE:",
    mlp_checkpoint.get(
        "best_validation_RMSE",
        np.nan,
    ),
)


# =============================================================================
# 11. FROZEN PREDICTION FUNCTIONS
# =============================================================================

def predict_plsr(X):
    Z = plsr_scaler.transform(
        X
    ).astype(
        np.float64,
        copy=False,
    )

    return best_plsr.predict(
        Z
    ).reshape(-1)


def predict_rf(X):
    return best_rf.predict(
        X
    ).reshape(-1)


def predict_mlp(X):
    Z = (
        (
            X.astype(
                np.float32,
                copy=False,
            )
            - mlp_mean[
                None,
                :
            ]
        )
        / mlp_std[
            None,
            :
        ]
    ).astype(
        np.float32,
        copy=False,
    )

    outputs = []

    with torch.no_grad():
        for start in range(
            0,
            len(test_idx),
            MLP_PRED_BATCH,
        ):
            indices = test_idx[
                start:
                start
                + MLP_PRED_BATCH
            ]

            xb = torch.from_numpy(
                Z[
                    indices
                ]
            ).float().to(
                device
            )

            outputs.append(
                mlp(
                    xb
                ).cpu().numpy()
            )

    return np.concatenate(
        outputs,
        axis=0,
    )


# =============================================================================
# 12. EVALUATE GENUINE MEASURED40 BASELINES
# =============================================================================

print("\n" + "=" * 118)
print("GENUINE MEASURED40 TEST BASELINES")
print("=" * 118)

y_test = y[
    test_idx
]

baseline_metrics = {}

# PLSR
plsr_measured_pred = predict_plsr(
    X_measured40
)[
    test_idx
]

baseline_metrics[
    "PLSR"
] = metrics(
    y_test,
    plsr_measured_pred,
)

# Random Forest
rf_measured_pred = predict_rf(
    X_measured40[
        test_idx
    ]
)

baseline_metrics[
    "RandomForest"
] = metrics(
    y_test,
    rf_measured_pred,
)

# MLP
mlp_measured_pred = predict_mlp(
    X_measured40
)

baseline_metrics[
    "MLP"
] = metrics(
    y_test,
    mlp_measured_pred,
)

for model_name in [
    "PLSR",
    "RandomForest",
    "MLP",
]:
    m = baseline_metrics[
        model_name
    ]

    print(
        f"  {model_name:<12s} | "
        f"R2={m['R2']:.4f} | "
        f"RMSE={m['RMSE_g']:.4f} g | "
        f"MAE={m['MAE_g']:.4f} g"
    )


# =============================================================================
# 13. APPLY EACH FROZEN MEASURED40 MODEL TO ALL RECONSTRUCTED INPUTS
# =============================================================================

print("\n" + "=" * 118)
print("FROZEN TRANSFER TO RECONSTRUCTED40")
print("=" * 118)

result_rows = []
prediction_rows = []

model_predictors = {
    "PLSR": predict_plsr,
    "RandomForest": predict_rf,
    "MLP": predict_mlp,
}

for model_name, predictor in model_predictors.items():
    base = baseline_metrics[
        model_name
    ]

    # Save genuine measured40 row first.
    result_rows.append(
        {
            "model_class": model_name,
            "representation": "measured40",
            "kind": "genuine_measured40",
            "source_file": str(
                MEASURED40_FEATURE_CACHE
            ),
            "R2": base[
                "R2"
            ],
            "RMSE_g": base[
                "RMSE_g"
            ],
            "MAE_g": base[
                "MAE_g"
            ],
            "delta_R2": 0.0,
            "delta_RMSE_g": 0.0,
            "delta_MAE_g": 0.0,
        }
    )

    if model_name == "PLSR":
        genuine_pred = (
            plsr_measured_pred
        )
    elif model_name == "RandomForest":
        genuine_pred = (
            rf_measured_pred
        )
    else:
        genuine_pred = (
            mlp_measured_pred
        )

    for j, row_index in enumerate(
        test_idx
    ):
        prediction_rows.append(
            {
                "model_class": model_name,
                "representation": "measured40",
                "row_index": int(
                    row_index
                ),
                "observed_yield_g": float(
                    y_test[j]
                ),
                "predicted_yield_g": float(
                    genuine_pred[j]
                ),
            }
        )

    for rep in representations[
        1:
    ]:
        X = rep[
            "X"
        ]

        if model_name == "PLSR":
            pred = predictor(
                X
            )[
                test_idx
            ]

        elif model_name == "RandomForest":
            pred = predictor(
                X[
                    test_idx
                ]
            )

        else:
            pred = predictor(
                X
            )

        m = metrics(
            y_test,
            pred,
        )

        delta_r2 = (
            base[
                "R2"
            ]
            - m[
                "R2"
            ]
        )

        delta_rmse = (
            m[
                "RMSE_g"
            ]
            - base[
                "RMSE_g"
            ]
        )

        delta_mae = (
            m[
                "MAE_g"
            ]
            - base[
                "MAE_g"
            ]
        )

        result_rows.append(
            {
                "model_class": model_name,
                "representation": rep[
                    "representation"
                ],
                "kind": rep[
                    "kind"
                ],
                "source_file": rep[
                    "source_file"
                ],
                "R2": m[
                    "R2"
                ],
                "RMSE_g": m[
                    "RMSE_g"
                ],
                "MAE_g": m[
                    "MAE_g"
                ],
                "delta_R2": float(
                    delta_r2
                ),
                "delta_RMSE_g": float(
                    delta_rmse
                ),
                "delta_MAE_g": float(
                    delta_mae
                ),
            }
        )

        for j, row_index in enumerate(
            test_idx
        ):
            prediction_rows.append(
                {
                    "model_class": model_name,
                    "representation": rep[
                        "representation"
                    ],
                    "row_index": int(
                        row_index
                    ),
                    "observed_yield_g": float(
                        y_test[j]
                    ),
                    "predicted_yield_g": float(
                        pred[j]
                    ),
                }
            )

        print(
            f"  {model_name:<12s} | "
            f"{rep['representation']:<70s} | "
            f"R2={m['R2']:.4f} | "
            f"delta_R2={delta_r2:.4f}"
        )


# =============================================================================
# 14. SAVE LONG-FORM RESULTS
# =============================================================================

results_df = pd.DataFrame(
    result_rows
)

predictions_df = pd.DataFrame(
    prediction_rows
)

results_path = (
    OUT
    / "model_class_transfer_results_long.csv"
)

predictions_path = (
    OUT
    / "model_class_transfer_predictions.csv"
)

results_df.to_csv(
    results_path,
    index=False,
)

predictions_df.to_csv(
    predictions_path,
    index=False,
)


# =============================================================================
# 15. SAVE MODEL-SELECTION / BASELINE SUMMARY
# =============================================================================

selection_summary = pd.DataFrame(
    [
        {
            "model_class": "PLSR",
            "selected_hyperparameters": json.dumps(
                {
                    "n_components": int(
                        best_plsr_key
                    )
                }
            ),
            "measured40_validation_RMSE_g": float(
                best_plsr_val_rmse
            ),
            "measured40_test_R2": baseline_metrics[
                "PLSR"
            ][
                "R2"
            ],
            "measured40_test_RMSE_g": baseline_metrics[
                "PLSR"
            ][
                "RMSE_g"
            ],
            "measured40_test_MAE_g": baseline_metrics[
                "PLSR"
            ][
                "MAE_g"
            ],
        },
        {
            "model_class": "RandomForest",
            "selected_hyperparameters": json.dumps(
                {
                    "n_estimators": RF_N_ESTIMATORS,
                    "max_depth": best_rf_params[
                        "max_depth"
                    ],
                    "min_samples_leaf": best_rf_params[
                        "min_samples_leaf"
                    ],
                    "max_features": best_rf_params[
                        "max_features"
                    ],
                    "random_state": RF_RANDOM_STATE,
                }
            ),
            "measured40_validation_RMSE_g": float(
                best_rf_val_rmse
            ),
            "measured40_test_R2": baseline_metrics[
                "RandomForest"
            ][
                "R2"
            ],
            "measured40_test_RMSE_g": baseline_metrics[
                "RandomForest"
            ][
                "RMSE_g"
            ],
            "measured40_test_MAE_g": baseline_metrics[
                "RandomForest"
            ][
                "MAE_g"
            ],
        },
        {
            "model_class": "MLP",
            "selected_hyperparameters": json.dumps(
                {
                    "architecture": (
                        "81-10-10-10-10-1"
                    ),
                    "seed": int(
                        mlp_seed
                    ),
                    "checkpoint": str(
                        mlp_checkpoint_path
                    ),
                }
            ),
            "measured40_validation_RMSE_g": float(
                mlp_checkpoint.get(
                    "best_validation_RMSE",
                    np.nan,
                )
            ),
            "measured40_test_R2": baseline_metrics[
                "MLP"
            ][
                "R2"
            ],
            "measured40_test_RMSE_g": baseline_metrics[
                "MLP"
            ][
                "RMSE_g"
            ],
            "measured40_test_MAE_g": baseline_metrics[
                "MLP"
            ][
                "MAE_g"
            ],
        },
    ]
)

selection_summary_path = (
    OUT
    / "measured40_reference_model_summary.csv"
)

selection_summary.to_csv(
    selection_summary_path,
    index=False,
)


# =============================================================================
# 16. MODEL-CLASS DEGRADATION SUMMARY ACROSS THE 11 DIRECT RECONSTRUCTIONS
# =============================================================================

direct_only = results_df.loc[
    results_df[
        "kind"
    ]
    == "direct_mlp_reconstructed40"
].copy()

degradation_summary = (
    direct_only
    .groupby(
        "model_class",
        as_index=False,
    )
    .agg(
        n_reconstructions=(
            "representation",
            "count",
        ),
        transfer_R2_mean=(
            "R2",
            "mean",
        ),
        transfer_R2_min=(
            "R2",
            "min",
        ),
        transfer_R2_max=(
            "R2",
            "max",
        ),
        delta_R2_mean=(
            "delta_R2",
            "mean",
        ),
        delta_R2_min=(
            "delta_R2",
            "min",
        ),
        delta_R2_max=(
            "delta_R2",
            "max",
        ),
        delta_RMSE_g_mean=(
            "delta_RMSE_g",
            "mean",
        ),
        delta_MAE_g_mean=(
            "delta_MAE_g",
            "mean",
        ),
    )
)

degradation_summary_path = (
    OUT
    / "model_class_transfer_degradation_summary.csv"
)

degradation_summary.to_csv(
    degradation_summary_path,
    index=False,
)


# =============================================================================
# 17. WIDE TABLES FOR MANUSCRIPT / FIGURE PREPARATION
# =============================================================================

r2_wide = results_df.pivot(
    index="representation",
    columns="model_class",
    values="R2",
).reset_index()

delta_r2_wide = results_df.pivot(
    index="representation",
    columns="model_class",
    values="delta_R2",
).reset_index()

rmse_wide = results_df.pivot(
    index="representation",
    columns="model_class",
    values="RMSE_g",
).reset_index()

delta_rmse_wide = results_df.pivot(
    index="representation",
    columns="model_class",
    values="delta_RMSE_g",
).reset_index()

r2_wide.to_csv(
    OUT
    / "transfer_R2_by_model_and_reconstruction.csv",
    index=False,
)

delta_r2_wide.to_csv(
    OUT
    / "transfer_delta_R2_by_model_and_reconstruction.csv",
    index=False,
)

rmse_wide.to_csv(
    OUT
    / "transfer_RMSE_by_model_and_reconstruction.csv",
    index=False,
)

delta_rmse_wide.to_csv(
    OUT
    / "transfer_delta_RMSE_by_model_and_reconstruction.csv",
    index=False,
)


# =============================================================================
# 18. SIMPLE FIGURES
# =============================================================================

import matplotlib.pyplot as plt


# ---- Figure A: measured40 baseline R2 for each model class ----

baseline_plot = selection_summary.copy()

fig, ax = plt.subplots(
    figsize=(6.5, 4.5)
)

ax.bar(
    baseline_plot[
        "model_class"
    ],
    baseline_plot[
        "measured40_test_R2"
    ],
)

ax.set_ylabel(
    "Measured40 test R²"
)

ax.set_title(
    "Reference-model performance on genuine measured HSI"
)

ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    OUT
    / "measured40_reference_model_R2.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ---- Figure B: distribution of delta R2 across direct reconstructions ----

model_order = [
    "PLSR",
    "RandomForest",
    "MLP",
]

box_data = [
    direct_only.loc[
        direct_only[
            "model_class"
        ]
        == model_name,
        "delta_R2",
    ].to_numpy()
    for model_name in model_order
]

fig, ax = plt.subplots(
    figsize=(7.0, 4.8)
)

ax.boxplot(
    box_data,
    labels=model_order,
)

ax.set_ylabel(
    "Prediction-model transfer degradation, ΔR²"
)

ax.set_title(
    "Transfer degradation across 11 reconstructed-HSI inputs"
)

ax.axhline(
    0.0,
    linewidth=1,
    linestyle="--",
)

ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    OUT
    / "transfer_degradation_delta_R2_by_model_class.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ---- Figure C: per-reconstruction delta R2 across model classes ----

plot_df = direct_only.copy()

# Short label = final directory name after ::
plot_df[
    "short_label"
] = (
    plot_df[
        "representation"
    ]
    .astype(str)
    .str.split("::")
    .str[-1]
)

ordered_labels = list(
    dict.fromkeys(
        plot_df[
            "short_label"
        ].tolist()
    )
)

fig, ax = plt.subplots(
    figsize=(11.0, 5.6)
)

x = np.arange(
    len(
        ordered_labels
    )
)

for model_name in model_order:
    values = []

    for label in ordered_labels:
        subset = plot_df.loc[
            (
                plot_df[
                    "model_class"
                ]
                == model_name
            )
            & (
                plot_df[
                    "short_label"
                ]
                == label
            ),
            "delta_R2",
        ]

        if len(subset) != 1:
            values.append(
                np.nan
            )
        else:
            values.append(
                float(
                    subset.iloc[0]
                )
            )

    ax.plot(
        x,
        values,
        marker="o",
        label=model_name,
    )

ax.set_xticks(
    x
)

ax.set_xticklabels(
    ordered_labels,
    rotation=55,
    ha="right",
)

ax.set_ylabel(
    "Prediction-model transfer degradation, ΔR²"
)

ax.set_title(
    "Model dependence of transfer degradation"
)

ax.axhline(
    0.0,
    linewidth=1,
    linestyle="--",
)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUT
    / "transfer_degradation_per_reconstruction.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(
    fig
)


# =============================================================================
# 19. SAVE RUN MANIFEST
# =============================================================================

manifest = {
    "root": str(
        ROOT
    ),
    "n_train": int(
        len(
            train_idx
        )
    ),
    "n_validation": int(
        len(
            val_idx
        )
    ),
    "n_test": int(
        len(
            test_idx
        )
    ),
    "n_direct_reconstructions": int(
        len(
            direct_representations
        )
    ),
    "include_gram_reference_outputs": False,
    "PLSR": {
        "selected_n_components": int(
            best_plsr_key
        ),
        "selection_metric": (
            "measured40 validation RMSE"
        ),
        "scaler": (
            "StandardScaler fit on measured40 train only; "
            "frozen for reconstructed inputs"
        ),
    },
    "RandomForest": {
        "selected_grid_index": int(
            best_rf_key
        ),
        "selected_params": best_rf_params,
        "n_estimators": int(
            RF_N_ESTIMATORS
        ),
        "random_state": int(
            RF_RANDOM_STATE
        ),
        "selection_metric": (
            "measured40 validation RMSE"
        ),
        "scaler": "none",
    },
    "MLP": {
        "checkpoint": str(
            mlp_checkpoint_path
        ),
        "seed": int(
            mlp_seed
        ),
        "selection_metric": (
            "measured40 validation RMSE from previous experiment"
        ),
        "scaler": (
            "saved measured40 training scaler from checkpoint; "
            "frozen for reconstructed inputs"
        ),
    },
    "transfer_rule": (
        "For every model class, all model parameters and preprocessing are "
        "fixed before reconstructed HSI is evaluated."
    ),
    "degradation_definition": {
        "delta_R2": (
            "R2_measured40 - R2_reconstructed"
        ),
        "delta_RMSE_g": (
            "RMSE_reconstructed - RMSE_measured40"
        ),
        "delta_MAE_g": (
            "MAE_reconstructed - MAE_measured40"
        ),
    },
}

with open(
    OUT
    / "run_manifest.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        manifest,
        file,
        indent=2,
    )


# =============================================================================
# 20. FINAL CONSOLE SUMMARY
# =============================================================================

print("\n" + "=" * 118)
print("MODEL-CLASS TRANSFER EXPERIMENT COMPLETE")
print("=" * 118)

print(
    "\nMeasured40 reference-model performance:"
)

print(
    selection_summary[
        [
            "model_class",
            "measured40_test_R2",
            "measured40_test_RMSE_g",
            "measured40_test_MAE_g",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print(
    "\nPrediction-model transfer degradation across the 11 direct reconstructions:"
)

print(
    degradation_summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print(
    "\nMain outputs:"
)

print(
    " ",
    results_path,
)

print(
    " ",
    selection_summary_path,
)

print(
    " ",
    degradation_summary_path,
)

print(
    " ",
    OUT
    / "transfer_delta_R2_by_model_and_reconstruction.csv",
)

print(
    " ",
    OUT
    / "transfer_degradation_delta_R2_by_model_class.png",
)

print(
    " ",
    OUT
    / "transfer_degradation_per_reconstruction.png",
)

print(
    "\nInterpretation reminder:"
)

print(
    "  PLSR, Random Forest and MLP differ in both flexibility and inductive bias."
)

print(
    "  Similar degradation across them supports a model-robust reconstruction effect."
)

print(
    "  Different degradation across them shows that task-model transfer is model-dependent."
)