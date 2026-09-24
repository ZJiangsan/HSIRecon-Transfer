#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STAGE 1 — Reproduce the original HSIwheat 190-band yield baseline first
=======================================================================

Goal
----
Do NOT evaluate sparse3 / reconstructed40 / measured40 yet.

First build a credible ORIGINAL-190 yield pipeline and audit it against the
published merged-field experiment from Moghimi et al. (2020).

Published merged-field targets used only for REPRODUCTION AUDIT
----------------------------------------------------------------
Total sub-plots:        51,710
Held-out plots:         50 = C3:20 + C4:10 + C9:20
Test sub-plots:         2,530
Training sub-plots:     44,261
Validation sub-plots:    4,919
Test actual yield sum:  59.36 kg
Sub-plot R²:            ~0.79
Sub-plot RMSE:          ~5.90 g
Plot R²:                ~0.41
Plot NRMSE:             ~0.14

Important limitation of the released dataset
--------------------------------------------
The paper obtained spike/leaf/soil/shadow endmember spectra from a separate
~5 m flight. Those numerical endmembers are not present in the released plot
cube dataset we currently have.

To avoid the previous incorrect "all nonzero pixels = wheat biomass" shortcut,
this script reconstructs a PAPER-LIKE SL segmentation from the released
original 190-band cubes only:

1) sample valid spectra from all three fields;
2) estimate four simplex vertices using a deterministic successive
   maximum-volume / affine-residual procedure (SVMAX-like);
3) average nearby spectra around each vertex, as the paper did to reduce noise;
4) perform fully-constrained least-squares unmixing by projected gradient:
       h >= 0, sum(h)=1;
5) test all 6 possible pairs of the four endmembers as the candidate
   spike+leaf (SL) pair;
6) choose the pair whose resulting per-field sub-plot counts most closely
   reproduce Table 3 of the paper.

Crucially, the SL-pair choice uses ONLY the published sub-plot counts,
NOT yield-prediction performance.

The script then reconstructs the merged test split. The exact 50 plot IDs and
random seed were not published. We therefore search yield-stratified selections
and choose the one that best matches TWO published properties:
    test sub-plots = 2,530
    actual test yield sum = 59.36 kg

Again, no model prediction is used to select the split.

Finally, ONLY original190 is trained:
    381 features = 190 mean + 190 std + SL area
    4 hidden layers x 10 units
    ReLU
    MSE
    Adam
    100 epochs
    lowest-validation-RMSE checkpoint

Because the paper did not publish the neural-network random seed, several
initialization seeds are run. The PRIMARY model is selected by validation RMSE
only; test performance is never used for model selection.

The script reports TWO R² definitions:
    R2_SSE   = 1 - SSE/SST (sklearn r2_score)
    R2_CORR2 = squared Pearson correlation

This is deliberate because the paper names "coefficient of determination"
without giving an explicit equation. The distinction can materially affect
plot-scale R² when predictions have bias.

Once this ORIGINAL190 reproduction is credible, freeze all saved masks,
sub-plots, split IDs, scaling, and model settings. A second script can then
apply exactly the same pipeline to:
    sparse3 / reconstructed40 / measured40 / original190.
"""

from __future__ import annotations

import copy
import json
import math
import pickle
import random
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path("/home/nibio/HSIwheat")
ROOT40 = Path("/home/nibio/HSIwheat_40")

YIELD_PICKLE = (
    ROOT
    / "Yield_data"
    / "Yield_data"
    / "yield_data.pickle"
)

WAVELENGTHS_190_PATH = ROOT40 / "wavelengths_190.npy"

ORIGINAL_CUBE_ROOTS = {
    "C3": ROOT / "C3_numpy",
    "C4": ROOT / "C4_numpy",
    "C9": ROOT / "C9_numpy",
}

OUT = ROOT40 / "original190_paper_reproduction"

ABUNDANCE_ROOT = OUT / "abundance4"
MASK_ROOT = OUT / "sl_masks"


# =============================================================================
# PAPER CONSTANTS
# =============================================================================

FIELDS = ("C3", "C4", "C9")

PAPER_SUBPLOTS_BY_FIELD = {
    "C3": 19287,
    "C4": 12773,
    "C9": 19650,
}

PAPER_TOTAL_SUBPLOTS = 51710

PAPER_TEST_PLOTS = {
    "C3": 20,
    "C4": 10,
    "C9": 20,
}

PAPER_TEST_SUBPLOTS = 2530
PAPER_TRAIN_SUBPLOTS = 44261
PAPER_VALID_SUBPLOTS = 4919
PAPER_TEST_YIELD_G = 59360.0

PAPER_SUBPLOT_R2 = 0.79
PAPER_SUBPLOT_RMSE = 5.90
PAPER_PLOT_R2 = 0.41
PAPER_PLOT_NRMSE = 0.14

SUB_H = 15
SUB_W = 15
SL_THRESHOLD = 0.5


# =============================================================================
# END-MEMBER ESTIMATION SETTINGS
# =============================================================================

# ~300 spectra/cube x 1021 cubes = ~306k spectra.
PIXELS_PER_CUBE_FOR_ENDMEMBERS = 300
ENDMEMBER_SAMPLE_SEED = 2020

# Paper averaged pixels around identified endmembers to reduce sensor noise.
# Numerical distance/radius was not reported, so use K nearest sampled spectra.
ENDMEMBER_NEIGHBORS_TO_AVERAGE = 100

# Projected-gradient FCLS.
FCLS_ITERATIONS = 120
FCLS_TOL = 1e-7
FCLS_CHUNK_SIZE = 50000

# Cache abundances so endmember-pair audit and feature extraction do not
# repeatedly solve the unmixing problem.
REUSE_ABUNDANCE_CACHE = True


# =============================================================================
# SPLIT CALIBRATION SETTINGS
# =============================================================================

# We know the paper used stratified sampling but not its random seed or exact
# plot IDs. Search deterministic candidate seeds and match published TEST
# metadata, not prediction performance.
N_TEST_SPLIT_SEEDS = 50000
TEST_STRATA_BINS = 10
TEST_SPLIT_SEARCH_SEED_START = 0

# Score weights for matching published test metadata.
TEST_COUNT_WEIGHT = 1.0
TEST_YIELD_WEIGHT = 1.0

VALIDATION_RANDOM_STATE = 2020
VALIDATION_STRATA_BINS = 20


# =============================================================================
# DNN SETTINGS
# =============================================================================

HIDDEN = (10, 10, 10, 10)
EPOCHS = 100

# Keras model.fit default batch size is 32 when unspecified.
BATCH_SIZE = 32

ADAM_LR = 1e-3

# Paper did not publish NN random seed.
# Select the primary run ONLY by validation RMSE.
MODEL_SEEDS = tuple(range(10))

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# =============================================================================
# HELPERS
# =============================================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_original_files(field):
    files = sorted(
        ORIGINAL_CUBE_ROOTS[field].rglob("*.npy")
    )

    if not files:
        raise RuntimeError(
            f"No .npy cubes under {ORIGINAL_CUBE_ROOTS[field]}"
        )

    return files


def load_yield_maps():
    with open(YIELD_PICKLE, "rb") as f:
        obj = pickle.load(f)

    maps = {}

    for field in FIELDS:
        df = obj[field]

        m = {}

        for _, row in df.iterrows():
            pid = str(row["plot_ID"]).strip()

            if pid.endswith(".0"):
                pid = pid[:-2]

            m[pid] = float(row["Yield"])

        maps[field] = m

    return maps


def plot_token(field, stem):
    prefix = field + "_"

    if stem.startswith(prefix):
        return stem[len(prefix):]

    return stem


def get_yield(yield_map, field, stem):
    token = plot_token(field, stem)

    for key in (
        token,
        stem,
        token.replace(".0", ""),
        stem.replace(".0", ""),
    ):
        if key in yield_map:
            return float(yield_map[key])

    raise KeyError(
        f"Yield label not found: {field}/{stem}"
    )


def corr2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if (
        len(y_true) < 2
        or np.std(y_true) == 0
        or np.std(y_pred) == 0
    ):
        return np.nan

    r = np.corrcoef(
        y_true,
        y_pred,
    )[0, 1]

    return float(r * r)


# =============================================================================
# STEP 1: SAMPLE ORIGINAL 190-BAND PIXELS
# =============================================================================

def sample_spectra():
    cache = OUT / "endmember_pixel_sample.npy"

    if cache.exists():
        X = np.load(cache)

        print(
            f"[sample] loading cached spectra: {cache} {X.shape}"
        )

        return X.astype(
            np.float32,
            copy=False,
        )

    rng = np.random.default_rng(
        ENDMEMBER_SAMPLE_SEED
    )

    chunks = []

    per_field = {
        f: 0
        for f in FIELDS
    }

    for field in FIELDS:
        files = find_original_files(field)

        for i, path in enumerate(
            files,
            start=1,
        ):
            cube = np.load(
                path
            ).astype(
                np.float32,
                copy=False,
            )

            if cube.ndim != 3 or cube.shape[-1] != 190:
                raise ValueError(
                    f"{path}: expected HxWx190, got {cube.shape}"
                )

            valid = np.any(
                cube > 0,
                axis=2,
            )

            pix = cube[
                valid
            ]

            if len(pix) == 0:
                continue

            n = min(
                PIXELS_PER_CUBE_FOR_ENDMEMBERS,
                len(pix),
            )

            idx = rng.choice(
                len(pix),
                size=n,
                replace=False,
            )

            chunks.append(
                pix[idx]
            )

            per_field[field] += n

        print(
            f"[sample] {field}: {per_field[field]:,} spectra"
        )

    X = np.concatenate(
        chunks,
        axis=0,
    ).astype(
        np.float32
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        cache,
        X,
    )

    print(
        f"[sample] total={len(X):,} spectra -> {cache}"
    )

    return X


# =============================================================================
# STEP 2: SVMAX-LIKE SIMPLEX VERTICES + LOCAL AVERAGING
# =============================================================================

def greedy_affine_simplex_vertices(X, n_vertices=4):
    """
    Deterministic successive affine-volume proxy.

    First vertex:
        spectrum farthest from global mean.

    Subsequent vertices:
        spectrum with largest squared residual distance from the affine span
        of already selected vertices.

    This is used because the numerical endmembers from the separate 5 m
    flight are unavailable in the released dataset.
    """

    X64 = np.asarray(
        X,
        dtype=np.float64,
    )

    center = X64.mean(
        axis=0
    )

    d2 = np.sum(
        (
            X64
            - center
        )
        ** 2,
        axis=1,
    )

    selected = [
        int(
            np.argmax(
                d2
            )
        )
    ]

    while len(selected) < n_vertices:
        anchor = X64[
            selected[0]
        ]

        D = np.stack(
            [
                X64[j]
                - anchor
                for j in selected[1:]
            ],
            axis=1,
        ) if len(selected) > 1 else None

        B = X64 - anchor

        if D is None or D.shape[1] == 0:
            residual = B

        else:
            # Orthonormal basis of current affine span.
            Q, _ = np.linalg.qr(
                D,
                mode="reduced",
            )

            residual = (
                B
                - (
                    B
                    @ Q
                )
                @ Q.T
            )

        r2 = np.sum(
            residual
            * residual,
            axis=1,
        )

        r2[
            selected
        ] = -np.inf

        selected.append(
            int(
                np.argmax(
                    r2
                )
            )
        )

    return selected


def average_near_vertices(
    X,
    vertex_indices,
):
    X64 = np.asarray(
        X,
        dtype=np.float64,
    )

    endmembers = []

    rows = []

    for j, idx in enumerate(
        vertex_indices
    ):
        v = X64[
            idx
        ]

        d2 = np.sum(
            (
                X64
                - v
            )
            ** 2,
            axis=1,
        )

        k = min(
            ENDMEMBER_NEIGHBORS_TO_AVERAGE,
            len(
                X64
            ),
        )

        nn = np.argpartition(
            d2,
            k - 1,
        )[
            :k
        ]

        e = X64[
            nn
        ].mean(
            axis=0
        )

        endmembers.append(
            e
        )

        rows.append(
            {
                "endmember": j,
                "vertex_sample_index": int(
                    idx
                ),
                "n_averaged": int(
                    k
                ),
                "mean_neighbor_distance": float(
                    np.mean(
                        np.sqrt(
                            d2[
                                nn
                            ]
                        )
                    )
                ),
            }
        )

    E = np.stack(
        endmembers,
        axis=0,
    ).astype(
        np.float32
    )

    return E, pd.DataFrame(
        rows
    )


def spectral_diagnostics(
    E,
    wavelengths,
):
    wl = np.asarray(
        wavelengths,
        dtype=np.float64,
    )

    def band_mean(
        spectrum,
        lo,
        hi,
    ):
        mask = (
            (wl >= lo)
            & (wl <= hi)
        )

        if not mask.any():
            idx = int(
                np.argmin(
                    np.abs(
                        wl
                        - (
                            lo
                            + hi
                        )
                        / 2
                    )
                )
            )

            return float(
                spectrum[
                    idx
                ]
            )

        return float(
            np.mean(
                spectrum[
                    mask
                ]
            )
        )

    rows = []

    for j, s in enumerate(
        E
    ):
        blue = band_mean(
            s,
            445,
            455,
        )

        red = band_mean(
            s,
            665,
            675,
        )

        nir = band_mean(
            s,
            780,
            830,
        )

        ndpsi = (
            (
                red
                - blue
            )
            / (
                red
                + blue
                + 1e-12
            )
        )

        ndvi_like = (
            (
                nir
                - red
            )
            / (
                nir
                + red
                + 1e-12
            )
        )

        rows.append(
            {
                "endmember": j,
                "mean_reflectance": float(
                    np.mean(
                        s
                    )
                ),
                "blue_445_455": blue,
                "red_665_675": red,
                "nir_780_830": nir,
                "NDPSI": float(
                    ndpsi
                ),
                "NDVI_like": float(
                    ndvi_like
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def estimate_endmembers():
    end_path = OUT / "estimated_endmembers_4x190.npy"

    diag_path = OUT / "estimated_endmember_diagnostics.csv"

    if (
        end_path.exists()
        and diag_path.exists()
    ):
        E = np.load(
            end_path
        )

        diag = pd.read_csv(
            diag_path
        )

        print(
            f"[endmembers] loading cached {end_path}"
        )

        return E, diag

    X = sample_spectra()

    idx = greedy_affine_simplex_vertices(
        X,
        n_vertices=4,
    )

    E, avg_info = average_near_vertices(
        X,
        idx,
    )

    wavelengths = np.load(
        WAVELENGTHS_190_PATH
    ).reshape(
        -1
    )

    diag = spectral_diagnostics(
        E,
        wavelengths,
    )

    diag = diag.merge(
        avg_info,
        on="endmember",
        how="left",
    )

    np.save(
        end_path,
        E,
    )

    diag.to_csv(
        diag_path,
        index=False,
    )

    print(
        "\nEstimated endmember diagnostics:"
    )

    print(
        diag.to_string(
            index=False
        )
    )

    save_endmember_plot(
        E,
        wavelengths,
        OUT
        / "estimated_endmembers.png",
    )

    return E, diag


def save_endmember_plot(
    E,
    wavelengths,
    path,
):
    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for i in range(
        E.shape[0]
    ):
        ax.plot(
            wavelengths,
            E[i],
            label=f"E{i}",
        )

    ax.set_xlabel(
        "Wavelength (nm)"
    )

    ax.set_ylabel(
        "Reflectance"
    )

    ax.set_title(
        "Estimated four endmembers from released HSI190 cubes"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# STEP 3: PROJECTED-GRADIENT FCLS
# =============================================================================

def project_rows_to_simplex(V):
    """
    Euclidean projection of every row of V onto:
        h >= 0, sum(h) = 1
    """

    U = np.sort(
        V,
        axis=1,
    )[
        :,
        ::-1
    ]

    cssv = np.cumsum(
        U,
        axis=1,
    ) - 1.0

    ind = np.arange(
        1,
        V.shape[1] + 1,
        dtype=np.float64,
    )

    cond = (
        U
        - cssv
        / ind[
            None,
            :
        ]
        > 0
    )

    rho = (
        cond.sum(
            axis=1
        )
        - 1
    )

    theta = (
        cssv[
            np.arange(
                len(
                    V
                )
            ),
            rho,
        ]
        / (
            rho
            + 1
        )
    )

    return np.maximum(
        V
        - theta[
            :,
            None
        ],
        0.0,
    )


def fcls_projected_gradient(
    X,
    E,
):
    """
    Solve min ||X - H E||² with H on the probability simplex.

    X: N x 190
    E: 4 x 190
    H: N x 4
    """

    X = np.asarray(
        X,
        dtype=np.float64,
    )

    E = np.asarray(
        E,
        dtype=np.float64,
    )

    G = E @ E.T  # 4x4

    B = X @ E.T  # Nx4

    # Smooth objective gradient: H G - B.
    eigmax = float(
        np.linalg.eigvalsh(
            G
        ).max()
    )

    step = 0.95 / max(
        eigmax,
        1e-12,
    )

    # Start from unconstrained least-squares then project.
    try:
        Ginv = np.linalg.pinv(
            G
        )

        H = B @ Ginv

    except np.linalg.LinAlgError:
        H = np.full(
            (
                len(
                    X
                ),
                4,
            ),
            0.25,
            dtype=np.float64,
        )

    H = project_rows_to_simplex(
        H
    )

    for _ in range(
        FCLS_ITERATIONS
    ):
        H_new = project_rows_to_simplex(
            H
            - step
            * (
                H @ G
                - B
            )
        )

        diff = float(
            np.max(
                np.abs(
                    H_new
                    - H
                )
            )
        )

        H = H_new

        if diff < FCLS_TOL:
            break

    return H.astype(
        np.float32
    )


def abundance_cache_path(
    field,
    stem,
):
    return (
        ABUNDANCE_ROOT
        / field
        / f"{stem}_abundance4.npy"
    )


def compute_and_cache_abundances(
    E,
):
    ABUNDANCE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for field in FIELDS:
        files = find_original_files(
            field
        )

        print(
            "\n"
            + "-" * 90
        )

        print(
            f"[FCLS] {field}: {len(files)} cubes"
        )

        for i, path in enumerate(
            files,
            start=1,
        ):
            stem = path.stem

            out_path = abundance_cache_path(
                field,
                stem,
            )

            if (
                REUSE_ABUNDANCE_CACHE
                and out_path.exists()
            ):
                if (
                    i == 1
                    or i % 100 == 0
                    or i == len(
                        files
                    )
                ):
                    print(
                        f"  {i}/{len(files)} cache exists: {stem}"
                    )

                continue

            cube = np.load(
                path
            ).astype(
                np.float32,
                copy=False,
            )

            valid = np.any(
                cube > 0,
                axis=2,
            )

            pix = cube[
                valid
            ]

            H = np.zeros(
                (
                    cube.shape[0],
                    cube.shape[1],
                    4,
                ),
                dtype=np.float32,
            )

            for start in range(
                0,
                len(
                    pix
                ),
                FCLS_CHUNK_SIZE,
            ):
                stop = min(
                    start
                    + FCLS_CHUNK_SIZE,
                    len(
                        pix
                    ),
                )

                h = fcls_projected_gradient(
                    pix[
                        start:stop
                    ],
                    E,
                )

                # Write chunk to flattened valid positions.
                valid_flat_idx = np.flatnonzero(
                    valid.reshape(
                        -1
                    )
                )[
                    start:stop
                ]

                H.reshape(
                    -1,
                    4
                )[
                    valid_flat_idx
                ] = h

            out_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.save(
                out_path,
                H,
            )

            if (
                i == 1
                or i % 50 == 0
                or i == len(
                    files
                )
            ):
                print(
                    f"  {i}/{len(files)} saved: {stem}"
                )


# =============================================================================
# STEP 4: CHOOSE SL ENDMEMBER PAIR BY PUBLISHED SUB-PLOT COUNTS
# =============================================================================

def subplot_count_from_mask(
    mask,
):
    H, W = mask.shape

    count = 0

    for r0 in range(
        0,
        H,
        SUB_H,
    ):
        r1 = min(
            H,
            r0
            + SUB_H,
        )

        for c0 in range(
            0,
            W,
            SUB_W,
        ):
            c1 = min(
                W,
                c0
                + SUB_W,
            )

            if mask[
                r0:r1,
                c0:c1
            ].any():
                count += 1

    return count


def audit_endmember_pairs():
    rows = []

    pair_list = list(
        combinations(
            range(4),
            2,
        )
    )

    field_counts = {
        pair: {
            f: 0
            for f in FIELDS
        }
        for pair in pair_list
    }

    for field in FIELDS:
        files = find_original_files(
            field
        )

        for i, path in enumerate(
            files,
            start=1,
        ):
            stem = path.stem

            H = np.load(
                abundance_cache_path(
                    field,
                    stem,
                )
            )

            for pair in pair_list:
                sl = (
                    H[
                        :,
                        :,
                        pair[0]
                    ]
                    + H[
                        :,
                        :,
                        pair[1]
                    ]
                    > SL_THRESHOLD
                )

                n_sub = subplot_count_from_mask(
                    sl
                )

                field_counts[
                    pair
                ][
                    field
                ] += n_sub

        print(
            f"[pair audit] processed {field}"
        )

    for pair in pair_list:
        errors = []

        total = 0

        for field in FIELDS:
            observed = field_counts[
                pair
            ][
                field
            ]

            expected = PAPER_SUBPLOTS_BY_FIELD[
                field
            ]

            rel = (
                observed
                - expected
            ) / expected

            errors.append(
                abs(
                    rel
                )
            )

            total += observed

        total_rel = (
            total
            - PAPER_TOTAL_SUBPLOTS
        ) / PAPER_TOTAL_SUBPLOTS

        score = float(
            np.mean(
                errors
                + [
                    abs(
                        total_rel
                    )
                ]
            )
        )

        rows.append(
            {
                "pair": f"{pair[0]}+{pair[1]}",
                "endmember_a": pair[0],
                "endmember_b": pair[1],
                "C3_subplots": field_counts[
                    pair
                ][
                    "C3"
                ],
                "C4_subplots": field_counts[
                    pair
                ][
                    "C4"
                ],
                "C9_subplots": field_counts[
                    pair
                ][
                    "C9"
                ],
                "ALL_subplots": total,
                "C3_rel_error": (
                    field_counts[
                        pair
                    ][
                        "C3"
                    ]
                    - PAPER_SUBPLOTS_BY_FIELD[
                        "C3"
                    ]
                )
                / PAPER_SUBPLOTS_BY_FIELD[
                    "C3"
                ],
                "C4_rel_error": (
                    field_counts[
                        pair
                    ][
                        "C4"
                    ]
                    - PAPER_SUBPLOTS_BY_FIELD[
                        "C4"
                    ]
                )
                / PAPER_SUBPLOTS_BY_FIELD[
                    "C4"
                ],
                "C9_rel_error": (
                    field_counts[
                        pair
                    ][
                        "C9"
                    ]
                    - PAPER_SUBPLOTS_BY_FIELD[
                        "C9"
                    ]
                )
                / PAPER_SUBPLOTS_BY_FIELD[
                    "C9"
                ],
                "ALL_rel_error": total_rel,
                "count_match_score": score,
            }
        )

    audit = pd.DataFrame(
        rows
    ).sort_values(
        "count_match_score"
    ).reset_index(
        drop=True
    )

    audit.to_csv(
        OUT
        / "endmember_pair_count_audit.csv",
        index=False,
    )

    print(
        "\nENDMEMBER PAIR / TABLE-3 COUNT AUDIT:"
    )

    print(
        audit.to_string(
            index=False
        )
    )

    best = audit.iloc[
        0
    ]

    pair = (
        int(
            best[
                "endmember_a"
            ]
        ),
        int(
            best[
                "endmember_b"
            ]
        ),
    )

    print(
        "\nSelected SL pair:",
        pair,
        "using published subplot counts only.",
    )

    return pair, audit


# =============================================================================
# STEP 5: FREEZE MASKS + EXTRACT ORIGINAL190 FEATURES
# =============================================================================

def build_original190_dataset(
    sl_pair,
):
    yield_maps = load_yield_maps()

    MASK_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    meta_rows = []

    features = []

    targets = []

    for field in FIELDS:
        files = find_original_files(
            field
        )

        print(
            "\n"
            + "-" * 90
        )

        print(
            f"[features] {field}"
        )

        for i, path in enumerate(
            files,
            start=1,
        ):
            cube = np.load(
                path
            ).astype(
                np.float32,
                copy=False,
            )

            stem = path.stem

            plot_yield = get_yield(
                yield_maps[
                    field
                ],
                field,
                stem,
            )

            H = np.load(
                abundance_cache_path(
                    field,
                    stem,
                )
            )

            sl = (
                H[
                    :,
                    :,
                    sl_pair[0]
                ]
                + H[
                    :,
                    :,
                    sl_pair[1]
                ]
                > SL_THRESHOLD
            )

            mask_path = (
                MASK_ROOT
                / field
                / f"{stem}_sl_mask.npy"
            )

            mask_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.save(
                mask_path,
                sl
            )

            total_sl = int(
                sl.sum()
            )

            if total_sl <= 0:
                raise RuntimeError(
                    f"No SL pixels: {field}/{stem}"
                )

            Hh, Ww = sl.shape

            subplot_index = 0

            for r0 in range(
                0,
                Hh,
                SUB_H,
            ):
                r1 = min(
                    Hh,
                    r0
                    + SUB_H,
                )

                for c0 in range(
                    0,
                    Ww,
                    SUB_W,
                ):
                    c1 = min(
                        Ww,
                        c0
                        + SUB_W,
                    )

                    m = sl[
                        r0:r1,
                        c0:c1
                    ]

                    n_sl = int(
                        m.sum()
                    )

                    if n_sl <= 0:
                        continue

                    pix = cube[
                        r0:r1,
                        c0:c1,
                        :
                    ][
                        m
                    ]

                    mean = pix.mean(
                        axis=0
                    )

                    std = pix.std(
                        axis=0,
                        ddof=0,
                    )

                    feat = np.concatenate(
                        [
                            mean,
                            std,
                            np.array(
                                [
                                    float(
                                        n_sl
                                    )
                                ],
                                dtype=np.float32,
                            ),
                        ]
                    ).astype(
                        np.float32
                    )

                    yi = (
                        float(
                            n_sl
                        )
                        / float(
                            total_sl
                        )
                        * float(
                            plot_yield
                        )
                    )

                    features.append(
                        feat
                    )

                    targets.append(
                        yi
                    )

                    meta_rows.append(
                        {
                            "field": field,
                            "plot": stem,
                            "plot_yield": float(
                                plot_yield
                            ),
                            "subplot_index": int(
                                subplot_index
                            ),
                            "row0": int(
                                r0
                            ),
                            "row1": int(
                                r1
                            ),
                            "col0": int(
                                c0
                            ),
                            "col1": int(
                                c1
                            ),
                            "n_SL_pixels": int(
                                n_sl
                            ),
                            "plot_total_SL_pixels": int(
                                total_sl
                            ),
                            "subplot_yield": float(
                                yi
                            ),
                        }
                    )

                    subplot_index += 1

            if (
                i == 1
                or i % 100 == 0
                or i == len(
                    files
                )
            ):
                print(
                    f"  {i}/{len(files)} {stem} | "
                    f"SL={total_sl} | subplots={subplot_index}"
                )

    meta = pd.DataFrame(
        meta_rows
    )

    X = np.stack(
        features
    ).astype(
        np.float32
    )

    y = np.asarray(
        targets,
        dtype=np.float32,
    )

    meta.to_csv(
        OUT
        / "subplot_metadata.csv",
        index=False,
    )

    np.savez_compressed(
        OUT
        / "original190_features.npz",
        X=X,
        y=y,
    )

    print(
        "\nFrozen original190 dataset:",
        X.shape,
        y.shape,
    )

    print(
        "Expected paper total subplots:",
        PAPER_TOTAL_SUBPLOTS,
    )

    return meta, X, y


# =============================================================================
# STEP 6: RECONSTRUCT THE PUBLISHED-STYLE 50-PLOT TEST SPLIT
# =============================================================================

def add_yield_deciles(
    plot_table,
):
    pieces = []

    for field in FIELDS:
        d = (
            plot_table[
                plot_table[
                    "field"
                ]
                == field
            ]
            .copy()
            .sort_values(
                [
                    "plot_yield",
                    "plot",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        # Rank first so every decile is stable even if yields repeat.
        ranks = d[
            "plot_yield"
        ].rank(
            method="first"
        )

        d[
            "stratum"
        ] = pd.qcut(
            ranks,
            q=TEST_STRATA_BINS,
            labels=False,
            duplicates="drop",
        ).astype(
            int
        )

        pieces.append(
            d
        )

    return pd.concat(
        pieces,
        ignore_index=True,
    )


def stratified_pick_for_seed(
    table,
    seed,
):
    selected = []

    for field in FIELDS:
        d = table[
            table[
                "field"
            ]
            == field
        ]

        n_test = PAPER_TEST_PLOTS[
            field
        ]

        strata = sorted(
            d[
                "stratum"
            ].unique()
        )

        # With 10 strata, allocations are exactly:
        # C3/C9: 2 per decile; C4: 1 per decile.
        base = n_test // len(
            strata
        )

        remainder = n_test % len(
            strata
        )

        rng = np.random.default_rng(
            seed
            + {
                "C3": 0,
                "C4": 1000003,
                "C9": 2000003,
            }[
                field
            ]
        )

        for pos, s in enumerate(
            strata
        ):
            ds = d[
                d[
                    "stratum"
                ]
                == s
            ]

            take = (
                base
                + (
                    1
                    if pos < remainder
                    else 0
                )
            )

            if take <= 0:
                continue

            chosen = rng.choice(
                ds.index.to_numpy(),
                size=take,
                replace=False,
            )

            selected.extend(
                chosen.tolist()
            )

    return table.loc[
        selected
    ].copy()


def search_test_split(
    meta,
):
    plot_table = (
        meta.groupby(
            [
                "field",
                "plot",
                "plot_yield",
            ],
            as_index=False,
        )
        .agg(
            n_subplots=(
                "subplot_index",
                "size",
            ),
        )
    )

    table = add_yield_deciles(
        plot_table
    )

    best = None

    top_rows = []

    for seed in range(
        TEST_SPLIT_SEARCH_SEED_START,
        TEST_SPLIT_SEARCH_SEED_START
        + N_TEST_SPLIT_SEEDS,
    ):
        d = stratified_pick_for_seed(
            table,
            seed,
        )

        n_sub = int(
            d[
                "n_subplots"
            ].sum()
        )

        yield_sum = float(
            d[
                "plot_yield"
            ].sum()
        )

        count_err = abs(
            n_sub
            - PAPER_TEST_SUBPLOTS
        ) / PAPER_TEST_SUBPLOTS

        yield_err = abs(
            yield_sum
            - PAPER_TEST_YIELD_G
        ) / PAPER_TEST_YIELD_G

        score = (
            TEST_COUNT_WEIGHT
            * count_err
            + TEST_YIELD_WEIGHT
            * yield_err
        )

        row = {
            "seed": int(
                seed
            ),
            "test_subplots": n_sub,
            "test_yield_g": yield_sum,
            "test_yield_kg": yield_sum
            / 1000.0,
            "count_relative_error": count_err,
            "yield_relative_error": yield_err,
            "score": score,
        }

        if (
            best is None
            or score
            < best[
                "score"
            ]
        ):
            best = row

        top_rows.append(
            row
        )

    search = (
        pd.DataFrame(
            top_rows
        )
        .sort_values(
            "score"
        )
        .reset_index(
            drop=True
        )
    )

    search.head(
        100
    ).to_csv(
        OUT
        / "test_split_search_top100.csv",
        index=False,
    )

    best_seed = int(
        search.iloc[
            0
        ][
            "seed"
        ]
    )

    test_plots = stratified_pick_for_seed(
        table,
        best_seed,
    ).sort_values(
        [
            "field",
            "plot",
        ]
    )

    test_plots.to_csv(
        OUT
        / "held_out_test_plots.csv",
        index=False,
    )

    print(
        "\nTEST SPLIT CALIBRATION:"
    )

    print(
        search.head(
            10
        ).to_string(
            index=False
        )
    )

    print(
        "\nSelected test seed:",
        best_seed,
    )

    print(
        "Selected test subplots:",
        int(
            test_plots[
                "n_subplots"
            ].sum()
        ),
        "(paper:",
        PAPER_TEST_SUBPLOTS,
        ")",
    )

    print(
        "Selected actual yield:",
        float(
            test_plots[
                "plot_yield"
            ].sum()
        )
        / 1000,
        "kg (paper: 59.36 kg)",
    )

    return test_plots, search


def make_train_val_test_indices(
    meta,
    y,
    test_plots,
):
    test_keys = set(
        zip(
            test_plots[
                "field"
            ],
            test_plots[
                "plot"
            ],
        )
    )

    is_test = np.array(
        [
            (
                f,
                p,
            )
            in test_keys
            for f, p in zip(
                meta[
                    "field"
                ],
                meta[
                    "plot"
                ],
            )
        ],
        dtype=bool,
    )

    test_idx = np.where(
        is_test
    )[0]

    dev_idx = np.where(
        ~is_test
    )[0]

    ranks = pd.Series(
        y[
            dev_idx
        ]
    ).rank(
        method="first"
    )

    strata = pd.qcut(
        ranks,
        q=VALIDATION_STRATA_BINS,
        labels=False,
        duplicates="drop",
    ).to_numpy()

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.10,
        random_state=VALIDATION_RANDOM_STATE,
    )

    dummy = np.zeros(
        (
            len(
                dev_idx
            ),
            1,
        ),
        dtype=np.float32,
    )

    tr_local, va_local = next(
        splitter.split(
            dummy,
            strata,
        )
    )

    train_idx = dev_idx[
        tr_local
    ]

    val_idx = dev_idx[
        va_local
    ]

    labels = np.full(
        len(
            meta
        ),
        "train",
        dtype="<U10",
    )

    labels[
        val_idx
    ] = "validation"

    labels[
        test_idx
    ] = "test"

    split_df = meta[
        [
            "field",
            "plot",
            "subplot_index",
        ]
    ].copy()

    split_df[
        "split"
    ] = labels

    split_df.to_csv(
        OUT
        / "split_assignments.csv",
        index=False,
    )

    print(
        "\nSPLIT COUNTS:"
    )

    print(
        f"train      {len(train_idx):,} "
        f"(paper {PAPER_TRAIN_SUBPLOTS:,})"
    )

    print(
        f"validation {len(val_idx):,} "
        f"(paper {PAPER_VALID_SUBPLOTS:,})"
    )

    print(
        f"test       {len(test_idx):,} "
        f"(paper {PAPER_TEST_SUBPLOTS:,})"
    )

    return train_idx, val_idx, test_idx


# =============================================================================
# STEP 7: PAPER DNN ON ORIGINAL190 ONLY
# =============================================================================

class OriginalPaperMLP(nn.Module):
    def __init__(
        self,
        input_dim,
    ):
        super().__init__()

        layers = []

        previous = input_dim

        for width in HIDDEN:
            dense = nn.Linear(
                previous,
                width,
            )

            nn.init.xavier_uniform_(
                dense.weight
            )

            nn.init.zeros_(
                dense.bias
            )

            layers.extend(
                [
                    dense,
                    nn.ReLU(),
                ]
            )

            previous = width

        out = nn.Linear(
            previous,
            1,
        )

        nn.init.xavier_uniform_(
            out.weight
        )

        nn.init.zeros_(
            out.bias
        )

        layers.append(
            out
        )

        self.net = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x,
    ):
        return self.net(
            x
        ).reshape(
            -1
        )


def standardize(
    X,
    train_idx,
):
    mean = X[
        train_idx
    ].mean(
        axis=0,
        dtype=np.float64,
    )

    std = X[
        train_idx
    ].std(
        axis=0,
        dtype=np.float64,
        ddof=0,
    )

    std[
        std < 1e-12
    ] = 1.0

    Z = (
        (
            X.astype(
                np.float64
            )
            - mean
        )
        / std
    ).astype(
        np.float32
    )

    return (
        Z,
        mean.astype(
            np.float32
        ),
        std.astype(
            np.float32
        ),
    )


def loader(
    X,
    y,
    idx,
    shuffle,
):
    ds = TensorDataset(
        torch.from_numpy(
            X[
                idx
            ]
        ).float(),
        torch.from_numpy(
            y[
                idx
            ]
        ).float(),
    )

    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False,
    )


def loader_mse(
    model,
    dl,
    device,
):
    model.eval()

    sse = 0.0
    n = 0

    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(
                device
            )

            yb = yb.to(
                device
            )

            pred = model(
                xb
            )

            sse += float(
                torch.sum(
                    (
                        pred
                        - yb
                    )
                    ** 2
                ).item()
            )

            n += int(
                yb.numel()
            )

    return sse / max(
        n,
        1,
    )


def batched_predict(
    model,
    X,
    idx,
    device,
):
    model.eval()

    out = []

    with torch.no_grad():
        for start in range(
            0,
            len(
                idx
            ),
            4096,
        ):
            ii = idx[
                start:
                start
                + 4096
            ]

            xb = torch.from_numpy(
                X[
                    ii
                ]
            ).float().to(
                device
            )

            out.append(
                model(
                    xb
                )
                .cpu()
                .numpy()
            )

    return np.concatenate(
        out
    )


def evaluate_predictions(
    meta,
    y,
    pred_subplot,
    test_idx,
):
    true_subplot = y[
        test_idx
    ]

    subplot_rmse = math.sqrt(
        mean_squared_error(
            true_subplot,
            pred_subplot,
        )
    )

    subplot = {
        "R2_SSE": float(
            r2_score(
                true_subplot,
                pred_subplot,
            )
        ),
        "R2_CORR2": corr2(
            true_subplot,
            pred_subplot,
        ),
        "RMSE": float(
            subplot_rmse
        ),
        "MAE": float(
            mean_absolute_error(
                true_subplot,
                pred_subplot,
            )
        ),
        "NRMSE": float(
            subplot_rmse
            / np.mean(
                true_subplot
            )
        ),
    }

    d = (
        meta.iloc[
            test_idx
        ]
        .reset_index(
            drop=True
        )
        .copy()
    )

    d[
        "true_subplot_yield"
    ] = true_subplot

    d[
        "pred_subplot_yield"
    ] = pred_subplot

    plots = (
        d.groupby(
            [
                "field",
                "plot",
                "plot_yield",
            ],
            as_index=False,
        )
        .agg(
            predicted_yield=(
                "pred_subplot_yield",
                "sum",
            ),
            n_subplots=(
                "subplot_index",
                "size",
            ),
        )
    )

    true_plot = plots[
        "plot_yield"
    ].to_numpy(
        dtype=np.float64
    )

    pred_plot = plots[
        "predicted_yield"
    ].to_numpy(
        dtype=np.float64
    )

    plot_rmse = math.sqrt(
        mean_squared_error(
            true_plot,
            pred_plot,
        )
    )

    plot = {
        "R2_SSE": float(
            r2_score(
                true_plot,
                pred_plot,
            )
        ),
        "R2_CORR2": corr2(
            true_plot,
            pred_plot,
        ),
        "RMSE": float(
            plot_rmse
        ),
        "MAE": float(
            mean_absolute_error(
                true_plot,
                pred_plot,
            )
        ),
        "NRMSE": float(
            plot_rmse
            / np.mean(
                true_plot
            )
        ),
        "actual_total_kg": float(
            np.sum(
                true_plot
            )
            / 1000.0
        ),
        "predicted_total_kg": float(
            np.sum(
                pred_plot
            )
            / 1000.0
        ),
    }

    return subplot, plot, d, plots


def train_seed(
    model_seed,
    Z,
    y,
    meta,
    train_idx,
    val_idx,
    test_idx,
    feature_mean,
    feature_std,
):
    seed_everything(
        model_seed
    )

    device = torch.device(
        DEVICE
    )

    train_loader = loader(
        Z,
        y,
        train_idx,
        True,
    )

    val_loader = loader(
        Z,
        y,
        val_idx,
        False,
    )

    model = OriginalPaperMLP(
        Z.shape[
            1
        ]
    ).to(
        device
    )

    opt = torch.optim.Adam(
        model.parameters(),
        lr=ADAM_LR,
    )

    mse = nn.MSELoss()

    best_val = np.inf
    best_epoch = -1
    best_state = None

    history = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        sse = 0.0
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(
                device
            )

            yb = yb.to(
                device
            )

            opt.zero_grad(
                set_to_none=True
            )

            pred = model(
                xb
            )

            loss = mse(
                pred,
                yb,
            )

            loss.backward()

            opt.step()

            sse += float(
                torch.sum(
                    (
                        pred.detach()
                        - yb
                    )
                    ** 2
                ).item()
            )

            n += int(
                yb.numel()
            )

        train_mse = sse / max(
            n,
            1,
        )

        val_mse = loader_mse(
            model,
            val_loader,
            device,
        )

        if val_mse < best_val:
            best_val = float(
                val_mse
            )

            best_epoch = int(
                epoch
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

        history.append(
            {
                "model_seed": model_seed,
                "epoch": epoch,
                "train_RMSE": math.sqrt(
                    train_mse
                ),
                "validation_RMSE": math.sqrt(
                    val_mse
                ),
                "best_validation_RMSE": math.sqrt(
                    best_val
                ),
                "best_epoch": best_epoch,
            }
        )

    model.load_state_dict(
        best_state
    )

    pred = batched_predict(
        model,
        Z,
        test_idx,
        device,
    )

    (
        subplot_metrics,
        plot_metrics,
        subplot_predictions,
        plot_predictions,
    ) = evaluate_predictions(
        meta,
        y,
        pred,
        test_idx,
    )

    ckpt_dir = OUT / "checkpoints"

    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_seed": model_seed,
            "best_epoch": best_epoch,
            "best_validation_RMSE": math.sqrt(
                best_val
            ),
            "model_state_dict": best_state,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "hidden_units": HIDDEN,
        },
        ckpt_dir
        / f"original190_seed{model_seed}_bestval.pth",
    )

    row = {
        "model_seed": model_seed,
        "best_epoch": best_epoch,
        "best_validation_RMSE": math.sqrt(
            best_val
        ),
        "subplot_R2_SSE": subplot_metrics[
            "R2_SSE"
        ],
        "subplot_R2_CORR2": subplot_metrics[
            "R2_CORR2"
        ],
        "subplot_RMSE": subplot_metrics[
            "RMSE"
        ],
        "subplot_NRMSE": subplot_metrics[
            "NRMSE"
        ],
        "plot_R2_SSE": plot_metrics[
            "R2_SSE"
        ],
        "plot_R2_CORR2": plot_metrics[
            "R2_CORR2"
        ],
        "plot_RMSE": plot_metrics[
            "RMSE"
        ],
        "plot_NRMSE": plot_metrics[
            "NRMSE"
        ],
        "actual_total_kg": plot_metrics[
            "actual_total_kg"
        ],
        "predicted_total_kg": plot_metrics[
            "predicted_total_kg"
        ],
    }

    return {
        "row": row,
        "history": pd.DataFrame(
            history
        ),
        "subplot_predictions": subplot_predictions,
        "plot_predictions": plot_predictions,
    }


def train_original190(
    meta,
    X,
    y,
    train_idx,
    val_idx,
    test_idx,
):
    Z, mean, std = standardize(
        X,
        train_idx,
    )

    runs = []

    for seed in MODEL_SEEDS:
        print(
            "\n"
            + "=" * 90
        )

        print(
            f"[DNN] original190 seed={seed}"
        )

        result = train_seed(
            seed,
            Z,
            y,
            meta,
            train_idx,
            val_idx,
            test_idx,
            mean,
            std,
        )

        runs.append(
            result
        )

        print(
            pd.DataFrame(
                [
                    result[
                        "row"
                    ]
                ]
            ).to_string(
                index=False
            )
        )

    summary = pd.DataFrame(
        [
            r[
                "row"
            ]
            for r in runs
        ]
    ).sort_values(
        "best_validation_RMSE"
    ).reset_index(
        drop=True
    )

    summary.to_csv(
        OUT
        / "original190_seed_summary.csv",
        index=False,
    )

    primary_seed = int(
        summary.iloc[
            0
        ][
            "model_seed"
        ]
    )

    primary = next(
        r
        for r in runs
        if int(
            r[
                "row"
            ][
                "model_seed"
            ]
        )
        == primary_seed
    )

    all_history = pd.concat(
        [
            r[
                "history"
            ]
            for r in runs
        ],
        ignore_index=True,
    )

    all_history.to_csv(
        OUT
        / "training_history_all_seeds.csv",
        index=False,
    )

    primary[
        "subplot_predictions"
    ].to_csv(
        OUT
        / "primary_test_subplot_predictions.csv",
        index=False,
    )

    primary[
        "plot_predictions"
    ].to_csv(
        OUT
        / "primary_test_plot_predictions.csv",
        index=False,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "ORIGINAL190 REPRODUCTION — PRIMARY RUN"
    )

    print(
        "selected ONLY by lowest validation RMSE"
    )

    print(
        "=" * 100
    )

    print(
        summary.head(
            10
        ).to_string(
            index=False
        )
    )

    return summary, primary_seed


# =============================================================================
# REPORT
# =============================================================================

def save_reproduction_report(
    pair,
    pair_audit,
    test_plots,
    seed_summary,
    primary_seed,
):
    primary = seed_summary[
        seed_summary[
            "model_seed"
        ]
        == primary_seed
    ].iloc[
        0
    ]

    report = {
        "selected_SL_endmember_pair": list(
            pair
        ),
        "SL_pair_selection_basis": (
            "minimum mismatch to published C3/C4/C9/ALL sub-plot counts; "
            "no yield-prediction metric used"
        ),
        "observed_total_subplots": int(
            pair_audit.iloc[
                0
            ][
                "ALL_subplots"
            ]
        ),
        "paper_total_subplots": PAPER_TOTAL_SUBPLOTS,
        "test_plot_count": int(
            len(
                test_plots
            )
        ),
        "test_subplot_count": int(
            test_plots[
                "n_subplots"
            ].sum()
        ),
        "paper_test_subplot_count": PAPER_TEST_SUBPLOTS,
        "test_actual_yield_kg": float(
            test_plots[
                "plot_yield"
            ].sum()
            / 1000.0
        ),
        "paper_test_actual_yield_kg": 59.36,
        "primary_model_seed": int(
            primary_seed
        ),
        "primary_selection_basis": (
            "lowest validation RMSE only"
        ),
        "our": {
            "subplot_R2_SSE": float(
                primary[
                    "subplot_R2_SSE"
                ]
            ),
            "subplot_R2_CORR2": float(
                primary[
                    "subplot_R2_CORR2"
                ]
            ),
            "subplot_RMSE": float(
                primary[
                    "subplot_RMSE"
                ]
            ),
            "subplot_NRMSE": float(
                primary[
                    "subplot_NRMSE"
                ]
            ),
            "plot_R2_SSE": float(
                primary[
                    "plot_R2_SSE"
                ]
            ),
            "plot_R2_CORR2": float(
                primary[
                    "plot_R2_CORR2"
                ]
            ),
            "plot_RMSE": float(
                primary[
                    "plot_RMSE"
                ]
            ),
            "plot_NRMSE": float(
                primary[
                    "plot_NRMSE"
                ]
            ),
            "predicted_total_kg": float(
                primary[
                    "predicted_total_kg"
                ]
            ),
        },
        "paper": {
            "subplot_R2": PAPER_SUBPLOT_R2,
            "subplot_RMSE": PAPER_SUBPLOT_RMSE,
            "plot_R2": PAPER_PLOT_R2,
            "plot_NRMSE": PAPER_PLOT_NRMSE,
            "actual_total_kg": 59.36,
            "predicted_total_kg": 59.49,
        },
        "important_limit": (
            "Original numerical endmember spectra and exact 50 test plot IDs "
            "were not released; this script estimates endmembers from the "
            "released HSI190 cubes and reconstructs a test split from published "
            "metadata. It is a controlled reproduction, not bit-identical."
        ),
    }

    (
        OUT
        / "reproduction_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 100
    )

    print(
        "STAGE 1: ORIGINAL190 PAPER-BASELINE REPRODUCTION"
    )

    print(
        "Do not evaluate sparse3 / reconstructed40 / measured40 yet."
    )

    print(
        "=" * 100
    )

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Target paper merged-model benchmark:"
    )

    print(
        "  sub-plot: R²≈0.79, RMSE≈5.90 g"
    )

    print(
        "  plot:     R²≈0.41, NRMSE≈0.14"
    )

    wavelengths = np.load(
        WAVELENGTHS_190_PATH
    ).reshape(
        -1
    )

    if wavelengths.shape != (190,):
        raise ValueError(
            f"Expected 190 wavelengths, got {wavelengths.shape}"
        )

    # 1–2. Estimate four paper-like endmembers from released HSI190.
    E, end_diag = estimate_endmembers()

    # 3. FCLS abundance maps.
    compute_and_cache_abundances(
        E
    )

    # 4. Identify the most plausible spike+leaf pair using Table-3 counts.
    pair, pair_audit = audit_endmember_pairs()

    # 5. Freeze masks and extract 381-dimensional original190 features.
    meta, X, y = build_original190_dataset(
        pair
    )

    # 6. Reconstruct paper-like held-out plot split from published metadata.
    test_plots, split_search = search_test_split(
        meta
    )

    train_idx, val_idx, test_idx = (
        make_train_val_test_indices(
            meta,
            y,
            test_plots,
        )
    )

    # 7. Train ONLY original190. Multiple seeds; choose by validation only.
    seed_summary, primary_seed = train_original190(
        meta,
        X,
        y,
        train_idx,
        val_idx,
        test_idx,
    )

    report = save_reproduction_report(
        pair,
        pair_audit,
        test_plots,
        seed_summary,
        primary_seed,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STAGE-1 COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    print(
        "\nOutput directory:"
    )

    print(
        OUT
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do not run the other three spectral representations until this "
        "original190 result is judged sufficiently comparable to the paper."
    )


if __name__ == "__main__":
    main()
