# HSIRecon-Transfer

Code accompanying the manuscript:

**Evaluating Hyperspectral Reconstruction for Downstream Analysis: Wheat Yield Prediction as a Case Study**

This repository reproduces the HSIwheat case study used to compare:

1. direct yield prediction from three measured bands;
2. spectral fidelity of a fixed 3-to-40-band reconstruction;
3. yield prediction after representation-specific retraining on reconstructed HSI;
4. unchanged-model transfer from genuine measured HSI to reconstructed HSI; and
5. model-class sensitivity of transfer degradation using PLSR, random forest, and MLP.

The public HSIwheat dataset is available from the University of Minnesota Data Repository under DOI **10.13020/0ch0-vb18**.

## Repository layout

```text
HSIRecon-Transfer/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── 00_prepare_40band_data.py
│   ├── 01_reproduce_frozen_hsiwheat_benchmark.py
│   ├── 02_run_representative_triplets.py
│   ├── 03_run_t12_yield_aware_control.py
│   ├── 04_run_frozen_mlp_transfer.py
│   └── 05_run_model_class_transfer.py
├── data/
│   └── README.md
├── results/
│   └── README.md
└── docs/
    ├── WORKFLOW.md
    ├── SCRIPT_AUDIT.md
    └── SOURCE_PROVENANCE.md
```

## Main workflow

The scripts are numbered in execution order.

### 00 — Prepare the 40-band working data

`00_prepare_40band_data.py`

Creates the fixed 40-band representation from the released 190-band cubes and writes the wavelength arrays used by later scripts.

### 01 — Reproduce and freeze the HSIwheat downstream benchmark

`01_reproduce_frozen_hsiwheat_benchmark.py`

Builds the paper-aligned spikes-and-leaves masks, subplot metadata, yield targets, and frozen train/validation/test split. These objects are then reused unchanged throughout the representation comparisons.

### 02 — Run the ten representative triplets

`02_run_representative_triplets.py`

Runs the fixed direct reconstruction experiment:

```text
3 measured bands -> 128 -> 128 -> 40 reconstructed bands
```

Only the three input wavelengths change among the ten representative triplets. The script also trains the measured40 yield reference and saves all ten validation-selected MLP checkpoints so that later transfer analyses can use exactly the same frozen models.

Main output:

```text
/home/nibio/HSIwheat_40/experiment2_direct_mlp_3to40/experiment2_results.csv
```

### 03 — Run the T12 yield-aware control

`03_run_t12_yield_aware_control.py`

Performs the greedy train/validation-only band selection used for T12, freezes the selected triplet, and runs the same final reconstruction and downstream protocol. The test labels are not used during band selection.

Main combined output:

```text
/home/nibio/HSIwheat_40/experiment2_direct_mlp_3to40/
    t12_yield_optimized_control/experiment2_plus_t12.csv
```

### 04 — Unchanged MLP transfer

`04_run_frozen_mlp_transfer.py`

Selects the measured40 MLP by measured40 validation RMSE, freezes its weights and measured40 training scaler, and applies the unchanged pipeline to all 11 reconstructed-HSI representations. No reconstructed-data scaler fitting, fine-tuning, or reconstruction-specific model selection is performed.

### 05 — Model-class sensitivity

`05_run_model_class_transfer.py`

Repeats unchanged-model transfer with:

- PLSR;
- random forest; and
- the frozen measured40 MLP.

PLSR and random-forest hyperparameters are selected using genuine measured40 validation data only. Every complete pipeline is frozen before reconstructed HSI is evaluated.

## Expected data roots

The release scripts retain the paths used for the manuscript experiments:

```text
/home/nibio/HSIwheat
/home/nibio/HSIwheat_40
```

If your data are stored elsewhere, edit the `ROOT` / `ROOT40` variables near the top of each script.

See `data/README.md` for the expected input layout.

## Environment

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The analysis uses NumPy, pandas, matplotlib, scikit-learn, and PyTorch. A CUDA-capable GPU is used automatically when available by the neural-network scripts.

## Reproducibility notes

- The reconstruction train/validation/test split is grouped by plot.
- The held-out reconstruction test plots are not used to fit the 3-to-40-band reconstruction model.
- The downstream subplot split is frozen and reused for every representation.
- The direct reconstruction architecture and training settings are fixed across representative triplets.
- T12 band selection uses training and validation yield data only.
- The primary measured40 MLP is selected only by measured40 validation RMSE.
- Unchanged-model transfer never refits a scaler or predictor on reconstructed HSI.
- Cross-triplet correlations are descriptive because the wavelength triplets were deliberately chosen rather than randomly sampled.

## Release refactoring

The final manuscript no longer contains the earlier decomposition/Gram/SparseHSR diagnostic experiments. Those historical scripts are therefore not part of the public execution path.

Two small reproducibility-oriented refactorings were made for this release without changing the reported model definitions or evaluation logic:

1. `02_run_representative_triplets.py` now saves the already-trained measured40 MLP checkpoints and their training-set scalers.
2. Scripts 04 and 05 load those measured40 features/checkpoints directly from the direct-MLP experiment, removing the previous dependency on an older SparseHSR four-way output directory.

The original source provenance and the scripts intentionally excluded from the final workflow are documented under `docs/`.

## Results

Large generated arrays, model checkpoints, and reconstructed cubes are not committed to Git. The scripts create them locally under `/home/nibio/HSIwheat_40`.

Small manuscript-facing CSV/figure outputs can be copied into `results/` after the complete run if desired.

## Citation

Please cite the associated manuscript when using this code. The full bibliographic citation can be added here after publication.
