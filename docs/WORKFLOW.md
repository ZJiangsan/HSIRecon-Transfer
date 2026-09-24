# Workflow

## Execution order

Run the scripts in numerical order.

```bash
python scripts/00_prepare_40band_data.py
python scripts/01_reproduce_frozen_hsiwheat_benchmark.py
python scripts/02_run_representative_triplets.py
python scripts/03_run_t12_yield_aware_control.py
python scripts/04_run_frozen_mlp_transfer.py
python scripts/05_run_model_class_transfer.py
```

## Dependency chain

```text
Released HSIwheat 190-band cubes + yield data
                  |
                  +--> 00: measured 40-band cubes + wavelength definitions
                  |
                  +--> 01: frozen SL masks, subplot metadata, targets, split
                                      |
                                      v
                  02: 10 representative triplets
                     - measured40 reference MLP
                     - direct sparse prediction
                     - 3->40 direct MLP reconstruction
                     - retrained reconstructed-HSI prediction
                     - spectral RMSE / SAM
                                      |
                                      v
                  03: T12 yield-aware selection + final T12 evaluation
                                      |
                         11 reconstructed-HSI feature sets
                            /                         \
                           v                           v
                  04: frozen MLP transfer       05: PLSR/RF/MLP transfer
```

## Important output dependencies

Script 02 creates:

```text
experiment2_direct_mlp_3to40/cache/measured40_features_frozen.npy
experiment2_direct_mlp_3to40/measured40_yield/checkpoints/seed_*_bestval.pth
```

Scripts 04 and 05 use these files directly.

Script 03 requires the completed 10-triplet table from script 02 and appends the T12 result to a new combined table; it does not overwrite the original 10-triplet result table.

## Scientific controls encoded in the workflow

### Reconstruction

For the ten representative triplets, only the observed wavelengths change. Architecture, optimizer, loss, split, early stopping, downstream masks, downstream split, and downstream predictor settings remain fixed.

### T12

Band selection is performed using the frozen train/validation yield data only. The held-out test labels are evaluated only after the final three bands are fixed.

### Adapted downstream utility

A new yield predictor is fitted to each reconstructed representation using that representation's training-set standardization.

### Unchanged-model transfer

The measured40 reference predictor and its measured40 training scaler are frozen before any reconstructed input is evaluated. No reconstructed-data fine-tuning, recalibration, or scaler fitting is allowed.

### Model-class sensitivity

PLSR and random forest are developed exclusively on genuine measured40 train/validation data. The existing measured40 MLP checkpoint is reused. The three model classes are not treated as a strict model-capacity ladder.
