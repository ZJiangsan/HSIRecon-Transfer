# Reviewer revision diagnostics (Remote Sensing)

These scripts reproduce **post-hoc analyses** prepared for the revision of
*Evaluating Hyperspectral Reconstruction for Downstream Analysis: Wheat Yield Prediction as a Case Study*.

They do **not** retrain or modify the original 3-to-40-band reconstruction models,
the genuine-HSI yield predictors, or the 11 frozen wavelength-triplet experiments.
They read existing HSIwheat-derived data and model outputs produced by the
numbered scripts in `../`.

## Source scripts

| Script | Analyses and outputs |
|---|---|
| `revision_analyses.py` | `audit` (input inventory), `bootstrap` (field-stratified plot-cluster resampling of paired test predictions), `transfer-table` (full 11-triplet × PLSR/RF/MLP table), `spectra` (per-wavelength RMSE using saved MLP checkpoints) |
| `feature_deviation_diagnostics.py` | Test-subplot deviations of the 40 mean and 40 SD yield features, standardized using **measured-HSI training-set** statistics; per-band and per-plot CSVs, summary tables and plots |
| `plot_real_spectra_only.py` | Illustrative measured-versus-reconstructed 40-band spectrum from a **held-out** plot, using all 11 frozen checkpoints |

The manuscript's **primary** analyses and training workflows remain documented in
[the repository root](../../README.md). These post-hoc analyses are supplemental
and should not be confused with independent external-dataset validation.

## Input data (not distributed in this repository)

Set `HSIWHEAT40_ROOT` to the root containing:

- `wavelengths_40.npy` and `target_indices_190.npy`
- `C3/hsi40/`, `C4/hsi40/`, `C9/hsi40/`
- `original190_paper_reproduction/{subplot_metadata.csv,split_assignments.csv}`
- `experiment2_direct_mlp_3to40/` (11 `best_direct_mlp_3to40.pth` checkpoints, 11 `yield_features.npz` caches, measured-HSI feature cache, plot split)
- `frozen_measured40_model_transfer/frozen_measured40_primary_predictions.csv`
- `prediction_model_transfer_model_classes/{model_class_transfer_predictions.csv,model_class_transfer_results_long.csv}`

The public source HSIwheat dataset is DOI **10.13020/0ch0-vb18**. Locally generated
caches, checkpoints, CSV outputs and potentially large input cubes are not
committed. The diagnostic scripts assume the exact original frozen split and
aligned rows; they check important dimensions before evaluating.

## Running

Requirements: `numpy`, `pandas`, `matplotlib`, `torch` (for spectral
checkpoint inference). Run from any Python environment with these packages.

```bash
export HSIWHEAT40_ROOT=/path/to/HSIwheat_40
python scripts/revision/revision_analyses.py audit --root "$HSIWHEAT40_ROOT"
python scripts/revision/revision_analyses.py bootstrap --root "$HSIWHEAT40_ROOT" --nboot 5000
python scripts/revision/revision_analyses.py transfer-table --root "$HSIWHEAT40_ROOT"
python scripts/revision/revision_analyses.py spectra --root "$HSIWHEAT40_ROOT" --device cuda --max-models 11
python scripts/revision/feature_deviation_diagnostics.py --root "$HSIWHEAT40_ROOT"
python scripts/revision/plot_real_spectra_only.py --root "$HSIWHEAT40_ROOT" --device cpu
```

Outputs are written to `$HSIWHEAT40_ROOT/revision_review2_diagnostics/` by
default. `spectra` is the only GPU-benefiting stage; the other commands can
run on CPU. Optional SLURM launcher: `sbatch --export=ALL,HSIWHEAT40_ROOT=/path/to/HSIwheat_40 scripts/revision/trinity_revision_diagnostics.slurm`.

**Interpretation limits:** The bootstrap resamples plots (not subplots as independent
clusters) while holding trained models fixed. Cross-triplet correlations are
descriptive (11 deliberately selected wavelength configurations, not random
sampling). An individual real spectrum is illustrative; feature deviation is
not by itself a causal explanation of downstream transfer loss.

## Provenance

These are the corrected scripts used for the November 2026 reviewer-diagnostic
work on the original Trinity HSIwheat split. The earlier preliminary spectral
script needed an adjustment because saved model-state keys are prefixed with
`net.`; the committed version handles that checkpoint naming.
