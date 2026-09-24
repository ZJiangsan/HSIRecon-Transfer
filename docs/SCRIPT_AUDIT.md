# Script audit

The user supplied nine candidate scripts for the GitHub release. They came from several stages of development, not all from the final manuscript.

## Included in the final execution path

| Final release script | Source | Role |
|---|---|---|
| `00_prepare_40band_data.py` | `HSIwheat_data_preparation_40bands_430to870nm.py` | Creates measured 40-band working cubes and wavelength definitions |
| `01_reproduce_frozen_hsiwheat_benchmark.py` | recovered prior source `hsiwheat_stage1_reproduce_original190_paper_baseline.py` | Creates the frozen downstream masks, subplot metadata, targets and split |
| `02_run_representative_triplets.py` | recovered original Experiment-2 source | Runs the ten representative direct-MLP triplets and measured40 reference |
| `03_run_t12_yield_aware_control.py` | uploaded `HSIwheat_superResolution_3to40bands_430to870nm_002_prediction_fullExperiment2.py` | Current uploaded version is the T12-only addition; it explicitly skips the ten representative triplets |
| `04_run_frozen_mlp_transfer.py` | uploaded `...ModelTransferabilityTest_001.py` | Unchanged measured40-MLP transfer |
| `05_run_model_class_transfer.py` | recovered prior source `hsiwheat_prediction_model_transfer_model_classes.py` | PLSR / random-forest / MLP transfer sensitivity |

## Not part of the final manuscript workflow

The following uploaded scripts belong to the earlier decomposition / SparseHSR / reference-mean / mean-removal development path and are intentionally excluded from the main public workflow:

- `HSIwheat_decomposition_40bands_430to870nm_002.py`
- `HSIwheat_superResolution_3to40bands_430to870nm_002_dataPreparation.py`
- `HSIwheat_superResolution_3to40bands_430to870nm_002_reconstruction.py`
- `HSIwheat_superResolution_3to40bands_430to870nm_002_prediction_stage_2_FULL.py`
- `HSIwheat_superResolution_3to40bands_430to870nm_002_prediction_stage_2_MeanRemoved.py`

`HSIwheat_superResolution_3to40bands_430to870nm_002_prediction_stage_2.py` is also from the earlier four-way SparseHSR workflow. It was historically useful because it saved the measured40 checkpoints later reused by transfer scripts, but publishing that entire legacy pipeline solely to create a measured40 checkpoint would make the final repository unnecessarily confusing.

For the release workflow, the same measured40 model training already present in the direct Experiment-2 script now saves its validation-selected checkpoints directly. This removes the legacy dependency while preserving the model architecture, seeds, preprocessing, validation selection rule and downstream split.

## Important correction found during the audit

The uploaded file named `HSIwheat_superResolution_3to40bands_430to870nm_002_prediction_fullExperiment2.py` is **not** the original ten-triplet script anymore. Its active main routine is the later T12-only addition. The original ten-triplet source was therefore recovered separately and is used as release script 02.
