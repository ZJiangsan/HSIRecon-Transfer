# Source provenance

SHA-256 hashes below refer to the source files from which the cleaned release scripts were prepared.

| Source | SHA-256 |
|---|---|
| `uploaded: HSIwheat_data_preparation_40bands_430to870nm.py` | `3ad0dda8ccdbc55354d9013c7ae106f8ed5a7bdaaed033a506e390c127c458a3` |
| `library: hsiwheat_stage1_reproduce_original190_paper_baseline.py` | `5e06930c1e0b68b8c3c15df32594dc75012f5ab5e57fbaa4121291329e431d5a` |
| `recovered original 10-triplet script` | `21d0ecd9c275779a35da2b4318a459c89aebd1ce2bb65a9d66653bf6af76a2fc` |
| `uploaded T12-only script` | `1c2d038eaa2a2f98143eca4c23bb5220b042e87da77260cba260cffc12b81426` |
| `uploaded frozen MLP transfer script` | `bdbf75c25b2995651dcd63dd2185d4eb6dc06c498b850aec5732b59e737791a4` |
| `library: hsiwheat_prediction_model_transfer_model_classes.py` | `e475c4f0ea4217407b80f6f226a15b545899febbc11774bfe39add39bc7bcad7` |

## Release-only changes

The public scripts are deliberately cleaner than the working-directory history:

- duplicate Spyder-style file headers that made `from __future__` imports invalid in two saved copies were removed;
- the unused four-band output in the 40-band preparation script was removed;
- the representative-triplet script now saves the measured40 checkpoints that it already trains;
- the frozen-transfer and model-class scripts now read the measured40 cache/checkpoints from the direct-MLP experiment instead of the legacy SparseHSR four-way directory;
- legacy Gram/SparseHSR discovery branches were removed from the final transfer scripts.

These changes alter file organization and checkpoint persistence, not the reported model architecture, data split, optimization settings, or evaluation definitions.
