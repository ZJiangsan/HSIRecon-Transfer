# Data

This repository does not redistribute the HSIwheat dataset.

The public HSIwheat dataset used in the manuscript is available from the University of Minnesota Data Repository under DOI:

**10.13020/0ch0-vb18**

The scripts retain the directory layout used in the experiments.

## Expected original-data layout

```text
/home/nibio/HSIwheat/
├── C3_numpy/
│   └── C3_numpy/
│       └── *.npy
├── C4_numpy/
│   └── C4_numpy/
│       └── *.npy
├── C9_numpy/
│   └── C9_numpy/
│       └── *.npy
└── Yield_data/
    └── Yield_data/
        └── yield_data.pickle
```

The Stage-1 benchmark script searches the field cube folders recursively, while the 40-band preparation script uses the nested `FIELD_numpy/FIELD_numpy` locations shown above.

## Generated working directory

The workflow writes derived data to:

```text
/home/nibio/HSIwheat_40/
```

Important generated subdirectories include:

```text
original190_paper_reproduction/
experiment2_direct_mlp_3to40/
frozen_measured40_model_transfer/
prediction_model_transfer_model_classes/
```

Edit `ROOT` / `ROOT40` near the top of the scripts if your data are stored elsewhere.
