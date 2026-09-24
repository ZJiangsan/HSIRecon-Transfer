#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 14:38:22 2026

@author: nibio
"""






from pathlib import Path
import numpy as np

root = Path("/home/nibio/HSIwheat")
out_root = Path("/home/nibio/HSIwheat_40")

fields = ["C3", "C4", "C9"]

# ---------------------------------------------------------
# Reconstructed nominal wavelength vector
# ---------------------------------------------------------

wl240 = np.linspace(400, 900, 240)

wl210 = wl240[
    (wl240 >= 430) &
    (wl240 <= 870)
]

keep = np.ones(210, dtype=bool)

# Empirically detected internal gaps
keep[159:163] = False       # 4 channels
keep[178:194] = False       # 16 channels

wl190 = wl210[keep]

assert len(wl190) == 190

# ---------------------------------------------------------
# 40 measured target channels used by the final manuscript
# ---------------------------------------------------------

TARGET_IDX_40 = np.array([
      0,   5,   9,  14,  19,  24,  29,  34,  38,  43,
     48,  53,  57,  62,  67,  72,  77,  81,  86,  91,
     96, 101, 105, 110, 115, 120, 124, 129, 134, 138,
    144, 148, 153, 158, 159, 166, 173, 174, 182, 189
])

wl40 = wl190[TARGET_IDX_40]

print("40 target wavelengths:")
for i, w in enumerate(wl40):
    print(f"{i:2d}: {w:.3f} nm")

# Save spectral definitions
out_root.mkdir(parents=True, exist_ok=True)

np.save(out_root / "wavelengths_190.npy", wl190)
np.save(out_root / "wavelengths_40.npy", wl40)
np.save(out_root / "target_indices_190.npy", TARGET_IDX_40)
# ---------------------------------------------------------
# Process all cubes
# ---------------------------------------------------------

for field in fields:

    input_folder = root / f"{field}_numpy" / f"{field}_numpy"

    target_folder = out_root / field / "hsi40"
    target_folder.mkdir(parents=True, exist_ok=True)

    files = sorted(input_folder.glob("*.npy"))

    print(f"\n{field}: {len(files)} cubes")

    for n, f in enumerate(files, 1):

        cube190 = np.load(f, mmap_mode="r")

        assert cube190.shape[-1] == 190, (
            f"Unexpected spectral dimension: "
            f"{f} {cube190.shape}"
        )

        # 40-channel measured HSI target
        cube40 = np.asarray(
            cube190[:, :, TARGET_IDX_40],
            dtype=np.float32
        )


        np.save(
            target_folder / f.name,
            cube40
        )


        if n % 50 == 0 or n == len(files):
            print(f"  {n}/{len(files)}")

print("\nFinished.")










