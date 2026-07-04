# Jin et al. (2021) Metrics — Detection Accuracy P_A and MMD

Derived from the ROC curves in `results.json` (no re-inference needed).
P_A is exact; MMD is a rank-based proxy (see script docstring).

## Detection Accuracy (P_A = 1 - P_E)

| Detector | XuNet | YeNet | YedroudjNet | SRNet | EfficientNetSteg |
| --- | --- | --- | --- | --- | --- |
| SteganoGAN-Dense | 0.980000 | 0.982500 | 0.959500 | 0.990000 | 0.921000 |
| Edge-UNet | 0.634500 | 0.971000 | 0.965500 | 0.994500 | 0.834000 |
| Edge-ASPP | 0.996000 | 0.972000 | 0.926500 | 0.989000 | 0.659000 |

## MMD (RBF kernel, rank-based proxy)

| Detector | XuNet | YeNet | YedroudjNet | SRNet | EfficientNetSteg |
| --- | --- | --- | --- | --- | --- |
| SteganoGAN-Dense | 0.991917 | 0.992056 | 0.975871 | 0.994938 | 0.938700 |
| Edge-UNet | 0.250537 | 0.984691 | 0.980535 | 0.997271 | 0.783513 |
| Edge-ASPP | 0.999656 | 0.984922 | 0.935243 | 0.993626 | 0.377645 |
