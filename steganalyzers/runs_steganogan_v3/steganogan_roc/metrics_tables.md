# Steganography Detector Performance Metrics

Evaluation results for different steganography models against multiple detectors.

## Accuracy

| Detector | XuNet | YeNet | YedroudjNet | SRNet | EfficientNetSteg |
| --- | --- | --- | --- | --- | --- |
| SteganoGAN-Dense | 0.968500 | 0.956500 | 0.959000 | 0.978500 | 0.644500 |
| Edge-UNet | 0.524500 | 0.937000 | 0.965000 | 0.993000 | 0.530500 |
| Edge-ASPP | 0.974500 | 0.926000 | 0.908000 | 0.970500 | 0.503000 |

## Balanced Accuracy

| Detector | XuNet | YeNet | YedroudjNet | SRNet | EfficientNetSteg |
| --- | --- | --- | --- | --- | --- |
| SteganoGAN-Dense | 0.968500 | 0.956500 | 0.959000 | 0.978500 | 0.644500 |
| Edge-UNet | 0.524500 | 0.937000 | 0.965000 | 0.993000 | 0.530500 |
| Edge-ASPP | 0.974500 | 0.926000 | 0.908000 | 0.970500 | 0.503000 |

## AUC-ROC

| Detector | XuNet | YeNet | YedroudjNet | SRNet | EfficientNetSteg |
| --- | --- | --- | --- | --- | --- |
| SteganoGAN-Dense | 0.997091 | 0.996539 | 0.991191 | 0.997167 | 0.978184 |
| Edge-UNet | 0.613253 | 0.993752 | 0.992530 | 0.997898 | 0.915857 |
| Edge-ASPP | 0.999677 | 0.993922 | 0.975497 | 0.996614 | 0.713363 |

## TPR at FPR=0.1

| Detector | XuNet | YeNet | YedroudjNet | SRNet | EfficientNetSteg |
| --- | --- | --- | --- | --- | --- |
| SteganoGAN-Dense | 0.989000 | 0.998000 | 0.981000 | 0.995000 | 0.936000 |
| Edge-UNet | 0.135000 | 0.986000 | 0.981000 | 0.997000 | 0.710000 |
| Edge-ASPP | 0.999000 | 0.989000 | 0.943000 | 0.996000 | 0.281000 |

## Precision

| Detector | XuNet | YeNet | YedroudjNet | SRNet | EfficientNetSteg |
| --- | --- | --- | --- | --- | --- |
| SteganoGAN-Dense | 0.951784 | 0.993514 | 0.968367 | 0.997919 | 1.000000 |
| Edge-UNet | 0.664430 | 0.993228 | 0.968750 | 0.997980 | 1.000000 |
| Edge-ASPP | 0.952336 | 0.993056 | 0.964692 | 0.997884 | 1.000000 |

## Recall

| Detector | XuNet | YeNet | YedroudjNet | SRNet | EfficientNetSteg |
| --- | --- | --- | --- | --- | --- |
| SteganoGAN-Dense | 0.987000 | 0.919000 | 0.949000 | 0.959000 | 0.289000 |
| Edge-UNet | 0.099000 | 0.880000 | 0.961000 | 0.988000 | 0.061000 |
| Edge-ASPP | 0.999000 | 0.858000 | 0.847000 | 0.943000 | 0.006000 |

## F1

| Detector | XuNet | YeNet | YedroudjNet | SRNet | EfficientNetSteg |
| --- | --- | --- | --- | --- | --- |
| SteganoGAN-Dense | 0.969072 | 0.954805 | 0.958586 | 0.978072 | 0.448410 |
| Edge-UNet | 0.172324 | 0.933192 | 0.964859 | 0.992965 | 0.114986 |
| Edge-ASPP | 0.975110 | 0.920601 | 0.902023 | 0.969666 | 0.011928 |

