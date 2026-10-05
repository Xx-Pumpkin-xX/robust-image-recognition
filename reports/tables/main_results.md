| Model | Clean acc | Precision | Recall | F1 | CE loss | mCA | Retention | Conf. drop | mCE |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | 98.55 | 98.54 | 98.55 | 98.54 | 0.045 | 89.80 | 91.13 | 6.93 | 100.0 |
| ConvNeXt-T | **99.69** | **99.70** | **99.69** | **99.69** | **0.015** | 90.77 | 91.05 | 5.57 | 66.2 |
| DeiT-S | 99.21 | 99.22 | 99.21 | 99.21 | 0.026 | **94.83** | **95.58** | **2.96** | **47.7** |
| Swin-T | 99.46 | 99.46 | 99.46 | 99.46 | 0.017 | 89.17 | 89.64 | 5.68 | 80.3 |

Test set: 3925 images. Clean acc / precision / recall / F1 / mCA / retention in %, CE loss = cross-entropy on clean images, conf. drop in percentage points of mean max-softmax, mCE relative to ResNet-50 (= 100, lower is better).
