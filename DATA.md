# Data

## 1. Dataset

| | |
|---|---|
| Name | **Imagenette** — a 10-class subset of ImageNet |
| Official source | https://github.com/fastai/imagenette (J. Howard, fast.ai) |
| Version used | **imagenette2-320** (version 2; images resized so the shorter side is 320 px) |
| Official download | https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz |
| Copy used in our runs | Kaggle dataset `xbinchen/imagenette2-320` — https://www.kaggle.com/datasets/xbinchen/imagenette2-320 |
| Path on Kaggle | `/kaggle/input/datasets/xbinchen/imagenette2-320/imagenette2-320/` (contains `train/` and `val/`) |
| License | see the official repository |

Images in Imagenette's `train/` folder come from the ImageNet **training** set; images in `val/`
come from the ImageNet **validation** set.

We do **not** create or publish a new dataset: corrupted images are generated on the fly from the
original images by deterministic code (Section 4), so the data can be reproduced exactly from the
official dataset plus this repository.

## 2. Data split

All four backbones are pretrained on ImageNet-1k, i.e. they have already seen the images of
Imagenette's `train/` folder. To avoid evaluating on images seen during pretraining, the **test set
is Imagenette's `val/` folder** and the validation set (used only to pick the best epoch) is taken
from `train/`:

| Split | Source | Share | Images | Used for |
|---|---|---|---|---|
| Train | `train/` folder | 80 % per class | 7,575 | fine-tuning |
| Val | `train/` folder | 20 % per class | 1,894 | selecting the best epoch (clean images only) |
| Test | `val/` folder | all | 3,925 | all reported results (clean + corrupted) |

- Stratified per class, random seed 42 (`random.Random(42)`, file lists sorted before shuffling).
- The exact file lists are saved in
  [`results/run_fix_data_leak/split.json`](results/run_fix_data_leak/split.json).
- Train, val and test share no image.

Per class (label = index in sorted folder-name order):

| Label | Folder | Class | Train | Val | Test |
|---|---|---|---|---|---|
| 0 | `n01440764` | tench | 770 | 193 | 387 |
| 1 | `n02102040` | English springer | 764 | 191 | 395 |
| 2 | `n02979186` | cassette player | 794 | 199 | 357 |
| 3 | `n03000684` | chain saw | 686 | 172 | 386 |
| 4 | `n03028079` | church | 753 | 188 | 409 |
| 5 | `n03394916` | French horn | 765 | 191 | 394 |
| 6 | `n03417042` | garbage truck | 769 | 192 | 389 |
| 7 | `n03425413` | gas pump | 745 | 186 | 419 |
| 8 | `n03445777` | golf ball | 761 | 190 | 399 |
| 9 | `n03888257` | parachute | 768 | 192 | 390 |
| | | **Total** | **7575** | **1894** | **3925** |

## 3. Preprocessing

| Split | Steps |
|---|---|
| Train | `RandomResizedCrop(224)` → `RandomHorizontalFlip()` → `ToTensor` ([0, 1]) → `Normalize` |
| Val / Test | `Resize(256)` → `CenterCrop(224)` → `ToTensor` ([0, 1]) → *degradation (test only)* → clip to [0, 1] → `Normalize` |

`Normalize` uses the ImageNet statistics, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225).
Grayscale / CMYK images are converted to RGB when loaded. No degradation is applied during training.

## 4. Corruptions (test time only)

5 types × 3 levels = 15 conditions, plus clean. Applied to the 224 × 224 test image in [0, 1],
then clipped to [0, 1].

| Group | Type | Level 1 (light) | Level 2 (medium) | Level 3 (heavy) |
|---|---|---|---|---|
| Blur | Gaussian blur | B1 (σ = 1) | B2 (σ = 2) | B3 (σ = 3) |
| Noise | Gaussian noise | N1 (σ = 0.04) | N2 (σ = 0.08) | N3 (σ = 0.12) |
| Illumination | Low light | L1 (f = 0.6, γ = 1.5) | L2 (f = 0.4, γ = 1.9) | L3 (f = 0.2, γ = 2.6) |
| Digital | JPEG compression | J30 (QF = 30) | J15 (QF = 15) | J7 (QF = 7) |
| Digital | Resolution decrease (pixel binning) | D2 (k = 2) | D3 (k = 3) | D4 (k = 4) |

- **Gaussian blur:** 2-D Gaussian kernel normalised to sum 1, size 2⌈3σ⌉ + 1 (7, 13, 19), reflect padding, per channel.
- **Gaussian noise:** x' = x + n, n ~ N(0, σ²), independent per pixel and channel.
- **Low light:** x' = f · x^γ (γ > 1 darkens shadows most, f < 1 scales the whole image down).
- **JPEG:** encode + decode with libjpeg (Pillow), YCbCr with 4:2:0 chroma subsampling, quality factor QF.
- **Pixel binning:** average-pool to round(224 / k) (area interpolation), then nearest-neighbour upsampling back to 224.

**Reproducibility:** the only random corruption is Gaussian noise. Its random generator is seeded per
image from `crc32(seed | image path | corruption | level)` with seed 42, so every run produces
identical corrupted pixels.

## 5. Scripts to reproduce the data

Everything is in [`kaggle/robust_training.ipynb`](kaggle/robust_training.ipynb):

| Block | Purpose |
|---|---|
| Data check | lists the dataset folders, class folders, image sizes and colour modes |
| Block 1 | dataset path, class folders, corruption parameters |
| Block 2 | corruption functions (`apply_degradation`, `FixedDegradation`) |
| Block 3 | split (`build_or_load_split` → `split.json`), preprocessing, data loaders |

Running the notebook with the same seed rebuilds the same `split.json`. The optional `export` step
(CONFIG `steps`) writes every corrupted test condition to `.pt` files if a frozen copy is needed.
