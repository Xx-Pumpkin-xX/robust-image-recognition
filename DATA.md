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
| License | See the official repository |

Images in Imagenette's `train/` folder come from the ImageNet **training** set; images in `val/`
come from the ImageNet **validation** set.

We do **not** create or publish a new dataset. Corrupted images are generated on the fly from the
original images. Gaussian noise is random in form but uses a deterministic per-image seed, so the
evaluation data can be reproduced from the official dataset plus this repository.

## 2. Data split

All four backbones use ImageNet-1k pretrained weights. To avoid evaluating on Imagenette images
originating from the ImageNet training set, the **final test set is Imagenette's `val/` folder**.
Imagenette's `train/` folder is split into fine-tuning and validation subsets:

| Split | Source | Share | Images | Used for |
|---|---|---:|---:|---|
| Train | `train/` folder | 80% per class | 7,575 | fine-tuning |
| Validation | `train/` folder | 20% per class | 1,894 | selecting the best epoch on clean images |
| Test | `val/` folder | all | 3,925 | all reported clean and corrupted results |

- The split is stratified per class with seed 42 using `random.Random(42)`.
- File names are sorted before shuffling, so the split is reproducible.
- The exact file lists are saved in [`results/run_fix_data_leak/split.json`](results/run_fix_data_leak/split.json).
- Train, validation and test share no image.

Per class (label = index in sorted folder-name order):

| Label | Folder | Class | Train | Validation | Test |
|---:|---|---|---:|---:|---:|
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
|  |  | **Total** | **7,575** | **1,894** | **3,925** |

## 3. Preprocessing

| Split | Steps |
|---|---|
| Train | convert to RGB → `RandomResizedCrop(224)` → `RandomHorizontalFlip()` → tensor in [0, 1] → `Normalize` |
| Validation | convert to RGB → `Resize(256)` → `CenterCrop(224)` → tensor in [0, 1] → `Normalize` |
| Test | convert to RGB → `Resize(256)` → `CenterCrop(224)` → tensor in [0, 1] → optional test-time degradation → clip to [0, 1] → `Normalize` |

`Normalize` uses the ImageNet statistics: mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225).
All images are converted to RGB when loaded. No test-time corruption is applied during fine-tuning.

## 4. Corruptions (test time only)

Five corruption types are evaluated at three severity levels, giving **15 corrupted conditions plus
one clean condition**. Corruptions are applied to the deterministic 224 × 224 center-cropped test
image in [0, 1], before ImageNet normalization. The result is clipped to [0, 1].

| Group | Type | Level 1 (light) | Level 2 (medium) | Level 3 (heavy) |
|---|---|---|---|---|
| Blur | Gaussian blur | B1 (σ = 1) | B2 (σ = 2) | B3 (σ = 3) |
| Noise | Gaussian noise | N1 (σ = 0.04) | N2 (σ = 0.08) | N3 (σ = 0.12) |
| Illumination | Low light | L1 (f = 0.6, γ = 1.5) | L2 (f = 0.4, γ = 1.9) | L3 (f = 0.2, γ = 2.6) |
| Digital | JPEG compression | J30 (QF = 30) | J15 (QF = 15) | J7 (QF = 7) |
| Digital | Resolution downscaling | D2 (k = 2) | D3 (k = 3) | D4 (k = 4) |

- **Gaussian blur:** a 2-D Gaussian kernel normalized to sum to 1, with kernel size `2*ceil(3σ)+1` (7, 13, 19), reflect padding, applied per channel.
- **Gaussian noise:** `x' = x + n`, where `n ~ N(0, σ²)` independently per pixel and channel.
- **Low light:** `x' = f * x^γ`; `γ > 1` darkens shadows more strongly and `f < 1` reduces overall intensity.
- **JPEG compression:** encode and decode with Pillow/libjpeg using the stated quality factor and 4:2:0 chroma subsampling.
- **Resolution downscaling:** area-downsample from 224 × 224 to `round(224/k) × round(224/k)`, then upsample back to 224 × 224 with nearest-neighbor interpolation. The internal code key remains `pixelate` because it was used in the completed experiment and saved result files. For D2 and D4, the reduced sizes are 112 × 112 and 56 × 56, so the resulting 2 × 2 and 4 × 4 blocks align exactly with the image grid. For D3, the image is reduced to 75 × 75, so the resulting blocks are only approximately 3 × 3 and are not exactly grid-aligned. This corruption is inspired by reduced sensor resolution and pixel-binning effects, but it is **not** a physical simulation of sensor pixel binning.

**Reproducibility.** Gaussian noise is the only random corruption. Its generator is seeded separately
for each image and condition from `crc32(seed | image path | degradation | level)` with seed 42, so
the same image receives the same noise on repeated runs. The other corruptions are deterministic.

## 5. Scripts and outputs

The data pipeline is implemented in [`kaggle/robust_training.ipynb`](kaggle/robust_training.ipynb):

| Notebook block | Purpose |
|---|---|
| Data-check cell | inspects dataset folders, class folders, image counts, image sizes, color modes and file types |
| Block 1 | dataset path, class names, corruption parameters and condition codes |
| Block 2 | corruption functions (`apply_degradation`, `FixedDegradation`) |
| Block 3 | reproducible split (`build_or_load_split` → `split.json`), preprocessing and data loaders |
| Block 7 | degradation preview and optional export of frozen degraded tensors |
| Block 9 | experiment configuration (`CONFIG`) and pipeline execution |

The reported run uses `steps = ['preview', 'train', 'evaluate']`, so degraded images are regenerated
at evaluation time rather than exported as a second dataset. The optional `export` step writes the
logical split(s) listed in `CONFIG['export_splits']` to `.pt` files; it is not required to reproduce the
reported results.

The raw reported outputs are stored under
[`results/run_fix_data_leak/`](results/run_fix_data_leak/). Derived tables and figures are generated
from those raw outputs by [`analysis/make_report.py`](analysis/make_report.py).
