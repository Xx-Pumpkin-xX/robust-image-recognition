# Results (raw output of Part 1)

Each folder is one Kaggle run, kept exactly as the notebook `kaggle/robust_training.ipynb` wrote it.
`analysis/make_report.py` turns it into the tables and figures in `reports/`.

| Run | Test set | Models | Notes |
|---|---|---|---|
| `run_fix_data_leak/` | Imagenette `val/` (3,925 images) | ResNet-50, ConvNeXt-T, DeiT-S, Swin-T | final run, reported results |

## Folder layout

```
run_fix_data_leak/
├── run_info.json          run settings: classes, models, conditions, training config, library versions, GPU
├── split.json             exact file lists of the train / val / test splits
├── summary.csv            one row per model: clean metrics + corruption summary
├── by_severity.csv        one row per model x condition: accuracy, retention, confidence, ...
├── labels.npy             (3925,)   int64    true class of every test image
├── sample_paths.json      (3925)             image path of every row in labels.npy / probs
├── degradation_preview.png                   one test image under every condition
├── logs/<model>_train.json                   validation loss / accuracy per epoch
├── figures/                                  quick-look plots made on Kaggle (final figures: reports/figures/)
└── <model>/                                  resnet50, convnext_tiny, deit_small, swin_tiny
    ├── metrics.json                          all metrics: clean (incl. per class), per condition, overall
    ├── train_history.json                    same as logs/<model>_train.json
    ├── probs/<code>.npy       (3925, 10) float32  softmax output for every test image
    ├── confusion/<code>.npy   (10, 10)   int64    confusion matrix (rows = true, cols = predicted)
    └── confusion_all.npy      (16, 10, 10) int64  all 16 confusion matrices, in condition order
```

## Condition codes (16 per model)

| Code | Condition |
|---|---|
| `clean` | no degradation |
| `B1` `B2` `B3` | Gaussian blur, σ = 1, 2, 3 |
| `N1` `N2` `N3` | Gaussian noise, σ = 0.04, 0.08, 0.12 |
| `L1` `L2` `L3` | low light, (f, γ) = (0.6, 1.5), (0.4, 1.9), (0.2, 2.6) |
| `J30` `J15` `J7` | JPEG, quality 30, 15, 7 |
| `D2` `D3` `D4` | pixel binning, k = 2, 3, 4 |

Details of the degradations: [DATA.md](../DATA.md).

## Loading

```python
from analysis.kaggle_results import load_run
run = load_run('results/run_fix_data_leak')
run.summary                       # pandas DataFrame
run.cm('deit_small', 'D4')        # confusion matrix
run.probs('swin_tiny', 'N3')      # softmax outputs
```
