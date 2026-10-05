"""
Load a results folder produced by the Kaggle notebook (Part 1).

    from kaggle_results import load_run
    run = load_run('results/run_fix_data_leak')     # a run folder or a <run_name>.zip

    run.summary                              # DataFrame, one row per model
    run.by_severity                          # DataFrame, one row per model x condition
    run.cm('resnet50', 'J15')                # (C, C) confusion matrix, rows = true, cols = predicted
    run.cm('resnet50', 'J15', normalize=True)
    run.probs('swin_tiny', 'N3')             # (N, C) softmax per test image
    run.labels                               # (N,) true labels
    run.metrics('deit_small')['clean']['per_class']
    run.history('resnet50')                  # per-epoch validation loss / accuracy

Condition codes: clean, B1-B3 (Gaussian blur), N1-N3 (Gaussian noise), L1-L3 (low light),
J30/J15/J7 (JPEG quality), D2-D4 (pixel binning).
Needs: numpy, pandas.
"""
import json
import os
import zipfile

import numpy as np
import pandas as pd


class Run:
    def __init__(self, folder):
        self.folder = folder
        with open(os.path.join(folder, 'run_info.json')) as f:
            self.info = json.load(f)
        self.name = self.info['run_name']
        self.classes = self.info['class_names']
        self.models = self.info['models']
        self.conditions = pd.DataFrame(self.info['conditions'])      # code, degradation, severity, param
        self.codes = list(self.conditions['code'])
        self.labels = np.load(os.path.join(folder, 'labels.npy'))
        with open(os.path.join(folder, 'sample_paths.json')) as f:
            self.sample_paths = json.load(f)
        self.summary = pd.read_csv(os.path.join(folder, 'summary.csv'))
        self.by_severity = pd.read_csv(os.path.join(folder, 'by_severity.csv'))

    def _check(self, model, code):
        if model not in self.models:
            raise KeyError(f'{model!r} not in this run. Models: {self.models}')
        if code not in self.codes:
            raise KeyError(f'{code!r} not in this run. Conditions: {self.codes}')

    def cm(self, model, code='clean', normalize=False):
        """Confusion matrix (rows = true class, cols = predicted). normalize=True -> each row sums to 1."""
        self._check(model, code)
        m = np.load(os.path.join(self.folder, model, 'confusion', f'{code}.npy'))
        return m / np.maximum(m.sum(1, keepdims=True), 1) if normalize else m

    def cm_all(self, model):
        """(K, C, C) stack in self.codes order."""
        return np.load(os.path.join(self.folder, model, 'confusion_all.npy'))

    def probs(self, model, code='clean'):
        """(N, C) softmax probabilities, row i = image self.sample_paths[i]."""
        self._check(model, code)
        return np.load(os.path.join(self.folder, model, 'probs', f'{code}.npy')).astype(np.float32)

    def preds(self, model, code='clean'):
        return self.probs(model, code).argmax(1)

    def metrics(self, model):
        with open(os.path.join(self.folder, model, 'metrics.json')) as f:
            return json.load(f)

    def history(self, model):
        """Per-epoch validation metrics recorded during training (None if not available)."""
        for p in (os.path.join(self.folder, model, 'train_history.json'),
                  os.path.join(self.folder, 'logs', f'{model}_train.json')):
            if os.path.exists(p):
                with open(p) as f:
                    h = json.load(f)
                return pd.DataFrame(h['history'] if isinstance(h, dict) else h)
        return None

    def __repr__(self):
        return (f"Run({self.name!r}: {len(self.models)} models, {len(self.classes)} classes, "
                f"{len(self.codes)} conditions, {len(self.labels)} test images)")


def _unzip(zip_path):
    """Unpack <run>.zip next to itself (once) and return the run folder."""
    root = os.path.dirname(os.path.abspath(zip_path))
    with zipfile.ZipFile(zip_path) as z:
        top = z.namelist()[0].split('/')[0]
        target = os.path.join(root, top)
        if not os.path.exists(os.path.join(target, 'run_info.json')):
            z.extractall(root)
    return target


def load_run(path):
    """path = a run folder or a <run_name>.zip"""
    return Run(_unzip(path) if path.endswith('.zip') else path)


def load_runs(folder):
    """All runs in a folder (zips are unpacked once). Returns {run_name: Run}."""
    for f in sorted(os.listdir(folder)):
        if f.endswith('.zip'):
            _unzip(os.path.join(folder, f))
    runs = {}
    for f in sorted(os.listdir(folder)):
        p = os.path.join(folder, f)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, 'run_info.json')):
            r = Run(p)
            runs[r.name] = r
    return runs
