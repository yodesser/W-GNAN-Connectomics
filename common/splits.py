"""Frozen CV splits shared by every model.

Design: 3 repeats x 5-fold stratified outer CV (seed 2026). Inside each outer-train
set, a stratified 20 % inner-validation set (seed 1000*repeat + fold).
    train -> fit parameters
    val   -> early stopping, hyper-parameter choice (e.g. C), decision threshold
    test  -> touched ONCE, predictions written to results/preds/<model>.csv

The file splits/cv_splits.csv is generated ONCE (python -m common.splits --make)
and then never regenerated. Every model reads it through iter_splits().

Quick mode for slow models (GAT/GCN/GNAN permutations): iter_splits(repeats=[0]).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import paths

N_REPEATS, N_FOLDS, SEED, VAL_FRAC = 3, 5, 2026, 0.20


def make_splits(y: np.ndarray, subjects: np.ndarray) -> pd.DataFrame:
    from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
    rkf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    rows = []
    for s, (otr, te) in enumerate(rkf.split(np.zeros(len(y)), y)):
        r, f = divmod(s, N_FOLDS)
        tr, va = train_test_split(otr, test_size=VAL_FRAC, stratify=y[otr], random_state=1000 * r + f)
        for role, idx in (("train", tr), ("val", va), ("test", te)):
            rows += [(r, f, int(i), subjects[i], role) for i in np.sort(idx)]
    return pd.DataFrame(rows, columns=["repeat", "fold", "idx", "subject", "role"])


def iter_splits(repeats=None):
    """Yield (repeat, fold, train_idx, val_idx, test_idx) as int numpy arrays."""
    df = pd.read_csv(paths.SPLITS)
    for (r, f), g in df.groupby(["repeat", "fold"], sort=True):
        if repeats is not None and r not in repeats:
            continue
        get = lambda role: g.loc[g.role == role, "idx"].to_numpy(int)
        yield int(r), int(f), get("train"), get("val"), get("test")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true", help="(re)generate the split file -- only ever done once")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.make and paths.SPLITS.exists() and not a.force:
        raise SystemExit("cv_splits.csv already exists and is FROZEN. Do not regenerate.")
    if a.make:
        from .data import load_meta
        meta = load_meta()
        df = make_splits(meta.y.to_numpy(), meta.subject.to_numpy())
        paths.SPLITS.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(paths.SPLITS, index=False)
        print(f"wrote {paths.SPLITS}: {len(df)} rows")
    for r, f, tr, va, te in iter_splits():
        print(r, f, len(tr), len(va), len(te))
