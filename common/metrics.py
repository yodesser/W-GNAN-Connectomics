"""Standard prediction format + all reported metrics.

EVERY model writes exactly one file results/preds/<model>.csv via PredWriter:
    model, repeat, fold, idx, y, score, threshold
  * score     : any monotone "more likely PTSD" score (logit or prob) on the TEST subjects
  * threshold : chosen on that fold's VAL set with choose_threshold() (same scale as score)
plus results/preds/<model>.json with free-form config (hyper-params, runtime, notes).

Reported numbers (make_tables.py):
  * AUROC, AUPRC (AP) pooled over the 5 test folds of a repeat, averaged over repeats
  * 95 % CI: subject-level bootstrap (1000x), same resample indices for every model
  * balanced accuracy, sensitivity, specificity, F1, MCC at the val-chosen thresholds
  * delta AUROC vs a reference model (default lr_edges) with paired bootstrap CI
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)

from . import paths

COLS = ["model", "repeat", "fold", "idx", "y", "score", "threshold"]


def choose_threshold(y_val: np.ndarray, s_val: np.ndarray) -> float:
    """Threshold maximising balanced accuracy on the validation fold."""
    y_val = np.asarray(y_val).astype(int)
    s_val = np.asarray(s_val, float)
    if y_val.min() == y_val.max():
        return float(np.median(s_val))
    cands = np.unique(s_val)
    best_t, best = float(np.median(s_val)), -1.0
    for t in cands:
        b = balanced_accuracy_score(y_val, (s_val >= t).astype(int))
        if b > best:
            best, best_t = b, float(t)
    return best_t


class PredWriter:
    """w = PredWriter('lr_edges'); w.add(r, f, test_idx, y[test_idx], scores, thr); w.save(config={...})"""

    def __init__(self, model: str):
        self.model, self.rows, self.t0 = model, [], time.time()

    def add(self, repeat, fold, idx, y, score, threshold):
        idx, y, score = np.asarray(idx), np.asarray(y), np.asarray(score, float)
        assert len(idx) == len(y) == len(score)
        assert np.isfinite(score).all(), f"{self.model}: non-finite scores in r{repeat}f{fold}"
        for i, yy, s in zip(idx, y, score):
            self.rows.append((self.model, int(repeat), int(fold), int(i), int(yy), float(s), float(threshold)))
        auc = roc_auc_score(y, score) if 0 < y.sum() < len(y) else float("nan")
        print(f"[{self.model}] r{repeat} f{fold}: test AUROC {auc:.3f} (n={len(y)}, pos={int(y.sum())})", flush=True)

    def save(self, config: dict | None = None):
        df = pd.DataFrame(self.rows, columns=COLS)
        out = paths.PREDS / f"{self.model}.csv"
        df.to_csv(out, index=False)
        meta = {"model": self.model, "runtime_s": round(time.time() - self.t0, 1),
                "saved": time.strftime("%Y-%m-%d %H:%M"), **(config or {})}
        (paths.PREDS / f"{self.model}.json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"[{self.model}] wrote {out} ({len(df)} rows)")
        print(summarize(df).to_string())
        return out


# ----------------------------------------------------------------------------- summaries
def _rank_within_fold(df: pd.DataFrame) -> pd.DataFrame:
    """Scores from different folds live on different scales; pooled AUROC uses per-fold
    percentile ranks so pooling is fair. Threshold metrics use the raw score/threshold."""
    df = df.copy()
    g = df.groupby(["repeat", "fold"])["score"]
    df["rscore"] = (g.rank(method="average") - 0.5) / g.transform("size")   # ties -> exactly 0.5
    return df


def _pooled(df: pd.DataFrame, fn) -> float:
    vals = []
    for _, g in df.groupby("repeat"):
        if 0 < g.y.sum() < len(g):
            vals.append(fn(g.y.to_numpy(), g.rscore.to_numpy()))
    return float(np.mean(vals)) if vals else float("nan")


def point_metrics(df: pd.DataFrame) -> dict:
    df = _rank_within_fold(df)
    pred = (df.score >= df.threshold).astype(int)
    y = df.y.to_numpy()
    sens = float(pred[y == 1].mean())
    spec = float(1 - pred[y == 0].mean())
    return {
        "AUROC": _pooled(df, roc_auc_score),
        "AUPRC": _pooled(df, average_precision_score),
        "balAcc": balanced_accuracy_score(y, pred),
        "sens": sens, "spec": spec,
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
        "n_repeats": int(df.repeat.nunique()),
    }


def bootstrap_ci(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0, ref: pd.DataFrame | None = None):
    """Subject-level bootstrap. Returns dict of (lo, hi) for AUROC/AUPRC and, if ref given,
    for delta AUROC (model - ref) on the SAME resamples (paired)."""
    rng = np.random.default_rng(seed)
    df = _rank_within_fold(df)
    subj = np.sort(df.idx.unique())
    by_rep = {r: g.set_index("idx") for r, g in df.groupby("repeat")}
    ref_rep = None
    if ref is not None:
        ref = _rank_within_fold(ref)
        ref_rep = {r: g.set_index("idx") for r, g in ref.groupby("repeat") if r in by_rep}
    A, P, D = [], [], []
    for _ in range(n_boot):
        s = rng.choice(subj, size=len(subj), replace=True)
        a, p, d = [], [], []
        for r, g in by_rep.items():
            gg = g.loc[s]
            if gg.y.sum() in (0, len(gg)):
                continue
            aa = roc_auc_score(gg.y, gg.rscore)
            a.append(aa); p.append(average_precision_score(gg.y, gg.rscore))
            if ref_rep is not None and r in ref_rep:
                rr = ref_rep[r].loc[s]
                d.append(aa - roc_auc_score(rr.y, rr.rscore))
        if a:
            A.append(np.mean(a)); P.append(np.mean(p))
        if d:
            D.append(np.mean(d))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)
    out = {"AUROC_ci": q(A), "AUPRC_ci": q(P)}
    if ref is not None:
        out["dAUROC"] = float(np.mean(D)) if D else np.nan
        out["dAUROC_ci"] = q(D)
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    m = point_metrics(df)
    return pd.DataFrame([m]).round(3)
