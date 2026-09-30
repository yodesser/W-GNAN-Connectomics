"""Non-graph baselines on the frozen splits (CPU, local).

    python -m baselines.train_linear                         # all five, 3 repeats
    python -m baselines.train_linear --models lr_strength    # just one

Models (all: StandardScaler fit on train, class_weight='balanced',
C chosen on the inner-val fold by AP from C_GRID, threshold chosen on val):
    majority     constant score (AUROC 0.5, AUPRC = prevalence)  -- the floor
    lr_scale     [log total streamlines, density]                  -- head-size sanity check
    lr_demo      [age, sex]                                        -- demographic sanity check
    lr_edges     93,096 normalised edges                           -- REFERENCE model
    lr_strength  432 node strengths                                -- coarse-signal model
  sanity rows (Table: sanity checks):
    lr_edges_log1p  93k edges, OLD pipeline (log1p, no per-subject normalisation)
    lr_edges_resid  93k normalised edges with age, sex, log-total regressed out (fit on train only)
Also saves per-fold coefficients for lr_strength -> results/contrib/lr_strength_coef.npy [15, 432]
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from common import paths
from common.data import edge_vectors, load_meta, node_features
from common.metrics import PredWriter, choose_threshold
from common.splits import iter_splits

warnings.filterwarnings("ignore")
C_GRID = [1e-3, 1e-2, 1e-1, 1.0]
ALL = ["majority", "lr_scale", "lr_demo", "lr_strength", "lr_edges", "lr_edges_log1p", "lr_edges_resid"]


def features(name, meta):
    if name == "lr_scale":
        nf = node_features()
        return np.column_stack([np.log(nf["total_raw"]), nf["density"]])
    if name == "lr_demo":
        return meta[["age", "sex"]].to_numpy(float)
    if name == "lr_strength":
        return node_features()["strength"].astype(np.float64)
    if name in ("lr_edges", "lr_edges_resid"):
        return edge_vectors("norm")
    if name == "lr_edges_log1p":
        return edge_vectors("log1p")
    raise ValueError(name)


def fit_lr(X, y, tr, va, wide):
    sc = StandardScaler().fit(X[tr])
    Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
    best = (-1, None)
    for C in C_GRID:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                 solver="liblinear", dual=wide)
        clf.fit(Xtr, y[tr])
        ap = average_precision_score(y[va], clf.decision_function(Xva))
        if ap > best[0]:
            best = (ap, clf)
    return sc, best[1]


def residualise(X, C, tr):
    """Remove the linear effect of confounds C from every column of X (betas fit on train only)."""
    C1 = np.column_stack([C, np.ones(len(C))])
    beta = np.linalg.pinv(C1[tr]) @ X[tr]
    return (X - C1 @ beta).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ALL)
    ap.add_argument("--repeats", nargs="+", type=int, default=None)
    a = ap.parse_args()
    meta = load_meta()
    y = meta.y.to_numpy()

    for name in a.models:
        w = PredWriter(name)
        chosen_C, coefs = [], []
        X = None if name == "majority" else features(name, meta)
        conf = None
        if name == "lr_edges_resid":
            conf = np.column_stack([meta.age, meta.sex, np.log(node_features()["total_raw"])])
        for r, f, tr, va, te in iter_splits(a.repeats):
            if name == "majority":
                s_va = np.full(len(va), y[tr].mean()); s_te = np.full(len(te), y[tr].mean())
            else:
                Xf = residualise(X, conf, tr) if name == "lr_edges_resid" else X
                sc, clf = fit_lr(Xf, y, tr, va, wide=Xf.shape[1] > len(tr))
                s_va = clf.decision_function(sc.transform(Xf[va]))
                s_te = clf.decision_function(sc.transform(Xf[te]))
                chosen_C.append(clf.C)
                if name == "lr_strength":
                    coefs.append(clf.coef_.ravel())
            w.add(r, f, te, y[te], s_te, choose_threshold(y[va], s_va))
        if coefs:
            np.save(paths.CONTRIB / "lr_strength_coef.npy", np.array(coefs, np.float32))
        w.save({"features": name, "C_grid": C_GRID, "chosen_C": chosen_C,
                "preprocessing": "norm (log10, relative to weakest edge)"})


if __name__ == "__main__":
    main()
