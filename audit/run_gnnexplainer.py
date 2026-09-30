"""GNNExplainer on the trained GAT (post-hoc arm) + agreement with the intrinsic W-GNAN contributions.

Leak-free: every subject is explained with the GAT of the repeat-0 fold in which it was a TEST subject.
Explained set: all 60 positives + 12 random test negatives per fold (seed 0) = 120 subjects.
Explanation target: the PTSD class (explanation_type="phenomenon", target=1) for every subject,
so masks answer "which regions support a PTSD prediction", comparable to W-GNAN contributions.

    python -m audit.run_gnnexplainer                       # uses common.data.load_mats("norm")
    python -m audit.run_gnnexplainer --csv-root <GML_project>   # loads only the 120 raw CSVs (no 650 MB cache)

Outputs (results/explain/):
    gnnexplainer_node_imp.npy   [120, 432] node masks
    gnnexplainer_meta.csv       fold, idx, y, gat_logit, gat_logit_saved (sanity: must match)
    agreement.json / agreement_folds.csv   per-fold comparisons (see keys below)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import hypergeom, spearmanr

from baselines.train_gat import build_graphs, load_fold
from common import paths
from common.data import N_NODES, PREREG_NETWORKS, _normalise_one, load_meta, networks
from common.splits import iter_splits

TOPK = 30


def select_subjects(y):
    rng = np.random.default_rng(0)
    rows = []
    for r, f, tr, va, te in iter_splits([0]):
        pos = te[y[te] == 1]
        neg = rng.choice(te[y[te] == 0], 12, replace=False)
        rows += [(f, int(i), 1) for i in pos] + [(f, int(i), 0) for i in neg]
    return pd.DataFrame(rows, columns=["fold", "idx", "y"])


def load_matrices(sel, csv_root):
    """dict idx -> normalised matrix. csv_root: rebuild from raw CSVs exactly like mats869 -> 'norm'."""
    if csv_root is None:
        from common.data import load_mats
        L = load_mats("norm")
        return {i: np.asarray(L[i]) for i in sel.idx}
    man = pd.read_csv(paths.MANIFEST)
    out = {}
    for i in sel.idx:
        A = pd.read_csv(Path(csv_root) / man.filepath[i], header=None).values.astype(np.float64)
        np.fill_diagonal(A, 0.0)
        A = np.log1p(A).astype(np.float32)                      # identical round trip to mats869.npy
        out[i] = _normalise_one(np.expm1(A.astype(np.float64)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-root", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from torch_geometric.explain import Explainer, GNNExplainer

    y = load_meta().y.to_numpy()
    sel = select_subjects(y)
    M = load_matrices(sel, a.csv_root)
    print(f"[explain] {len(sel)} subjects ({sel.y.sum()} positives), matrices loaded", flush=True)

    imps, meta = [], []
    t0 = time.time()
    for f in sorted(sel.fold.unique()):
        model, mask, fs, es = load_fold(0, f)
        sub = sel[sel.fold == f]
        graphs = build_graphs(M, sub.idx.tolist(), mask, fs, es, y)
        saved = np.load(paths.MODELS / f"gat_preds_r0_f{f}.npz")
        saved_map = dict(zip(saved["te"].tolist(), saved["test_scores"].tolist()))

        class Wrap(torch.nn.Module):
            def __init__(s, m):
                super().__init__(); s.m = m
            def forward(s, x, edge_index, edge_attr=None, batch=None):
                return s.m(x, edge_index, edge_attr, batch).view(-1, 1)

        wm = Wrap(model).eval()
        explainer = Explainer(model=wm, algorithm=GNNExplainer(epochs=a.epochs, lr=0.01),
                              explanation_type="phenomenon", node_mask_type="object", edge_mask_type="object",
                              model_config=dict(mode="binary_classification", task_level="graph", return_type="raw"))
        for g, (_, row) in zip(graphs, sub.iterrows()):
            b = torch.zeros(N_NODES, dtype=torch.long)
            with torch.no_grad():
                logit = float(wm(g.x, g.edge_index, g.edge_attr, b))
            torch.manual_seed(0)
            e = explainer(g.x, g.edge_index, target=torch.tensor([1]), edge_attr=g.edge_attr, batch=b)
            imps.append(e.node_mask.view(-1).detach().numpy())
            meta.append(dict(fold=f, idx=row.idx, y=row.y, gat_logit=logit, gat_logit_saved=saved_map[row.idx]))
        np.save(paths.EXPLAIN / "gnnexplainer_node_imp.npy", np.array(imps, np.float32))
        pd.DataFrame(meta).to_csv(paths.EXPLAIN / "gnnexplainer_meta.csv", index=False)
        print(f"[explain] fold {f} done ({len(imps)} subjects, {time.time()-t0:.0f}s)", flush=True)

    analyse(np.array(imps), pd.DataFrame(meta), M)


def analyse(I, meta, M):
    nets = networks()
    W = np.load(paths.CONTRIB / "wgnan_learned_node_contrib_oof.npy")[:5]      # repeat 0, folds 0-4
    dev = np.abs(meta.gat_logit - meta.gat_logit_saved).max()
    rows = []
    for f in sorted(meta.fold.unique()):
        m = (meta.fold == f).to_numpy()
        pos, neg = m & (meta.y == 1).to_numpy(), m & (meta.y == 0).to_numpy()
        Ip, In = I[pos].mean(0), I[neg].mean(0)
        wabs = np.abs(W[f])
        strength = np.mean([M[i].sum(1) for i in meta.idx[m]], axis=0)
        top_e, top_w = set(np.argsort(-Ip)[:TOPK]), set(np.argsort(-wabs)[:TOPK])
        k = len(top_e & top_w)
        rows.append(dict(fold=f,
                         rho_gnnexpl_vs_wgnan=spearmanr(Ip, wabs)[0],
                         top30_overlap=k, top30_overlap_p=hypergeom.sf(k - 1, N_NODES, TOPK, TOPK),
                         rho_pos_vs_neg_masks=spearmanr(Ip, In)[0],
                         rho_gnnexpl_vs_strength=spearmanr(Ip, strength)[0],
                         rho_wgnan_vs_strength=spearmanr(wabs, strength)[0]))
    df = pd.DataFrame(rows)
    df.to_csv(paths.EXPLAIN / "agreement_folds.csv", index=False)
    Ipf = np.array([I[((meta.fold == f) & (meta.y == 1)).to_numpy()].mean(0) for f in sorted(meta.fold.unique())])
    stab = [spearmanr(Ipf[i], Ipf[j])[0] for i in range(len(Ipf)) for j in range(i + 1, len(Ipf))]
    z = (Ipf - Ipf.mean(1, keepdims=True)) / Ipf.std(1, keepdims=True)
    out = {
        "n_explained": int(len(meta)), "max_abs_logit_mismatch_vs_saved": float(dev),
        "gat_auroc_on_explained": None,
        "rho_gnnexpl_vs_wgnan_mean": float(df.rho_gnnexpl_vs_wgnan.mean()),
        "rho_gnnexpl_vs_wgnan_min_max": [float(df.rho_gnnexpl_vs_wgnan.min()), float(df.rho_gnnexpl_vs_wgnan.max())],
        "top30_overlap_mean": float(df.top30_overlap.mean()), "top30_overlap_chance": TOPK * TOPK / N_NODES,
        "gnnexpl_fold_stability_rho": float(np.mean(stab)),
        "rho_pos_vs_neg_masks_mean": float(df.rho_pos_vs_neg_masks.mean()),
        "rho_gnnexpl_vs_strength_mean": float(df.rho_gnnexpl_vs_strength.mean()),
        "rho_wgnan_vs_strength_mean": float(df.rho_wgnan_vs_strength.mean()),
        "network_mean_z_gnnexpl": {n: float(z[:, nets == n].mean()) for n in PREREG_NETWORKS},
    }
    from sklearn.metrics import roc_auc_score
    out["gat_auroc_on_explained"] = float(roc_auc_score(meta.y, meta.gat_logit))
    json.dump(out, open(paths.EXPLAIN / "agreement.json", "w"), indent=2)
    print(df.round(3).to_string(index=False))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
