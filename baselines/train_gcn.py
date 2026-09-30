"""GCN baseline (optional) -- same graphs / training / outputs as baselines.train_gat, only the convs differ.

    python -m baselines.train_gcn --smoke              # fold 0, 5 epochs: timing + save/load/forward check
    python -m baselines.train_gcn --repeats 0          # full run -> preds/gcn.csv + models/gcn_r0_f*.pt
    python -m baselines.train_gcn --repeats 0 --folds 3 4
    python -m baselines.train_gcn --assemble           # rebuild preds/gcn.csv from saved folds

Frozen config:
  graphs    : identical to gat.py (consistency_mask, build_graphs, 440-dim node x) -- reused, not copied
  model     : GCNConv(440->64) + BatchNorm + ReLU -> GCNConv(64->64) + BatchNorm
              -> concat(mean_pool, max_pool) -> dropout -> Linear(128->1)
  edge wt   : the NON-NEGATIVE norm weight L_ij (not the z-score). build_graphs stores the z-scored
              weight in edge_attr[:,0]; the model undoes the z-score with the fold's edge scaler
              (kept as buffers), so L_ij = z*scale + mean exactly. Same forward signature as the GAT.
  training  : same as GAT (BCE pos_weight, Adam 5e-4 / wd 1e-4, batch 32, val-AP early stopping, patience 30)
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

from common import paths
from common.data import N_NODES, load_mats, load_meta
from common.metrics import PredWriter, choose_threshold
from common.splits import iter_splits
from baselines.train_gat import CFG, _scaler, _scores, build_graphs, consistency_mask, fit_scalers

torch.set_num_threads(4)


class GCNConnectome(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.6, in_dropout=0.6, edge_mean=0.0, edge_scale=1.0):
        super().__init__()
        self.in_dropout, self.dropout = in_dropout, dropout
        self.conv1, self.bn1 = GCNConv(in_dim, hidden), nn.BatchNorm1d(hidden)
        self.conv2, self.bn2 = GCNConv(hidden, hidden), nn.BatchNorm1d(hidden)
        self.head = nn.Linear(2 * hidden, 1)
        # edge scaler of this fold, so we can go z-score -> raw non-negative weight
        self.register_buffer("e_mean", torch.tensor(float(edge_mean)))
        self.register_buffer("e_scale", torch.tensor(float(edge_scale)))

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        w = None
        if edge_attr is not None:
            w = (edge_attr[:, 0] * self.e_scale + self.e_mean).clamp_min(0)   # L_ij >= 0
        x = F.dropout(x, self.in_dropout, self.training)
        x = F.relu(self.bn1(self.conv1(x, edge_index, w)))
        x = self.bn2(self.conv2(x, edge_index, w))
        g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], 1)
        g = F.dropout(g, self.dropout, self.training)
        return self.head(g).view(-1)


def train_fold(r, f, tr, va, te, M, y, cfg) -> dict:
    cfg = {**CFG, "hidden": 64, **(cfg or {})}
    dev = cfg["device"]
    torch.manual_seed(cfg["seed"] + 100 * r + f); np.random.seed(cfg["seed"] + 100 * r + f)
    t0 = time.time()
    mask = consistency_mask((M[i] for i in tr), cfg["keep_frac"])
    fs, es = fit_scalers(M, tr, mask)
    g_tr, g_va, g_te = (build_graphs(M, ix, mask, fs, es, y) for ix in (tr, va, te))
    print(f"[gcn] r{r} f{f}: graphs built in {time.time()-t0:.0f}s", flush=True)

    model = GCNConnectome(g_tr[0].x.shape[1], cfg["hidden"], cfg["dropout"], cfg["in_dropout"],
                          es.mean_[0], es.scale_[0]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    npos = float(y[tr].sum())
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(tr) - npos) / npos, device=dev))
    loader = DataLoader(g_tr, batch_size=cfg["batch"], shuffle=True,
                        generator=torch.Generator().manual_seed(cfg["seed"] + 100 * r + f))

    best, best_state, best_ep, wait, hist = (-1.0, -1.0), None, 0, 0, []
    for ep in range(1, cfg["max_epochs"] + 1):
        te0 = time.time(); model.train(); tot = 0.0
        for b in loader:
            b = b.to(dev); opt.zero_grad()
            loss = lossf(model(b.x, b.edge_index, b.edge_attr, b.batch), b.y.view(-1))
            loss.backward(); opt.step()
            tot += loss.item() * b.num_graphs
        sv = _scores(model, g_va, dev)
        ap, auc = average_precision_score(y[va], sv), roc_auc_score(y[va], sv)
        hist.append((ep, tot / len(g_tr), ap, auc, time.time() - te0))
        if (ap, auc) > best:
            best, best_state, best_ep, wait = (ap, auc), copy.deepcopy(model.state_dict()), ep, 0
        else:
            wait += 1
        if ep == 1 or ep % 10 == 0:
            print(f"[gcn] r{r} f{f} ep{ep:3d} loss {tot/len(g_tr):.4f} val AP {ap:.3f} AUROC {auc:.3f} "
                  f"({time.time()-te0:.1f}s/ep)", flush=True)
        if ep >= cfg["min_epochs"] and wait >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    sv, st = _scores(model, g_va, dev), _scores(model, g_te, dev)
    print(f"[gcn] r{r} f{f}: best ep {best_ep}, val AP {best[0]:.3f}, {time.time()-t0:.0f}s total", flush=True)
    return dict(state={k: v.cpu() for k, v in model.state_dict().items()}, mask=mask, feat_scaler=fs,
                edge_scaler=es, val_scores=sv, test_scores=st, best_epoch=best_ep, history=hist,
                in_dim=int(g_tr[0].x.shape[1]), cfg=cfg, va=va, te=te)


def _fold_path(r, f):
    return paths.MODELS / f"gcn_r{r}_f{f}.pt"


def save_fold(bundle, r, f):
    fs, es = bundle["feat_scaler"], bundle["edge_scaler"]
    torch.save(dict(state=bundle["state"], mask=bundle["mask"], in_dim=bundle["in_dim"],
                    feat_mean=fs.mean_, feat_scale=fs.scale_, edge_mean=es.mean_, edge_scale=es.scale_,
                    cfg={k: v for k, v in bundle["cfg"].items() if k != "device"},
                    best_epoch=bundle["best_epoch"]), _fold_path(r, f))
    np.savez(paths.MODELS / f"gcn_preds_r{r}_f{f}.npz", va=bundle["va"], te=bundle["te"],
             val_scores=bundle["val_scores"], test_scores=bundle["test_scores"],
             best_epoch=bundle["best_epoch"], history=np.array(bundle["history"]))


def load_fold(r, f):
    """-> (model in eval mode on CPU, mask, feat_scaler, edge_scaler)"""
    b = torch.load(_fold_path(r, f), map_location="cpu", weights_only=False)
    c = b["cfg"]
    model = GCNConnectome(b["in_dim"], c["hidden"], c["dropout"], c["in_dropout"],
                          b["edge_mean"][0], b["edge_scale"][0])
    model.load_state_dict(b["state"]); model.eval()
    return model, b["mask"], _scaler(b["feat_mean"], b["feat_scale"]), _scaler(b["edge_mean"], b["edge_scale"])


def assemble(y, repeats=None):
    w, done = PredWriter("gcn"), []
    for r, f, tr, va, te in iter_splits(repeats):
        p = paths.MODELS / f"gcn_preds_r{r}_f{f}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        assert (z["te"] == te).all(), f"saved fold r{r}f{f} does not match the frozen splits"
        w.add(r, f, te, y[te], z["test_scores"], choose_threshold(y[va], z["val_scores"]))
        done.append((r, f, int(z["best_epoch"])))
    if not done:
        print("[gcn] no saved folds to assemble"); return
    used = torch.load(_fold_path(*done[0][:2]), map_location="cpu", weights_only=False)["cfg"]
    w.save({"folds": done, "config": used, "edge_weight": "non-negative norm L_ij",
            "preprocessing": "norm (log10, relative to weakest edge)"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", nargs="+", type=int, default=[0])
    ap.add_argument("--folds", nargs="+", type=int, default=None)
    ap.add_argument("--device", default=CFG["device"])
    ap.add_argument("--max-epochs", type=int, default=CFG["max_epochs"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    a = ap.parse_args()
    cfg = {**CFG, "hidden": 64, "device": a.device, "max_epochs": a.max_epochs}
    y = load_meta().y.to_numpy()
    print(f"[gcn] device {cfg['device']}", flush=True)
    if a.assemble:
        assemble(y, a.repeats); return

    M = load_mats("norm")
    if a.smoke:
        r, f, tr, va, te = next(iter_splits([0]))
        b = train_fold(r, f, tr, va, te, M, y, {**cfg, "max_epochs": 5, "min_epochs": 5, "patience": 99})
        per_ep = np.mean([h[4] for h in b["history"]])
        save_fold(b, 99, 0)
        model, mask, fs, es = load_fold(99, 0)
        g = build_graphs(M, te[:1], mask, fs, es, y)[0]
        with torch.no_grad():
            out = model(g.x, g.edge_index, edge_attr=g.edge_attr, batch=torch.zeros(N_NODES, dtype=torch.long))
        assert out.shape == (1,) and torch.isfinite(out).all()
        _fold_path(99, 0).unlink(); (paths.MODELS / "gcn_preds_r99_f0.npz").unlink()
        print(f"SMOKE OK: {per_ep:.1f}s/epoch -> worst case {per_ep*cfg['max_epochs']/60:.0f} min/fold")
        return

    for r, f, tr, va, te in iter_splits(a.repeats):
        if a.folds is not None and f not in a.folds:
            continue
        b = train_fold(r, f, tr, va, te, M, y, cfg)
        save_fold(b, r, f)
        aps = [h[2] for h in b["history"]]
        if max(aps) - min(aps) < 1e-6 or np.std(b["test_scores"]) < 1e-6:
            print(f"[gcn] WARNING r{r} f{f} looks collapsed", flush=True)
        print(f"[gcn] saved {_fold_path(r, f)}", flush=True)
    assemble(y, a.repeats)


if __name__ == "__main__":
    main()
