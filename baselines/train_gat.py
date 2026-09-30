"""GAT baseline (binary PTSD, PCL-5 >= 33) on the frozen splits.

    python -m baselines.train_gat --smoke                       # fold 0, 5 epochs: timing + save/load/forward check
    python -m baselines.train_gat --repeats 0                   # full run, 5 folds -> preds/gat.csv + models/gat_r0_f*.pt
    python -m baselines.train_gat --repeats 0 --folds 3 4       # rerun only missing folds (crash / OOM)
    python -m baselines.train_gat --assemble                    # rebuild preds/gat.csv from whatever folds are saved
    python -m baselines.train_gat --repeats 0 --lr 1e-3 --in-drop 0.2   # the ONE allowed rerun if it collapses

Frozen config:
  topology  : consistency mask on TRAIN mats (CV = sd/mean per edge), keep 20 % most consistent, symmetric, no diag
  node x    : full 432-dim norm row (StandardScaler on train rows) + Yeo-7 one-hot + participation coeff -> 440
  edge attr : [z-scored weight (scaler on train mask edges), dist / max dist]
  model     : GATConv(440->32, heads=2, edge_dim=2) -> ELU -> GATConv(64->32, heads=1) -> ELU
              -> concat(mean_pool, max_pool) -> dropout -> Linear(64->1); dropout 0.6 on inputs / attention / head
  train     : BCEWithLogits(pos_weight=n_neg/n_pos train), Adam lr 5e-4 wd 1e-4, batch 32,
              max 150 ep, early stop on val AP (ties -> val AUROC), patience 30, min 10, restore best
  output    : score = test logit, threshold = choose_threshold(y_val, logit_val), PredWriter("gat")

Each finished fold is saved on its own (models/gat_r{r}_f{f}.pt + small preds npz), so a crash
never loses the folds already done; preds/gat.csv is always assembled from all saved folds.
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool

from common import paths
from common.data import N_NODES, load_mats, load_meta
from common.metrics import PredWriter, choose_threshold
from common.splits import iter_splits

torch.set_num_threads(4)  # laptop / colab survival

CFG = dict(keep_frac=0.20, hidden=32, heads=2, dropout=0.6, in_dropout=0.6,
           lr=5e-4, wd=1e-4, batch=32, max_epochs=150, patience=30, min_epochs=10,
           seed=0, device="cuda" if torch.cuda.is_available() else "cpu")

IU = np.triu_indices(N_NODES, 1)


# ------------------------------------------------------------------ fixed node / edge extras
def _yeo():
    Y = np.load(paths.YEO7).astype(np.float32)
    assert Y.shape[0] == N_NODES, Y.shape
    return Y


def _modules(yeo):
    # module id per node for the participation coeff; rows with no yeo net (subcortical) = own module
    mod = yeo.argmax(1)
    mod[yeo.sum(1) == 0] = yeo.shape[1]
    return np.eye(yeo.shape[1] + 1, dtype=np.float32)[mod]      # [432, n_mod] one-hot


def compute_participation_coefficient(W, mod_onehot):
    """P_i = 1 - sum_m (k_im / k_i)^2 on a weighted adjacency W [432,432]; 0 for isolated nodes."""
    k = W.sum(1)
    kim = W @ mod_onehot
    with np.errstate(divide="ignore", invalid="ignore"):
        P = 1.0 - ((kim / k[:, None]) ** 2).sum(1)
    P[k <= 0] = 0.0
    return P.astype(np.float32)


def _edge_dist():
    D = np.load(paths.EDGE_DIST).astype(np.float32)
    assert D.shape == (N_NODES, N_NODES), D.shape
    return D / D.max()


# ------------------------------------------------------------------ required API
def consistency_mask(M_train, keep_frac=0.20) -> np.ndarray:
    """bool [432,432]: keep the keep_frac most consistent (lowest sd/mean over TRAIN subjects) edges."""
    s = np.zeros(len(IU[0])); ss = np.zeros(len(IU[0]))
    n = 0
    for A in M_train:                      # stream over subjects, M_train can be a memmap slice
        v = np.asarray(A, dtype=np.float64)[IU]
        s += v; ss += v * v; n += 1
    mean = s / n
    sd = np.sqrt(np.maximum(ss / n - mean ** 2, 0))
    cv = np.full_like(mean, np.inf)
    ok = mean > 0
    cv[ok] = sd[ok] / mean[ok]
    k = int(round(keep_frac * len(cv)))
    keep = np.argsort(cv, kind="stable")[:k]
    mask = np.zeros((N_NODES, N_NODES), bool)
    mask[IU[0][keep], IU[1][keep]] = True
    mask |= mask.T
    np.fill_diagonal(mask, False)
    return mask


def fit_scalers(M, tr, mask):
    """feat scaler on the train node rows (streamed), edge scaler on train mask-edge weights."""
    fs, es = StandardScaler(), StandardScaler()
    for i in tr:
        A = np.asarray(M[i], dtype=np.float32)
        fs.partial_fit(A)
        es.partial_fit(A[mask].reshape(-1, 1))
    return fs, es


def build_graphs(M, idx, mask, feat_scaler, edge_scaler, y) -> list:
    yeo = _yeo(); mods = _modules(yeo); dist = _edge_dist()
    src, dst = np.nonzero(mask)
    ei = torch.from_numpy(np.stack([src, dst])).long()
    d_attr = dist[src, dst]
    out = []
    for i in idx:
        A = np.asarray(M[i], dtype=np.float32)
        x = feat_scaler.transform(A).astype(np.float32)
        pc = compute_participation_coefficient(A * mask, mods)
        x = np.concatenate([x, yeo, pc[:, None]], 1)
        w = edge_scaler.transform(A[src, dst].reshape(-1, 1)).ravel().astype(np.float32)
        ea = np.stack([w, d_attr], 1)
        out.append(Data(x=torch.from_numpy(x), edge_index=ei, edge_attr=torch.from_numpy(ea),
                        y=torch.tensor([float(y[i])]), idx=int(i)))
    return out


class GATConnectome(nn.Module):
    def __init__(self, in_dim, hidden=32, heads=2, dropout=0.6, in_dropout=0.6):
        super().__init__()
        self.in_dropout, self.dropout = in_dropout, dropout
        self.conv1 = GATConv(in_dim, hidden, heads=heads, edge_dim=2, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, dropout=dropout)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = F.dropout(x, self.in_dropout, self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, self.dropout, self.training)
        x = F.elu(self.conv2(x, edge_index))
        g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], 1)
        g = F.dropout(g, self.dropout, self.training)
        return self.head(g).view(-1)          # one logit per graph


@torch.no_grad()
def _scores(model, graphs, device, bs=64):
    model.eval()
    out = []
    for b in DataLoader(graphs, batch_size=bs, shuffle=False):
        b = b.to(device)
        out.append(model(b.x, b.edge_index, b.edge_attr, b.batch).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def train_fold(r, f, tr, va, te, M, y, cfg) -> dict:
    cfg = {**CFG, **(cfg or {})}
    dev = cfg["device"]
    torch.manual_seed(cfg["seed"] + 100 * r + f); np.random.seed(cfg["seed"] + 100 * r + f)
    t0 = time.time()
    mask = consistency_mask((M[i] for i in tr), cfg["keep_frac"])
    fs, es = fit_scalers(M, tr, mask)
    g_tr = build_graphs(M, tr, mask, fs, es, y)
    g_va = build_graphs(M, va, mask, fs, es, y)
    g_te = build_graphs(M, te, mask, fs, es, y)
    print(f"[gat] r{r} f{f}: graphs built in {time.time()-t0:.0f}s, {int(mask.sum()//2)} edges, "
          f"in_dim {g_tr[0].x.shape[1]}", flush=True)

    model = GATConnectome(g_tr[0].x.shape[1], cfg["hidden"], cfg["heads"], cfg["dropout"], cfg["in_dropout"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    npos = float(y[tr].sum())
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(tr) - npos) / npos, device=dev))
    loader = DataLoader(g_tr, batch_size=cfg["batch"], shuffle=True,
                        generator=torch.Generator().manual_seed(cfg["seed"] + 100 * r + f))

    best, best_state, best_ep, wait, hist = (-1.0, -1.0), None, 0, 0, []
    for ep in range(1, cfg["max_epochs"] + 1):
        te0 = time.time()
        model.train()
        tot = 0.0
        for b in loader:
            b = b.to(dev)
            opt.zero_grad()
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
            print(f"[gat] r{r} f{f} ep{ep:3d} loss {tot/len(g_tr):.4f} val AP {ap:.3f} AUROC {auc:.3f} "
                  f"({time.time()-te0:.1f}s/ep)", flush=True)
        if ep >= cfg["min_epochs"] and wait >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    sv, st = _scores(model, g_va, dev), _scores(model, g_te, dev)
    print(f"[gat] r{r} f{f}: best ep {best_ep}, val AP {best[0]:.3f}, {time.time()-t0:.0f}s total", flush=True)
    return dict(state={k: v.cpu() for k, v in model.state_dict().items()}, mask=mask, feat_scaler=fs,
                edge_scaler=es, val_scores=sv, test_scores=st, best_epoch=best_ep, history=hist,
                in_dim=int(g_tr[0].x.shape[1]), cfg=cfg, va=va, te=te)


def _fold_path(r, f):
    return paths.MODELS / f"gat_r{r}_f{f}.pt"


def save_fold(bundle, r, f):
    fs, es = bundle["feat_scaler"], bundle["edge_scaler"]
    torch.save(dict(state=bundle["state"], mask=bundle["mask"], in_dim=bundle["in_dim"],
                    feat_mean=fs.mean_, feat_scale=fs.scale_, edge_mean=es.mean_, edge_scale=es.scale_,
                    cfg={k: v for k, v in bundle["cfg"].items() if k != "device"},
                    best_epoch=bundle["best_epoch"]), _fold_path(r, f))
    # small per-fold preds so gat.csv can be re-assembled after crashes / partial reruns
    np.savez(paths.MODELS / f"gat_preds_r{r}_f{f}.npz", va=bundle["va"], te=bundle["te"],
             val_scores=bundle["val_scores"], test_scores=bundle["test_scores"],
             best_epoch=bundle["best_epoch"], history=np.array(bundle["history"]))


def _scaler(mean, scale):
    s = StandardScaler()
    s.mean_, s.scale_ = np.asarray(mean), np.asarray(scale)
    s.var_, s.n_features_in_ = s.scale_ ** 2, len(s.mean_)
    return s


def load_fold(r, f):
    """-> (model in eval mode on CPU, mask, feat_scaler, edge_scaler)"""
    b = torch.load(_fold_path(r, f), map_location="cpu", weights_only=False)
    c = b["cfg"]
    model = GATConnectome(b["in_dim"], c["hidden"], c["heads"], c["dropout"], c["in_dropout"])
    model.load_state_dict(b["state"]); model.eval()
    return model, b["mask"], _scaler(b["feat_mean"], b["feat_scale"]), _scaler(b["edge_mean"], b["edge_scale"])


# ------------------------------------------------------------------ CLI
def assemble(y, cfg, repeats=None):
    w = PredWriter("gat")
    done = []
    for r, f, tr, va, te in iter_splits(repeats):
        p = paths.MODELS / f"gat_preds_r{r}_f{f}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        assert (z["te"] == te).all(), f"saved fold r{r}f{f} does not match the frozen splits"
        w.add(r, f, te, y[te], z["test_scores"], choose_threshold(y[va], z["val_scores"]))
        done.append((r, f, int(z["best_epoch"])))
    if not done:
        print("[gat] no saved folds to assemble"); return
    r0, f0, _ = done[0]   # report the config the folds were actually trained with
    used = torch.load(_fold_path(r0, f0), map_location="cpu", weights_only=False)["cfg"]
    w.save({"folds": done, "config": used,
            "preprocessing": "norm (log10, relative to weakest edge)"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", nargs="+", type=int, default=[0])
    ap.add_argument("--folds", nargs="+", type=int, default=None)
    ap.add_argument("--device", default=CFG["device"])
    ap.add_argument("--lr", type=float, default=CFG["lr"])
    ap.add_argument("--in-drop", type=float, default=CFG["in_dropout"])
    ap.add_argument("--max-epochs", type=int, default=CFG["max_epochs"])
    ap.add_argument("--smoke", action="store_true", help="fold 0, 5 epochs, save/load/forward, no preds")
    ap.add_argument("--assemble", action="store_true", help="only rebuild preds/gat.csv from saved folds")
    a = ap.parse_args()
    cfg = {**CFG, "device": a.device, "lr": a.lr, "in_dropout": a.in_drop, "max_epochs": a.max_epochs}
    y = load_meta().y.to_numpy()
    print(f"[gat] device {cfg['device']}, cfg {cfg}", flush=True)

    if a.assemble:
        assemble(y, cfg, a.repeats); return

    M = load_mats("norm")
    if a.smoke:
        r, f, tr, va, te = next(iter_splits([0]))
        b = train_fold(r, f, tr, va, te, M, y, {**cfg, "max_epochs": 5, "min_epochs": 5, "patience": 99})
        per_ep = np.mean([h[4] for h in b["history"]])
        save_fold(b, 99, 0)                              # test slot, won't clash with real folds
        model, mask, fs, es = load_fold(99, 0)
        g = build_graphs(M, te[:2], mask, fs, es, y)[0]
        with torch.no_grad():
            out = model(g.x, g.edge_index, edge_attr=g.edge_attr, batch=torch.zeros(N_NODES, dtype=torch.long))
        assert out.shape == (1,) and torch.isfinite(out).all()
        _fold_path(99, 0).unlink(); (paths.MODELS / "gat_preds_r99_f0.npz").unlink()
        worst = per_ep * cfg["max_epochs"] / 60
        print(f"SMOKE OK: {per_ep:.1f}s/epoch -> worst case {worst:.0f} min/fold (150 ep). "
              f"{'OK here' if worst <= 20 else 'TOO SLOW here -> use a Colab T4 (--device cuda)'}")
        return

    for r, f, tr, va, te in iter_splits(a.repeats):
        if a.folds is not None and f not in a.folds:
            continue
        b = train_fold(r, f, tr, va, te, M, y, cfg)
        save_fold(b, r, f)
        aps = [h[2] for h in b["history"]]
        if max(aps) - min(aps) < 1e-6 or np.std(b["test_scores"]) < 1e-6:
            print(f"[gat] WARNING r{r} f{f} looks collapsed (flat val AP / constant test scores)", flush=True)
        print(f"[gat] saved {_fold_path(r, f)}", flush=True)
    assemble(y, cfg, a.repeats)


if __name__ == "__main__":
    main()
