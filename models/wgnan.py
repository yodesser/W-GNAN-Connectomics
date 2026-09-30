"""Weighted-GNAN core model and CV routines.
Implementation of the fixed W-GNAN model with region-specific shape functions.
"""
from __future__ import annotations

import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from common import data, paths, metrics

def interp1d(x, knots, values):
    x = torch.clamp(x, float(knots[0]), float(knots[-1]))
    y = torch.zeros_like(x) + values[0]
    s_prev = 0.0
    for i in range(len(knots) - 1):
        s_i = (values[i+1] - values[i]) / torch.clamp(knots[i+1] - knots[i], min=1e-8)
        delta_s = s_i - s_prev
        y = y + delta_s * torch.nn.functional.relu(x - knots[i])
        s_prev = s_i
    return y

class AnchoredShape(nn.Module):
    def __init__(self, n_knots):
        super().__init__()
        self.register_buffer("knots", torch.linspace(-3.0, 3.0, n_knots))
        self.values = nn.Parameter(torch.zeros(n_knots))
        self.center = n_knots // 2

    def anchored(self):
        return self.values - self.values[self.center]

    def forward(self, x):
        return interp1d(torch.clamp(x, -3, 3), self.knots, self.anchored())

    def smoothness(self):
        v = self.anchored()
        d2 = v[:-2] - 2 * v[1:-1] + v[2:]
        return (d2 ** 2).mean()

class RegionShapes(nn.Module):
    def __init__(self, n_regions, n_knots=7):
        super().__init__()
        self.register_buffer("knots", torch.linspace(-3.0, 3.0, n_knots))
        self.values = nn.Parameter(torch.zeros(n_regions, n_knots))
        self.center = n_knots // 2
        
    def anchored(self):
        return self.values - self.values[:, self.center:self.center+1]
        
    def forward(self, x):
        # x: [batch, n_regions]
        b, r = x.shape
        x_flat = x.view(-1)
        knots_rep = self.knots.unsqueeze(0).expand(b * r, -1)
        vals_rep = self.anchored().repeat(b, 1)
        
        x_c = torch.clamp(x_flat, float(self.knots[0]), float(self.knots[-1])).unsqueeze(1)
        idx = torch.sum(knots_rep < x_c, dim=1).clamp(1, len(self.knots)-1)
        
        idx_m1 = idx - 1
        x0 = knots_rep[torch.arange(b*r), idx_m1]
        x1 = knots_rep[torch.arange(b*r), idx]
        y0 = vals_rep[torch.arange(b*r), idx_m1]
        y1 = vals_rep[torch.arange(b*r), idx]
        
        t = (x_flat - x0) / torch.clamp(x1 - x0, min=1e-8)
        return (y0 + t * (y1 - y0)).view(b, r)

    def smoothness(self):
        v = self.anchored()
        d2 = v[:, :-2] - 2 * v[:, 1:-1] + v[:, 2:]
        return (d2 ** 2).mean()

class MonotoneGate(nn.Module):
    def __init__(self, n_knots):
        super().__init__()
        self.register_buffer("knots", torch.linspace(0.0, 1.0, n_knots))
        import math
        init = math.log(math.exp(1.0) - 1.0)
        self.raw_inc = nn.Parameter(torch.full((n_knots - 1,), init))

    def knot_values(self):
        inc = F.softplus(self.raw_inc) + 1e-6
        vals = torch.cat([torch.zeros(1, device=inc.device), torch.cumsum(inc, 0)])
        return vals / vals[-1]

    def forward(self, x):
        return interp1d(torch.clamp(x, 0, 1), self.knots, self.knot_values())

    def smoothness(self):
        v = self.knot_values()
        d2 = v[:-2] - 2 * v[1:-1] + v[2:]
        return (d2 ** 2).mean()

class WGNAN(nn.Module):
    def __init__(self, variant, train_stats):
        super().__init__()
        self.variant = variant
        self.n_nodes = len(train_stats["mean_s"])
        
        # train_stats: mean_s, sd_s, mean_d, sd_d (all [432]), q99
        self.register_buffer("mean_s", torch.tensor(train_stats["mean_s"], dtype=torch.float32))
        self.register_buffer("sd_s", torch.tensor(train_stats["sd_s"], dtype=torch.float32))
        self.register_buffer("mean_d", torch.tensor(train_stats["mean_d"], dtype=torch.float32))
        self.register_buffer("sd_d", torch.tensor(train_stats["sd_d"], dtype=torch.float32))
        self.register_buffer("q99", torch.tensor(float(train_stats["q99"]), dtype=torch.float32))
        
        self.region_shape = RegionShapes(self.n_nodes, 7)
        self.degree_shape = AnchoredShape(9)
        
        if variant == "learned":
            self.gate = MonotoneGate(9)
            
        import math
        init = math.log(math.exp(1.0) - 1.0)
        self.raw_lambda = nn.Parameter(torch.tensor(init))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def graph_lambda(self):
        if self.variant == "nodeonly":
            return 0.0
        return F.softplus(self.raw_lambda)
        
    def forward_components(self, L, S, D):
        zs = (S - self.mean_s) / torch.clamp(self.sd_s, min=1e-6)
        zd = (D - self.mean_d) / torch.clamp(self.sd_d, min=1e-6)
        
        u = self.region_shape(zs) + self.degree_shape(zd)
        
        if self.variant == "nodeonly":
            total = u
            mask = None
            gate_w = None
        else:
            w = torch.clamp(L / torch.clamp(self.q99, min=1e-6), 0, 1)
            mask = L > 0
            
            if self.variant == "topology":
                gate_w = mask.float()
            elif self.variant == "linear":
                gate_w = torch.where(mask, w, torch.zeros_like(w))
            else: # learned
                gate_w = torch.where(mask, self.gate(w), torch.zeros_like(w))
                
            n_neigh = mask.sum(dim=-1, keepdim=True).float().clamp(min=1.0)
            neigh = torch.bmm(gate_w, u.unsqueeze(-1)).squeeze(-1) / n_neigh.squeeze(-1)
            
            total = u + self.graph_lambda() * neigh
            
        logit = self.bias + total.sum(dim=-1)
        return {"logit": logit, "total": total}
        
    def forward(self, L, S, D):
        return self.forward_components(L, S, D)["logit"]
        
    def regularization(self, l2):
        reg = l2 * (self.region_shape.values ** 2).mean()
        smooth = 1e-3 * (self.region_shape.smoothness() + self.degree_shape.smoothness())
        if self.variant == "learned":
            smooth = smooth + 1e-3 * self.gate.smoothness()
        return reg + smooth

class WGNANDataset(Dataset):
    def __init__(self, idxs, L_all, S_all, D_all, y_all):
        self.idxs = idxs
        self.L = L_all
        self.S = S_all
        self.D = D_all
        self.y = y_all
        
    def __len__(self):
        return len(self.idxs)
        
    def __getitem__(self, i):
        idx = self.idxs[i]
        return (
            torch.from_numpy(np.array(self.L[idx], copy=True)),
            torch.from_numpy(np.array(self.S[idx], copy=True)),
            torch.from_numpy(np.array(self.D[idx], copy=True)),
            torch.tensor(self.y[idx], dtype=torch.float32),
            idx
        )

def get_train_stats(L, S, D, tr):
    # L [N, 432, 432], S [N, 432], D [N, 432]
    mean_s = S[tr].mean(axis=0)
    sd_s = S[tr].std(axis=0)
    mean_d = D[tr].mean(axis=0)
    sd_d = D[tr].std(axis=0)
    
    # 99th percentile of non-zero L in train
    L_tr = L[tr]
    nz = L_tr[L_tr > 0]
    q99 = np.percentile(nz, 99) if len(nz) > 0 else 1.0
    
    return {
        "mean_s": mean_s, "sd_s": sd_s, "mean_d": mean_d, "sd_d": sd_d, "q99": q99
    }

def fit_wgnan(variant, tr, va, y, cfg=None) -> tuple[WGNAN, dict]:
    cfg = cfg or {}
    device = cfg.get("device", "cpu")
    epochs = cfg.get("epochs", 200)
    patience = cfg.get("patience", 25)
    min_epochs = cfg.get("min_epochs", 10)
    fast = cfg.get("fast", False)
    if fast:
        epochs = min(epochs, 80)
        patience = min(patience, 15)
        
    l2 = cfg.get("l2", 1e-3)
    
    L_all = data.load_mats("norm")
    feats = data.node_features()
    S_all = feats["strength"]
    D_all = feats["degree"]
    
    stats = get_train_stats(L_all, S_all, D_all, tr)
    
    model = WGNAN(variant, stats).to(device)
    n1 = y[tr].sum()
    n0 = len(tr) - n1
    pos_weight = float(n0 / max(n1, 1))
    
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0)
    
    tr_loader = DataLoader(WGNANDataset(tr, L_all, S_all, D_all, y), batch_size=64, shuffle=True)
    va_loader = DataLoader(WGNANDataset(va, L_all, S_all, D_all, y), batch_size=64, shuffle=False)
    
    best_key = (-float('inf'), -float('inf'))
    best_state = None
    best_epoch = 0
    bad = 0
    
    for ep in range(1, epochs + 1):
        model.train()
        for L_b, S_b, D_b, y_b, _ in tr_loader:
            L_b, S_b, D_b, y_b = L_b.to(device), S_b.to(device), D_b.to(device), y_b.to(device)
            opt.zero_grad()
            logits = model(L_b, S_b, D_b)
            loss = loss_fn(logits, y_b) + model.regularization(l2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            
        # val
        model.eval()
        va_y, va_s = [], []
        with torch.no_grad():
            for L_b, S_b, D_b, y_b, _ in va_loader:
                L_b, S_b, D_b = L_b.to(device), S_b.to(device), D_b.to(device)
                logits = model(L_b, S_b, D_b)
                va_s.append(logits.cpu().numpy())
                va_y.append(y_b.numpy())
        va_s = np.concatenate(va_s)
        va_y = np.concatenate(va_y)
        
        from sklearn.metrics import average_precision_score, roc_auc_score
        ap = float(average_precision_score(va_y, va_s)) if va_y.sum() > 0 else 0.0
        auc = float(roc_auc_score(va_y, va_s)) if va_y.sum() > 0 else 0.0
        
        key = (ap, auc)
        if key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep
            bad = 0
        else:
            bad += 1
            
        if ep >= min_epochs and bad >= patience:
            break
            
    if best_state is not None:
        model.load_state_dict(best_state)
        
    return model, {"best_epoch": best_epoch, "val_ap": best_key[0], "lambda": float(model.graph_lambda().detach())}

@torch.no_grad()
def predict(model, idx, device="cpu") -> np.ndarray:
    model.eval()
    model.to(device)
    L_all = data.load_mats("norm")
    feats = data.node_features()
    S_all, D_all = feats["strength"], feats["degree"]
    
    loader = DataLoader(WGNANDataset(idx, L_all, S_all, D_all, np.zeros(L_all.shape[0])), batch_size=64, shuffle=False)
    res = []
    for L_b, S_b, D_b, _, _ in loader:
        logits = model(L_b.to(device), S_b.to(device), D_b.to(device))
        res.append(logits.cpu().numpy())
    return np.concatenate(res)

@torch.no_grad()
def node_contrib(model, idx, device="cpu") -> np.ndarray:
    model.eval()
    model.to(device)
    L_all = data.load_mats("norm")
    feats = data.node_features()
    S_all, D_all = feats["strength"], feats["degree"]
    
    loader = DataLoader(WGNANDataset(idx, L_all, S_all, D_all, np.zeros(L_all.shape[0])), batch_size=64, shuffle=False)
    res = []
    for L_b, S_b, D_b, _, _ in loader:
        c = model.forward_components(L_b.to(device), S_b.to(device), D_b.to(device))
        res.append(c["total"].cpu().numpy())
    return np.concatenate(res)

def contrib_diff(model, te, y, device="cpu") -> np.ndarray:
    c = node_contrib(model, te, device)
    pos = y[te] == 1
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return np.zeros(c.shape[1], dtype=np.float32)
    return c[pos].mean(axis=0) - c[neg].mean(axis=0)

def gate_curve(model, n=101) -> tuple[np.ndarray, np.ndarray]:
    if model.variant != "learned":
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    x = np.linspace(0, 1, n, dtype=np.float32)
    with torch.no_grad():
        g = model.gate(torch.tensor(x, device=next(model.parameters()).device)).cpu().numpy()
    return x, g

def run_cv_contrib(variant, y, repeats=(0,), fast=True, device="cpu") -> tuple[np.ndarray, list]:
    splits = pd.read_csv(paths.SPLITS)
    aucs = []
    contribs = []
    for rep in repeats:
        for fold in range(5):
            mask = (splits["repeat"] == rep) & (splits["fold"] == fold)
            if not mask.any(): continue
            s = splits[mask]
            tr = s[s["role"] == "train"]["idx"].values
            va = s[s["role"] == "val"]["idx"].values
            te = s[s["role"] == "test"]["idx"].values
            
            cfg = {"fast": fast, "device": device}
            model, info = fit_wgnan(variant, tr, va, y, cfg)
            
            c_diff = contrib_diff(model, te, y, device)
            contribs.append(c_diff)
            
            preds = predict(model, te, device)
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y[te], preds) if y[te].sum() > 0 else 0.5
            aucs.append(auc)
            print(f"Rep {rep} Fold {fold}: AUC {auc:.3f}")
            
    return np.array(contribs), aucs

def run_cv(variant, repeats=(0, 1, 2), device="cpu") -> None:
    splits = pd.read_csv(paths.SPLITS)
    meta = data.load_meta()
    y = meta["y"].values
    
    writer = metrics.PredWriter(f"wgnan_{variant}")
    contribs = []
    lams = []
    gate_curves = []
    region_shapes = []
    
    for rep in repeats:
        for fold in range(5):
            mask = (splits["repeat"] == rep) & (splits["fold"] == fold)
            if not mask.any(): continue
            s = splits[mask]
            tr = s[s["role"] == "train"]["idx"].values
            va = s[s["role"] == "val"]["idx"].values
            te = s[s["role"] == "test"]["idx"].values
            
            cfg = {"device": device}
            model, info = fit_wgnan(variant, tr, va, y, cfg)
            
            val_preds = predict(model, va, device)
            thresh = metrics.choose_threshold(y[va], val_preds)
            
            te_preds = predict(model, te, device)
            writer.add(rep, fold, te, y[te], te_preds, thresh)
            
            c_diff = contrib_diff(model, te, y, device)
            contribs.append(c_diff)
            lams.append({"repeat": rep, "fold": fold, "lambda": info["lambda"]})
            
            if variant == "learned":
                x, g = gate_curve(model, 101)
                for i in range(len(x)):
                    gate_curves.append({"repeat": rep, "fold": fold, "x": float(x[i]), "g": float(g[i])})
                with torch.no_grad():
                    region_shapes.append(model.region_shape.anchored().cpu().numpy())

            
    writer.save()
    
    # Save contribs
    np.save(paths.CONTRIB / f"wgnan_{variant}_node_contrib_oof.npy", np.array(contribs))
    pd.DataFrame(lams).to_csv(paths.CONTRIB / f"wgnan_{variant}_lambda.csv", index=False)
    
    if variant == "learned":
        pd.DataFrame(gate_curves).to_csv(paths.CONTRIB / "wgnan_learned_gate_curves.csv", index=False)
        np.save(paths.CONTRIB / "wgnan_learned_region_shapes.npy", np.array(region_shapes))

    
if __name__ == "__main__":
    # Smoke test for audit/models
    print("Running W-GNAN smoke test...")
    y = data.load_meta()["y"].values
    c, aucs = run_cv_contrib("learned", y, repeats=(0,), fast=True, device="cpu")
    print("SMOKE TEST DONE. Shapes:", c.shape)
