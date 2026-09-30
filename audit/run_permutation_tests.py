import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import json
import torch
import sys
import os

from common import data, paths

def get_lr_strength_contributions(y, seed, repeats=(0, 1, 2)):
    # Load matrices to compute strengths
    L_all = data.load_mats("norm")
    S_all = L_all.sum(axis=-1)
    
    splits = pd.read_csv(paths.SPLITS)
    contribs = []
    
    # Try to load chosen C from the observed LR run, fallback to 0.1
    c_val = 0.1
    try:
        with open(paths.PREDS / "lr_strength.json") as f:
            meta = json.load(f)
            c_val = meta.get("chosen_C", 0.1)
    except:
        pass

    for rep in repeats:
        for fold in range(5):
            mask = (splits["repeat"] == rep) & (splits["fold"] == fold)
            if not mask.any(): continue
            s = splits[mask]
            tr = s[s["role"] == "train"]["idx"].values
            te = s[s["role"] == "test"]["idx"].values
            
            X_tr = S_all[tr]
            y_tr = y[tr]
            X_te = S_all[te]
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            
            if isinstance(c_val, list):
                C = float(c_val[rep * 5 + fold])
            else:
                C = float(c_val)
                
            clf = LogisticRegression(C=C, class_weight='balanced', solver='liblinear', random_state=seed)
            clf.fit(X_tr_s, y_tr)
            
            coef = clf.coef_[0]
            
            # c_k = coef * (mean z_test | y=1 - mean z_test | y=0)
            pos_te = y[te] == 1
            neg_te = ~pos_te
            
            if pos_te.sum() > 0 and neg_te.sum() > 0:
                mean_pos = X_te_s[pos_te].mean(axis=0)
                mean_neg = X_te_s[neg_te].mean(axis=0)
                c_k = coef * (mean_pos - mean_neg)
            else:
                c_k = np.zeros_like(coef)
                
            contribs.append(c_k)
            
    return np.array(contribs)

def run_permutation(backend, P, device="cpu"):
    meta = data.load_meta()
    y_true = meta["y"].values
    networks = data.networks()
    
    # For LR we do 3 repeats, for WGNAN we do 1 repeat (fold 0-4)
    repeats = (0, 1, 2) if backend == "lr_strength" else (0,)
    
    def get_contribs(y_iter, seed):
        if backend == "lr_strength":
            return get_lr_strength_contributions(y_iter, seed, repeats)
        else:
            from models import wgnan
            var = backend.split("_")[1] # e.g. wgnan_learned -> learned
            c, _ = wgnan.run_cv_contrib(var, y_iter, repeats=repeats, fast=True, device=device)
            return c

    print("Computing observed statistic...")
    obs_c = get_contribs(y_true, seed=42)
    
    # Stat calculation
    # For each network M_net = mean_k mean_{i in net} z_k(|c_k|)_i
    def calc_stats(c_arr):
        # c_arr is [n_folds, 432]
        # z-score across nodes within fold
        mu = c_arr.mean(axis=1, keepdims=True)
        sigma = c_arr.std(axis=1, keepdims=True)
        z_abs = (np.abs(c_arr) - mu) / np.clip(sigma, 1e-8, None)
        z_signed = (c_arr - mu) / np.clip(sigma, 1e-8, None)
        
        stats = {}
        for net in data.PREREG_NETWORKS:
            mask = networks == net
            stats[f"{net}_abs"] = z_abs[:, mask].mean()
            stats[f"{net}_signed"] = z_signed[:, mask].mean()
            
        # Stability: rho_bar
        n_f = c_arr.shape[0]
        rhos = []
        for i in range(n_f):
            for j in range(i+1, n_f):
                r, _ = spearmanr(c_arr[i], c_arr[j])
                rhos.append(r)
        stats["stability"] = float(np.mean(rhos)) if rhos else 0.0
        return stats
        
    obs_stats = calc_stats(obs_c)
    
    print("Running null permutations...")
    null_dist = []
    rng = np.random.default_rng(123)
    
    for p in range(P):
        print(f"Permutation {p+1}/{P}")
        y_p = rng.permutation(y_true)
        c_p = get_contribs(y_p, seed=123+p)
        null_stats = calc_stats(c_p)
        null_stats["perm"] = p + 1
        null_dist.append(null_stats)
        
    null_df = pd.DataFrame(null_dist)
    null_df.to_csv(paths.PERM / f"{backend}_null.csv", index=False)
    
    obs_stats_serializable = {k: float(v) for k, v in obs_stats.items()}
    with open(paths.PERM / f"{backend}_observed.json", "w") as f:
        json.dump(obs_stats_serializable, f, indent=2)
        
    print("\n--- Results ---")
    for net in data.PREREG_NETWORKS:
        k = f"{net}_abs"
        obs_val = obs_stats[k]
        null_vals = null_df[k].values
        pval = (1 + (null_vals >= obs_val).sum()) / (P + 1)
        print(f"Network {net}: Obs Z = {obs_val:.3f}, p = {pval:.4f}")
        
    obs_stab = obs_stats["stability"]
    null_stab = null_df["stability"].values
    pval_stab = (1 + (null_stab >= obs_stab).sum()) / (P + 1)
    print(f"Stability rho: Obs = {obs_stab:.3f}, p = {pval_stab:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["lr_strength", "wgnan_learned", "wgnan_nodeonly"], required=True)
    parser.add_argument("--n-perm", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_permutation(args.backend, args.n_perm, args.device)
