import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from models import wgnan
from common import data, paths

def calc_stats(c_arr, networks):
    mu = c_arr.mean(axis=1, keepdims=True)
    sigma = c_arr.std(axis=1, keepdims=True)
    z_abs = (np.abs(c_arr) - mu) / np.clip(sigma, 1e-8, None)
    z_signed = (c_arr - mu) / np.clip(sigma, 1e-8, None)
    
    stats = {}
    for net in data.PREREG_NETWORKS:
        mask = networks == net
        stats[f"{net}_abs"] = z_abs[:, mask].mean()
        stats[f"{net}_signed"] = z_signed[:, mask].mean()
        
    n_f = c_arr.shape[0]
    rhos = []
    for i in range(n_f):
        for j in range(i+1, n_f):
            r, _ = spearmanr(c_arr[i], c_arr[j])
            rhos.append(r)
    stats["stability"] = float(np.mean(rhos)) if rhos else 0.0
    return stats

def main():
    print("Recalculating observed statistics locally...")
    y_true = data.load_meta()["y"].values
    networks = data.networks()

    obs_c, _ = wgnan.run_cv_contrib("learned", y_true, repeats=(0,), fast=True)
    obs_stats = calc_stats(obs_c, networks)

    # Save JSON correctly
    obs_stats_serializable = {k: float(v) for k, v in obs_stats.items()}
    with open(paths.PERM / "wgnan_learned_observed.json", "w") as f:
        json.dump(obs_stats_serializable, f, indent=2)

    # Load the saved nulls and compute p-values!
    null_df = pd.read_csv(paths.PERM / "wgnan_learned_null.csv")
    P = len(null_df)

    print("\n--- Final Results ---")
    for net in data.PREREG_NETWORKS:
        k = f"{net}_abs"
        obs_val = obs_stats[k]
        null_vals = null_df[k].values
        pval = (1 + (null_vals >= obs_val).sum()) / (P + 1)
        print(f"Network {net}: Obs Z = {obs_val:.3f}, p = {pval:.4f}")

if __name__ == "__main__":
    main()
