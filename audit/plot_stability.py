import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Add the repository root to sys.path so we can import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common import paths

def pairwise_spearman(C):
    n = C.shape[0]
    corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = spearmanr(C[i], C[j])
            corrs.append(r)
    return np.array(corrs)

def sign_consistency(C, top_k=30, threshold=0.8):
    n_rows = C.shape[0]
    mean_abs = np.abs(C).mean(axis=0)
    top_nodes = np.argsort(mean_abs)[::-1][:top_k]
    
    consistent_count = 0
    for node in top_nodes:
        vals = C[:, node]
        pos = (vals > 0).sum()
        neg = (vals < 0).sum()
        if max(pos, neg) / n_rows >= threshold:
            consistent_count += 1
            
    return consistent_count / top_k

def main():
    print("Loading C_in and C_oof...")
    C_in = np.load(paths.OLD_C_IN)
    C_oof = np.load(paths.OLD_C_OOF)
    
    print(f"C_in shape: {C_in.shape}, C_oof shape: {C_oof.shape}")
    
    # Compute correlations
    print("Computing correlations...")
    corrs_in = pairwise_spearman(C_in)
    corrs_oof = pairwise_spearman(C_oof)
    
    # Compute sign consistency
    print("Computing sign consistencies...")
    sc_in = sign_consistency(C_in)
    sc_oof = sign_consistency(C_oof)
    
    print(f"In-sample mean rho: {corrs_in.mean():.3f}, sign consistency: {sc_in:.2f}")
    print(f"OOF mean rho: {corrs_oof.mean():.3f}, sign consistency: {sc_oof:.2f}")
    
    # Save JSON
    out_json = paths.EXPLAIN / "insample_vs_oof.json"
    res = {
        "in_sample": {
            "mean_rho": float(corrs_in.mean()),
            "median_rho": float(np.median(corrs_in)),
            "sign_consistency": float(sc_in)
        },
        "out_of_fold": {
            "mean_rho": float(corrs_oof.mean()),
            "median_rho": float(np.median(corrs_oof)),
            "sign_consistency": float(sc_oof)
        }
    }
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved {out_json}")
    
    # Plot
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Jitter
    jitter_in = np.random.normal(1, 0.04, size=len(corrs_in))
    jitter_oof = np.random.normal(2, 0.04, size=len(corrs_oof))
    
    ax.scatter(jitter_in, corrs_in, alpha=0.6, s=15, color='blue', edgecolors='none')
    ax.scatter(jitter_oof, corrs_oof, alpha=0.3, s=15, color='red', edgecolors='none')
    
    ax.boxplot([corrs_in, corrs_oof], positions=[1, 2], widths=0.4, showfliers=False, 
               medianprops=dict(color="black", linewidth=2),
               boxprops=dict(color="black"))
               
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"In-sample\n(n={len(corrs_in)} pairs)", f"Out-of-fold\n(n={len(corrs_oof)} pairs)"])
    ax.set_ylabel("Pairwise Spearman Correlation (ρ)")
    ax.set_title(f"Stability of Contributions\nSign consistency: In-sample {sc_in:.2f} vs OOF {sc_oof:.2f}")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(-0.2, 1.05)
    
    # Save figure
    out_fig = paths.FIGURES / "fig4a_insample_oof.pdf"
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    print(f"Saved {out_fig}")

if __name__ == "__main__":
    main()
