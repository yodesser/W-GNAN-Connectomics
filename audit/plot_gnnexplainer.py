import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import time

from torch_geometric.explain import Explainer, GNNExplainer

from common import data, paths
from baselines.train_gat import load_fold, build_graphs

def explain_gat(device="cpu"):
    print("Loading GAT Fold 0...")
    model, mask, fs, es = load_fold(0, 0)
    model.to(device)
    model.eval()
    
    splits = pd.read_csv(paths.SPLITS)
    s = splits[(splits["repeat"] == 0) & (splits["fold"] == 0)]
    te = s[s["role"] == "test"]["idx"].values
    y = data.load_meta()["y"].values
    
    M = data.load_mats("norm")
    graphs = build_graphs(M, te, mask, fs, es, y)
    
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='object',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='graph',
            return_type='raw',
        ),
    )
    
    node_imps = []
    print(f"Running GNNExplainer on {len(graphs)} test graphs...")
    t0 = time.time()
    for i, g in enumerate(graphs):
        x = g.x.to(device)
        ei = g.edge_index.to(device)
        ea = g.edge_attr.to(device)
        batch = torch.zeros(g.x.size(0), dtype=torch.long, device=device)
        
        explanation = explainer(x, ei, edge_attr=ea, batch=batch)
        node_imps.append(explanation.node_mask.detach().cpu().numpy().ravel())
        
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(graphs)} done...")
            
    print(f"GNNExplainer finished in {time.time()-t0:.1f}s")
    return np.array(node_imps), te

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        wgnan_diffs = np.load(paths.CONTRIB / "wgnan_learned_node_contrib_oof.npy")
        wgnan_diff = wgnan_diffs[0] # fold 0
    except FileNotFoundError:
        print("W-GNAN contribs not found. Please wait for Colab models runs to finish!")
        return

    gat_imps, te = explain_gat(device)
    y = data.load_meta()["y"].values
    
    pos = y[te] == 1
    neg = ~pos
    
    if pos.sum() > 0 and neg.sum() > 0:
        gat_diff = gat_imps[pos].mean(axis=0) - gat_imps[neg].mean(axis=0)
    else:
        gat_diff = gat_imps.mean(axis=0)
        
    from scipy.stats import spearmanr
    rho, p = spearmanr(gat_diff, wgnan_diff)
    
    plt.figure(figsize=(6, 5))
    sns.regplot(x=wgnan_diff, y=gat_diff, scatter_kws={'alpha':0.5, 'color': '#2b5c8f'}, line_kws={'color': '#d9534f'})
    plt.xlabel("Intrinsic Node Contribution (W-GNAN)")
    plt.ylabel("Post-hoc Node Mask Importance (GAT + GNNExplainer)")
    plt.title(f"Post-hoc vs Intrinsic Explanations\nSpearman $\\rho$ = {rho:.3f} (p={p:.1e})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths.FIGURES / "fig4b_explainer.pdf")
    print(f"Saved {paths.FIGURES / 'fig4b_explainer.pdf'}")
    
    # Save the raw mask differences for reporting
    df = pd.DataFrame({"wgnan_diff": wgnan_diff, "gat_gnnexplainer_diff": gat_diff})
    df.to_csv(paths.CONTRIB / "explainer_comparison.csv", index=False)

if __name__ == "__main__":
    main()
