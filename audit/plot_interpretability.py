"""Figure 4 (interpretability audit), double column.

    python -m audit.plot_interpretability  -> results/figures/fig4_interp.pdf/.png

(a) agreement of region rankings between runs: in-sample (5 seeds, all data) vs out-of-fold (25 held-out folds)
(b) permutation audit: observed statistic vs label-permuted null (median, 95% interval), both backends
(c) GNNExplainer (GAT) per fold: agreement with W-GNAN, and PTSD-vs-control mask similarity
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import paths

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#8a8a85"
plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})


def pair_rhos(C):
    return [spearmanr(C[i], C[j])[0] for i in range(len(C)) for j in range(i + 1, len(C))]


def main():
    fig, ax = plt.subplots(1, 3, figsize=(6.3, 1.95), gridspec_kw=dict(width_ratios=[0.8, 1.5, 1.2]))

    # (a) in-sample vs out-of-fold
    rin, roof = pair_rhos(np.load(paths.OLD_C_IN)), pair_rhos(np.load(paths.OLD_C_OOF))
    rng = np.random.default_rng(0)
    for x, r, c in [(0, rin, BLUE), (1, roof, ORANGE)]:
        ax[0].scatter(x + rng.uniform(-.15, .15, len(r)), r, s=6, color=c, alpha=.55, lw=0)
        ax[0].plot([x - .25, x + .25], [np.mean(r)] * 2, color=INK, lw=1.5)
    ax[0].set_xticks([0, 1], ["in-sample", "held-out"])
    ax[0].set_ylabel("Spearman $\\rho$ between runs")
    ax[0].set_ylim(-0.3, 1.0); ax[0].axhline(0, color=MUTED, lw=.6, ls=":")
    ax[0].set_title("(a) ranking agreement", fontsize=7.5, loc="left")

    # (b) permutation audit
    stats = [("Cont_abs", "Control"), ("SalVentAttn_abs", "Salience"), ("Default_abs", "Default"),
             ("stability", "Stability")]
    for k, (b, lab, c, dy) in enumerate([("lr_strength", "LR", BLUE, -.15), ("wgnan_learned", "W-GNAN", ORANGE, .15)]):
        null = pd.read_csv(paths.PERM / f"{b}_null.csv"); obs = json.load(open(paths.PERM / f"{b}_observed.json"))
        for i, (s, _) in enumerate(stats):
            x = null[s].to_numpy()
            lo, md, hi = np.percentile(x, [2.5, 50, 97.5])
            # standardise each statistic by its null so both backends share one axis
            z = lambda v: (v - x.mean()) / x.std()
            ax[1].plot([z(lo), z(hi)], [i + dy] * 2, color=c, lw=2.2, alpha=.35, solid_capstyle="round")
            ax[1].scatter([z(obs[s])], [i + dy], s=14, color=c, zorder=3, label=lab if i == 0 else None)
    ax[1].axvline(0, color=MUTED, lw=.6, ls=":")
    ax[1].set_yticks(range(len(stats)), [s[1] for s in stats])
    ax[1].set_xlabel("observed, in SD of permutation null")
    ax[1].set_ylim(4.35, -0.45)
    ax[1].legend(frameon=False, fontsize=6, loc="lower center", ncol=2, handletextpad=.2, columnspacing=1.2)
    ax[1].set_title("(b) permutation audit", fontsize=7.5, loc="left")

    # (c) GNNExplainer
    d = pd.read_csv(paths.EXPLAIN / "agreement_folds.csv")
    xs = np.arange(len(d))
    ax[2].bar(xs - .2, d.rho_gnnexpl_vs_wgnan, .36, color=BLUE, label="vs W-GNAN")
    ax[2].bar(xs + .2, d.rho_pos_vs_neg_masks, .36, color=ORANGE, label="PTSD vs control masks")
    ax[2].axhline(0, color=MUTED, lw=.6)
    ax[2].set_xticks(xs, [f"f{i}" for i in d.fold]); ax[2].set_ylim(-0.2, 1.45)
    ax[2].set_ylabel("Spearman $\\rho$"); ax[2].legend(frameon=False, fontsize=6, loc="upper left", ncol=1)
    ax[2].set_title("(c) GNNExplainer on GAT", fontsize=7.5, loc="left")

    fig.tight_layout(w_pad=1.0)
    fig.savefig(paths.FIGURES / "fig4_interp.pdf"); fig.savefig(paths.FIGURES / "fig4_interp.png", dpi=200)
    print("wrote", paths.FIGURES / "fig4_interp.pdf", "| in-sample rho %.2f, held-out %.2f" % (np.mean(rin), np.mean(roof)))


if __name__ == "__main__":
    main()
