"""Synthetic validation v2: isolates the W-GNAN interaction module (fixes the v1 design problems).

Why v1 (models/synthetic.py) cannot support the paper's claims:
  * the node term f_i(strength_i) already sees the weighted signal (strength = sum_j w_ij), so
    topology == g(w)=w == learned g (all ~0.85-0.89 AUROC): the ablation shows no benefit of weights;
  * `binary` case: g_true = 1 on every entry -> latent is constant -> labels are pure noise (AUROC 0.50);
  * learned-gate correlation 0.90 / 0.92 on threshold / saturating (< 0.95 claimed in 05_results.tex).

v2 changes (models/wgnan.py is NOT modified; the real-data model is untouched):
  * graph cases use EdgeChannelWGNAN = WGNAN with the node term replaced by a learned per-region
    constant alpha_i (centred), i.e. exactly the data-generating structure u_j = alpha_j. The interaction
    module (g, lambda, neighbour mean) is the unchanged WGNAN code. Say this in §3/§5.1.
  * `binary`: every subject gets its own random binary topology (70 % edge keep), all weights = 1,
    so labels depend on topology and topology/linear/learned must coincide (g(1) = 1).
  * `regions` (node term only): unchanged (plain WGNAN nodeonly).
  * gate recovery reported on the full grid AND on the support of the observed weights (5th-95th pct).
  * early stopping on the 20 % val split (patience 20, max 300 epochs) instead of a fixed 100 epochs.

Run (from the repository root, CPU ok, ~5-10 min; GPU faster):
    python -m models.train_synthetic
    python -m models.train_synthetic --fig-only        # just redraw fig2 from the saved curves
Outputs: results/synthetic/synthetic_v2b_results.csv, synthetic_v2b_gate_curves.csv,
         results/tables/synthetic_v2b.tex, results/figures/fig2_synthetic_v2b.pdf
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from common import paths
from models import wgnan
from models.synthetic import true_gate

CASES = ["linear", "threshold", "saturating", "binary", "regions"]


class EdgeChannelWGNAN(wgnan.WGNAN):
    """WGNAN whose node term is a centred per-region constant alpha_i (graph-channel test)."""

    def __init__(self, variant, stats):
        super().__init__(variant, stats)
        self.roi_effect = nn.Parameter(torch.zeros(self.n_nodes))

    def alpha(self):
        return self.roi_effect - self.roi_effect.mean()

    def forward_components(self, L, S, D):
        u = self.alpha().unsqueeze(0).expand(L.shape[0], -1)
        w = torch.clamp(L / torch.clamp(self.q99, min=1e-6), 0, 1)
        mask = L > 0
        if self.variant == "topology":
            gw = mask.float()
        elif self.variant == "linear":
            gw = torch.where(mask, w, torch.zeros_like(w))
        else:
            gw = torch.where(mask, self.gate(w), torch.zeros_like(w))
        n = mask.sum(-1).float().clamp(min=1.0)
        neigh = torch.bmm(gw, u.unsqueeze(-1)).squeeze(-1) / n
        total = u + self.graph_lambda() * neigh
        return {"logit": self.bias + total.sum(-1), "total": total}

    def regularization(self, l2):
        r = l2 * (self.roi_effect ** 2).mean()
        if self.variant == "learned":
            r = r + 1e-3 * self.gate.smoothness()
        return r


def make_synthetic(case, seed, n_subjects=700, n_nodes=36):
    """Same generator as models/synthetic.py except the binary case (per-subject random topology)."""
    rng = np.random.default_rng(seed)
    mask = np.triu(rng.random((n_nodes, n_nodes)) < 0.25, 1); mask = mask | mask.T
    for i in range(n_nodes):
        if not mask[i].any():
            j = (i + 1) % n_nodes; mask[i, j] = mask[j, i] = True
    alpha = rng.normal(size=n_nodes); alpha -= alpha.mean()
    gamma = np.zeros(n_nodes)
    if case == "regions":
        gamma[rng.choice(n_nodes, 4, replace=False)] = [2.0, -2.0, 1.5, -1.5]
    mats = np.zeros((n_subjects, n_nodes, n_nodes), np.float32)
    latent = np.zeros(n_subjects)
    for s in range(n_subjects):
        w = rng.beta(2, 2, size=(n_nodes, n_nodes)); w = (w + w.T) / 2; w *= mask; np.fill_diagonal(w, 0)
        if case == "binary":
            keep = np.triu(rng.random((n_nodes, n_nodes)) < 0.7, 1); keep = keep | keep.T
            w = ((w > 0) & keep).astype(np.float64)
        mats[s] = w
        if case == "regions":
            latent[s] = np.sum(gamma * w.sum(1))
        else:
            g = (w > 0).astype(float) if case == "binary" else true_gate(w, case)
            deg = (w > 0).sum(1).clip(min=1)
            latent[s] = np.mean(alpha + 2 * (g @ alpha) / deg)
    latent += rng.normal(0, 0.12 * latent.std(), size=n_subjects)
    return mats, (latent >= np.median(latent)).astype(int), (gamma if case == "regions" else alpha)


def fit(model, mats, S, D, y, tr, va, dev, max_ep=300, patience=20):
    T = lambda a: torch.tensor(a).to(dev)
    Ltr, Str, Dtr, ytr = T(mats[tr]), T(S[tr]), T(D[tr]), torch.tensor(y[tr], dtype=torch.float32, device=dev)
    Lva, Sva, Dva = T(mats[va]), T(S[va]), T(D[va])
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0)
    lossf = nn.BCEWithLogitsLoss()
    best, state, bad = -1.0, None, 0
    for ep in range(max_ep):
        model.train(); opt.zero_grad()
        loss = lossf(model(Ltr, Str, Dtr), ytr) + model.regularization(1e-3)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            auc = roc_auc_score(y[va], model(Lva, Sva, Dva).cpu().numpy())
        if auc > best:
            best, state, bad = auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience and ep >= 30:
                break
    model.load_state_dict(state)
    return model


def run(seeds, dev):
    rows, curves = [], []
    for case in CASES:
        for seed in seeds:
            mats, y, truth = make_synthetic(case, seed)
            idx = np.arange(len(y))
            otr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
            tr, va = train_test_split(otr, test_size=0.25, stratify=y[otr], random_state=seed + 1)
            S = mats.sum(-1).astype(np.float32)
            D = ((mats > 0).sum(-1) / (mats.shape[1] - 1)).astype(np.float32)
            stats = wgnan.get_train_stats(mats, S, D, tr); stats["q99"] = 1.0
            nz = mats[tr][mats[tr] > 0]
            lo, hi = np.percentile(nz, 5), np.percentile(nz, 95)
            variants = ["nodeonly"] if case == "regions" else ["topology", "linear", "learned"]
            for var in variants:
                torch.manual_seed(seed)
                cls = wgnan.WGNAN if case == "regions" else EdgeChannelWGNAN
                model = fit(cls(var, stats).to(dev), mats, S, D, y, tr, va, dev)
                with torch.no_grad():
                    s = model(torch.tensor(mats[te]).to(dev), torch.tensor(S[te]).to(dev),
                              torch.tensor(D[te]).to(dev)).cpu().numpy()
                r = dict(case=case, seed=seed, variant=var, auroc=roc_auc_score(y[te], s),
                         ap=average_precision_score(y[te], s), lam=float(model.graph_lambda()))
                if var == "learned" and case != "binary":
                    x = np.linspace(0, 1, 101, dtype=np.float32)
                    with torch.no_grad():
                        gl = model.gate(torch.tensor(x).to(dev)).cpu().numpy()
                    gt = true_gate(x, case)
                    sup = (x >= lo) & (x <= hi)
                    r.update(gate_corr=float(np.corrcoef(gl, gt)[0, 1]),
                             gate_corr_support=float(np.corrcoef(gl[sup], gt[sup])[0, 1]),
                             gate_rmse=float(np.sqrt(np.mean((gl - gt) ** 2))))
                    curves += [dict(case=case, seed=seed, x=float(a), g_true=float(b), g_learned=float(c),
                                    in_support=bool(d)) for a, b, c, d in zip(x, gt, gl, sup)]
                if case != "regions":
                    r["alpha_rho"] = float(spearmanr(model.alpha().detach().cpu().numpy(), truth)[0])
                else:
                    n = mats.shape[1]
                    with torch.no_grad():
                        amp = (model.region_shape(torch.full((1, n), 2.0, device=dev))[0]
                               - model.region_shape(torch.full((1, n), -2.0, device=dev))[0]).abs().cpu().numpy()
                    planted = np.where(truth != 0)[0]
                    r.update(topk_prec=len(set(np.argsort(-amp)[:4]) & set(planted)) / 4,
                             roi_rho=float(spearmanr(amp, np.abs(truth))[0]))
                rows.append(r)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(paths.SYNTH / "synthetic_v2b_results.csv", index=False)
    pd.DataFrame(curves).to_csv(paths.SYNTH / "synthetic_v2b_gate_curves.csv", index=False)
    return res


def table(res):
    g = res.groupby(["case", "variant"])
    ms = lambda c: g[c].agg(["mean", "std"])
    au = ms("auroc")
    out = []
    for case in CASES:
        vs = ["nodeonly"] if case == "regions" else ["topology", "linear", "learned"]
        cells = [f"{au.loc[(case, v), 'mean']:.2f}$\\pm${au.loc[(case, v), 'std']:.2f}" for v in vs]
        if case == "regions":
            tk = res[res.case == "regions"].topk_prec.mean()
            out.append(f"regions (node term) & \\multicolumn{{3}}{{c}}{{{cells[0]}}} & top-4 prec. {tk:.2f} \\\\")
        else:
            gc = res[(res.case == case) & (res.variant == "learned")]
            gtxt = "--" if case == "binary" else f"{gc.gate_corr_support.mean():.3f}"
            out.append(f"{case} & " + " & ".join(cells) + f" & {gtxt} \\\\")
    tex = ("\\begin{tabular}{lcccc}\\toprule\n"
           "Case & Topology & $g(w)=w$ & Learned $g$ & corr$(g,\\hat g)$ \\\\\\midrule\n"
           + "\n".join(out) + "\n\\bottomrule\\end{tabular}\n")
    (paths.TABLES / "synthetic_v2b.tex").write_text(tex)
    print(res.groupby(["case", "variant"]).mean(numeric_only=True).round(3).to_string())


def figure():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = pd.read_csv(paths.SYNTH / "synthetic_v2b_gate_curves.csv")
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 1.8), sharey=True)
    for ax, case in zip(axes, ["linear", "threshold", "saturating"]):
        d = c[c.case == case].groupby("x").agg(t=("g_true", "first"), m=("g_learned", "mean"), s=("g_learned", "std"))
        sup = c[(c.case == case) & c.in_support].x
        ax.axvspan(sup.min(), sup.max(), color="0.92", lw=0, zorder=0)
        ax.fill_between(d.index, d.m - d.s, d.m + d.s, alpha=.3, lw=0)
        ax.plot(d.index, d.m, lw=1.2, label="learned (mean$\\pm$sd)")
        ax.plot(d.index, d.t, "k--", lw=1, label="true")
        ax.set_title(case, fontsize=8); ax.set_xlabel("$w$", fontsize=7); ax.tick_params(labelsize=6)
    axes[0].set_ylabel("$g(w)$", fontsize=7); axes[0].legend(fontsize=5.5, frameon=False, loc="upper left")
    ax = axes[3]
    try:
        r = pd.read_csv(paths.CONTRIB / "wgnan_learned_gate_curves.csv").groupby("x").g
        ax.fill_between(r.median().index, r.min(), r.max(), color="grey", alpha=.3, lw=0)
        ax.plot(r.median().index, r.median().values, color="grey", lw=1.2, label="real data (15 folds)")
        ax.plot([0, 1], [0, 1], "k:", lw=.8, label="identity (init)")
        ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    except FileNotFoundError:
        pass
    ax.set_title("real data", fontsize=8); ax.set_xlabel("$w$", fontsize=7); ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(paths.FIGURES / "fig2_synthetic_v2b.pdf"); fig.savefig(paths.FIGURES / "fig2_synthetic_v2b.png", dpi=200)
    print("wrote", paths.FIGURES / "fig2_synthetic_v2b.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--fig-only", action="store_true")
    a = ap.parse_args()
    if not a.fig_only:
        torch.set_num_threads(4)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        table(run(a.seeds, dev))
    figure()
