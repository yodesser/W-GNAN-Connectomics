"""Permutation-audit table (Table: interp) from results/perm/*_null.csv + *_observed.json.

    python -m audit.tabulate_interpretability          -> results/tables/interp.tex + interp.csv

Statistic per network = mean over folds of the network's normalised |contribution|
((|c| - mean c) / sd c across the 432 regions, as computed identically for observed and null runs).
p = (1 + #{null >= observed}) / (n_perm + 1), one-sided (larger than chance).
"""
import json

import numpy as np
import pandas as pd

from common import paths

BACKENDS = [("lr_strength", "LR (node strengths)"), ("wgnan_learned", "W-GNAN (learned $g$)")]
STATS = [("Cont_abs", "Control"), ("SalVentAttn_abs", "Salience"), ("Default_abs", "Default mode"),
         ("stability", "Fold-to-fold stability ($\\bar\\rho$)")]


def main():
    rows = []
    for b, bname in BACKENDS:
        try:
            null = pd.read_csv(paths.PERM / f"{b}_null.csv")
            obs = json.load(open(paths.PERM / f"{b}_observed.json"))
        except FileNotFoundError:
            print("missing", b); continue
        for k, kname in STATS:
            x = null[k].to_numpy(); v = float(obs[k])
            rows.append(dict(backend=bname, stat=kname, observed=v, null_mean=x.mean(),
                             null_p95=np.percentile(x, 95), p=(1 + (x >= v).sum()) / (len(x) + 1), n_perm=len(x)))
    t = pd.DataFrame(rows)
    t.to_csv(paths.TABLES / "interp.csv", index=False)
    lines = ["\\begin{tabular}{llcccc}", "\\toprule",
             "Model & Statistic & Observed & Null mean & Null 95\\% & $p$ \\\\", "\\midrule"]
    for i, (bname, g) in enumerate(t.groupby("backend", sort=False)):
        if i:
            lines.append("\\midrule")
        for j, r in enumerate(g.itertuples()):
            lab = f"{bname} ({r.n_perm} perm.)" if j == 0 else ""
            lines.append(f"{lab} & {r.stat} & {r.observed:.3f} & {r.null_mean:.3f} & {r.null_p95:.3f} & {r.p:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (paths.TABLES / "interp.tex").write_text("\n".join(lines) + "\n")
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
