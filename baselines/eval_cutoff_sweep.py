"""Optional: does the weak signal depend on the PCL-5 >= 33 cutoff?

    python -m baselines.eval_cutoff_sweep            # ~1 min on CPU

lr_strength (the fast model) re-fit at cutoffs 20 / 25 / 30 / 33. For EACH cutoff:
own 5-fold x 1 repeat stratified on that cutoff's label (seed 2026), inner 20 % val
(same recipe as common/splits.py), C + threshold picked on val exactly like bench_linear.
These splits are NOT the frozen ones (the label changes), so nothing goes to results/preds/
and nothing touches the benchmark table. Output: results/tables/cutoff_sweep.csv (+ .tex).
The 33 row should land near the frozen-split lr_strength (0.572) -- if it's way off, say so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from common import paths
from common.data import load_meta, node_features
from common.metrics import bootstrap_ci, choose_threshold, point_metrics
from baselines.train_linear import fit_lr

CUTOFFS = [20, 25, 30, 33]
SEED, N_FOLDS, VAL_FRAC = 2026, 5, 0.20


def run_cutoff(X, pcl5, cut):
    y = (pcl5 >= cut).astype(int)
    rows = []
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for f, (otr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        tr, va = train_test_split(otr, test_size=VAL_FRAC, stratify=y[otr], random_state=f)
        sc, clf = fit_lr(X, y, tr, va, wide=False)
        s_va = clf.decision_function(sc.transform(X[va]))
        s_te = clf.decision_function(sc.transform(X[te]))
        thr = choose_threshold(y[va], s_va)
        rows += [("lr_strength", 0, f, int(i), int(yy), float(s), thr) for i, yy, s in zip(te, y[te], s_te)]
    df = pd.DataFrame(rows, columns=["model", "repeat", "fold", "idx", "y", "score", "threshold"])
    m, ci = point_metrics(df), bootstrap_ci(df, n_boot=1000)
    return dict(cutoff=cut, n_pos=int(y.sum()), prevalence=round(float(y.mean()), 3),
                AUROC=m["AUROC"], AUROC_lo=ci["AUROC_ci"][0], AUROC_hi=ci["AUROC_ci"][1],
                AUPRC=m["AUPRC"], AUPRC_lo=ci["AUPRC_ci"][0], AUPRC_hi=ci["AUPRC_ci"][1],
                balAcc=m["balAcc"])


def main():
    meta = load_meta()
    pcl5 = meta.pcl5.to_numpy()
    X = node_features()["strength"].astype(np.float64)
    out = []
    for cut in CUTOFFS:
        r = run_cutoff(X, pcl5, cut)
        print(f"[cutoff {cut}] n_pos={r['n_pos']} ({r['prevalence']:.1%})  AUROC {r['AUROC']:.3f} "
              f"[{r['AUROC_lo']:.2f}, {r['AUROC_hi']:.2f}]  AUPRC {r['AUPRC']:.3f} (chance {r['prevalence']:.3f})",
              flush=True)
        out.append(r)
    t = pd.DataFrame(out)
    t.to_csv(paths.TABLES / "cutoff_sweep.csv", index=False)
    L = [r"\begin{tabular}{rrcc}", r"\toprule",
         r"PCL-5 cutoff & positives & AUROC [95\% CI] & AUPRC (chance) \\", r"\midrule"]
    for r in out:
        L.append(f"{r['cutoff']} & {r['n_pos']} ({100*r['prevalence']:.1f}\\%) & {r['AUROC']:.3f} "
                 f"[{r['AUROC_lo']:.2f}, {r['AUROC_hi']:.2f}] & {r['AUPRC']:.3f} ({r['prevalence']:.3f}) \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (paths.TABLES / "cutoff_sweep.tex").write_text("\n".join(L) + "\n")
    print(f"wrote {paths.TABLES / 'cutoff_sweep.csv'}")


if __name__ == "__main__":
    main()
