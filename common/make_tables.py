"""Aggregate every results/preds/<model>.csv into the benchmark table.

    python -m common.make_tables                  # reference = lr_edges
    python -m common.make_tables --ref lr_strength

Writes results/tables/benchmark.csv, benchmark.md and benchmark.tex (booktabs).
Safe to run any time, by anyone; it only reads preds/ and overwrites tables/benchmark.*.
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import paths
from .metrics import bootstrap_ci, point_metrics

# display order + pretty names; unknown models are appended at the end
ORDER = [
    ("majority", "Majority class"),
    ("lr_scale", "LR: scale only (total, density)"),
    ("lr_demo", "LR: age + sex"),
    ("lr_edges", "LR: 93k edges"),
    ("lr_strength", "LR: 432 node strengths"),
    ("lr_edges_log1p", "LR: 93k edges, old log1p pipeline"),
    ("lr_edges_resid", "LR: 93k edges, age/sex/total removed"),
    ("gcn", "GCN"),
    ("gat", "GAT"),
    ("wgnan_nodeonly", "W-GNAN: node terms only"),
    ("wgnan_topology", "W-GNAN: binary topology"),
    ("wgnan_linear", "W-GNAN: g(w)=w"),
    ("wgnan_learned", "W-GNAN: learned g(w)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="lr_edges")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()

    files = sorted(paths.PREDS.glob("*.csv"))
    preds = {f.stem: pd.read_csv(f) for f in files}
    if not preds:
        raise SystemExit("no prediction files in results/preds yet")
    ref = preds.get(a.ref)
    names = dict(ORDER)
    order = [k for k, _ in ORDER if k in preds] + sorted(k for k in preds if k not in names)

    rows = []
    for k in order:
        df = preds[k]
        m = point_metrics(df)
        ci = bootstrap_ci(df, n_boot=a.n_boot, ref=ref if (ref is not None and k != a.ref) else None)
        row = {"model": k, "name": names.get(k, k), **m,
               "AUROC_lo": ci["AUROC_ci"][0], "AUROC_hi": ci["AUROC_ci"][1],
               "AUPRC_lo": ci["AUPRC_ci"][0], "AUPRC_hi": ci["AUPRC_ci"][1]}
        if "dAUROC" in ci:
            row.update(dAUROC=ci["dAUROC"], dAUROC_lo=ci["dAUROC_ci"][0], dAUROC_hi=ci["dAUROC_ci"][1])
        rows.append(row)
        print(f"done {k}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(paths.TABLES / "benchmark.csv", index=False)

    fmt = lambda v, lo, hi: f"{v:.3f} [{lo:.2f}, {hi:.2f}]"
    show = pd.DataFrame({
        "Model": t.name,
        "AUROC [95% CI]": [fmt(r.AUROC, r.AUROC_lo, r.AUROC_hi) for r in t.itertuples()],
        "AUPRC [95% CI]": [fmt(r.AUPRC, r.AUPRC_lo, r.AUPRC_hi) for r in t.itertuples()],
        "balAcc": t.balAcc.map("{:.3f}".format), "F1": t.F1.map("{:.3f}".format),
        f"dAUROC vs {a.ref}": [
            (fmt(r.dAUROC, r.dAUROC_lo, r.dAUROC_hi) if pd.notna(getattr(r, "dAUROC", float("nan"))) else "--")
            for r in t.itertuples()],
        "reps": t.n_repeats,
    })
    try:
        md = show.to_markdown(index=False)
    except ImportError:                      # tabulate not installed
        md = show.to_string(index=False)
    (paths.TABLES / "benchmark.md").write_text(md + "\n\nAUPRC chance level = 0.069\n")
    try:
        tex = show.to_latex(index=False, escape=True)
    except ImportError:                      # jinja2 not installed
        tex = "\n".join(" & ".join(map(str, r)) + r" \\" for r in [show.columns, *show.values])
    (paths.TABLES / "benchmark.tex").write_text(tex)
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
