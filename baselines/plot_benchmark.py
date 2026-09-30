"""Figure 3 + sanity table + trimmed main table, all from results/tables/benchmark.csv.

    python -m common.make_tables          # first, so benchmark.csv is fresh
    python -m baselines.plot_benchmark

Writes
    results/figures/fig3_benchmark.pdf (+ .png preview)   single column, 3.2 in wide
    results/tables/benchmark_main.tex    Table 1: majority, lr_edges, lr_strength, gat, (gcn), 4 wgnan rows
    results/tables/sanity.tex            Table 2: scale / demo / log1p / resid + the 2 old regression rows

benchmark.tex (from make_tables) keeps ALL rows and gets overwritten on every make_tables run,
so the paper should \\input benchmark_main.tex + sanity.tex, not benchmark.tex.
Rows that don't exist yet (e.g. gcn, wgnan_learned) are just skipped -- rerun after new preds land.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import paths
from common.make_tables import ORDER

REF = "lr_edges"
CHANCE_AUPRC = 0.069
MAIN = ["majority", "lr_edges", "lr_strength", "gat", "gcn",
        "wgnan_nodeonly", "wgnan_topology", "wgnan_linear", "wgnan_learned"]
SANITY = ["lr_scale", "lr_demo", "lr_edges_log1p", "lr_edges_resid"]
SHORT = {  # short labels so the fig fits 3.2 in
    "majority": "Majority", "lr_edges": "LR edges (ref.)", "lr_strength": "LR strengths",
    "gat": "GAT", "gcn": "GCN", "wgnan_nodeonly": "W-GNAN node-only",
    "wgnan_topology": "W-GNAN binary topo.", "wgnan_linear": "W-GNAN $g(w)=w$",
    "wgnan_learned": "W-GNAN learned $g$",
}
# regression rows copied from the earlier five-fold pipeline's report (paper, Table 3)
REGRESSION = [("Constant (predict 3)", "8.09", "--", "--"),
              ("Ridge on edges", "9.40", "$-0.000$", "0.031 ($p=0.35$)")]

INK, MUTED, GRID, MARK = "#1f1f1f", "#6b6b6b", "#d9d9d9", "#2f5d8a"


def load():
    t = pd.read_csv(paths.TABLES / "benchmark.csv")
    order = [k for k, _ in ORDER]
    t["o"] = t.model.map(lambda m: order.index(m) if m in order else 99)
    return t.sort_values("o").set_index("model")


def ci(v, lo, hi, d=3):
    return f"{v:.{d}f} [{lo:.2f}, {hi:.2f}]"


def delta(r):
    if pd.isna(r.get("dAUROC", float("nan"))):
        return "--"
    return f"${r.dAUROC:+.3f}$ [${r.dAUROC_lo:+.2f}$, ${r.dAUROC_hi:+.2f}$]"   # math mode = real minus signs


def fig3(t):
    rows = [m for m in MAIN if m in t.index]
    n = len(rows)
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    fig, axs = plt.subplots(1, 2, figsize=(3.2, 0.22 * n + 0.75), sharey=True,
                            gridspec_kw={"wspace": 0.08})
    ys = list(range(n))[::-1]           # first model on top
    for ax, met, chance, lab in ((axs[0], "AUROC", 0.5, "AUROC"), (axs[1], "AUPRC", CHANCE_AUPRC, "AUPRC")):
        if met == "AUROC" and REF in t.index:   # grey band = reference CI
            ax.axvspan(t.loc[REF, "AUROC_lo"], t.loc[REF, "AUROC_hi"], color=GRID, alpha=0.6, lw=0, zorder=0)
        ax.axvline(chance, color=MUTED, ls="--", lw=0.7, zorder=1)
        for y, m in zip(ys, rows):
            r = t.loc[m]
            is_ref = m == REF
            ax.plot([r[f"{met}_lo"], r[f"{met}_hi"]], [y, y], color=INK if is_ref else MARK, lw=1.2,
                    solid_capstyle="round", zorder=2)
            ax.plot(r[met], y, "o", ms=3.6, mfc="white" if is_ref else MARK, mec=INK if is_ref else MARK,
                    mew=0.9, zorder=3)
        ax.set_xlabel(lab, color=INK, labelpad=2)
        ax.grid(axis="x", color=GRID, lw=0.4); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axs[0].set_yticks(ys); axs[0].set_yticklabels([SHORT.get(m, m) for m in rows], color=INK)
    axs[0].set_xlim(0.35, 0.75)
    axs[1].set_xlim(0.0, max(0.2, float(t.loc[rows, "AUPRC_hi"].max()) + 0.02))
    axs[0].set_title("(a)", loc="left", fontsize=7, color=INK, pad=2)
    axs[1].set_title("(b)", loc="left", fontsize=7, color=INK, pad=2)
    fig.subplots_adjust(left=0.36, right=0.98, top=0.93, bottom=0.12 if n > 6 else 0.18)
    for ext in ("pdf", "png"):
        fig.savefig(paths.FIGURES / f"fig3_benchmark.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[fig3] {n} models -> {paths.FIGURES / 'fig3_benchmark.pdf'}")


def main_table(t):
    rows = [m for m in MAIN if m in t.index]
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Model & AUROC [95\% CI] & AUPRC [95\% CI] & $\Delta$AUROC vs.\ LR edges & reps \\", r"\midrule"]
    for m in rows:
        r = t.loc[m]
        name = SHORT.get(m, m).replace(" (ref.)", "")
        L.append(f"{name} & {ci(r.AUROC, r.AUROC_lo, r.AUROC_hi)} & {ci(r.AUPRC, r.AUPRC_lo, r.AUPRC_hi)} "
                 f"& {delta(r)} & {int(r.n_repeats)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (paths.TABLES / "benchmark_main.tex").write_text("\n".join(L) + "\n")
    print(f"[table1] {len(rows)} rows -> benchmark_main.tex")


def sanity_table(t):
    names = {"lr_scale": "LR: scale only (total, density)", "lr_demo": "LR: age + sex",
             "lr_edges_log1p": "LR edges, old log1p pipeline", "lr_edges_resid": "LR edges, age/sex/total removed"}
    L = [r"\begin{tabular}{lcc}", r"\toprule",
         r"\multicolumn{3}{l}{\emph{Classification (PCL-5 $\geq$ 33)}} \\",
         r"Model & AUROC [95\% CI] & $\Delta$AUROC vs.\ LR edges \\", r"\midrule"]
    for m in SANITY:
        if m in t.index:
            r = t.loc[m]
            L.append(f"{names[m]} & {ci(r.AUROC, r.AUROC_lo, r.AUROC_hi)} & {delta(r)} \\\\")
    L += [r"\midrule", r"\multicolumn{3}{l}{\emph{Regression on PCL-5 score}$^\dagger$} \\",
          r"Model & MAE & $R^2$ / Spearman $\rho$ \\", r"\midrule"]
    for name, mae, r2, rho in REGRESSION:
        L.append(f"{name} & {mae} & {r2 if r2 != '--' else '--'}" + (f" / {rho}" if rho != "--" else "") + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]   # dagger note lives in the caption (keeps the table narrow)
    (paths.TABLES / "sanity.tex").write_text("\n".join(L) + "\n")
    print("[table2] -> sanity.tex")


if __name__ == "__main__":
    t = load()
    fig3(t); main_table(t); sanity_table(t)
