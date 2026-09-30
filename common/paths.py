"""Single source of truth for every path used in this repository.

The frozen inputs below follow the (private) project data folder layout; they are not
distributed with the repository.

Works unchanged on:
  * Windows + Google Drive for Desktop  (G:/My Drive/GML_project)
  * macOS Google Drive                  (~/Library/CloudStorage/GoogleDrive-*/My Drive/GML_project)
  * Colab                               (/content/drive/MyDrive/GML_project)
Override with the env var GML_ROOT if your Drive is mounted somewhere else.

Heavy derived caches (normalised matrices, edge vectors) go to a LOCAL folder
(GML_CACHE, default ~/.gml_cache) -- never to Drive -- so three people don't
sync 650 MB files at each other.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path


def _find_root() -> Path:
    env = os.environ.get("GML_ROOT")
    if env:
        return Path(env).expanduser()
    here = Path(__file__).resolve()
    # <repo>/common/paths.py -> project root is two levels up
    if (here.parents[2] / "ProjectProposal.pdf").exists():
        return here.parents[2]
    candidates = [
        "/content/drive/MyDrive/GML_project",
        "G:/My Drive/GML_project",
        os.path.expanduser("~/Google Drive/My Drive/GML_project"),
        *glob.glob(os.path.expanduser("~/Library/CloudStorage/GoogleDrive-*/My Drive/GML_project")),
    ]
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    raise FileNotFoundError("Could not find GML_project. Set env var GML_ROOT to its path.")


ROOT = _find_root()
FS = ROOT / "final_stage"

# ---- frozen inputs (read-only, never edit) ---------------------------------
MANIFEST = ROOT / "Adam Code" / "pcl5_manifest_33.csv"          # 869 rows, label = PCL-5 >= 33
DEMOGRAPHICS = ROOT / "pcl-5_after_dropout_age_sex.csv"          # same row order as MANIFEST (verified)
MATS_LOG1P = ROOT / "Adam Code" / "mats869.npy"                  # [869,432,432] float32 = log1p(raw), diag = 0
LEGEND = ROOT / "matrix_legend.json"
EDGE_DIST = ROOT / "edge_distance_matrix.npy"
YEO7 = ROOT / "Yasha Code" / "yeo_7_onehot.npy"
OLD_C_IN = ROOT / "Yasha Code" / "Claude_Debunk_1208-1800" / "results" / "C_in.npy"
OLD_C_OOF = ROOT / "Yasha Code" / "Claude_Debunk_1208-1800" / "results" / "C_oof.npy"

# ---- shared, frozen split file ----------------------------------------------
SPLITS = FS / "splits" / "cv_splits.csv"

# ---- outputs ------------------------------------------------------------------
RESULTS = FS / "results"
PREDS = RESULTS / "preds"          # <model>.csv, standard format (see common/metrics.py)
CONTRIB = RESULTS / "contrib"      # <model>_node_contrib_oof.npy  [n_folds_total, 432]
MODELS = RESULTS / "models"        # saved fold weights (GAT/GCN), small files only
SYNTH = RESULTS / "synthetic"
PERM = RESULTS / "perm"
EXPLAIN = RESULTS / "explain"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
LOGS = RESULTS / "logs"

# ---- local cache (NOT on Drive) ------------------------------------------------
CACHE = Path(os.environ.get("GML_CACHE", os.path.expanduser("~/.gml_cache")))

for _p in (PREDS, CONTRIB, MODELS, SYNTH, PERM, EXPLAIN, TABLES, FIGURES, LOGS, CACHE):
    _p.mkdir(parents=True, exist_ok=True)
