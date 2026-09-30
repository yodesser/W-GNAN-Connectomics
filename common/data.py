"""Shared data loading. Every script loads data through here.

Frozen decisions (do not re-open):
  * cohort      : 869 subjects, one scan each (manifest path in common/paths.py)
  * target      : y = 1[PCL-5 >= 33]  -> 60 positives (6.9 %)
  * preprocessing ("norm", used for ALL reported models):
        raw = expm1(mats869)            (mats869 = log1p(raw), diag already 0)
        A   = raw / raw.sum()           (removes head-size / streamline-count scale)
        L   = log10(A) on non-zeros, shifted so the weakest edge = 0; zeros stay 0
  * node features: strength_i = sum_j L_ij ; degree_i = #nonzero_ij / 431
  * networks    : Schaefer-7 name parsed from the legend; subcortical = "Subcortical"
    pre-registered networks for the interpretability test: Cont, SalVentAttn, Default

Usage:
    from common.data import load_meta, load_mats, edge_vectors, node_features, networks
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import paths

N_NODES = 432
CUTOFF = 33
PREREG_NETWORKS = ["Cont", "SalVentAttn", "Default"]
IU = np.triu_indices(N_NODES, 1)


# ----------------------------------------------------------------------------- labels
def load_meta() -> pd.DataFrame:
    """One row per subject, row order == matrix order. Columns:
    idx, subject, session, pcl5, y, age, sex (1 = male, 0 = female; 1 NaN imputed with mode)."""
    m = pd.read_csv(paths.MANIFEST)
    d = pd.read_csv(paths.DEMOGRAPHICS)
    assert len(m) == len(d) == 869, (len(m), len(d))
    assert (m["subject"].astype(str).values == d["Udi"].astype(str).values).all(), "row order mismatch"
    assert (m["pcl5"].values == d["pcl-5"].values).all(), "PCL-5 mismatch"
    out = pd.DataFrame({
        "idx": np.arange(len(m)),
        "subject": m["subject"].astype(str),
        "session": m["session"].astype(str),
        "pcl5": m["pcl5"].astype(float),
        "age": d["age"].astype(float),
        "sex": d["sex"].fillna(d["sex"].mode()[0]).astype(float),
    })
    out["y"] = (out["pcl5"] >= CUTOFF).astype(int)
    assert out["y"].sum() == 60
    return out


# ----------------------------------------------------------------------------- matrices
def _normalise_one(raw: np.ndarray) -> np.ndarray:
    A = raw.astype(np.float64, copy=True)
    np.fill_diagonal(A, 0.0)
    A /= A.sum() + 1e-12
    L = np.zeros_like(A)
    nz = A > 0
    L[nz] = np.log10(A[nz] + 1e-8)
    L[nz] -= L[nz].min()
    return L.astype(np.float32)


def load_mats(kind: str = "norm", mmap: bool = True) -> np.ndarray:
    """kind: 'norm' (default, use this), 'log1p' (old pipeline), 'raw'.
    'norm' is built once (~1-2 min) and cached locally in ~/.gml_cache."""
    src = np.load(paths.MATS_LOG1P, mmap_mode="r")
    assert src.shape == (869, N_NODES, N_NODES), src.shape
    if kind == "log1p":
        return src if mmap else np.array(src)
    if kind == "raw":
        return np.expm1(np.asarray(src, dtype=np.float32))
    if kind != "norm":
        raise ValueError(kind)
    cache = paths.CACHE / "mats869_norm.npy"
    if not cache.exists():
        print(f"[data] building normalised matrices -> {cache} (one-off)")
        out = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float32, shape=src.shape)
        for i in range(src.shape[0]):
            out[i] = _normalise_one(np.expm1(src[i].astype(np.float64)))
        out.flush()
        del out
    return np.load(cache, mmap_mode="r" if mmap else None)


def edge_vectors(kind: str = "norm") -> np.ndarray:
    """[869, 93096] upper-triangle edge features (cached locally)."""
    cache = paths.CACHE / f"edges869_{kind}.npy"
    if not cache.exists():
        M = load_mats(kind)
        E = np.empty((M.shape[0], len(IU[0])), dtype=np.float32)
        for i in range(M.shape[0]):
            E[i] = M[i][IU]
        np.save(cache, E)
    return np.load(cache)


def node_features() -> dict:
    """dict of [869, 432] arrays: strength (norm), degree, plus [869] scale covariates
    total_raw (streamline total before normalisation) and density."""
    cache = paths.CACHE / "node_feats869.npz"
    if not cache.exists():
        L = load_mats("norm")
        src = np.load(paths.MATS_LOG1P, mmap_mode="r")
        n = L.shape[0]
        strength = np.empty((n, N_NODES), np.float32)
        degree = np.empty((n, N_NODES), np.float32)
        total_raw = np.empty(n, np.float64)
        density = np.empty(n, np.float64)
        for i in range(n):
            raw = np.expm1(src[i].astype(np.float64))
            strength[i] = L[i].sum(1)
            degree[i] = (raw > 0).sum(1) / (N_NODES - 1)
            total_raw[i] = raw.sum()
            density[i] = (raw[IU] > 0).mean()
        np.savez(cache, strength=strength, degree=degree, total_raw=total_raw, density=density)
    z = np.load(cache)
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------- atlas
def region_labels() -> list[str]:
    return json.load(open(paths.LEGEND))["region_labels"]


def networks() -> np.ndarray:
    """[432] network name per node: Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default, Subcortical."""
    out = []
    for lab in region_labels():
        out.append(lab.split("_")[1] if lab.startswith(("LH_", "RH_")) else "Subcortical")
    return np.array(out)
