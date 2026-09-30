"""Run first, from the repository root:   python -m common.smoke_test
Builds the local caches (~2-4 min once) and checks every frozen input."""
import time

import numpy as np

from . import paths
from .data import load_meta, load_mats, node_features, networks, edge_vectors, PREREG_NETWORKS
from .splits import iter_splits

t0 = time.time()
print("ROOT  :", paths.ROOT)
print("CACHE :", paths.CACHE)
meta = load_meta()
print(f"meta  : {len(meta)} subjects, {meta.y.sum()} positives, age {meta.age.mean():.1f}, male {meta.sex.mean():.2f}")
M = load_mats("norm")
print("mats  :", M.shape, M.dtype, "subject0 nonzero min/max", float(M[0][M[0] > 0].min()), float(M[0].max()))
assert np.allclose(M[0], M[0].T) and M[0].diagonal().max() == 0
nf = node_features()
print("nodes :", {k: v.shape for k, v in nf.items()})
E = edge_vectors("norm")
print("edges :", E.shape)
nets = networks()
print("nets  :", {n: int((nets == n).sum()) for n in PREREG_NETWORKS})
spl = list(iter_splits())
assert len(spl) == 15
for r, f, tr, va, te in spl:
    assert len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0 and len(set(tr) & set(va)) == 0
print("splits: 15 folds OK; test positives per fold:", sorted({int(meta.y.values[te].sum()) for *_, te in spl}))
print(f"SMOKE TEST PASSED in {time.time()-t0:.0f}s")
