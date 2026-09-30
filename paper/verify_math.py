"""Data-free checks for the revised W-GNAN paper. Run: python verify_math.py"""
from __future__ import annotations
import numpy as np


def implemented_transform(raw: np.ndarray) -> np.ndarray:
    a = raw.astype(float).copy()
    np.fill_diagonal(a, 0.0)
    a /= a.sum() + 1e-12
    out = np.zeros_like(a)
    present = a > 0
    out[present] = np.log10(a[present] + 1e-8)
    out[present] -= out[present].min()
    return out


def check_preprocessing() -> None:
    raw = np.array([[0.0, 2.0, 8.0], [2.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    got = implemented_transform(raw)
    a = raw / (raw.sum() + 1e-12)
    expected = np.zeros_like(raw)
    present = raw > 0
    z = np.log10(a[present] + 1e-8)
    expected[present] = z - z.min()
    np.testing.assert_allclose(got, expected)
    assert raw[0, 1] > 0 and got[0, 1] == 0  # weakest present edge
    assert raw[1, 2] == 0 and got[1, 2] == 0  # absent edge
    assert not (got > 0)[0, 1]               # implemented interaction mask
    q99 = np.percentile(got[got > 0], 99)
    w = np.clip(got / max(q99, 1e-6), 0, 1)
    assert np.isfinite(w).all() and w.min() >= 0 and w.max() <= 1


def check_decomposition() -> None:
    w = np.array([[0.0, .2, .8, 0.0], [.2, 0.0, .5, 0.0],
                  [.8, .5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    mask = w > 0
    assert np.allclose(w, w.T) and not np.diag(mask).any()
    u = np.array([.4, -.7, 1.2, .3])
    lam, bias, gate = .9, -.15, w**2
    n = np.maximum(1, mask.sum(axis=1))
    neigh = (gate @ u) / n
    direct = bias + np.sum(u + lam * neigh)
    pair_sum = sum(gate[i, j] * (u[j] / n[i] + u[i] / n[j])
                   for i in range(len(u)) for j in range(i + 1, len(u)))
    decomposed = bias + u.sum() + lam * pair_sum
    np.testing.assert_allclose(direct, decomposed, rtol=0, atol=1e-12)
    assert neigh[3] == 0  # isolate policy


if __name__ == "__main__":
    check_preprocessing()
    check_decomposition()
    print("PASS: preprocessing boundary/domain and fixed-graph logit decomposition")
