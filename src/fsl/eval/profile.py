"""Ablation profile from the n_e = n_cr contour (GUIDE.md §2.6, notes/decisions.md I-18).

Width  = total length of the r interval(s) where n_e ≥ n_cr at z = 0.
Depth  = largest z where n_e ≥ n_cr at r = 0.
Crossings are linearly interpolated between grid points. Shared by FDM and PINN.
All inputs SI, outputs SI (None when the threshold is never reached).
"""

from __future__ import annotations

import numpy as np


def _edges(x: np.ndarray, f: np.ndarray) -> list[float]:
    """Sorted boundaries of {x : f ≥ 0}: interpolated zero crossings, plus domain ends when
    the region touches them. Uses the ≥0 / <0 transition (not sign products), so a grid
    point with f exactly 0 is handled."""
    above = f >= 0
    xs: list[float] = [float(x[0])] if above[0] else []
    for i in np.where(above[:-1] != above[1:])[0]:
        xs.append(float(x[i] - f[i] * (x[i + 1] - x[i]) / (f[i + 1] - f[i])))
    if above[-1]:
        xs.append(float(x[-1]))
    return xs


def ablation_width(r: np.ndarray, ne_z0: np.ndarray, n_cr: float) -> float | None:
    """Measure of {r : n_e(r, z=0) ≥ n_cr}."""
    f = np.asarray(ne_z0, dtype=float) - n_cr
    if not (f >= 0).any():
        return None
    xs = _edges(r, f)
    return float(sum(b - a for a, b in zip(xs[0::2], xs[1::2])))


def ablation_depth(z: np.ndarray, ne_r0: np.ndarray, n_cr: float) -> float | None:
    """Largest z with n_e(r=0, z) ≥ n_cr."""
    f = np.asarray(ne_r0, dtype=float) - n_cr
    above = f >= 0
    if not above.any():
        return None
    last = int(np.where(above)[0][-1])
    if last == len(z) - 1:
        return float(z[-1])  # ablated through the whole z domain
    return float(z[last] - f[last] * (z[last + 1] - z[last]) / (f[last + 1] - f[last]))


def ablation_profile(r: np.ndarray, z: np.ndarray, ne_rz: np.ndarray, n_cr: float):
    """(width, depth) from an n_e(r, z) field. ne_rz shape (nr, nz); z[0] must be 0."""
    ir0 = int(np.argmin(np.abs(r)))
    return ablation_width(r, ne_rz[:, 0], n_cr), ablation_depth(z, ne_rz[ir0, :], n_cr)
