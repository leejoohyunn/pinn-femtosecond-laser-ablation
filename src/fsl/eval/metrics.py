"""Error metrics PINN vs FDM (GUIDE.md §4.5; paper Eq. 5.1).

All functions take arrays of the same shape (any dimensionality) in SI and return floats.
"""

from __future__ import annotations

import numpy as np


def l2re(u_nn: np.ndarray, u_ref: np.ndarray) -> float:
    """Paper Eq. (5.1): sqrt(Σ(u_NN − u_M)² / Σ u_M²) over all grid points."""
    u_nn, u_ref = np.asarray(u_nn, float), np.asarray(u_ref, float)
    den = float(np.sum(u_ref**2))
    return float(np.sqrt(np.sum((u_nn - u_ref) ** 2) / den)) if den > 0 else float("nan")


def max_rel_pointwise(u_nn: np.ndarray, u_ref: np.ndarray, floor: float = 1e-3) -> float:
    """max |u_NN − u_M| / |u_M| over points with |u_M| ≥ floor·max|u_M| (compare with the paper's %)."""
    u_nn, u_ref = np.asarray(u_nn, float), np.asarray(u_ref, float)
    mask = np.abs(u_ref) >= floor * np.max(np.abs(u_ref))
    if not mask.any():
        return float("nan")
    return float(np.max(np.abs(u_nn[mask] - u_ref[mask]) / np.abs(u_ref[mask])))


def max_rel_global(u_nn: np.ndarray, u_ref: np.ndarray) -> float:
    """max |u_NN − u_M| / max|u_M| (internal judgment)."""
    u_nn, u_ref = np.asarray(u_nn, float), np.asarray(u_ref, float)
    m = float(np.max(np.abs(u_ref)))
    return float(np.max(np.abs(u_nn - u_ref)) / m) if m > 0 else float("nan")


def all_metrics(u_nn: np.ndarray, u_ref: np.ndarray) -> dict[str, float]:
    return {"l2re": l2re(u_nn, u_ref),
            "max_rel_pointwise": max_rel_pointwise(u_nn, u_ref),
            "max_rel_global": max_rel_global(u_nn, u_ref)}
