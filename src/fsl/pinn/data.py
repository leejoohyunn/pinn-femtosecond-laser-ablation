"""Geometry and collocation points (GUIDE.md §4.7; decisions D18, D21).

The nondimensional box (r̃, z̃, t̃) ∈ [0,1] × [0,1] × [0, t̃_max] is treated as a plain
3-D geometry: no ``GeometryXTime`` and no BC/IC objects, because IC and BC are hard
constraints (net.py). Points are sampled once at construction and kept fixed for the
whole training (D18: "pseudo" = uniform random; no resampling is our assumption).

Surface collocation (D21, I-34, 2026-10-05): the reflectivity R(t, r) is decided by n_e on the
plane z̃ = 0, which has zero measure in a 3-D sample — the first full run put essentially no
residual weight there, let the surface n_e stay below n_cr (even negative) so that R ≈ 0, and
grew n_e to 1e22 cm⁻³ in a thin layer just below the surface. ``num_surface`` extra points
(r̃, 0, t̃) are added as DeepXDE ``anchors``: the same PDE residuals are enforced on them.
"""

from __future__ import annotations

import deepxde as dde
import numpy as np

from fsl.scaling import Scales


def make_geometry(sc: Scales) -> dde.geometry.Cuboid:
    lo, hi = sc.box
    return dde.geometry.Cuboid(lo, hi)


def surface_points(sc: Scales, num_surface: int, seed: int = 0) -> np.ndarray:
    """``num_surface`` uniform random points on the surface plane z̃ = 0, shape (N, 3)."""
    rng = np.random.default_rng(seed)
    r_t = rng.random(num_surface)
    t_t = rng.random(num_surface) * sc.t_tilde_max
    return np.column_stack([r_t, np.zeros(num_surface), t_t])


def make_data(pde, sc: Scales, num_domain: int, distribution: str = "pseudo",
              num_surface: int = 0, seed: int = 0) -> dde.data.PDE:
    anchors = surface_points(sc, num_surface, seed) if num_surface > 0 else None
    return dde.data.PDE(make_geometry(sc), pde, [], num_domain=num_domain, num_boundary=0,
                        train_distribution=distribution, anchors=anchors)
