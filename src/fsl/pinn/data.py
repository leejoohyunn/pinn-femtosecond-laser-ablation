"""Geometry and collocation points (GUIDE.md §4.7; decision D18).

The nondimensional box (r̃, z̃, t̃) ∈ [0,1] × [0,1] × [0, t̃_max] is treated as a plain
3-D geometry: no ``GeometryXTime`` and no BC/IC objects, because IC and BC are hard
constraints (net.py). Points are sampled once at construction and kept fixed for the
whole training (D18: "pseudo" = uniform random; no resampling is our assumption).
"""

from __future__ import annotations

import deepxde as dde

from fsl.scaling import Scales


def make_geometry(sc: Scales) -> dde.geometry.Cuboid:
    lo, hi = sc.box
    return dde.geometry.Cuboid(lo, hi)


def make_data(pde, sc: Scales, num_domain: int, distribution: str = "pseudo") -> dde.data.PDE:
    return dde.data.PDE(make_geometry(sc), pde, [], num_domain=num_domain, num_boundary=0,
                        train_distribution=distribution)
