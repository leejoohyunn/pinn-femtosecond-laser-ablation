"""Nondimensionalization for the PINN (GUIDE.md §4.3; decisions D7, D15, D16, D17).

    r̃ = (r − a)/(b − a) ∈ [0, 1]      a, b  = PINN r domain
    z̃ = z / z_max        ∈ [0, 1]      z_max = PINN z domain
    t̃ = t / t_ref                       t_ref = 100 fs → t̃ ∈ [0, t_max/t_ref]
    ñ = n_e / n_ref                     n_ref = 1e21 cm⁻³ (shared by all materials, D15)
    T̃ = (T_e − T₀) / T_ref              T_ref = 1e4 K, T₀ = 300 K

Every method is plain arithmetic, so it works on numpy arrays and torch tensors alike
(the same rule as ``physics.py``). All dimensional quantities are SI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fsl import constants as C
from fsl.config import MaterialParams


@dataclass(frozen=True)
class Scales:
    r_min: float   # a  [m]
    r_max: float   # b  [m]
    z_max: float   # [m]
    t_max: float   # [s]
    t_ref: float   # [s]
    n_ref: float   # [m⁻³]
    T_ref: float   # [K]
    T0: float = C.T_ROOM

    @classmethod
    def from_material(cls, mat: MaterialParams) -> "Scales":
        d, s = mat.domain_pinn, mat.scaling
        return cls(r_min=d.r_min, r_max=d.r_max, z_max=d.z_max, t_max=d.t_max,
                   t_ref=s.t_ref, n_ref=s.n_ref, T_ref=s.T_ref)

    # -- domain in nondimensional form -------------------------------------------
    @property
    def t_tilde_max(self) -> float:
        return self.t_max / self.t_ref

    @property
    def box(self) -> tuple[list[float], list[float]]:
        """Corners of the nondimensional training box (r̃, z̃, t̃)."""
        return [0.0, 0.0, 0.0], [1.0, 1.0, self.t_tilde_max]

    # -- inputs ---------------------------------------------------------------------
    def to_tilde(self, r, z, t):
        return (r - self.r_min) / (self.r_max - self.r_min), z / self.z_max, t / self.t_ref

    def from_tilde(self, r_t, z_t, t_t):
        return self.r_min + r_t * (self.r_max - self.r_min), z_t * self.z_max, t_t * self.t_ref

    # -- outputs --------------------------------------------------------------------
    def n_from_tilde(self, n_t):
        return n_t * self.n_ref

    def T_from_tilde(self, T_t):
        return self.T0 + T_t * self.T_ref

    def n_to_tilde(self, n_e):
        return n_e / self.n_ref

    def T_to_tilde(self, T_e):
        return (T_e - self.T0) / self.T_ref

    # -- grids (evaluation) ---------------------------------------------------------
    def grid_inputs(self, r: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        """(nr·nz, 3) array of (r̃, z̃, t̃) for an (r, z) grid at one time, r-major order
        (reshape the prediction with ``.reshape(len(r), len(z))``)."""
        rr, zz = np.meshgrid(r, z, indexing="ij")
        r_t, z_t, t_t = self.to_tilde(rr.ravel(), zz.ravel(), float(t))
        return np.column_stack([r_t, z_t, np.full(r_t.shape, t_t)])
