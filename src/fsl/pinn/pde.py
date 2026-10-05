"""PDE residuals for DeepXDE (GUIDE.md §4.3, §4.7; decisions D2, D7).

``pde(x, y)`` receives the nondimensional inputs x = (r̃, z̃, t̃) and the transformed
outputs y = (ñ, T̃, φ) and returns the three residuals

    R₁ = ∂ñ/∂t̃ − t_ref (α_i I ñ + δ_N I^N / n_ref)
    R₂ = ñ ∂T̃/∂t̃ − t_ref α_h I / (c_e n_ref T_ref)
    R_φ = ∂φ/∂z̃ − z_max α

All physics is evaluated in SI by the *same* ``physics.py`` functions the FDM uses
(GUIDE §4.4); only the residuals are nondimensional.

Surface query (the one non-standard piece, §4.7): R(t, r) needs n_e at z = 0, so the
network is called a second time at (r̃, 0, t̃). The net is captured by closure; the
output transform is applied automatically. Gradients flow through this call.

``residual_terms`` exposes every intermediate quantity (for tests and scripts/diagnose_run.py);
``make_pde`` wraps it for DeepXDE.
"""

from __future__ import annotations

import deepxde as dde
import torch

from fsl import physics as P
from fsl.config import MaterialParams
from fsl.scaling import Scales


def surface_density(net, x: torch.Tensor, n_ref: float) -> torch.Tensor:
    """n_e(r, z = 0, t) in m⁻³ for each row of x = (r̃, z̃, t̃): re-evaluate the net at z̃ = 0."""
    x_s = torch.cat([x[:, 0:1], torch.zeros_like(x[:, 1:2]), x[:, 2:3]], dim=1)
    return net(x_s)[:, 0:1] * n_ref


def residual_terms(net, mat: MaterialParams, sc: Scales, x: torch.Tensor, y: torch.Tensor) -> dict:
    """All intermediate quantities of the residuals at the points x (y = net(x), transformed)."""
    a, b = sc.r_min, sc.r_max
    t_ref, n_ref, T_ref, z_max = sc.t_ref, sc.n_ref, sc.T_ref, sc.z_max
    n_t, T_t, phi = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    dn_dt = dde.grad.jacobian(y, x, i=0, j=2)
    dT_dt = dde.grad.jacobian(y, x, i=1, j=2)
    dphi_dz = dde.grad.jacobian(y, x, i=2, j=1)

    r = a + x[:, 0:1] * (b - a)          # [m]
    t = x[:, 2:3] * t_ref                # [s]
    n_e = n_t * n_ref                    # [m⁻³]

    n_surf = surface_density(net, x, n_ref)
    R_surf, _ = P.surface_optics(n_surf, mat)                 # z = 0 → R(t, r)
    _, ah = P.surface_optics(n_e, mat)                        # local α_h(t, r, z)
    I = P.intensity(t, r, phi, R_surf, mat)                   # Eq. (2.5) [W/m²]
    alpha = P.alpha_total(ah, n_e, mat.alpha_i, mat.U1)       # Eq. (2.11) [1/m]

    # I-32: fold t_ref / n_ref into the rate constants — the SI rates (~4e40 m⁻³ s⁻¹) overflow
    # float32, the nondimensional sources (~1) do not. Mathematically identical to
    # t_ref * rate / n_ref, which is what the float64 tests check against.
    src_impact = P.impact_rate(I, n_e, mat, scale=t_ref / n_ref)
    src_photo = P.photoionization_rate(I, mat, scale=t_ref / n_ref)
    heat = ah * I * (t_ref / (P.c_e() * n_ref * T_ref))
    return {
        "n_t": n_t, "T_t": T_t, "phi": phi, "dn_dt": dn_dt, "dT_dt": dT_dt, "dphi_dz": dphi_dz,
        "n_surf_t": n_surf / n_ref, "R_surf": R_surf, "I_over_I0": I / P.peak_intensity(mat),
        "alpha_h": ah, "alpha": alpha, "zmax_alpha": z_max * alpha,
        "src_impact": src_impact, "src_photo": src_photo, "heat": heat,
        "R1": dn_dt - src_impact - src_photo,
        "R2": n_t * dT_dt - heat,
        "Rphi": dphi_dz - z_max * alpha,
    }


def make_pde(net, mat: MaterialParams, sc: Scales):
    """Build ``pde(x, y) -> [R1, R2, Rphi]`` for DeepXDE with ``net`` captured by closure."""

    def pde(x, y):
        q = residual_terms(net, mat, sc, x, y)
        return [q["R1"], q["R2"], q["Rphi"]]

    return pde


LOSS_NAMES = ("loss_ne", "loss_Te", "loss_phi")
