"""Physics functions of GUIDE.md §2 — the single implementation shared by FDM and PINN.

Rules (GUIDE §4.2, §4.4, notes/decisions.md I-4, I-5):
- All inputs and outputs are SI (m, s, K, J, W/m², m⁻³) — except δ_N, see ``photoionization_rate``.
- Every function accepts numpy arrays *or* torch tensors and returns the same kind.
  Only ``sqrt``/``exp``/``maximum`` differ between the two libraries, so each function
  picks the module once via ``_xp`` and otherwise uses plain arithmetic.
- τ is a constant taken from ``mat.tau`` (D1). Pass ``tau=`` explicitly to override it
  (Phase 2 scan, or a future τ(n_e, T_e) model).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from fsl import constants as C

# ----------------------------------------------------------------------------- helpers


def _xp(*args):
    """Return ``torch`` if any argument is a tensor, else ``numpy``."""
    return torch if any(isinstance(a, torch.Tensor) for a in args) else np


def _relu(x):
    """max(x, 0) — guards the sqrt in ``nk_from_eps`` against −1e-17 round-off."""
    xp = _xp(x)
    return xp.maximum(x, xp.zeros_like(x))


# ----------------------------------------------------------------------------- constants of the model


def omega(lambda_m: float) -> float:
    """Angular frequency ω = 2πc/λ  [rad/s]."""
    return 2.0 * math.pi * C.c / lambda_m


def critical_density(lambda_m: float) -> float:
    """n_cr = 4π² c² m_e ε₀ / (λ² e²)  [m⁻³]  (GUIDE §2.3)."""
    return 4.0 * math.pi**2 * C.c**2 * C.m_e * C.eps0 / (lambda_m**2 * C.e**2)


def c_e() -> float:
    """Electron heat capacity per electron, c_e = 3/2 k_B  [J/K]  (GUIDE §2.3)."""
    return 1.5 * C.k_B


# ----------------------------------------------------------------------------- Drude chain (GUIDE §2.3)


def plasma_freq_sq(n_e):
    """ω_p² = n_e e² / (m_e ε₀)  [rad²/s²]."""
    return n_e * C.e**2 / (C.m_e * C.eps0)


def drude_eps(n_e, omega_: float, tau):
    """Drude permittivity (ε_r, ε_i) for electron density n_e and relaxation time τ."""
    wp2 = plasma_freq_sq(n_e)
    wt2 = (omega_ * tau) ** 2
    eps_r = 1.0 - wp2 * tau**2 / (1.0 + wt2)
    eps_i = wp2 * tau / (omega_ * (1.0 + wt2))
    return eps_r, eps_i


def nk_from_eps(eps_r, eps_i):
    """Refractive index n and extinction k from ε = ε_r + i ε_i."""
    xp = _xp(eps_r, eps_i)
    abs_eps = xp.sqrt(eps_r**2 + eps_i**2)
    n = xp.sqrt(_relu((eps_r + abs_eps) / 2.0))
    k = xp.sqrt(_relu((-eps_r + abs_eps) / 2.0))
    return n, k


def reflectivity(n, k):
    """Normal-incidence Fresnel reflectivity R = ((n−1)² + k²) / ((n+1)² + k²)."""
    return ((n - 1.0) ** 2 + k**2) / ((n + 1.0) ** 2 + k**2)


def alpha_h(k, omega_: float):
    """Free-carrier (heating) absorption coefficient α_h = 2ωk/c  [1/m]."""
    return 2.0 * omega_ * k / C.c


def alpha_total(alpha_h_, n_e, alpha_i: float, U1: float):
    """Total absorption α = α_h + α_i n_e U₁  [1/m]  (used in the φ = ∫α dz integral)."""
    return alpha_h_ + alpha_i * n_e * U1


def surface_optics(n_e_surf, mat, tau=None):
    """Whole Drude chain at the surface: n_e(z=0) → (R, α_h).

    ``tau`` defaults to the constant ``mat.tau`` (D1). Pass an array to try a
    density/temperature-dependent τ without touching this function.
    """
    tau = mat.tau if tau is None else tau
    w = omega(mat.lambda_)
    eps_r, eps_i = drude_eps(n_e_surf, w, tau)
    n, k = nk_from_eps(eps_r, eps_i)
    return reflectivity(n, k), alpha_h(k, w)


# ----------------------------------------------------------------------------- ionization (GUIDE §2.1)

# δ_N is stored in the paper's units cm⁻³ ps⁻¹ (cm²/TW)^N (GUIDE §4.2 rule 4).
# These two factors are the *only* place that unit is converted.
_I_SI_TO_TW_CM2 = 1e-16   # W/m² → W/cm² (×1e-4) → TW/cm² (×1e-12)
_RATE_CM3PS_TO_SI = 1e18  # cm⁻³ → m⁻³ (×1e6), ps⁻¹ → s⁻¹ (×1e12)


def photoionization_rate(I, mat):
    """Multiphoton term δ_N I^N of Eq. (2.1), returned in m⁻³ s⁻¹ for I in W/m²."""
    I_tw = I * _I_SI_TO_TW_CM2
    return mat.delta_N * I_tw**mat.N * _RATE_CM3PS_TO_SI


def impact_rate(I, n_e, mat):
    """Impact (avalanche) term α_i I n_e of Eq. (2.1)  [m⁻³ s⁻¹]."""
    return mat.alpha_i * I * n_e


# ----------------------------------------------------------------------------- laser intensity (GUIDE §2.2)


def peak_intensity(mat) -> float:
    """I₀ = 2F / (sqrt(π/ln2) t_p)  [W/m²] — prefactor of Eq. (2.5)."""
    return 2.0 * mat.F / (math.sqrt(math.pi / math.log(2.0)) * mat.tp)


def intensity(t, r, phi, R, mat):
    """Eq. (2.5): I = I₀ (1−R) exp(−r²/r₀² − 4 ln2 ((t−t_c)/t_p)² − φ)  [W/m²].

    ``R`` is the surface reflectivity at the same (t, r); ``phi`` the optical depth ∫₀ᶻ α dz'.
    """
    xp = _xp(t, r, phi, R)
    arg = -(r / mat.r0) ** 2 - 4.0 * math.log(2.0) * ((t - mat.tc) / mat.tp) ** 2 - phi
    return peak_intensity(mat) * (1.0 - R) * xp.exp(arg)
