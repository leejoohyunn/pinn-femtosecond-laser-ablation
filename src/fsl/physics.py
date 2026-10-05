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
    """max(x, 0). No longer used by ``nk_from_eps`` (complex sqrt, I-27); kept for callers."""
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


_WP2_PER_NE = C.e**2 / (C.m_e * C.eps0)   # 3.18e3 rad² s⁻² m³, folded in float64 (I-32)


def plasma_freq_sq(n_e):
    """ω_p² = n_e e² / (m_e ε₀)  [rad²/s²].

    The constant is folded first: evaluated left to right on a float32 tensor, ``n_e * e**2``
    would be divided by ``m_e * eps0`` = 8e-42, a float32 denormal (I-32)."""
    return n_e * _WP2_PER_NE


def drude_eps(n_e, omega_: float, tau):
    """Drude permittivity (ε_r, ε_i) for electron density n_e and relaxation time τ."""
    wp2 = plasma_freq_sq(n_e)
    wt2 = (omega_ * tau) ** 2
    eps_r = 1.0 - wp2 * tau**2 / (1.0 + wt2)
    eps_i = wp2 * tau / (omega_ * (1.0 + wt2))
    return eps_r, eps_i


def nk_from_eps(eps_r, eps_i):
    """Refractive index n and extinction k from ε = ε_r + i ε_i (paper Eq. 2.7).

    Computed as the principal complex square root, n + ik = sqrt(ε_r + i|ε_i|), which is
    algebraically identical to Eq. (2.7) but has no singular gradient: the real-valued
    form k = sqrt((−ε_r + |ε|)/2) suffers catastrophic cancellation for n_e → 0 and its
    sqrt has an infinite derivative at 0, which produced NaN in PINN training (I-27).
    |ε_i| keeps k ≥ 0 (α_h ≥ 0) even where a network transiently predicts n_e < 0; for
    the FDM (n_e ≥ 0) it changes nothing.
    """
    xp = _xp(eps_r, eps_i)
    f = xp.sqrt(eps_r + 1j * xp.abs(eps_i))
    return f.real, f.imag


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


def photoionization_rate(I, mat, scale: float = 1.0):
    """Multiphoton term δ_N I^N of Eq. (2.1), returned in m⁻³ s⁻¹ for I in W/m², times ``scale``.

    ``scale`` is folded into the unit constant *before* it touches the tensor (I-32): the SI
    rate itself is ~4e40 m⁻³ s⁻¹ at the glass peak, beyond float32 (3.4e38). The PINN passes
    ``scale = t_ref / n_ref`` and gets the nondimensional source (≈ 4) directly; FDM (float64)
    uses the default."""
    I_tw = I * _I_SI_TO_TW_CM2
    return mat.delta_N * I_tw**mat.N * (_RATE_CM3PS_TO_SI * scale)


def impact_rate(I, n_e, mat, scale: float = 1.0):
    """Impact (avalanche) term α_i I n_e of Eq. (2.1)  [m⁻³ s⁻¹], times ``scale`` (I-32: the
    SI value is ~5e40 at the glass peak; the PINN passes scale = t_ref / n_ref).

    Order matters in float32: ``alpha_i * scale`` = 1e-44 is a denormal (3 bits of precision,
    5 % error), ``alpha_i * I`` ≈ 5e13 and ``n_e * scale`` ≈ 1e-13 are both safe."""
    return (mat.alpha_i * I) * (n_e * scale)


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
