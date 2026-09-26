"""Phase 1: physics.py against the paper / GUIDE hand calculations (GUIDE.md Phase 1 test list)."""

import math

import numpy as np
import pytest
import torch

from fsl import constants as C
from fsl import physics as P
from fsl.config import load_material

torch.set_default_dtype(torch.float64)  # compare numpy and torch at the same precision


@pytest.fixture(scope="module")
def glass():
    return load_material("glass")


# ---------------------------------------------------------------- critical density


def test_critical_density():
    assert P.critical_density(780e-9) * 1e-6 == pytest.approx(1.83e21, rel=1e-2)
    assert P.critical_density(1030e-9) * 1e-6 == pytest.approx(1.05e21, rel=1e-2)


# ---------------------------------------------------------------- Drude chain limits


def test_zero_density_is_vacuum(glass):
    w = P.omega(glass.lambda_)
    eps_r, eps_i = P.drude_eps(0.0, w, glass.tau)
    assert (eps_r, eps_i) == (1.0, 0.0)
    n, k = P.nk_from_eps(eps_r, eps_i)
    assert (n, k) == (1.0, 0.0)
    assert P.reflectivity(n, k) == 0.0
    assert P.alpha_h(k, w) == 0.0
    R, ah = P.surface_optics(np.array([0.0]), glass)
    assert R[0] == 0.0 and ah[0] == 0.0


def test_infinite_density_is_mirror(glass):
    R, _ = P.surface_optics(np.array([1e40]), glass)
    assert R[0] == pytest.approx(1.0, abs=1e-6)


def test_nk_sqrt_guard_round_off():
    # (−ε_r + |ε|)/2 can round to −1e-17 for ε_i = 0, ε_r > 0; must give k = 0, not nan
    n, k = P.nk_from_eps(np.array([1.0 + 1e-15]), np.array([0.0]))
    assert np.isfinite(k).all() and k[0] == 0.0


# ---------------------------------------------------------------- D1 hand-calculation table (GUIDE §3)


@pytest.mark.parametrize(
    "tau_fs, R_ncr, R_113, ah_113",
    [
        (1, 0.15, 0.20, 6.9e6),
        (5, 0.44, 0.64, 6.0e6),
        (10, 0.56, 0.80, 5.9e6),
        (20, 0.66, 0.89, 5.8e6),
        (50, 0.77, 0.96, 5.8e6),
        (100, 0.83, 0.977, 5.8e6),
    ],
)
def test_d1_table(glass, tau_fs, R_ncr, R_113, ah_113):
    n_cr = glass.n_cr
    ne = np.array([n_cr, 1.13 * n_cr])
    R, ah = P.surface_optics(ne, glass, tau=tau_fs * 1e-15)
    assert R[0] == pytest.approx(R_ncr, abs=0.01)
    assert R[1] == pytest.approx(R_113, abs=0.01)
    assert ah[1] == pytest.approx(ah_113, rel=0.03)


def test_alpha_total_impact_share(glass):
    """GUIDE D1: α_i n_e U₁ adds only ≈1.6e5 /m at 1.13 n_cr."""
    ne = 1.13 * glass.n_cr
    extra = P.alpha_total(0.0, ne, glass.alpha_i, glass.U1)
    assert extra == pytest.approx(1.6e5, rel=0.05)


# ---------------------------------------------------------------- intensity and ionization (GUIDE D6)


def test_peak_intensity(glass):
    I0 = P.peak_intensity(glass)
    assert I0 * 1e-4 == pytest.approx(1.69e13, rel=1e-2)  # W/cm²
    # at t = t_c, r = 0, φ = 0, R = 0 the intensity equals I₀
    assert P.intensity(glass.tc, 0.0, 0.0, 0.0, glass) == pytest.approx(I0)


def test_photoionization_units(glass):
    """δ₃ I₀³ = 3.39e21 cm⁻³ ps⁻¹ (GUIDE D6) = 3.39e39 m⁻³ s⁻¹."""
    I0 = P.peak_intensity(glass)
    rate = P.photoionization_rate(I0, glass)
    I0_tw = I0 * 1e-4 * 1e-12
    hand = glass.delta_N * I0_tw**3 * 1e6 * 1e12
    assert rate == pytest.approx(hand)
    assert rate == pytest.approx(3.39e39, rel=1e-2)


def test_photo_only_50fs_integral(glass):
    """GUIDE D6: photoionization alone (R=0, t_c=0) gives ≈1.4e20 cm⁻³ at 50 fs
    with ∫₀^50fs exp(−12 ln2 (t/t_p)²) dt = 42.6 fs."""
    t = np.linspace(0, 50e-15, 20001)
    I = P.intensity(t, 0.0, 0.0, 0.0, glass)
    n = np.trapezoid(P.photoionization_rate(I, glass), t)
    assert n * 1e-6 == pytest.approx(1.4e20, rel=0.05)
    eff = np.trapezoid(np.exp(-12 * math.log(2) * (t / glass.tp) ** 2), t)
    assert eff == pytest.approx(42.6e-15, rel=1e-2)


def test_impact_rate(glass):
    assert P.impact_rate(2.0, 3.0, glass) == pytest.approx(glass.alpha_i * 6.0)


def test_c_e():
    assert P.c_e() == pytest.approx(1.5 * C.k_B)


# ---------------------------------------------------------------- numpy ↔ torch parity


def test_numpy_torch_parity(glass):
    ne_np = np.logspace(24, 28, 50)  # 1e18 – 1e22 cm⁻³
    ne_t = torch.tensor(ne_np)
    R_np, ah_np = P.surface_optics(ne_np, glass)
    R_t, ah_t = P.surface_optics(ne_t, glass)
    assert isinstance(R_t, torch.Tensor)
    np.testing.assert_allclose(R_t.numpy(), R_np, rtol=1e-12)
    np.testing.assert_allclose(ah_t.numpy(), ah_np, rtol=1e-12)

    t = np.linspace(0, 200e-15, 7)
    r = np.linspace(-5e-6, 5e-6, 7)
    phi = np.linspace(0, 2, 7)
    I_np = P.intensity(t, r, phi, R_np[:7], glass)
    I_t = P.intensity(torch.tensor(t), torch.tensor(r), torch.tensor(phi), R_t[:7], glass)
    np.testing.assert_allclose(I_t.numpy(), I_np, rtol=1e-12)
    np.testing.assert_allclose(
        P.photoionization_rate(I_t, glass).numpy(), P.photoionization_rate(I_np, glass), rtol=1e-12
    )


def test_torch_gradients_flow(glass):
    """PINN needs dR/dn_e through the whole chain (Phase 3 surface query)."""
    ne = torch.tensor([1.0 * glass.n_cr], requires_grad=True)
    R, ah = P.surface_optics(ne, glass)
    (R + ah).sum().backward()
    assert torch.isfinite(ne.grad).all()
