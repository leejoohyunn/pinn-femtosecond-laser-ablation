"""Phase 2: FDM solver checks (GUIDE.md Phase 2 DoD, notes/decisions.md I-14 … I-18)."""

from dataclasses import replace

import numpy as np
import pytest

from fsl import physics as P
from fsl.config import FS, load_material
from fsl.eval.profile import ablation_depth, ablation_profile, ablation_width
from fsl.fdm import FDMConfig, Solution, make_grid, metrics, solve, time_to_reach


@pytest.fixture(scope="module")
def glass():
    return load_material("glass")


def test_grid_matches_paper(glass):
    r, z, t = make_grid(glass)
    assert len(r) == 41 and len(z) == 31 and len(t) == 201     # Δr 0.25 µm, Δz 0.02 µm, Δt 1 fs
    assert z[0] == 0.0 and r[20] == 0.0
    assert np.isclose(r[1] - r[0], 0.25e-6) and np.isclose(t[1] - t[0], 1e-15)


def test_photo_only_matches_analytic_integral(glass):
    """α_i = 0, r = z = 0: n_e(50 fs) = ∫ δ₃ (I₀(1−R) e^{-4ln2 (t/tp)²})³ dt with R ≈ 0 (GUIDE D6 ≈ 1.4e20 cm⁻³).
    Uses the paper's Table 1 pulse (t_p 200 fs, t_c 0) explicitly — the YAML default is the calibrated set (§3.3)."""
    mat = load_material("glass", alpha_i_cm2J=0, tp_fs=200, tc_fs=0)
    sol = solve(mat, FDMConfig(integrator="rk4", dt=0.1 * FS))
    i50 = sol.it(50 * FS)
    t = np.linspace(0, 50 * FS, 50001)
    I = P.intensity(t, 0.0, 0.0, 0.0, mat)
    analytic = np.trapezoid(P.photoionization_rate(I, mat), t)
    n50 = sol.n_e[i50, sol.ir0, 0]
    assert n50 == pytest.approx(analytic, rel=5e-3)  # R(1e20 cm⁻³) ~ 1e-3 is the only difference
    assert n50 * 1e-6 == pytest.approx(1.4e20, rel=0.05)
    assert sol.T_e[i50, sol.ir0, 0] > 300.0          # heating happened
    assert sol.n_e[0].max() == 0.0 and sol.T_e[0].min() == 300.0


def test_euler_vs_rk4_convergence(glass):
    """GUIDE Phase 2 DoD: Euler Δt=1 fs vs RK4 Δt=0.1 fs differ by < 1% (final n_e, T_e at r = z = 0)."""
    eu = solve(glass, FDMConfig(integrator="euler"))
    rk = solve(glass, FDMConfig(integrator="rk4", dt=0.1 * FS))
    for name in ("n_e", "T_e"):
        a, b = eu.center(name)[-1], rk.center(name)[-1]
        assert abs(a - b) / abs(b) < 1e-2, f"{name}: euler {a:.4e} vs rk4 {b:.4e}"
    assert np.allclose(eu.t, rk.t)  # same output grid regardless of the integration step


def test_points_are_independent(glass):
    """No r/z derivatives → the r = 0 column is the same on a narrower r domain (GUIDE §2.5.1)."""
    wide = solve(glass, FDMConfig())
    narrow_mat = replace(glass, domain_fdm=replace(glass.domain_fdm, r_min=-1e-6, r_max=1e-6))
    narrow = solve(narrow_mat, FDMConfig())
    np.testing.assert_allclose(narrow.n_e[:, narrow.ir0, :], wide.n_e[:, wide.ir0, :], rtol=1e-12)
    np.testing.assert_allclose(narrow.T_e[:, narrow.ir0, :], wide.T_e[:, wide.ir0, :], rtol=1e-12)


def test_phi_attenuates_with_depth(glass):
    sol = solve(glass, FDMConfig())
    I_last = sol.I[-1, sol.ir0, :]
    assert np.all(np.diff(I_last) <= 0)             # I decreases with z (φ ≥ 0, cumulative)
    assert sol.n_e[-1, sol.ir0, 0] >= sol.n_e[-1, sol.ir0, -1]


def test_shared_R_reproduces_full_run(glass):
    """Feeding a run its own R/α must reproduce it exactly (I-19 plumbing check)."""
    full = solve(glass, FDMConfig())
    again = solve(glass, FDMConfig(), shared=full)
    np.testing.assert_allclose(again.n_e, full.n_e, rtol=1e-10)
    np.testing.assert_allclose(again.R, full.R, rtol=1e-10)


def test_save_load_roundtrip(tmp_path, glass):
    sol = solve(glass, FDMConfig())
    sol.save(tmp_path / "s.npz")
    back = Solution.load(tmp_path / "s.npz")
    np.testing.assert_array_equal(back.n_e, sol.n_e)
    assert back.meta["material"] == "glass"
    assert back.it(50 * FS) == 50
    with pytest.raises(ValueError):
        back.it(50.7 * FS)


def test_metrics_keys(glass):
    sol = solve(glass, FDMConfig(report_times=(50 * FS, 200 * FS)))
    m = metrics(sol, glass, FDMConfig(report_times=(50 * FS, 200 * FS)))
    assert set(m["center_at_fs"]) == {"50", "200"}
    assert m["max_ne_cm3"] > 0 and m["max_Te_K"] >= 300


# ---------------------------------------------------------------- profile helpers


def test_time_to_reach():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    assert time_to_reach(t, np.array([0.0, 1.0, 3.0, 4.0]), 2.0) == pytest.approx(1.5)
    assert time_to_reach(t, np.array([0.0, 1.0, 1.5, 1.9]), 2.0) is None


def test_ablation_width_and_depth_interpolate():
    r = np.linspace(-4, 4, 9)                        # Δr = 1
    ne = 10 - r**2                                   # ≥ 6 for |r| ≤ 2 exactly
    assert ablation_width(r, ne, 6.0) == pytest.approx(4.0)
    assert ablation_width(r, ne, 9.5) == pytest.approx(1.0)        # crossings at ±0.5 by linear interp
    assert ablation_width(r, ne, 11.0) is None
    z = np.linspace(0, 1, 11)
    assert ablation_depth(z, 1 - z, 0.55) == pytest.approx(0.45)
    assert ablation_depth(z, 1 - z, 2.0) is None
    assert ablation_depth(z, np.ones_like(z), 0.5) == pytest.approx(1.0)  # through the whole domain
    field = (10 - r**2)[:, None] * (1 - z)[None, :]
    w, d = ablation_profile(r, z, field, 6.0)
    assert w == pytest.approx(4.0) and d == pytest.approx(0.4)
