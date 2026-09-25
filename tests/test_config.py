"""Phase 0: config loading + unit conversion (GUIDE.md §4.2, Phase 0 DoD)."""

import math

import pytest

from fsl import constants as C
from fsl.config import load_material

MATERIALS = ["glass", "sic", "gan"]


@pytest.mark.parametrize("name", MATERIALS)
def test_material_loads(name):
    m = load_material(name)
    assert m.name == name
    assert m.domain_fdm.r_min < 0 < m.domain_fdm.r_max
    assert m.domain_pinn.z_max > 0 and m.domain_pinn.t_max > 0


def test_glass_si_values():
    g = load_material("glass")
    assert g.lambda_ == pytest.approx(780e-9)
    assert g.tp == pytest.approx(200e-15)
    assert g.r0 == pytest.approx(5e-6)
    assert g.F == pytest.approx(3.6e4)            # 3.6 J/cm² = 3.6e4 J/m²
    assert g.U1 == pytest.approx(4.0 * C.eV)
    assert g.alpha_i == pytest.approx(1.2e-4)     # cm²/J → m²/J
    assert g.delta_N == pytest.approx(7.0e17)     # kept in paper units (rule 4)
    assert g.N == 3
    assert g.tc == 0.0


def test_critical_density_dod():
    """DoD: glass n_cr = 1.83e27 m⁻³ (= 1.83e21 cm⁻³); GaN/SiC ≈ 1.05e21 cm⁻³."""
    assert load_material("glass").n_cr == pytest.approx(1.83e27, rel=1e-2)
    assert load_material("sic").n_cr_cm3 == pytest.approx(1.05e21, rel=1e-2)
    assert load_material("gan").n_cr_cm3 == pytest.approx(1.05e21, rel=1e-2)


def test_photon_energy_supports_N3():
    """GUIDE §2.4: N=3 inferred from U1 / photon energy (glass 4.0/1.59, SiC 3.26/1.20)."""
    g = load_material("glass")
    assert g.photon_energy / C.eV == pytest.approx(1.59, rel=1e-2)
    assert math.ceil(g.U1 / g.photon_energy) == 3
    s = load_material("sic")
    assert s.photon_energy / C.eV == pytest.approx(1.20, rel=1e-2)
    assert math.ceil(s.U1 / s.photon_energy) == 3


def test_tc_options_and_overrides():
    g = load_material("glass", tc_fs="tp")
    assert g.tc == pytest.approx(g.tp)
    g2 = load_material("glass", tc_fs="tp/2", tau_fs=100, alpha_i_cm2J=0)
    assert g2.tc == pytest.approx(0.5 * g2.tp)
    assert g2.tau == pytest.approx(100e-15)
    assert g2.alpha_i == 0.0
    with pytest.raises(KeyError):
        load_material("glass", not_a_key=1)
