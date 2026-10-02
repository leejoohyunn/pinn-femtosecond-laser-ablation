"""YAML loading + unit conversion → SI dataclasses (GUIDE.md §4.2, §4.4).

This is the *only* file that converts human units (nm, fs, µm, J/cm², eV, cm²/J)
to SI. Keys in the YAML carry their unit as a suffix (``lambda_nm``, ``tp_fs``, ...).

Exception (rule 4): δ_N is stored *as written* in the paper's mixed units
(cm⁻³ ps⁻¹ (cm²/TW)^N). Its conversion lives in ``physics.photoionization_rate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from fsl import constants as C
from fsl.physics import critical_density

# --- unit factors (the only place they are allowed to appear) -------------------
NM = 1e-9      # nm  → m
UM = 1e-6      # µm  → m
FS = 1e-15     # fs  → s
J_CM2 = 1e4    # J/cm² → J/m²
CM2_J = 1e-4   # cm²/J → m²/J
CM3 = 1e6      # cm⁻³ → m⁻³
M3_TO_CM3 = 1e-6  # m⁻³ → cm⁻³ (for printing only)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _f(x: Any) -> float:
    """YAML 1.1 (PyYAML) reads ``1.0e21`` as a *string* (it wants ``1.0e+21``); cast explicitly."""
    return float(x)
MATERIALS_DIR = REPO_ROOT / "configs" / "materials"


@dataclass(frozen=True)
class Domain:
    """Computational box in SI (GUIDE §2.5.1). z always starts at 0 (φ integral)."""

    r_min: float  # [m]
    r_max: float  # [m]
    z_max: float  # [m]
    t_max: float  # [s]

    @classmethod
    def from_yaml(cls, d: dict[str, Any]) -> "Domain":
        return cls(
            r_min=_f(d["r_min_um"]) * UM,
            r_max=_f(d["r_max_um"]) * UM,
            z_max=_f(d["z_max_um"]) * UM,
            t_max=_f(d["t_max_fs"]) * FS,
        )


@dataclass(frozen=True)
class Grid:
    """FDM grid spacing in SI (paper: Δr 0.25 µm, Δz 0.02 µm, Δt 1 fs; GaN Δr per D19)."""

    dr: float  # [m]
    dz: float  # [m]
    dt: float  # [s]

    @classmethod
    def from_yaml(cls, d: dict[str, Any]) -> "Grid":
        return cls(dr=_f(d["dr_um"]) * UM, dz=_f(d["dz_um"]) * UM, dt=_f(d["dt_fs"]) * FS)


@dataclass(frozen=True)
class Scaling:
    """Nondimensionalization references (GUIDE §4.3, D7/D15). Shared by all materials."""

    n_ref: float  # [m⁻³]
    T_ref: float  # [K]
    t_ref: float  # [s]

    @classmethod
    def from_yaml(cls, d: dict[str, Any]) -> "Scaling":
        return cls(n_ref=_f(d["n_ref_cm3"]) * CM3, T_ref=_f(d["T_ref_K"]), t_ref=_f(d["t_ref_fs"]) * FS)


@dataclass(frozen=True)
class MaterialParams:
    """All material / laser parameters of GUIDE §2.4 in SI, plus domains and scaling."""

    name: str
    lambda_: float   # wavelength           [m]
    tp: float        # pulse duration (FWHM) [s]
    r0: float        # beam radius          [m]
    F: float         # fluence (Eq. 2.5 parameter, see §2.2 factor-of-2 caution) [J/m²]
    U1: float        # band gap             [J]
    alpha_i: float   # impact-ionization coefficient [m²/J]
    delta_N: float   # photoionization coefficient, paper units cm⁻³ps⁻¹(cm²/TW)^N (rule 4)
    N: int           # photon number of the multiphoton process
    tau: float       # Drude relaxation time [s] (D1; constant in this implementation)
    tc: float        # pulse center [s] (D3)
    domain_fdm: Domain
    domain_pinn: Domain
    grid: Grid
    scaling: Scaling
    notes: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)  # YAML as loaded (+overrides), for run records

    # --- derived -------------------------------------------------------------
    @property
    def n_cr(self) -> float:
        """Critical density [m⁻³]."""
        return critical_density(self.lambda_)

    @property
    def n_cr_cm3(self) -> float:
        return self.n_cr * M3_TO_CM3

    @property
    def photon_energy(self) -> float:
        """ħω = hc/λ [J]."""
        return C.h * C.c / self.lambda_

    def __repr__(self) -> str:  # human-readable, paper units
        return (
            f"MaterialParams({self.name}: λ={self.lambda_ / NM:.0f} nm, t_p={self.tp / FS:.0f} fs, "
            f"r0={self.r0 / UM:g} µm, F={self.F / J_CM2:g} J/cm², U1={self.U1 / C.eV:g} eV, "
            f"α_i={self.alpha_i / CM2_J:g} cm²/J, δ_{self.N}={self.delta_N:.3g}, "
            f"τ={self.tau / FS:g} fs, t_c={self.tc / FS:g} fs, "
            f"n_cr={self.n_cr:.3e} m⁻³ = {self.n_cr_cm3:.3e} cm⁻³)"
        )


def _parse_tc(value: Any, tp_s: float) -> float:
    """t_c may be a number in fs, or one of the D3 options 'tp' / 'tp/2'."""
    if isinstance(value, str):
        v = value.strip().lower().replace(" ", "")
        if v == "tp":
            return tp_s
        if v == "tp/2":
            return 0.5 * tp_s
        raise ValueError(f"tc_fs must be a number, 'tp' or 'tp/2'; got {value!r}")
    return float(value) * FS


def material_from_dict(d: dict[str, Any]) -> MaterialParams:
    tp = _f(d["tp_fs"]) * FS
    return MaterialParams(
        name=d["name"],
        lambda_=_f(d["lambda_nm"]) * NM,
        tp=tp,
        r0=_f(d["r0_um"]) * UM,
        F=_f(d["F_Jcm2"]) * J_CM2,
        U1=_f(d["U1_eV"]) * C.eV,
        alpha_i=_f(d["alpha_i_cm2J"]) * CM2_J,
        delta_N=_f(d["delta_N_cm3ps_cm2TW"]),
        N=int(d["N"]),
        tau=_f(d["tau_fs"]) * FS,
        tc=_parse_tc(d.get("tc_fs", 0), tp),
        domain_fdm=Domain.from_yaml(d["domain"]["fdm"]),
        domain_pinn=Domain.from_yaml(d["domain"]["pinn"]),
        grid=Grid.from_yaml(d["grid"]),
        scaling=Scaling.from_yaml(d["scaling"]),
        notes=dict(d.get("notes", {})),
        raw=dict(d),
    )


def load_material(name_or_path: str | Path, **overrides: Any) -> MaterialParams:
    """Load ``configs/materials/<name>.yaml`` (or an explicit path).

    ``overrides`` are applied to the raw YAML keys before conversion, e.g.
    ``load_material("glass", tau_fs=100, alpha_i_cm2J=0)`` for the Phase 2 scans.
    """
    p = Path(name_or_path)
    if p.suffix != ".yaml":
        p = MATERIALS_DIR / f"{name_or_path}.yaml"
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    unknown = set(overrides) - set(raw)
    if unknown:
        raise KeyError(f"unknown override keys {sorted(unknown)} for {p.name}")
    raw.update(overrides)
    return material_from_dict(raw)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Plain YAML loader for run configs (fdm.yaml, forward_glass.yaml, ...)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
