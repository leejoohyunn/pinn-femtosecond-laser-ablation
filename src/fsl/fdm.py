"""Reference FDM solver for paper Sec.2.2 (GUIDE.md Phase 2, notes/decisions.md I-14 … I-19).

The PDEs (2.1)(2.2) have no spatial derivatives, so every (r, z) grid point is an
independent ODE in t, coupled only through R (surface n_e at z = 0) and
φ = ∫₀ᶻ α dz' (cumulative along z). One time step is therefore:

    ① Drude chain on the whole grid → R(r) from the z = 0 row, α_h(r, z)
    ② α = α_h + α_i n_e U₁
    ③ φ = cumulative trapezoid of α along z (φ = 0 at z = 0)
    ④ I = Eq. (2.5)
    ⑤ dn_e/dt = α_i I n_e + δ_N I^N,   dT_e/dt = α_h I / (c_e n_e)   (n_e ≥ floor)

Everything is SI. Grid axes are stored in SI too; convert only for plotting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid

from fsl import constants as C
from fsl import physics as P
from fsl.config import FS, MaterialParams, load_yaml


# ----------------------------------------------------------------------------- config


@dataclass(frozen=True)
class FDMConfig:
    integrator: str = "euler"        # "euler" (paper) | "rk4" (convergence check, I-14)
    dt: float | None = None          # integration step [s]; None → material grid dt
    ne_floor: float = 1e6            # T_e is not updated below this n_e [m⁻³] (I-15)
    report_times: tuple[float, ...] = ()  # times at which metrics are reported [s]

    @classmethod
    def from_yaml(cls, path: str | Path, material: str) -> "FDMConfig":
        d = load_yaml(path)
        rt = d.get("report_times_fs", {})
        if isinstance(rt, dict):
            rt = rt.get(material, [])
        dt = d.get("dt_fs")
        return cls(
            integrator=str(d.get("integrator", "euler")),
            dt=None if dt is None else float(dt) * FS,
            ne_floor=float(d.get("ne_floor_m3", 1e6)),
            report_times=tuple(float(x) * FS for x in rt),
        )

    def replace(self, **kw: Any) -> "FDMConfig":
        return replace(self, **kw)


# ----------------------------------------------------------------------------- solution container


@dataclass
class Solution:
    """Full time history on the FDM grid (I-17). Shapes: t (nt,), r (nr,), z (nz,)."""

    r: np.ndarray
    z: np.ndarray
    t: np.ndarray
    n_e: np.ndarray      # (nt, nr, nz) [m⁻³]
    T_e: np.ndarray      # (nt, nr, nz) [K]
    R: np.ndarray        # (nt, nr)     surface reflectivity
    alpha: np.ndarray    # (nt, nr, nz) [1/m]
    I: np.ndarray        # (nt, nr, nz) [W/m²]
    meta: dict[str, Any] = field(default_factory=dict)

    # -- indexing helpers
    def it(self, t_s: float) -> int:
        """Index of the stored time equal to t_s (up to round-off); raises if t_s is off-grid."""
        i = int(np.argmin(np.abs(self.t - t_s)))
        dt = self.t[1] - self.t[0] if len(self.t) > 1 else 1.0
        if abs(self.t[i] - t_s) > 1e-6 * dt:
            raise ValueError(f"t = {t_s / FS:g} fs is not on the stored time grid")
        return i

    @property
    def ir0(self) -> int:
        return int(np.argmin(np.abs(self.r)))

    def center(self, name: str) -> np.ndarray:
        """Time series of a field at r = 0, z = 0 (Fig.3)."""
        a = getattr(self, name)
        return a[:, self.ir0, 0] if a.ndim == 3 else a[:, self.ir0]

    # -- io
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, r=self.r, z=self.z, t=self.t, n_e=self.n_e, T_e=self.T_e,
            R=self.R, alpha=self.alpha, I=self.I, meta=json.dumps(self.meta),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Solution":
        with np.load(path) as f:
            return cls(
                r=f["r"], z=f["z"], t=f["t"], n_e=f["n_e"], T_e=f["T_e"],
                R=f["R"], alpha=f["alpha"], I=f["I"], meta=json.loads(str(f["meta"])),
            )


# ----------------------------------------------------------------------------- grid


def make_grid(mat: MaterialParams, dt_out: float | None = None):
    """FDM grid of GUIDE §2.5.1 in SI. z starts at 0 (φ integral). t is the *output* grid."""
    d, g = mat.domain_fdm, mat.grid
    nr = int(round((d.r_max - d.r_min) / g.dr)) + 1
    nz = int(round(d.z_max / g.dz)) + 1
    dt_out = g.dt if dt_out is None else dt_out
    nt = int(round(d.t_max / dt_out)) + 1
    r = np.linspace(d.r_min, d.r_max, nr)
    r[np.abs(r) < 1e-9 * g.dr] = 0.0          # make the r = 0 node exact (Fig.3 uses r = z = 0)
    z = np.linspace(0.0, d.z_max, nz)
    t = np.linspace(0.0, d.t_max, nt)
    return r, z, t


# ----------------------------------------------------------------------------- solver


class _Shared:
    """R(t, r) and α(t, r, z) taken from another run (I-19, hypothesis H1), linearly interpolated in t."""

    def __init__(self, sol: Solution, r: np.ndarray, z: np.ndarray):
        if sol.R.shape[1] != len(r) or sol.alpha.shape[2] != len(z):
            raise ValueError("shared run must use the same (r, z) grid")
        self.sol = sol

    def at(self, t: float):
        ts = self.sol.t
        k = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2))
        w = (t - ts[k]) / (ts[k + 1] - ts[k])
        w = float(np.clip(w, 0.0, 1.0))
        R = (1 - w) * self.sol.R[k] + w * self.sol.R[k + 1]
        a = (1 - w) * self.sol.alpha[k] + w * self.sol.alpha[k + 1]
        return R, a


def _rhs(t: float, n: np.ndarray, mat: MaterialParams, cfg: FDMConfig,
         r_col: np.ndarray, z: np.ndarray, shared: _Shared | None):
    """Right-hand side of (2.1)(2.2) on the grid + diagnostics (R_surf, α, I)."""
    R_grid, ah = P.surface_optics(n, mat)          # Drude chain on the whole grid
    if shared is None:
        R_surf = R_grid[:, 0]
        alpha = P.alpha_total(ah, n, mat.alpha_i, mat.U1)
    else:
        R_surf, alpha = shared.at(t)
    phi = cumulative_trapezoid(alpha, z, axis=1, initial=0.0)
    I = P.intensity(t, r_col, phi, R_surf[:, None], mat)
    dn = P.photoionization_rate(I, mat) + P.impact_rate(I, n, mat)
    dT = np.where(n >= cfg.ne_floor, ah * I / (P.c_e() * np.maximum(n, cfg.ne_floor)), 0.0)
    return dn, dT, (R_surf, alpha, I)


def solve(mat: MaterialParams, cfg: FDMConfig, shared: Solution | None = None) -> Solution:
    """Time-march the whole (r, z) grid. Returns the full history on the output grid (Δt of the material)."""
    r, z, t_out = make_grid(mat)
    nr, nz, nt = len(r), len(z), len(t_out)
    dt_out = t_out[1] - t_out[0]
    dt = dt_out if cfg.dt is None else cfg.dt
    nsub = int(round(dt_out / dt))
    if nsub < 1 or abs(nsub * dt - dt_out) > 1e-6 * dt_out:
        raise ValueError(f"dt = {dt / FS:g} fs must divide the output step {dt_out / FS:g} fs")
    dt = dt_out / nsub
    if cfg.integrator not in ("euler", "rk4"):
        raise ValueError(f"unknown integrator {cfg.integrator!r}")

    r_col = r[:, None]
    sh = None if shared is None else _Shared(shared, r, z)

    n = np.zeros((nr, nz))
    T = np.full((nr, nz), C.T_ROOM)
    out = Solution(
        r=r, z=z, t=t_out,
        n_e=np.empty((nt, nr, nz)), T_e=np.empty((nt, nr, nz)),
        R=np.empty((nt, nr)), alpha=np.empty((nt, nr, nz)), I=np.empty((nt, nr, nz)),
        meta={"material": mat.name, "integrator": cfg.integrator, "dt_fs": float(dt / FS),
              "ne_floor_m3": float(cfg.ne_floor), "shared_R": shared is not None,
              "material_yaml": mat.raw},
    )

    def step(tk: float, n: np.ndarray, T: np.ndarray):
        if cfg.integrator == "euler":
            dn, dT, _ = _rhs(tk, n, mat, cfg, r_col, z, sh)
            return n + dt * dn, T + dt * dT
        # RK4 (T does not feed back, so its stages only need the n stages)
        k1n, k1T, _ = _rhs(tk, n, mat, cfg, r_col, z, sh)
        k2n, k2T, _ = _rhs(tk + dt / 2, n + dt / 2 * k1n, mat, cfg, r_col, z, sh)
        k3n, k3T, _ = _rhs(tk + dt / 2, n + dt / 2 * k2n, mat, cfg, r_col, z, sh)
        k4n, k4T, _ = _rhs(tk + dt, n + dt * k3n, mat, cfg, r_col, z, sh)
        return (n + dt / 6 * (k1n + 2 * k2n + 2 * k3n + k4n),
                T + dt / 6 * (k1T + 2 * k2T + 2 * k3T + k4T))

    for k in range(nt):
        tk = t_out[k]
        _, _, (R_surf, alpha, I) = _rhs(tk, n, mat, cfg, r_col, z, sh)   # diagnostics at t_k
        out.n_e[k], out.T_e[k], out.R[k], out.alpha[k], out.I[k] = n, T, R_surf, alpha, I
        if k == nt - 1:
            break
        for s in range(nsub):
            n, T = step(tk + s * dt, n, T)

    return out


# ----------------------------------------------------------------------------- metrics (GUIDE 2.6 table)


def time_to_reach(t: np.ndarray, y: np.ndarray, level: float) -> float | None:
    """First time y crosses `level` from below (linear interpolation); None if never."""
    above = y >= level
    if not above.any():
        return None
    k = int(np.argmax(above))
    if k == 0:
        return float(t[0])
    f = (level - y[k - 1]) / (y[k] - y[k - 1])
    return float(t[k - 1] + f * (t[k] - t[k - 1]))


def metrics(sol: Solution, mat: MaterialParams, cfg: FDMConfig) -> dict[str, Any]:
    from fsl.eval.profile import ablation_profile

    ne0, Te0, R0, a0 = sol.center("n_e"), sol.center("T_e"), sol.center("R"), sol.center("alpha")
    t_ncr = time_to_reach(sol.t, ne0, mat.n_cr)
    width, depth = ablation_profile(sol.r, sol.z, sol.n_e[-1], mat.n_cr)
    at = {}
    for ts in tuple(cfg.report_times) + (float(sol.t[-1]),):
        i = sol.it(ts)
        at[f"{ts / FS:g}"] = {"ne_cm3": float(ne0[i] * 1e-6), "Te_K": float(Te0[i]),
                              "R": float(R0[i]), "alpha_m": float(a0[i])}
    return {
        "material": mat.name,
        "tau_fs": mat.tau / FS, "tc_fs": mat.tc / FS,
        "alpha_i_cm2J": mat.raw.get("alpha_i_cm2J"), "delta_N": mat.delta_N, "N": mat.N,
        "integrator": cfg.integrator, "dt_fs": sol.meta["dt_fs"], "shared_R": sol.meta["shared_R"],
        "t_ncr_fs": None if t_ncr is None else t_ncr / FS,
        "max_ne_cm3": float(sol.n_e.max() * 1e-6),
        "max_Te_K": float(sol.T_e.max()),
        "center_at_fs": at,
        "width_um": None if width is None else width * 1e6,
        "depth_nm": None if depth is None else depth * 1e9,
    }
