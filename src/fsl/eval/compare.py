"""PINN ↔ FDM comparison helpers (GUIDE.md §4.5, Phase 4.2–4.4; notes/decisions.md I-33).

Everything here works on plain numpy arrays so that the figure functions in ``plots.py`` can
be tested with FDM fields standing in for the PINN. The only place the network is called is
``pinn_fields`` / ``pinn_center_series``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fsl import physics as P
from fsl.config import FS, MaterialParams
from fsl.eval.metrics import all_metrics
from fsl.eval.profile import ablation_profile
from fsl.fdm import Solution
from fsl.pinn.train import predict_fields
from fsl.scaling import Scales


# ----------------------------------------------------------------------------- PINN evaluation


def pinn_fields(model, sc: Scales, sol: Solution, t_s: float):
    """(n_e, T_e, φ) of the PINN on the FDM (r, z) grid at time t_s [s]; shapes (nr, nz)."""
    return predict_fields(model, sc, sol.r, sol.z, t_s)


def pinn_center_series(model, sc: Scales, mat: MaterialParams, t: np.ndarray) -> dict[str, np.ndarray]:
    """The PINN's Fig.3 curves: n_e, T_e at r = z = 0 for every t [s], plus R and α computed
    from that surface density with the same physics functions the FDM uses."""
    t = np.asarray(t, dtype=float)
    r_t, z_t, t_t = sc.to_tilde(np.zeros_like(t), np.zeros_like(t), t)
    y = model.predict(np.column_stack([r_t, z_t, t_t]))
    n_e = sc.n_from_tilde(y[:, 0])
    T_e = sc.T_from_tilde(y[:, 1])
    R, ah = P.surface_optics(n_e, mat)
    return {"t": t, "n_e": n_e, "T_e": T_e, "R": R,
            "alpha": P.alpha_total(ah, n_e, mat.alpha_i, mat.U1), "phi": y[:, 2]}


def fdm_center_series(sol: Solution) -> dict[str, np.ndarray]:
    return {"t": sol.t, "n_e": sol.center("n_e"), "T_e": sol.center("T_e"),
            "R": sol.center("R"), "alpha": sol.center("alpha")}


def compare_at_times(model, sc: Scales, sol: Solution, times_s) -> dict[str, dict[str, Any]]:
    """Per report time: PINN fields on the FDM grid and the §4.5 metrics against the FDM.
    Keys are the times in fs as strings ('50', '100', …) to match metrics.json."""
    out = {}
    for t in times_s:
        i = sol.it(t)
        n_nn, T_nn, phi_nn = pinn_fields(model, sc, sol, t)
        out[f"{t / FS:g}"] = {
            "t_s": float(t), "n_e": n_nn, "T_e": T_nn, "phi": phi_nn,
            "metrics": {"n_e": all_metrics(n_nn, sol.n_e[i]), "T_e": all_metrics(T_nn, sol.T_e[i])},
        }
    return out


def profile_comparison(sol: Solution, mat: MaterialParams, n_nn: np.ndarray, t_s: float | None = None) -> dict:
    i = -1 if t_s is None else sol.it(t_s)
    w_nn, d_nn = ablation_profile(sol.r, sol.z, n_nn, mat.n_cr)
    w_fd, d_fd = ablation_profile(sol.r, sol.z, sol.n_e[i], mat.n_cr)
    f = lambda v, s: None if v is None else float(v * s)
    return {
        "t_fs": float(sol.t[i] / FS),
        "pinn": {"width_um": f(w_nn, 1e6), "depth_nm": f(d_nn, 1e9)},
        "fdm": {"width_um": f(w_fd, 1e6), "depth_nm": f(d_fd, 1e9)},
        "max_ne_cm3": {"pinn": float(n_nn.max() * 1e-6), "fdm": float(sol.n_e[i].max() * 1e-6)},
    }


# ----------------------------------------------------------------------------- DoD and Table 2


def dod_check(at_fs: dict[str, dict], profile: dict, l2re_ne: float = 1e-2, l2re_Te: float = 2e-2,
              tol: float = 0.10) -> list[dict[str, Any]]:
    """GUIDE Phase 4 DoD: L2RE at the final time (n_e < 1e-2, T_e < 2e-2), width and depth within
    ±10 % of the FDM, and L2RE not monotonically increasing in t. Returns one dict per check."""
    times = sorted(at_fs, key=float)
    last = times[-1]
    m = lambda t, q: at_fs[t][q]["l2re"] if "l2re" in at_fs[t][q] else at_fs[t]["metrics"][q]["l2re"]
    checks = [
        {"name": f"L2RE n_e at {last} fs < {l2re_ne:g}", "value": m(last, "n_e"), "ok": m(last, "n_e") < l2re_ne},
        {"name": f"L2RE T_e at {last} fs < {l2re_Te:g}", "value": m(last, "T_e"), "ok": m(last, "T_e") < l2re_Te},
    ]
    for key, unit in (("width_um", "µm"), ("depth_nm", "nm")):
        a, b = profile["pinn"][key], profile["fdm"][key]
        if b is None:
            checks.append({"name": f"{key} within ±{tol:.0%} of FDM", "value": None, "ok": a is None,
                           "note": "FDM has no profile" + ("" if a is None else "; PINN has one")})
        else:
            rel = None if a is None else abs(a - b) / abs(b)
            checks.append({"name": f"{key} within ±{tol:.0%} of FDM", "value": rel,
                           "ok": rel is not None and rel <= tol,
                           "note": f"PINN {a if a is None else round(a, 3)} vs FDM {round(b, 3)} {unit}"})
    for q in ("n_e", "T_e"):
        seq = [m(t, q) for t in times]
        increasing = all(b > a for a, b in zip(seq, seq[1:])) if len(seq) > 1 else False
        checks.append({"name": f"L2RE {q} not monotonically increasing in t", "value": seq, "ok": not increasing})
    return checks


def table2_markdown(at_fs: dict[str, dict], paper: dict | None = None, label: str = "ours") -> str:
    """Paper Table 2 / Table 3 format: L2RE of n_e and T_e per time, with the paper's values
    beside ours when ``paper`` = {t_fs: [...], n_e: [...], T_e: [...]} is given."""
    times = sorted(at_fs, key=float)
    get = lambda t, q, k: (at_fs[t][q] if "l2re" in at_fs[t][q] else at_fs[t]["metrics"][q])[k]
    pp = {}
    if paper:
        for j, t in enumerate(paper.get("t_fs", [])):
            pp[f"{float(t):g}"] = (float(paper["n_e"][j]), float(paper["T_e"][j]))
    head = f"| t (fs) | L2RE n_e ({label}) | L2RE n_e (paper) | L2RE T_e ({label}) | L2RE T_e (paper) | max rel n_e | max rel T_e |"
    lines = [head, "|---|---|---|---|---|---|---|"]
    for t in times:
        p = pp.get(t)
        lines.append(f"| {t} | {get(t, 'n_e', 'l2re'):.2e} | {p[0]:.2e} | {get(t, 'T_e', 'l2re'):.2e} | {p[1]:.2e} | "
                     f"{get(t, 'n_e', 'max_rel_pointwise'):.1%} | {get(t, 'T_e', 'max_rel_pointwise'):.1%} |"
                     if p else
                     f"| {t} | {get(t, 'n_e', 'l2re'):.2e} | — | {get(t, 'T_e', 'l2re'):.2e} | — | "
                     f"{get(t, 'n_e', 'max_rel_pointwise'):.1%} | {get(t, 'T_e', 'max_rel_pointwise'):.1%} |")
    return "\n".join(lines) + "\n"
