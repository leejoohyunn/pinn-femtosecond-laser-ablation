"""Figures reproducing the paper's plots (GUIDE.md §4.4: all figures go through here).

Function names follow the paper's numbering. Phase 2 provides fig3a, fig3b, fig5;
the rest (fig4, fig6, fig7, fig9–13) are added in Phase 4/6/7.
Axes use the paper's units (µm, nm, fs, cm⁻³, K, 1/m) — GUIDE §4.2 rule 5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fsl.config import FS, REPO_ROOT, MaterialParams, load_yaml  # noqa: E402
from fsl.eval.profile import ablation_profile  # noqa: E402
from fsl.fdm import Solution  # noqa: E402

PAPER_REF_PATH = REPO_ROOT / "configs" / "paper_reference.yaml"


def paper_ref(material: str) -> dict[str, Any]:
    return load_yaml(PAPER_REF_PATH).get(material, {})


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- Fig.3


def fig3a(sol: Solution, mat: MaterialParams, path: str | Path,
          overlays: dict[str, Solution] | None = None) -> Path:
    """n_e(t) at r = z = 0. `overlays` = {label: Solution} drawn dashed (e.g. photo-only runs)."""
    fig, ax = plt.subplots(figsize=(5, 3.6))
    t_fs = sol.t / FS
    ax.plot(t_fs, sol.center("n_e") * 1e-6, "k-", lw=1.8, label="FDM (photo + impact)")
    for label, s in (overlays or {}).items():
        ax.plot(s.t / FS, s.center("n_e") * 1e-6, "--", lw=1.5, label=label)
    ax.axhline(mat.n_cr_cm3, color="gray", ls=":", lw=1, label=f"n_cr = {mat.n_cr_cm3:.2e}")
    ref = paper_ref(mat.name).get("fig3a", {})
    for key, marker in (("full", "o"), ("photo_only", "s")):
        if key in ref:
            pts = np.array(ref[key], dtype=float)
            ax.plot(pts[:, 0], pts[:, 1], marker, mfc="none", ms=7, color="C3",
                    label=f"paper Fig.3a ({key})")
    ax.set_xlabel("t (fs)")
    ax.set_ylabel("n_e at r = z = 0 (cm⁻³)")
    ax.set_title(f"{mat.name}: τ = {mat.tau / FS:g} fs, t_c = {mat.tc / FS:g} fs")
    ax.legend(fontsize=7)
    return _save(fig, path)


def fig3b(sol: Solution, mat: MaterialParams, path: str | Path) -> Path:
    """R(t) and α(t) at r = 0 (surface), twin axes, with the paper's reference points."""
    fig, ax1 = plt.subplots(figsize=(5, 3.6))
    t_fs = sol.t / FS
    ax1.plot(t_fs, sol.center("R"), "b-", lw=1.8, label="R (FDM)")
    ax1.set_xlabel("t (fs)")
    ax1.set_ylabel("R", color="b")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(t_fs, sol.center("alpha"), "r-", lw=1.8, label="α (FDM)")
    ax2.set_ylabel("α (1/m)", color="r")
    ref = paper_ref(mat.name).get("fig3b", {})
    if "R" in ref:
        p = np.array(ref["R"], dtype=float)
        ax1.plot(p[:, 0], p[:, 1], "bo", mfc="none", ms=7, label="R (paper)")
    if "alpha_m" in ref:
        p = np.array(ref["alpha_m"], dtype=float)
        ax2.plot(p[:, 0], p[:, 1], "rs", mfc="none", ms=7, label="α (paper)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    ax1.set_title(f"{mat.name}: τ = {mat.tau / FS:g} fs, t_c = {mat.tc / FS:g} fs")
    return _save(fig, path)


# ----------------------------------------------------------------------------- Fig.5


def fig5(sol: Solution, mat: MaterialParams, path: str | Path, t_s: float | None = None) -> Path:
    """n_e(r, z) map at the final time with the n_e = n_cr ablation contour and width/depth."""
    i = -1 if t_s is None else sol.it(t_s)
    ne = sol.n_e[i] * 1e-6  # (nr, nz) cm⁻³
    r_um, z_nm = sol.r * 1e6, sol.z * 1e9
    width, depth = ablation_profile(sol.r, sol.z, sol.n_e[i], mat.n_cr)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    pm = ax.pcolormesh(r_um, z_nm, ne.T, shading="auto", cmap="inferno")
    fig.colorbar(pm, ax=ax, label="n_e (cm⁻³)")
    if np.nanmax(ne) >= mat.n_cr_cm3:
        ax.contour(r_um, z_nm, ne.T, levels=[mat.n_cr_cm3], colors="cyan", linewidths=1.5)
    ax.invert_yaxis()
    ax.set_xlabel("r (µm)")
    ax.set_ylabel("z (nm)")
    ref = paper_ref(mat.name).get("profile", {})
    w = "—" if width is None else f"{width * 1e6:.2f} µm"
    d = "—" if depth is None else f"{depth * 1e9:.0f} nm"
    ax.set_title(f"{mat.name} t = {sol.t[i] / FS:g} fs — width {w}, depth {d}"
                 + (f"  (paper {ref.get('width_um')} µm / {ref.get('depth_nm')} nm)" if ref else ""),
                 fontsize=8)
    return _save(fig, path)


# ----------------------------------------------------------------------------- Fig.9 / Fig.11 (loss curves)


def fig9(history_csv: str | Path, path: str | Path, title: str = "") -> Path:
    """Training loss per term vs iteration (paper Fig.9: Loss_ne magenta, Loss_Te orange, total black;
    follow the legend, not the caption — GUIDE §1.2). Reads history.csv written by pinn/train.py."""
    import csv

    rows = list(csv.DictReader(open(history_csv, encoding="utf-8")))
    step = np.array([float(r["step"]) for r in rows])
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for key, color, label in (("total", "k", "Total loss"), ("loss_ne", "m", "Loss_ne"),
                              ("loss_Te", "orange", "Loss_Te"), ("loss_phi", "c", "Loss_phi")):
        if key in rows[0]:
            ax.semilogy(step, [float(r[key]) for r in rows], color=color, lw=1.5, label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training loss")
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=7)
    return _save(fig, path)


# ============================================================================= Phase 4: PINN vs FDM
# All functions below take plain arrays for the PINN side (computed by eval/compare.py), so
# they can be exercised with FDM fields standing in for the network (tests). I-33.

_T_COLORS = ("C0", "C1", "C2", "C3", "C4", "C5")


def _rz_axes(ax, sol: Solution):
    ax.invert_yaxis()
    ax.set_xlabel("r (µm)")
    ax.set_ylabel("z (nm)")


def fig3(sol: Solution, mat: MaterialParams, path: str | Path, pinn: dict | None = None,
         photo: Solution | None = None) -> Path:
    """Paper Fig.3: (a) n_e(t) at r = z = 0 — FDM, PINN, photoionization-only FDM, paper points;
    (b) R(t) and α(t) at the surface r = 0 — FDM vs PINN (R, α of the PINN come from its own
    surface n_e through the same Drude chain, see compare.pinn_center_series)."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 3.6))
    t_fs = sol.t / FS
    ax_a.plot(t_fs, sol.center("n_e") * 1e-6, "k-", lw=1.8, label="FDM (photo + impact)")
    if pinn is not None:
        ax_a.plot(pinn["t"] / FS, pinn["n_e"] * 1e-6, "r--", lw=1.6, label="PINN")
    if photo is not None:
        ax_a.plot(photo.t / FS, photo.center("n_e") * 1e-6, "b:", lw=1.6, label="FDM photoionization only")
    ax_a.axhline(mat.n_cr_cm3, color="gray", ls=":", lw=1, label=f"n_cr = {mat.n_cr_cm3:.2e}")
    ref = paper_ref(mat.name).get("fig3a", {})
    for key, marker in (("full", "o"), ("photo_only", "s")):
        if key in ref:
            pts = np.array(ref[key], dtype=float)
            ax_a.plot(pts[:, 0], pts[:, 1], marker, mfc="none", ms=7, color="C3", label=f"paper ({key})")
    ax_a.set_xlabel("t (fs)")
    ax_a.set_ylabel("n_e at r = z = 0 (cm⁻³)")
    ax_a.set_title("(a) free-electron density", fontsize=9)
    ax_a.legend(fontsize=7)

    ax_b.plot(t_fs, sol.center("R"), "b-", lw=1.8, label="R (FDM)")
    if pinn is not None:
        ax_b.plot(pinn["t"] / FS, pinn["R"], "b--", lw=1.4, label="R (PINN)")
    ax_b.set_xlabel("t (fs)")
    ax_b.set_ylabel("R", color="b")
    ax_b.set_ylim(0, 1.05)
    ax_c = ax_b.twinx()
    ax_c.plot(t_fs, sol.center("alpha"), "r-", lw=1.8, label="α (FDM)")
    if pinn is not None:
        ax_c.plot(pinn["t"] / FS, pinn["alpha"], "r--", lw=1.4, label="α (PINN)")
    ax_c.set_ylabel("α (1/m)", color="r")
    ref = paper_ref(mat.name).get("fig3b", {})
    if "R" in ref:
        p = np.array(ref["R"], dtype=float)
        ax_b.plot(p[:, 0], p[:, 1], "bo", mfc="none", ms=7, label="R (paper)")
    if "alpha_m" in ref:
        p = np.array(ref["alpha_m"], dtype=float)
        ax_c.plot(p[:, 0], p[:, 1], "rs", mfc="none", ms=7, label="α (paper)")
    h1, l1 = ax_b.get_legend_handles_labels()
    h2, l2 = ax_c.get_legend_handles_labels()
    ax_b.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    ax_b.set_title("(b) surface reflectivity and absorption", fontsize=9)
    fig.suptitle(f"{mat.name}: τ = {mat.tau / FS:g} fs, t_c = {mat.tc / FS:g} fs, t_p = {mat.tp / FS:g} fs", fontsize=9)
    return _save(fig, path)


def _rel_error_map(u_nn: np.ndarray, u_ref: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Pointwise |u_NN − u_M| / |u_M|, zero where |u_M| < floor·max|u_M| (metrics.max_rel_pointwise)."""
    mask = np.abs(u_ref) >= floor * np.max(np.abs(u_ref))
    err = np.zeros_like(u_ref, dtype=float)
    err[mask] = np.abs(u_nn[mask] - u_ref[mask]) / np.abs(u_ref[mask])
    return err


def fig4(sol: Solution, mat: MaterialParams, path: str | Path, t_s: float,
         n_nn: np.ndarray, T_nn: np.ndarray) -> Path:
    """Paper Fig.4 at time t_s: (a) n_e FDM, (b) n_e PINN, (c) relative error; (d–f) the same for T_e.
    The error maps use the pointwise definition of §4.5 (denominator floor 1e-3·max); the titles
    give the pointwise and global maxima."""
    from fsl.eval.metrics import max_rel_global, max_rel_pointwise

    i = sol.it(t_s)
    r_um, z_nm = sol.r * 1e6, sol.z * 1e9
    rows = (("n_e (cm⁻³)", sol.n_e[i] * 1e-6, n_nn * 1e-6, "inferno"),
            ("T_e (K)", sol.T_e[i], T_nn, "viridis"))
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.2))
    for k, (label, ref, nn, cmap) in enumerate(rows):
        vmin, vmax = float(min(ref.min(), nn.min())), float(max(ref.max(), nn.max()))
        for j, (title, fld) in enumerate(((f"FDM {label}", ref), (f"PINN {label}", nn))):
            ax = axes[k, j]
            pm = ax.pcolormesh(r_um, z_nm, fld.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(pm, ax=ax)
            _rz_axes(ax, sol)
            ax.set_title(f"({'abcdef'[3 * k + j]}) {title}", fontsize=9)
        err = _rel_error_map(nn, ref)
        ax = axes[k, 2]
        pm = ax.pcolormesh(r_um, z_nm, err.T, shading="auto", cmap="magma")
        fig.colorbar(pm, ax=ax)
        _rz_axes(ax, sol)
        ax.set_title(f"({'abcdef'[3 * k + 2]}) relative error — max {max_rel_pointwise(nn, ref):.2%} pointwise, "
                     f"{max_rel_global(nn, ref):.2%} of max", fontsize=8)
    ref4 = paper_ref(mat.name + "_tables").get("fig4_max_rel", {})
    fig.suptitle(f"{mat.name} t = {sol.t[i] / FS:g} fs — PINN vs FDM"
                 + (f"  (paper max rel. error: n_e {float(ref4['n_e']):.1%}, T_e {float(ref4['T_e']):.1%})" if ref4 else ""),
                 fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def fig5a(sol: Solution, mat: MaterialParams, path: str | Path, n_nn: np.ndarray, t_s: float | None = None) -> Path:
    """Paper Fig.5a: ablation contour n_e = n_cr at the final time, FDM (black) vs PINN (red dashed),
    on top of the PINN density map; widths/depths and the paper's values in the title."""
    i = -1 if t_s is None else sol.it(t_s)
    r_um, z_nm = sol.r * 1e6, sol.z * 1e9
    ne_fd, ne_nn = sol.n_e[i] * 1e-6, n_nn * 1e-6
    w_fd, d_fd = ablation_profile(sol.r, sol.z, sol.n_e[i], mat.n_cr)
    w_nn, d_nn = ablation_profile(sol.r, sol.z, n_nn, mat.n_cr)
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    pm = ax.pcolormesh(r_um, z_nm, ne_nn.T, shading="auto", cmap="inferno")
    fig.colorbar(pm, ax=ax, label="PINN n_e (cm⁻³)")
    if np.nanmax(ne_fd) >= mat.n_cr_cm3:
        ax.contour(r_um, z_nm, ne_fd.T, levels=[mat.n_cr_cm3], colors="white", linewidths=2.2)
        ax.contour(r_um, z_nm, ne_fd.T, levels=[mat.n_cr_cm3], colors="black", linewidths=1.2)
    if np.nanmax(ne_nn) >= mat.n_cr_cm3:
        ax.contour(r_um, z_nm, ne_nn.T, levels=[mat.n_cr_cm3], colors="red", linewidths=1.4, linestyles="--")
    _rz_axes(ax, sol)
    ax.plot([], [], "k-", label="FDM n_e = n_cr")
    ax.plot([], [], "r--", label="PINN n_e = n_cr")
    ax.legend(fontsize=7, loc="lower right")
    f = lambda v, s, u: "—" if v is None else f"{v * s:.{0 if u == 'nm' else 2}f} {u}"
    ref = paper_ref(mat.name).get("profile", {})
    ax.set_title(f"{mat.name} t = {sol.t[i] / FS:g} fs — width / depth\n"
                 f"FDM {f(w_fd, 1e6, 'µm')} / {f(d_fd, 1e9, 'nm')},  PINN {f(w_nn, 1e6, 'µm')} / {f(d_nn, 1e9, 'nm')}"
                 + (f",  paper {ref.get('width_um')} µm / {ref.get('depth_nm')} nm" if ref else ""), fontsize=8)
    return _save(fig, path)


def fig10(sol: Solution, mat: MaterialParams, path: str | Path, fields: dict[str, tuple[np.ndarray, np.ndarray]],
          z_m: float = 200e-9) -> Path:
    """Paper Fig.10: n_e(r) and T_e(r) on the slice z = 200 nm at several times — FDM lines, PINN
    markers — plus (c)(d) absolute-error panels |PINN − FDM| (added by us; the paper reports
    the n_e error peaking at 50 fs and the T_e error at 150 fs).
    ``fields`` = {t_fs_string: (n_nn, T_nn)} with arrays on the FDM grid."""
    iz = int(np.argmin(np.abs(sol.z - z_m)))
    r_um = sol.r * 1e6
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for c, (t_key, (n_nn, T_nn)) in zip(_T_COLORS, sorted(fields.items(), key=lambda kv: float(kv[0]))):
        i = sol.it(float(t_key) * FS)
        ne_fd, Te_fd = sol.n_e[i][:, iz] * 1e-6, sol.T_e[i][:, iz]
        ne_nn, Te_nn = n_nn[:, iz] * 1e-6, T_nn[:, iz]
        axes[0, 0].plot(r_um, ne_fd, "-", color=c, lw=1.6, label=f"FDM {t_key} fs")
        axes[0, 0].plot(r_um[::2], ne_nn[::2], "o", color=c, ms=3.5, mfc="none", label=f"PINN {t_key} fs")
        axes[0, 1].plot(r_um, Te_fd, "-", color=c, lw=1.6, label=f"FDM {t_key} fs")
        axes[0, 1].plot(r_um[::2], Te_nn[::2], "o", color=c, ms=3.5, mfc="none", label=f"PINN {t_key} fs")
        axes[1, 0].plot(r_um, np.abs(ne_nn - ne_fd), "-", color=c, lw=1.4, label=f"{t_key} fs")
        axes[1, 1].plot(r_um, np.abs(Te_nn - Te_fd), "-", color=c, lw=1.4, label=f"{t_key} fs")
    axes[0, 0].set_ylabel("n_e (cm⁻³)")
    axes[0, 1].set_ylabel("T_e (K)")
    axes[1, 0].set_ylabel("|Δn_e| (cm⁻³)")
    axes[1, 1].set_ylabel("|ΔT_e| (K)")
    for k, ax in enumerate(axes.ravel()):
        ax.set_xlabel("r (µm)")
        ax.legend(fontsize=6, ncol=2)
        ax.set_title(f"({'abcd'[k]}) " + ["free-electron density", "free-electron temperature",
                                            "absolute error n_e", "absolute error T_e"][k], fontsize=9)
    ref = paper_ref(mat.name + "_tables").get("fig10", {})
    fig.suptitle(f"{mat.name} slice z = {sol.z[iz] * 1e9:.0f} nm"
                 + (f"  (paper: max n_e ≈ {float(ref['max_ne_cm3']):.2e} cm⁻³, max T_e ≈ {float(ref['max_Te_K']):.2e} K at 200 fs)"
                    if ref else ""), fontsize=9)   # float(): PyYAML reads 1.9e21 as a string (I-4)
    fig.tight_layout()
    return _save(fig, path)
