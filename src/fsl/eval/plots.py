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
