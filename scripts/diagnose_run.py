"""Print what the network actually produces at its collocation points (GUIDE.md Phase 3 debugging).

    python scripts/diagnose_run.py --run outputs/forward/<run>
    python scripts/diagnose_run.py --run outputs/forward/<run> --ckpt best-1000.pt
    python scripts/diagnose_run.py --run outputs/forward/<run> --fdm outputs/fdm/<fdm_run>

For the restored checkpoint, evaluates every intermediate quantity of the residuals
(ñ, φ, R, I/I₀, α, z_max·α, ∂φ/∂z̃, R₁, R₂, R_φ, …) on the run's own training points
(same seed → same points) and prints percentiles plus the location of the largest |ñ|.
With --fdm, also prints the same quantities evaluated from the FDM solution at the
final time for comparison (what the PINN *should* produce).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from fsl import physics as P
from fsl.config import FS
from fsl.fdm import Solution
from fsl.pinn.pde import residual_terms
from fsl.pinn.train import load_run

QUANT = ["n_t", "n_surf_t", "T_t", "phi", "dphi_dz", "R_surf", "I_over_I0", "alpha", "zmax_alpha",
         "dn_dt", "src_impact", "src_photo", "heat", "R1", "R2", "Rphi"]
PCT = [0, 1, 50, 99, 100]


def _table(rows: dict[str, np.ndarray], title: str) -> None:
    print(f"\n{title}")
    print(f"{'quantity':>12} " + " ".join(f"{f'p{p}':>11}" for p in PCT) + f" {'rms':>11}")
    for k, v in rows.items():
        v = np.asarray(v, float).ravel()
        pc = np.percentile(v, PCT)
        print(f"{k:>12} " + " ".join(f"{x:11.3e}" for x in pc) + f" {np.sqrt(np.mean(v**2)):11.3e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", help="checkpoint file name in <run>/ckpt (default: newest 'best')")
    ap.add_argument("--fdm", help="FDM run dir to print the reference quantities")
    args = ap.parse_args()

    model, mat, sc, cfg = load_run(args.run, "best", args.ckpt)
    net = model.net
    x = torch.as_tensor(model.data.train_x).requires_grad_()
    y = net(x)
    q = residual_terms(net, mat, sc, x, y)
    rows = {k: q[k].detach().cpu().numpy() for k in QUANT}
    _table(rows, f"PINN at its {len(x)} collocation points — {args.run} / {args.ckpt or 'best'}")
    for k in ("R1", "R2", "Rphi"):
        print(f"  MSE {k}: {np.mean(rows[k] ** 2):.3e}")

    i = int(np.argmax(np.abs(rows["n_t"])))
    xi = x[i].detach().cpu().numpy()
    r, z, t = sc.from_tilde(xi[0], xi[1], xi[2])
    print(f"\nlargest |ñ| = {rows['n_t'][i, 0]:.3e} at r = {r * 1e6:.2f} µm, z = {z * 1e9:.1f} nm, "
          f"t = {t / FS:.1f} fs  (R_surf {rows['R_surf'][i, 0]:.3f}, I/I0 {rows['I_over_I0'][i, 0]:.3e}, "
          f"z_max·α {rows['zmax_alpha'][i, 0]:.3e}, ∂φ/∂z̃ {rows['dphi_dz'][i, 0]:.3e})")

    if args.fdm:
        sol = Solution.load(Path(args.fdm) / "solution.npz")
        k = -1
        n_e, T_e, alpha, I, R = sol.n_e[k], sol.T_e[k], sol.alpha[k], sol.I[k], sol.R[k]
        ref = {
            "n_t": sc.n_to_tilde(n_e), "T_t": sc.T_to_tilde(T_e), "R_surf": R,
            "I_over_I0": I / P.peak_intensity(mat), "alpha": alpha, "zmax_alpha": sc.z_max * alpha,
        }
        _table(ref, f"FDM reference at t = {sol.t[k] / FS:g} fs on its grid — {args.fdm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
