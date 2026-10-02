"""Evaluate a forward PINN run against an FDM run on the FDM grid (GUIDE.md Phase 3.8 / 4.2).

    python scripts/evaluate.py --run outputs/forward/<run> --fdm outputs/fdm/<fdm_run>
    python scripts/evaluate.py --run ... --fdm ... --which latest --times 50,100,150,200

Both runs must use the same material YAML (same physics) — this is checked. Writes
<run>/eval/metrics.json with L2RE / max relative errors of n_e and T_e at each time and
the ablation width/depth of PINN vs FDM at the final time. Figures come in Phase 4.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fsl.config import FS, REPO_ROOT, load_yaml
from fsl.eval.metrics import all_metrics
from fsl.eval.profile import ablation_profile
from fsl.fdm import Solution
from fsl.pinn.train import load_run, predict_fields

PHYSICS_KEYS = ("lambda_nm", "tp_fs", "r0_um", "F_Jcm2", "U1_eV", "alpha_i_cm2J",
                "delta_N_cm3ps_cm2TW", "N", "tau_fs", "tc_fs")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="forward PINN run dir")
    ap.add_argument("--fdm", required=True, help="FDM run dir (solution.npz)")
    ap.add_argument("--which", default="best", choices=["best", "latest"])
    ap.add_argument("--times", help="comma-separated times in fs (default: fdm.yaml report_times + final)")
    args = ap.parse_args()

    run_dir, fdm_dir = Path(args.run), Path(args.fdm)
    sol = Solution.load(fdm_dir / "solution.npz")
    model, mat, sc, cfg = load_run(run_dir, args.which)

    fdm_mat = sol.meta.get("material_yaml", {})
    diff = {k: (mat.raw.get(k), fdm_mat.get(k)) for k in PHYSICS_KEYS if mat.raw.get(k) != fdm_mat.get(k)}
    if diff:
        print(f"WARNING: physics differs between PINN and FDM runs: {diff}")

    if args.times:
        times = [float(x) * FS for x in args.times.split(",")]
    else:
        rt = load_yaml(REPO_ROOT / "configs" / "fdm.yaml").get("report_times_fs", {}).get(mat.name, [])
        times = sorted({float(x) * FS for x in rt} | {float(sol.t[-1])})

    out = {"run": str(run_dir), "fdm": str(fdm_dir), "which": args.which, "ckpt": None,
           "physics_mismatch": diff, "at_fs": {}}
    print(f"{'t (fs)':>7} {'L2RE n_e':>10} {'L2RE T_e':>10} {'maxrel n_e':>11} {'maxrel T_e':>11}")
    for t in times:
        i = sol.it(t)
        n_nn, T_nn, _ = predict_fields(model, sc, sol.r, sol.z, t)
        mn, mT = all_metrics(n_nn, sol.n_e[i]), all_metrics(T_nn, sol.T_e[i])
        out["at_fs"][f"{t / FS:g}"] = {"n_e": mn, "T_e": mT}
        print(f"{t / FS:7.0f} {mn['l2re']:10.3e} {mT['l2re']:10.3e} {mn['max_rel_pointwise']:11.3e} {mT['max_rel_pointwise']:11.3e}")

    n_nn, _, _ = predict_fields(model, sc, sol.r, sol.z, float(sol.t[-1]))
    w_nn, d_nn = ablation_profile(sol.r, sol.z, n_nn, mat.n_cr)
    w_fd, d_fd = ablation_profile(sol.r, sol.z, sol.n_e[-1], mat.n_cr)
    out["profile_final"] = {
        "pinn": {"width_um": None if w_nn is None else w_nn * 1e6, "depth_nm": None if d_nn is None else d_nn * 1e9},
        "fdm": {"width_um": None if w_fd is None else w_fd * 1e6, "depth_nm": None if d_fd is None else d_fd * 1e9},
        "max_ne_cm3": {"pinn": float(n_nn.max() * 1e-6), "fdm": float(sol.n_e[-1].max() * 1e-6)},
    }
    print(f"profile @ {sol.t[-1] / FS:g} fs — PINN {out['profile_final']['pinn']}  FDM {out['profile_final']['fdm']}")
    print(f"max n_e (cm⁻³) — PINN {out['profile_final']['max_ne_cm3']['pinn']:.3e}  FDM {out['profile_final']['max_ne_cm3']['fdm']:.3e}")

    (run_dir / "eval").mkdir(exist_ok=True)
    with open(run_dir / "eval" / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"→ {run_dir / 'eval' / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
