"""Phase 2.10: Δt convergence check of the reference FDM with the adopted (YAML default) settings.

    python scripts/check_dt.py --material glass
    python scripts/check_dt.py --material glass --cases euler:1 euler:0.5 rk4:0.1
    python scripts/check_dt.py --material sic

Runs every (integrator, Δt) case on the material's own FDM grid, takes the LAST case as the
reference (finest), and prints for each case the r = z = 0 values at the report times, t(n_cr),
width/depth, and the relative difference to the reference. GUIDE Phase 2 DoD: Euler Δt = 1 fs vs
RK4 Δt = 0.1 fs differ by < 1 % in final n_e, T_e at r = z = 0. Δt = 0.5 fs is the step used in [27].
Writes outputs/fdm/dt_check_<material>_<stamp>/summary.json.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from fsl.config import FS, REPO_ROOT, load_material
from fsl.eval.metrics import l2re
from fsl.fdm import FDMConfig, metrics, solve


def _rel(a, b):
    if a is None or b is None:
        return None
    return abs(a - b) / abs(b) if b else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--material", required=True, choices=["glass", "sic", "gan"])
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "fdm.yaml"))
    ap.add_argument("--cases", nargs="*", default=["euler:1", "euler:0.5", "rk4:1", "rk4:0.1"],
                    metavar="INTEGRATOR:DT_FS", help="last one is the reference")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="material YAML overrides")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "fdm"))
    args = ap.parse_args()

    ov = {}
    for it in args.set:
        k, v = it.split("=", 1)
        try:
            ov[k] = float(v)
        except ValueError:
            ov[k] = v
    mat = load_material(args.material, **ov)
    base = FDMConfig.from_yaml(args.config, args.material)

    sols, rows = [], []
    for case in args.cases:
        integ, dt = case.split(":")
        cfg = base.replace(integrator=integ, dt=float(dt) * FS)
        t0 = time.time()
        sol = solve(mat, cfg)
        m = metrics(sol, mat, cfg)
        m["case"] = case
        m["wall_s"] = round(time.time() - t0, 3)
        sols.append(sol)
        rows.append(m)

    ref, mref = sols[-1], rows[-1]
    times = list(mref["center_at_fs"])
    print(f"{mat!r}")
    print(f"reference = {mref['case']}   (times in fs; rel = |case − ref| / ref)\n")
    head = f"{'case':>10} | " + " ".join(f"{'ne(' + k + ')':>10}" for k in times) + " | " \
        + " ".join(f"{'Te(' + k + ')':>9}" for k in times) + f" | {'t_ncr':>6} {'width':>6} {'depth':>6} | {'wall':>6}"
    print(head)
    for m in rows:
        at = m["center_at_fs"]
        print(f"{m['case']:>10} | " + " ".join(f"{at[k]['ne_cm3']:10.4e}" for k in times) + " | "
              + " ".join(f"{at[k]['Te_K']:9.3e}" for k in times)
              + f" | {m['t_ncr_fs'] if m['t_ncr_fs'] is None else round(m['t_ncr_fs'], 2)!s:>6}"
              + f" {m['width_um'] if m['width_um'] is None else round(m['width_um'], 3)!s:>6}"
              + f" {m['depth_nm'] if m['depth_nm'] is None else round(m['depth_nm'], 1)!s:>6} | {m['wall_s']:6.2f}")

    print("\nrelative difference to the reference")
    print(f"{'case':>10} | {'ne_end r=z=0':>13} {'Te_end r=z=0':>13} | {'L2RE ne(end)':>13} {'L2RE Te(end)':>13} | "
          f"{'t_ncr':>7} {'width':>7} {'depth':>7}")
    summary = {"material": mat.name, "material_yaml": mat.raw, "reference": mref["case"], "cases": []}
    for sol, m in zip(sols, rows):
        atr, at = mref["center_at_fs"], m["center_at_fs"]
        last = times[-1]
        d = {
            "case": m["case"],
            "ne_end": _rel(at[last]["ne_cm3"], atr[last]["ne_cm3"]),
            "Te_end": _rel(at[last]["Te_K"], atr[last]["Te_K"]),
            "l2re_ne_end": float(l2re(sol.n_e[-1], ref.n_e[-1])),
            "l2re_Te_end": float(l2re(sol.T_e[-1] - 300.0, ref.T_e[-1] - 300.0)),
            "t_ncr": _rel(m["t_ncr_fs"], mref["t_ncr_fs"]),
            "width": _rel(m["width_um"], mref["width_um"]),
            "depth": _rel(m["depth_nm"], mref["depth_nm"]),
        }
        summary["cases"].append({**d, "metrics": m})
        f = lambda x: "    —" if x is None else f"{x:.2e}"
        print(f"{m['case']:>10} | {f(d['ne_end']):>13} {f(d['Te_end']):>13} | {f(d['l2re_ne_end']):>13} {f(d['l2re_Te_end']):>13} | "
              f"{f(d['t_ncr']):>7} {f(d['width']):>7} {f(d['depth']):>7}")

    worst = max(max(c["ne_end"] or 0.0, c["Te_end"] or 0.0) for c in summary["cases"][:-1])
    verdict = "PASS" if worst < 1e-2 else "FAIL"
    print(f"\nDoD (< 1 % in final n_e, T_e at r = z = 0 for every case vs reference): {verdict}")
    summary["dod_pass"] = verdict == "PASS"

    out_dir = Path(args.out) / f"dt_check_{mat.name}_{datetime.now().strftime('%Y%m%d-%H%M')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"→ {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
