"""Evaluate a forward PINN run against an FDM reference on the FDM grid (GUIDE.md Phase 4.2–4.4).

    python scripts/evaluate.py --run outputs/forward/<run> --fdm outputs/fdm/<fdm_run>
    python scripts/evaluate.py --run ... --fdm ... --photo outputs/fdm/<photo_only_run> --which latest
    python scripts/evaluate.py --run ... --fdm ... --times 50,100,150,200 --no-figs

Both runs must use the same material YAML (same physics) — checked, warning printed otherwise.
Writes under <run>/eval/:
    metrics.json   L2RE / max relative errors per time, ablation width/depth PINN vs FDM, DoD checks,
                   FDM integrator and the checkpoint that was evaluated
    table2.md      paper Table 2 / Table 3 format (ours beside the paper's values)
    figs/          fig3 (r = z = 0 curves + R, α), fig4 (maps + relative error at the final time),
                   fig5a (ablation contours), fig10 (z = 200 nm slices + absolute errors)
The photoionization-only FDM run for Fig.3a is found automatically next to --fdm
(<stem>_photo_<stamp>) unless --photo is given.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from fsl.config import FS, REPO_ROOT, load_yaml
from fsl.eval import plots
from fsl.eval.compare import (compare_at_times, dod_check, fdm_center_series, pinn_center_series,
                              profile_comparison, table2_markdown)
from fsl.fdm import Solution
from fsl.pinn.train import latest_checkpoint, load_run

PHYSICS_KEYS = ("lambda_nm", "tp_fs", "r0_um", "F_Jcm2", "U1_eV", "alpha_i_cm2J",
                "delta_N_cm3ps_cm2TW", "N", "tau_fs", "tc_fs")


def find_photo_run(fdm_dir: Path) -> Path | None:
    """<stem>_photo_<stamp> next to <stem>_<stamp> (e.g. glass_ref_rk4_photo_… for glass_ref_rk4_…)."""
    m = re.match(r"^(.*?)_(\d{8}-\d{4})$", fdm_dir.name)
    stem = m.group(1) if m else fdm_dir.name
    hits = sorted(p for p in fdm_dir.parent.glob(f"{stem}_photo_*") if (p / "solution.npz").exists())
    return hits[-1] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="forward PINN run dir")
    ap.add_argument("--fdm", required=True, help="FDM reference run dir (solution.npz)")
    ap.add_argument("--photo", help="photoionization-only FDM run dir for Fig.3a (default: auto-detect)")
    ap.add_argument("--which", default="best", choices=["best", "latest"])
    ap.add_argument("--times", help="comma-separated times in fs (default: fdm.yaml report_times + final)")
    ap.add_argument("--z-slice-nm", type=float, default=200.0, help="depth of the Fig.10 slice")
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()

    run_dir, fdm_dir = Path(args.run), Path(args.fdm)
    sol = Solution.load(fdm_dir / "solution.npz")
    model, mat, sc, cfg = load_run(run_dir, args.which)
    ckpt = latest_checkpoint(run_dir, "best" if args.which == "best" else "periodic")

    fdm_mat = sol.meta.get("material_yaml", {})
    diff = {k: (mat.raw.get(k), fdm_mat.get(k)) for k in PHYSICS_KEYS if mat.raw.get(k) != fdm_mat.get(k)}
    if diff:
        print(f"WARNING: physics differs between PINN and FDM runs: {diff}")
    if sol.meta.get("integrator") != "rk4":
        print(f"NOTE: FDM reference integrator is {sol.meta.get('integrator')!r} (I-31 expects rk4 for PINN evaluation)")

    if args.times:
        times = [float(x) * FS for x in args.times.split(",")]
    else:
        rt = load_yaml(REPO_ROOT / "configs" / "fdm.yaml").get("report_times_fs", {}).get(mat.name, [])
        times = sorted({float(x) * FS for x in rt} | {float(sol.t[-1])})

    comp = compare_at_times(model, sc, sol, times)
    at_fs = {k: v["metrics"] for k, v in comp.items()}
    print(f"{'t (fs)':>7} {'L2RE n_e':>10} {'L2RE T_e':>10} {'maxrel n_e':>11} {'maxrel T_e':>11} {'global n_e':>11} {'global T_e':>11}")
    for k, m in at_fs.items():
        print(f"{float(k):7.0f} {m['n_e']['l2re']:10.3e} {m['T_e']['l2re']:10.3e} {m['n_e']['max_rel_pointwise']:11.3e} "
              f"{m['T_e']['max_rel_pointwise']:11.3e} {m['n_e']['max_rel_global']:11.3e} {m['T_e']['max_rel_global']:11.3e}")

    last_key = max(comp, key=float)
    profile = profile_comparison(sol, mat, comp[last_key]["n_e"], comp[last_key]["t_s"])
    print(f"profile @ {profile['t_fs']:g} fs — PINN {profile['pinn']}  FDM {profile['fdm']}")
    print(f"max n_e (cm⁻³) — PINN {profile['max_ne_cm3']['pinn']:.3e}  FDM {profile['max_ne_cm3']['fdm']:.3e}")

    checks = dod_check(at_fs, profile)
    print("\nPhase 4 DoD:")
    for c in checks:
        v = c["value"]
        vs = "—" if v is None else (f"{v:.3e}" if isinstance(v, float) else ", ".join(f"{x:.2e}" for x in v))
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {vs}" + (f"  ({c['note']})" if c.get("note") else ""))
    print(f"  → {'ALL PASS' if all(c['ok'] for c in checks) else 'NOT PASSED'}")

    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    tables = plots.paper_ref(mat.name + "_tables")
    paper_tbl = tables.get("table2") or (tables.get("table3") or {}).get("basic")
    if paper_tbl and "t_fs" not in paper_tbl:
        paper_tbl = dict(paper_tbl, t_fs=tables["table3"]["t_fs"])
    md = table2_markdown(at_fs, paper_tbl, label=f"PINN {cfg.profile}")
    (eval_dir / "table2.md").write_text(md, encoding="utf-8")
    print("\n" + md)

    run_metrics = {}
    if (run_dir / "metrics.json").exists():
        rm = json.load(open(run_dir / "metrics.json", encoding="utf-8"))
        run_metrics = {k: rm.get(k) for k in ("iterations_total", "n_params", "net", "dtype", "device", "wall_s")}
    out = {"run": str(run_dir), "fdm": str(fdm_dir), "which": args.which, "ckpt": None if ckpt is None else ckpt.name,
           "fdm_integrator": sol.meta.get("integrator"), "fdm_dt_fs": sol.meta.get("dt_fs"),
           "physics_mismatch": diff, "train": run_metrics, "at_fs": at_fs, "profile_final": profile,
           "dod": checks, "dod_pass": all(c["ok"] for c in checks)}
    with open(eval_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"→ {eval_dir / 'metrics.json'}, {eval_dir / 'table2.md'}")

    if args.no_figs:
        return 0
    photo_dir = Path(args.photo) if args.photo else find_photo_run(fdm_dir)
    photo = Solution.load(photo_dir / "solution.npz") if photo_dir and (photo_dir / "solution.npz").exists() else None
    figs = eval_dir / "figs"
    center = pinn_center_series(model, sc, mat, sol.t)
    plots.fig3(sol, mat, figs / "fig3.png", pinn=center, photo=photo)
    plots.fig4(sol, mat, figs / "fig4.png", comp[last_key]["t_s"], comp[last_key]["n_e"], comp[last_key]["T_e"])
    plots.fig5a(sol, mat, figs / "fig5a.png", comp[last_key]["n_e"], comp[last_key]["t_s"])
    plots.fig10(sol, mat, figs / "fig10.png", {k: (v["n_e"], v["T_e"]) for k, v in comp.items()},
                z_m=args.z_slice_nm * 1e-9)
    print(f"→ figures in {figs} (fig3, fig4, fig5a, fig10; fig9 is in {run_dir / 'figs'})"
          + ("" if photo is not None else "  [no photoionization-only run found for Fig.3a]"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
