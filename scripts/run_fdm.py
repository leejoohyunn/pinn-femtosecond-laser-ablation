"""Run the reference FDM (GUIDE.md Phase 2.4–2.8).

    python scripts/run_fdm.py --material glass
    python scripts/run_fdm.py --material glass --sweep tau_fs=1,5,10,20,50,100 tc_fs=0,tp
    python scripts/run_fdm.py --material glass --ionization photo_only
    python scripts/run_fdm.py --material glass --ionization photo_only_shared_R --shared_from outputs/fdm/<full_run>
    python scripts/run_fdm.py --material glass --integrator rk4 --dt_fs 0.1        # convergence check
    python scripts/run_fdm.py --material glass --set tau_fs=100 tc_fs=tp             # single override run

Each run writes outputs/fdm/<name>_<YYYYmmdd-HHMM>/{config.yaml, solution.npz, metrics.json, figs/}.
A sweep writes one sub-directory per combination plus sweep.csv / sweep.md.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from fsl.config import REPO_ROOT, MaterialParams, load_material
from fsl.eval import plots
from fsl.fdm import FDMConfig, Solution, metrics, solve

IONIZATION = ("full", "photo_only", "photo_only_shared_R")


def _parse_value(s: str) -> Any:
    try:
        return float(s)
    except ValueError:
        return s  # e.g. "tp", "tp/2"


def _parse_kv(items: list[str], multi: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items:
        k, v = it.split("=", 1)
        vals = [_parse_value(x) for x in v.split(",")]
        out[k] = vals if multi else vals[0]
    return out


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def run_one(mat: MaterialParams, cfg: FDMConfig, out_dir: Path, ionization: str,
            shared: Solution | None, overlays: dict[str, Solution]) -> dict[str, Any]:
    t0 = time.time()
    sol = solve(mat, cfg, shared=shared)
    m = metrics(sol, mat, cfg)
    m["ionization"] = ionization
    m["wall_s"] = round(time.time() - t0, 3)

    out_dir.mkdir(parents=True, exist_ok=True)
    sol.save(out_dir / "solution.npz")
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"material": mat.raw, "fdm": {"integrator": cfg.integrator,
                        "dt_fs": sol.meta["dt_fs"], "ne_floor_m3": cfg.ne_floor},
                        "ionization": ionization}, f, sort_keys=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    figs = out_dir / "figs"
    plots.fig3a(sol, mat, figs / "fig3a.png", overlays=overlays)
    plots.fig3b(sol, mat, figs / "fig3b.png")
    plots.fig5(sol, mat, figs / "fig5.png")
    return m


def _print_metrics(m: dict[str, Any]) -> None:
    at = m["center_at_fs"]
    keys = list(at)
    print(f"  τ={m['tau_fs']:g} fs  t_c={m['tc_fs']:g} fs  [{m['ionization']}, {m['integrator']} dt={m['dt_fs']:g}]  {m['wall_s']} s")
    print(f"  t(n_cr)={_fmt(m['t_ncr_fs'])} fs   t90={_fmt(m['t90_fs'])} fs   n_e(25)/n_e(50)={_fmt(m['ne_ratio_25_50'])}"
          f"   max n_e={m['max_ne_cm3']:.3g} cm⁻³   max T_e={m['max_Te_K']:.3g} K")
    print("  " + "   ".join(f"R({k})={at[k]['R']:.3f}" for k in keys))
    print("  " + "   ".join(f"α({k})={at[k]['alpha_m']:.3g}" for k in keys))
    print(f"  width={_fmt(m['width_um'])} µm   depth={_fmt(m['depth_nm'])} nm")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--material", required=True, choices=["glass", "sic", "gan"])
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "fdm.yaml"))
    ap.add_argument("--ionization", default="full", choices=IONIZATION)
    ap.add_argument("--shared_from", help="run dir whose R/α are reused (photo_only_shared_R)")
    ap.add_argument("--overlay", nargs="*", default=[], help="run dirs drawn dashed in fig3a (label=dir or dir)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="material YAML overrides")
    ap.add_argument("--sweep", nargs="*", default=[], metavar="KEY=V1,V2", help="grid of overrides")
    ap.add_argument("--integrator", choices=["euler", "rk4"])
    ap.add_argument("--dt_fs", type=float)
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "fdm"))
    ap.add_argument("--name")
    args = ap.parse_args()

    cfg = FDMConfig.from_yaml(args.config, args.material)
    if args.integrator:
        cfg = cfg.replace(integrator=args.integrator)
    if args.dt_fs:
        cfg = cfg.replace(dt=args.dt_fs * 1e-15)

    base = _parse_kv(args.set, multi=False)
    if args.ionization.startswith("photo_only"):
        base["alpha_i_cm2J"] = 0.0  # GUIDE 2.8: impact term and α_i n_e U₁ both off
    shared = None
    if args.ionization == "photo_only_shared_R":
        if not args.shared_from:
            raise SystemExit("--shared_from <run_dir> is required for photo_only_shared_R")
        shared = Solution.load(Path(args.shared_from) / "solution.npz")
    overlays: dict[str, Solution] = {}
    for ov in args.overlay:
        label, _, d = ov.rpartition("=") if "=" in ov else ("", "", ov)
        overlays[label or Path(d).name] = Solution.load(Path(d) / "solution.npz")

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out_root = Path(args.out)

    if not args.sweep:
        mat = load_material(args.material, **base)
        name = args.name or f"{args.material}_{args.ionization}_tau{mat.tau * 1e15:g}_tc{mat.tc * 1e15:g}"
        run_dir = out_root / f"{name}_{stamp}"
        m = run_one(mat, cfg, run_dir, args.ionization, shared, overlays)
        print(f"→ {run_dir}")
        _print_metrics(m)
        return 0

    grid = _parse_kv(args.sweep, multi=True)
    keys = list(grid)
    sweep_dir = out_root / f"{args.name or args.material + '_sweep'}_{stamp}"
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        ov = dict(base, **dict(zip(keys, combo)))
        mat = load_material(args.material, **ov)
        tag = "_".join(f"{k}{v}" for k, v in zip(keys, combo)).replace("/", "over")
        m = run_one(mat, cfg, sweep_dir / tag, args.ionization, shared, overlays)
        print(f"→ {tag}")
        _print_metrics(m)
        at = m["center_at_fs"]
        tk = list(at)
        row = {k: v for k, v in zip(keys, combo)}
        row.update({"t_ncr_fs": m["t_ncr_fs"], "t90_fs": m["t90_fs"], "ne_ratio_25_50": m["ne_ratio_25_50"],
                    "max_ne_cm3": m["max_ne_cm3"], "max_Te_K": m["max_Te_K"]})
        for k in (tk[0], tk[-1]):
            row[f"R@{k}"] = at[k]["R"]
            row[f"alpha@{k}"] = at[k]["alpha_m"]
        row.update({"width_um": m["width_um"], "depth_nm": m["depth_nm"]})
        rows.append(row)

    cols = list(rows[0])
    with open(sweep_dir / "sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(sweep_dir / "sweep.md", "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(_fmt(r[c]) for c in cols) + " |\n")
    print(f"\n→ {sweep_dir}/sweep.md")
    print(open(sweep_dir / "sweep.md", encoding="utf-8").read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
