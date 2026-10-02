"""Score a Phase 2.9 calibration scan against the GUIDE.md 2.9 acceptance criteria (D3 / D6).

    python scripts/judge_d6.py --full outputs/fdm/D6_scan_<stamp> --photo outputs/fdm/D6_scan_photo_<stamp>

Reads sweep.csv of the full run and of the photoionization-only run (same sweep keys),
joins them by (tp_fs, tc_fs, F_Jcm2, …) and prints one row per combination with the
paper's reference values, the measured values and a pass count. Criteria (all from GUIDE 2.9):

  photo-only  P1 n_e(200 fs) = 1.55e21 ± 15 %     P2 flattening t90 ≤ 70 fs     P3 convex start n_e(25)/n_e(50) < 0.5
  full        F1 n_cr reached at 45–55 fs          F2 n_e(200 fs) in 1.9–2.2e21    F3 R(200 fs) ≥ 0.9
              F4 width within 4–16 µm             F5 depth within 150–1500 nm  (order of magnitude of 8 µm / 450 nm)

Nothing is adopted automatically: the table is the input to the D1/D3/D6 decision (user confirmation).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _read(sweep_dir: Path) -> tuple[list[str], list[dict]]:
    rows = list(csv.DictReader(open(sweep_dir / "sweep.csv", encoding="utf-8")))
    keys = [c for c in rows[0] if c in ("tau_fs", "tc_fs", "tp_fs", "F_Jcm2", "alpha_i_cm2J", "delta_N_cm3ps_cm2TW")]
    return keys, rows


def _key(row, keys):
    return tuple(row[k] for k in keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", required=True)
    ap.add_argument("--photo", required=True)
    ap.add_argument("--photo-target", type=float, default=1.55e21, help="paper Fig.3a plateau [cm⁻³]")
    args = ap.parse_args()

    keys, full = _read(Path(args.full))
    keys_p, photo = _read(Path(args.photo))
    photo_by = {_key(r, keys_p): r for r in photo}

    results = []
    for fr in full:
        pr = photo_by.get(_key(fr, keys))
        ne200_p = _f(pr["max_ne_cm3"]) if pr else None
        t90_p = _f(pr["t90_fs"]) if pr else None
        ratio_p = _f(pr["ne_ratio_25_50"]) if pr else None
        t_ncr = _f(fr["t_ncr_fs"]); ne200 = _f(fr["max_ne_cm3"]); R200 = _f(fr.get("R@200"))
        width = _f(fr["width_um"]); depth = _f(fr["depth_nm"])
        checks = {
            "P1": ne200_p is not None and abs(ne200_p / args.photo_target - 1) <= 0.15,
            "P2": t90_p is not None and t90_p <= 70,
            "P3": ratio_p is not None and ratio_p < 0.5,
            "F1": t_ncr is not None and 45 <= t_ncr <= 55,
            "F2": ne200 is not None and 1.9e21 <= ne200 <= 2.2e21,
            "F3": R200 is not None and R200 >= 0.9,
            "F4": width is not None and 4 <= width <= 16,
            "F5": depth is not None and 150 <= depth <= 1500,
        }
        results.append((sum(checks.values()), fr, pr, checks, dict(ne200_p=ne200_p, t90_p=t90_p, ratio_p=ratio_p,
                                                                   t_ncr=t_ncr, ne200=ne200, R200=R200, width=width, depth=depth)))

    results.sort(key=lambda x: -x[0])
    head = " ".join(f"{k:>8}" for k in keys)
    print(f"{'pass':>4} {head} | {'photo200':>9} {'t90':>5} {'r25/50':>6} | {'t_ncr':>6} {'ne200':>9} {'R200':>5} {'width':>6} {'depth':>6} | checks")
    print(f"{'ref':>4} {' ' * len(head)} | {args.photo_target:9.2e} {'≤70':>5} {'<0.5':>6} | {'45–55':>6} {'1.9–2.2e21':>9} {'≥0.9':>5} {'8':>6} {'450':>6} |")
    for n, fr, pr, checks, v in results:
        kv = " ".join(f"{fr[k]:>8}" for k in keys)
        fmt = lambda x, f: "—" if x is None else format(x, f)
        flags = "".join(k[0] + k[1] if ok else "  " for k, ok in checks.items())
        print(f"{n:>4} {kv} | {fmt(v['ne200_p'], '9.2e')} {fmt(v['t90_p'], '5.0f')} {fmt(v['ratio_p'], '6.2f')} | "
              f"{fmt(v['t_ncr'], '6.1f')} {fmt(v['ne200'], '9.2e')} {fmt(v['R200'], '5.2f')} {fmt(v['width'], '6.2f')} {fmt(v['depth'], '6.0f')} | {flags}")
    print("\nP = photoionization-only criteria, F = full-run criteria (GUIDE 2.9). 8/8 = all criteria met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
