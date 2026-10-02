"""Train the forward PINN (GUIDE.md Phase 3 smoke / Phase 4 full).

    python scripts/train_forward.py --config configs/forward_glass.yaml --profile smoke
    python scripts/train_forward.py --config configs/forward_glass.yaml --profile full
    python scripts/train_forward.py --config configs/forward_glass.yaml --profile full --resume outputs/forward/<run>

Writes outputs/forward/<name>_<YYYYmmdd-HHMM>/{config.yaml, history.csv, ckpt/, metrics.json, figs/fig9.png}.
With --resume the restored run's directory is reused (new checkpoints get a _r<k> suffix).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from fsl.config import REPO_ROOT
from fsl.pinn.train import TrainConfig, train  # sets DDE_BACKEND before importing deepxde


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "forward_glass.yaml"))
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--name")
    ap.add_argument("--resume", help="run dir to continue (restores the newest periodic checkpoint)")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "forward"))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="override profile values, e.g. iterations=200 display_every=10 dtype=float32")
    ap.add_argument("--debug", action="store_true", help="torch anomaly detection (find the NaN op); slow")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config, args.profile)
    if args.set:
        d = cfg.__dict__ | {}
        for kv in args.set:
            k, v = kv.split("=", 1)
            if k not in d:
                raise SystemExit(f"unknown train option {k!r}; choices: {sorted(d)}")
            if k == "stages":
                d[k] = v                       # '1000:1,0,0:ne;500:0,0,1:phi' → parsed in from_dict
            elif isinstance(d[k], bool):
                d[k] = v.lower() in ("1", "true", "yes")
            elif isinstance(d[k], tuple):
                d[k] = tuple(float(x) for x in v.split(","))
            else:
                d[k] = type(d[k])(v)
        cfg = TrainConfig.from_dict(d)
    if args.resume:
        run_dir = Path(args.resume)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        run_dir = Path(args.out) / f"{args.name or cfg.material + '_' + cfg.profile}_{stamp}"
    print(f"→ {run_dir}")
    m = train(cfg, run_dir, resume=args.resume, debug=args.debug)
    print(json.dumps({k: m[k] for k in ("device", "dtype", "net", "n_params", "iterations_total", "wall_s",
                                        "stages", "loss_first", "loss_final", "loss_best",
                                        "drop_orders", "drop_orders_ne_raw", "nan")}, indent=2))
    return 1 if m["nan"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
