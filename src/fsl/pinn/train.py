"""Training loop around ``dde.Model`` (GUIDE.md §4.4, §4.7, Phase 3/4; decisions D8, D13).

One forward run = one directory ``outputs/forward/<name>_<stamp>/`` with
    config.yaml   resolved training config + material YAML (for exact reruns / reloads)
    history.csv   step, loss_ne, loss_Te, loss_phi, total   (every ``display_every`` iterations)
    ckpt/         best-<step>.pt (lowest training loss) and periodic-<step>.pt (resume)
    metrics.json  final / best losses, NaN flag, wall time, device, dtype, parameter count
    figs/fig9.png loss curves

Resume (Colab disconnects, §4.6): ``--resume <run_dir>`` restores the newest periodic
checkpoint and trains the remaining iterations; new checkpoints get the prefix
``*_r<k>`` because DeepXDE restarts its step counter after ``restore``. ``ckpt/offsets.json``
records the global iteration at which each resume round started, so a checkpoint's global
step is ``offsets[suffix] + step_in_name``. Staged runs (curriculum / Gauss–Seidel, I-28/I-29)
resume inside the schedule: fully finished stages are skipped and the interrupted stage runs
for its remaining iterations (I-24 addendum). History rows after the restored step are dropped.
"""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import deepxde as dde
import numpy as np
import torch
import yaml

from fsl.config import MaterialParams, load_material, load_yaml, material_from_dict
from fsl.pinn.data import make_data
from fsl.pinn.net import build_net, set_trainable_heads
from fsl.pinn.pde import LOSS_NAMES, make_pde
from fsl.scaling import Scales


# ----------------------------------------------------------------------------- config


@dataclass(frozen=True)
class TrainConfig:
    material: str
    profile: str
    layers: int
    width: int
    num_domain: int
    iterations: int
    lr: float
    loss_weights: tuple[float, float, float]
    display_every: int
    ckpt_every: int
    dtype: str                 # "float64" (smoke on CPU) | "float32" (full on GPU), I-21
    seed: int
    distribution: str          # D18
    activation: str
    initializer: str
    k: int                     # hard-constraint order, D9
    device: str = "auto"       # auto → cuda if available else cpu (never mps, I-26) | cpu | cuda
    net: str = "fnn"           # fnn (paper) | pfnn (independent sub-net per output, I-28)
    # Curriculum (I-28) / block Gauss–Seidel (I-29): stages of (iterations, loss_weights, heads),
    # heads = None (all trainable) or a tuple of 'ne' | 'Te' | 'phi'. The stage list is repeated
    # `rounds` times. When set, `iterations` and `loss_weights` above are the total / last stage's.
    stages: tuple[tuple[int, tuple[float, float, float], tuple[str, ...] | None], ...] = ()
    rounds: int = 1
    phi_positive: bool = True  # φ = z̃·softplus(NN_φ − 2) (I-29); False → paper-like z̃·NN_φ

    @classmethod
    def from_yaml(cls, path: str | Path, profile: str) -> "TrainConfig":
        d = load_yaml(path)
        p = d["profiles"][profile]
        stages = parse_stages(p.get("stages", []))
        rounds = int(p.get("rounds", 1))
        lw = p.get("loss_weights", d.get("loss_weights", [1.0, 1.0, 1.0]))
        if stages:
            lw = stages[-1][1]
        iterations = rounds * sum(s[0] for s in stages) if stages else int(p["iterations"])
        return cls(
            material=str(d["material"]), profile=profile,
            layers=int(p["layers"]), width=int(p["width"]), num_domain=int(p["num_domain"]),
            iterations=iterations, lr=float(p.get("lr", d.get("lr", 1e-3))),
            loss_weights=tuple(float(w) for w in lw),
            display_every=int(p.get("display_every", 100)), ckpt_every=int(p.get("ckpt_every", 1000)),
            dtype=str(p.get("dtype", d.get("dtype", "float32"))),
            seed=int(d.get("seed", 0)), distribution=str(d.get("distribution", "pseudo")),
            activation=str(d.get("activation", "silu")), initializer=str(d.get("initializer", "Glorot normal")),
            k=int(d.get("k", 1)), device=str(p.get("device", d.get("device", "auto"))),
            net=str(p.get("net", d.get("net", "fnn"))), stages=stages, rounds=rounds,
            phi_positive=bool(p.get("phi_positive", d.get("phi_positive", True))),
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        d = dict(d)
        d["loss_weights"] = tuple(float(w) for w in d["loss_weights"])
        d["stages"] = parse_stages(d.get("stages", ()))
        d["rounds"] = int(d.get("rounds", 1))
        if d["stages"]:
            d["iterations"] = d["rounds"] * sum(s[0] for s in d["stages"])
            d["loss_weights"] = d["stages"][-1][1]
        return cls(**d)

    def schedule(self):
        """The stages to run, rounds expanded: list of (round, iterations, loss_weights, heads)."""
        base = self.stages or ((self.iterations, self.loss_weights, None),)
        return [(r, *s) for r in range(self.rounds if self.stages else 1) for s in base]


def parse_stages(value: Any):
    """Stages from YAML dicts ({iterations, loss_weights, heads?}), tuples/lists, or the CLI
    string '1000:1,0,0:ne;500:0,0,1:phi;500:0,1,0:Te' (heads part optional)."""
    if not value:
        return ()
    out = []
    if isinstance(value, str):
        for part in value.split(";"):
            bits = part.split(":")
            heads = tuple(bits[2].split("+")) if len(bits) > 2 and bits[2] else None
            out.append((int(bits[0]), tuple(float(w) for w in bits[1].split(",")), heads))
        return tuple(out)
    for s in value:
        if isinstance(s, dict):
            heads = s.get("heads")
            out.append((int(s["iterations"]), tuple(float(w) for w in s["loss_weights"]),
                        tuple(heads) if heads else None))
        else:
            heads = s[2] if len(s) > 2 else None
            out.append((int(s[0]), tuple(float(w) for w in s[1]), tuple(heads) if heads else None))
    return tuple(out)


_HEAD_BIT = {"ne": 1, "Te": 2, "phi": 4}


def _heads_mask(heads) -> int:
    return 7 if heads is None else sum(_HEAD_BIT[h] for h in heads)


# ----------------------------------------------------------------------------- model setup


def select_device(choice: str = "auto") -> str:
    """Set torch's default device. DeepXDE 1.15 picks MPS on Apple Silicon at import time;
    we never use MPS (no float64, double-backward unverified) — D13 / I-26."""
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    torch.set_default_device(choice)
    return choice


def setup(cfg: TrainConfig, mat: MaterialParams):
    """Device, precision, seed, then net → pde → data → compiled Model. Returns (model, net, scales)."""
    select_device(cfg.device)
    dde.config.set_default_float(cfg.dtype)
    dde.config.set_random_seed(cfg.seed)
    sc = Scales.from_material(mat)
    net = build_net(cfg.layers, cfg.width, cfg.activation, cfg.initializer, cfg.k, cfg.net,
                    cfg.phi_positive)
    pde = make_pde(net, mat, sc)
    data = make_data(pde, sc, cfg.num_domain, cfg.distribution)
    model = dde.Model(data, net)
    model.compile("adam", lr=cfg.lr, loss_weights=list(cfg.schedule()[0][2]))
    return model, net, sc


def device_name() -> str:
    """The device tensors are actually created on (after select_device)."""
    dev = torch.get_default_device().type
    return f"cuda ({torch.cuda.get_device_name(0)})" if dev == "cuda" else dev


# ----------------------------------------------------------------------------- head freezing


class HeadFreezer(dde.callbacks.Callback):
    """Re-apply the per-head freezing before every optimizer step.

    DeepXDE's ``Model._outputs_losses`` (called by ``_test`` at the start of ``train`` and every
    ``display_every`` iterations) does ``net.requires_grad_(False)`` and then
    ``net.requires_grad_()`` — i.e. it re-enables *all* parameters, silently undoing any
    freezing done before ``compile``. Found on 2026-10-02 when the "n_e-only" stage trained the
    φ head too (I-29 addendum). ``on_epoch_begin`` runs right before ``_train_step``.
    """

    def __init__(self, net, heads):
        super().__init__()
        self.net, self.heads = net, heads

    def on_train_begin(self):
        set_trainable_heads(self.net, self.heads)

    def on_epoch_begin(self):
        set_trainable_heads(self.net, self.heads)


# ----------------------------------------------------------------------------- checkpoints


_CKPT_RE = re.compile(r"^(best|periodic)(?:_r(\d+))?-(\d+)\.pt$")


def list_checkpoints(run_dir: str | Path, kind: str) -> list[Path]:
    """Checkpoints of one kind ('best' | 'periodic'), oldest → newest by modification time."""
    ck = Path(run_dir) / "ckpt"
    files = [p for p in ck.glob("*.pt") if (m := _CKPT_RE.match(p.name)) and m.group(1) == kind]
    return sorted(files, key=lambda p: p.stat().st_mtime)


def latest_checkpoint(run_dir: str | Path, kind: str) -> Path | None:
    files = list_checkpoints(run_dir, kind)
    return files[-1] if files else None


def _resume_round(run_dir: Path) -> int:
    rounds = [int(m.group(2)) for p in (run_dir / "ckpt").glob("*.pt")
              if (m := _CKPT_RE.match(p.name)) and m.group(2)]
    return (max(rounds) + 1) if rounds else 1


def _ckpt_parts(path: Path) -> tuple[str, int]:
    """(suffix '' | '_r<k>', step in the file name) of a checkpoint file."""
    m = _CKPT_RE.match(path.name)
    if not m:
        raise ValueError(f"not a checkpoint file name: {path.name}")
    return (f"_r{m.group(2)}" if m.group(2) else ""), int(m.group(3))


def _load_offsets(run_dir: Path) -> dict[str, int]:
    p = Path(run_dir) / "ckpt" / "offsets.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return {k: int(v) for k, v in json.load(f).items()}


def _save_offsets(run_dir: Path, offsets: dict[str, int]) -> None:
    with open(Path(run_dir) / "ckpt" / "offsets.json", "w", encoding="utf-8") as f:
        json.dump(offsets, f, indent=2)


def checkpoint_global_step(run_dir: str | Path, path: Path, hist_last: int | None = None) -> int:
    """Global iteration a checkpoint corresponds to. DeepXDE restarts ``train_state.iteration``
    after every ``restore``, so the number in the file name counts from the start of that resume
    round; ``ckpt/offsets.json`` holds where each round started. Runs recorded before
    offsets.json existed fall back to the last history step."""
    suffix, step = _ckpt_parts(path)
    offsets = _load_offsets(Path(run_dir))
    if suffix in offsets:
        return offsets[suffix] + step
    if suffix == "":
        return step
    return hist_last if hist_last is not None else step


def _truncate_history(path: Path, max_step: int) -> int:
    """Drop history rows after ``max_step`` (iterations that ran after the restored checkpoint
    and are about to be redone). Returns the number of rows dropped."""
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    keep = [r for r in body if int(r[0]) <= max_step]
    dropped = len(body) - len(keep)
    if dropped:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(keep)
    return dropped


def plan_stages(schedule, done: int):
    """Stages still to run after ``done`` iterations of ``TrainConfig.schedule()``:
    list of (stage_index, round, iterations_left, loss_weights, heads). A stage that was
    interrupted keeps its weights/heads and runs only its remaining iterations."""
    out, cum = [], 0
    for si, (rnd_i, it, w, h) in enumerate(schedule):
        start, cum = cum, cum + it
        if cum <= done:
            continue
        out.append((si, rnd_i, cum - max(done, start), w, h))
    return out


# ----------------------------------------------------------------------------- history


def _append_history(path: Path, steps, losses, offset: int, weights=(1.0, 1.0, 1.0), heads_mask: int = 7) -> None:
    """DeepXDE records *weighted* losses; we store them together with the weights (and which
    heads were trainable, bitmask ne=1 Te=2 phi=4) so the raw residual MSE (loss / weight) can be
    recovered and curriculum / Gauss–Seidel stages stay comparable."""
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["step", *LOSS_NAMES, "total", "w_ne", "w_Te", "w_phi", "heads"])
        for s, l in zip(steps, losses):
            if offset and int(s) == 0:
                continue  # the pre-training evaluation of a resumed/continued run duplicates the previous last row
            l = np.asarray(l, dtype=float)
            w.writerow([int(s) + offset, *[f"{v:.6e}" for v in l], f"{l.sum():.6e}",
                        *[f"{x:g}" for x in weights], heads_mask])


def read_history(path: str | Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


# ----------------------------------------------------------------------------- training


def train(cfg: TrainConfig, out_dir: str | Path, resume: str | Path | None = None,
          mat: MaterialParams | None = None, debug: bool = False) -> dict[str, Any]:
    out_dir = Path(out_dir)
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    mat = mat or load_material(cfg.material)
    if debug:  # pinpoints the op that first produces NaN/inf in backward (slow; smoke only)
        torch.autograd.set_detect_anomaly(True)
    model, net, sc = setup(cfg, mat)

    offset, rnd, suffix = 0, 0, ""
    if resume is not None:
        src = Path(resume)
        last = latest_checkpoint(src, "periodic")
        if last is None:
            raise FileNotFoundError(f"no periodic checkpoint to resume in {src / 'ckpt'}")
        hist = read_history(src / "history.csv") if (src / "history.csv").exists() else None
        hist_last = int(hist["step"][-1]) if hist is not None and len(hist["step"]) else None
        offset = checkpoint_global_step(src, last, hist_last)
        rnd = _resume_round(src)
        suffix = f"_r{rnd}"
        dropped = _truncate_history(src / "history.csv", offset)
        model.restore(str(last), device=torch.get_default_device().type, verbose=1)
        print(f"resumed from {last.name} = global iteration {offset} "
              f"(resume round {rnd}; {dropped} history rows after it dropped)")
    offsets = _load_offsets(out_dir)
    offsets[suffix] = offset
    _save_offsets(out_dir, offsets)

    plan = plan_stages(cfg.schedule(), offset)   # staged runs resume inside the schedule (I-24)
    iterations = sum(p[2] for p in plan)

    def make_callbacks(heads=None):
        # Fresh per stage: the 'best' monitor compares weighted totals, which are only comparable
        # within one stage. latest_checkpoint() picks by mtime, so 'best' = best of the last stage.
        cbs = [
            dde.callbacks.ModelCheckpoint(str(out_dir / "ckpt" / f"best{suffix}"),
                                          save_better_only=True, period=cfg.display_every),
            dde.callbacks.ModelCheckpoint(str(out_dir / "ckpt" / f"periodic{suffix}"),
                                          save_better_only=False, period=cfg.ckpt_every),
        ]
        if heads is not None:
            cbs.insert(0, HeadFreezer(net, heads))   # must run before the optimizer step
        return cbs

    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"train": asdict(cfg), "material": mat.raw,
                        "resume_from": None if resume is None else str(resume),
                        "device": device_name()}, f, sort_keys=False)

    t0 = time.time()
    stage_log = []
    if plan:
        # Curriculum (I-28): one compile+train per stage; the model, its step counter and the
        # checkpoints continue across stages (only the Adam state restarts at each compile —
        # and therefore also at a resume, which always lands at or inside a stage).
        n_hist = 0
        for si, rnd_i, st_iters, st_w, st_heads in plan:
            n_train = set_trainable_heads(net, st_heads)   # I-29: freeze the other heads (PFNN)
            model.compile("adam", lr=cfg.lr, loss_weights=list(st_w))
            print(f"--- round {rnd_i} stage {si}: {st_iters} it, weights {st_w}, heads {st_heads or 'all'}, "
                  f"{n_train} trainable params")
            losshistory, train_state = model.train(iterations=st_iters, display_every=cfg.display_every,
                                                   callbacks=make_callbacks(st_heads), disregard_previous_best=True)
            steps, losses = losshistory.steps[n_hist:], losshistory.loss_train[n_hist:]
            # DeepXDE evaluates once before each stage's loop; from stage 2 on that row repeats
            # the previous step number (same weights state, new weights) — keep it, it is informative.
            _append_history(out_dir / "history.csv", steps, losses, offset, st_w, _heads_mask(st_heads))
            n_hist = len(losshistory.steps)
            raw = lambda l: [float(v / w) if w else None for v, w in zip(np.asarray(l, float), st_w)]
            stage_log.append({"round": rnd_i, "stage": si, "iterations": int(st_iters),
                              "loss_weights": list(st_w), "heads": list(st_heads) if st_heads else "all",
                              "raw_first": raw(losses[0]), "raw_last": raw(losses[-1])})
            if not np.isfinite(np.asarray(losses[-1], float)).all():
                break
        set_trainable_heads(net, None)
    wall = time.time() - t0

    hist = read_history(out_dir / "history.csv")
    total = hist["total"]
    final = {k: float(hist[k][-1]) for k in LOSS_NAMES}
    i_best = int(np.nanargmin(total)) if np.isfinite(total).any() else -1
    # drop_orders is judged on the raw (unweighted) n_e residual across the whole run, which stays
    # comparable between curriculum stages; the total is weighted and only meaningful within a stage.
    raw_ne = hist["loss_ne"] / np.where(hist["w_ne"] > 0, hist["w_ne"], np.nan)
    ok = np.isfinite(raw_ne)
    drop_ne = float(np.log10(raw_ne[ok][0] / np.nanmin(raw_ne[ok]))) if ok.any() and np.nanmin(raw_ne[ok]) > 0 else None
    m = {
        "material": mat.name, "profile": cfg.profile, "dtype": cfg.dtype, "device": device_name(),
        "net": cfg.net, "layers": cfg.layers, "width": cfg.width, "n_params": int(net.num_trainable_parameters()),
        "num_domain": cfg.num_domain, "iterations_total": int(hist["step"][-1]),
        "iterations_this_run": int(iterations), "resume_round": rnd, "resume_from_step": int(offset),
        "wall_s": round(wall, 1),
        "loss_weights": list(cfg.loss_weights), "stages": stage_log,
        "loss_final": {**final, "total": float(total[-1])},
        "loss_first": {k: float(hist[k][0]) for k in LOSS_NAMES} | {"total": float(total[0])},
        "loss_best": {"total": float(total[i_best]), "step": int(hist["step"][i_best])} if i_best >= 0 else None,
        "drop_orders": float(np.log10(total[0] / total[i_best])) if i_best >= 0 and total[i_best] > 0 else None,
        "drop_orders_ne_raw": drop_ne,
        "nan": bool(~np.isfinite(total).all()),
        "ckpt_best": (p.name if (p := latest_checkpoint(out_dir, "best")) else None),
        "ckpt_latest": (p.name if (p := latest_checkpoint(out_dir, "periodic")) else None),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)

    from fsl.eval import plots  # matplotlib import kept out of the hot path
    plots.fig9(out_dir / "history.csv", out_dir / "figs" / "fig9.png",
               title=f"{mat.name} {cfg.profile}: {{{cfg.layers},{cfg.width}}}, N={cfg.num_domain}")
    return m


# ----------------------------------------------------------------------------- reload / predict


def load_run(run_dir: str | Path, which: str = "best", ckpt: str | Path | None = None):
    """Rebuild the model of a finished run and restore its 'best' / 'latest' checkpoint, or an
    explicit checkpoint file (``ckpt``, e.g. ckpt/best-1000.pt from an earlier stage).

    Returns (model, mat, scales, cfg)."""
    run_dir = Path(run_dir)
    d = load_yaml(run_dir / "config.yaml")
    cfg = TrainConfig.from_dict(d["train"])
    mat = material_from_dict(d["material"])
    model, _, sc = setup(cfg, mat)
    if ckpt is not None:
        ck = Path(ckpt) if Path(ckpt).is_absolute() or Path(ckpt).exists() else run_dir / "ckpt" / ckpt
    else:
        kind = "best" if which == "best" else "periodic"
        ck = latest_checkpoint(run_dir, kind)
        if ck is None:
            raise FileNotFoundError(f"no {kind} checkpoint in {run_dir / 'ckpt'}")
    model.restore(str(ck), device=torch.get_default_device().type, verbose=1)
    return model, mat, sc, cfg


def predict_fields(model, sc: Scales, r: np.ndarray, z: np.ndarray, t: float):
    """n_e [m⁻³], T_e [K], φ on an (r, z) grid (SI inputs) at time t [s]; shapes (nr, nz)."""
    X = sc.grid_inputs(r, z, t)
    y = model.predict(X)
    n_e = sc.n_from_tilde(y[:, 0]).reshape(len(r), len(z))
    T_e = sc.T_from_tilde(y[:, 1]).reshape(len(r), len(z))
    phi = y[:, 2].reshape(len(r), len(z))
    return n_e, T_e, phi
