"""Approximator network + hard constraints (GUIDE.md §4.3, §4.7; decisions D2, D9, D10; I-28, I-29).

Inputs  x = (r̃, z̃, t̃)         nondimensional, see ``scaling.py``
Outputs y = (ñ, T̃, φ)           after the output transform:

    ñ = t̃^k · r̃(1 − r̃) · NN_n           → ñ = 0 at t̃ = 0 (IC) and at r̃ ∈ {0, 1} (BC)
    T̃ = t̃^k · r̃(1 − r̃) · NN_T           → T_e = 300 K at t̃ = 0 and r̃ ∈ {0, 1}
    φ = z̃ · softplus(NN_φ − 2)          → φ = 0 at the surface (D2) and φ ≥ 0 everywhere
                                           (optical depth cannot be negative: a negative φ
                                           amplifies I with depth and blew n_e up, I-29).
                                           softplus(−2) ≈ 0.13 keeps the initial attenuation small.
      (phi_positive=False restores the paper-like unconstrained φ = z̃ · NN_φ)

k = 1 as in the paper ("T(t) = t", D9). No positivity constraint on ñ (D10).

Two network kinds (I-28):
    fnn  — one fully-connected net, hidden layers shared by the 3 outputs (the paper's net)
    pfnn — three independent sub-networks (``dde.nn.PFNN``); needed for head-wise
           (block Gauss–Seidel) training, where only one output's parameters are trainable
           at a time (I-29).
"""

from __future__ import annotations

import deepxde as dde
import torch

HEADS = {"ne": 0, "Te": 1, "phi": 2}   # output index of each head


def hard_constraint_transform(x: torch.Tensor, y: torch.Tensor, k: int = 1,
                              phi_positive: bool = True, phi_ref: float = 1.0) -> torch.Tensor:
    r, z, t = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    g = (t**k) * r * (1.0 - r)
    # ñ stays the paper's linear form g·NN_n. A softplus form g·softplus(NN_n) was tried on
    # 2026-10-05 (I-34) and collapsed to ñ ≡ 0: pushing NN_n → −∞ kills the gradient (dead
    # zone), so the network never recovers where the source needs ñ > 0. Positivity is instead
    # enforced in the physics (pde.py, n_positive: the physics sees max(n_e, 0)).
    phi_raw = y[:, 2:3]
    # phi_ref (I-35): output scale of the optical depth. With the calibrated physics z_max·α reaches
    # 3–5 above n_cr, but softplus(NN_φ − 2) starts at 0.13 and Adam moves the output by ≲ lr per
    # step, so the φ head stayed at ≈ 0.13 z̃ in the smoke runs (I-34 checks). phi_ref = 1 keeps D2.
    phi = z * phi_ref * (torch.nn.functional.softplus(phi_raw - 2.0) if phi_positive else phi_raw)
    return torch.cat([g * y[:, 0:1], g * y[:, 1:2], phi], dim=1)


def build_net(layers: int, width: int, activation: str = "silu",
              initializer: str = "Glorot normal", k: int = 1, kind: str = "fnn",
              phi_positive: bool = True, phi_ref: float = 1.0):
    """Network 3 → [width]×layers → 3 with the hard-constraint output transform applied.
    ``dde.config.set_default_float`` must be called *before* this (layer dtype)."""
    if kind == "fnn":
        net = dde.nn.FNN([3] + [width] * layers + [3], activation, initializer)
    elif kind == "pfnn":
        net = dde.nn.PFNN([3] + [[width] * 3] * layers + [3], activation, initializer)
    else:
        raise ValueError(f"unknown net kind {kind!r} (fnn | pfnn)")
    net.apply_output_transform(lambda x, y: hard_constraint_transform(x, y, k, phi_positive, phi_ref))
    return net


def head_parameters(net, head: int) -> list[torch.nn.Parameter]:
    """Parameters that belong exclusively to output `head` of a PFNN (parallel layers only)."""
    if not isinstance(net, dde.nn.PFNN):
        raise TypeError("head_parameters needs a PFNN; an FNN shares all parameters")
    return [p for layer in net.layers if isinstance(layer, torch.nn.ModuleList)
            for p in layer[head].parameters()]


def set_trainable_heads(net, heads) -> int:
    """Make only the given heads ('ne' | 'Te' | 'phi', or None = all) trainable.
    Shared layers (if any) stay trainable. Returns the number of trainable parameters."""
    if heads is None:
        for p in net.parameters():
            p.requires_grad_(True)
    else:
        if not isinstance(net, dde.nn.PFNN):
            raise TypeError("head-wise training needs net: pfnn")
        wanted = {HEADS[h] if isinstance(h, str) else int(h) for h in heads}
        for layer in net.layers:
            if isinstance(layer, torch.nn.ModuleList):
                for j, sub in enumerate(layer):
                    for p in sub.parameters():
                        p.requires_grad_(j in wanted)
            else:
                for p in layer.parameters():
                    p.requires_grad_(True)
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
