"""Phase 4.2–4.4: comparison helpers, Table 2 and the PINN-vs-FDM figures (GUIDE.md Phase 4, I-33).

The figure functions take plain arrays, so the FDM solution stands in for the PINN here
(identical fields → zero error → every DoD check passes). The only network call is tested
with an untrained model from ``train.setup``.
"""

from __future__ import annotations

import numpy as np

import fsl.pinn  # noqa: F401  (DDE_BACKEND before deepxde)
from fsl.config import FS, load_material
from fsl.eval import plots
from fsl.eval.compare import (dod_check, fdm_center_series, pinn_center_series, profile_comparison,
                              table2_markdown)
from fsl.eval.metrics import all_metrics
from fsl.fdm import FDMConfig, solve


def _sol():
    mat = load_material("glass")
    return mat, solve(mat, FDMConfig(integrator="euler"))


def test_table2_dod_and_figures_with_identical_fields(tmp_path):
    mat, sol = _sol()
    times = [50e-15, 100e-15, 150e-15, 200e-15]
    at_fs, fields = {}, {}
    for t in times:
        i = sol.it(t)
        at_fs[f"{t / FS:g}"] = {"n_e": all_metrics(sol.n_e[i], sol.n_e[i]), "T_e": all_metrics(sol.T_e[i], sol.T_e[i])}
        fields[f"{t / FS:g}"] = (sol.n_e[i], sol.T_e[i])
    profile = profile_comparison(sol, mat, sol.n_e[-1])
    assert profile["pinn"] == profile["fdm"] and profile["fdm"]["width_um"] is not None

    checks = dod_check(at_fs, profile)
    assert len(checks) == 6 and all(c["ok"] for c in checks)

    paper = plots.paper_ref("glass_tables")["table2"]
    md = table2_markdown(at_fs, paper, label="test")
    assert md.count("\n") == 6 and "| 200 | 0.00e+00 | 2.24e-03 |" in md

    p3 = plots.fig3(sol, mat, tmp_path / "fig3.png", pinn=fdm_center_series(sol), photo=sol)
    p4 = plots.fig4(sol, mat, tmp_path / "fig4.png", 200e-15, sol.n_e[-1], sol.T_e[-1])
    p5 = plots.fig5a(sol, mat, tmp_path / "fig5a.png", sol.n_e[-1])
    p10 = plots.fig10(sol, mat, tmp_path / "fig10.png", fields, z_m=200e-9)
    for p in (p3, p4, p5, p10):
        assert p.exists() and p.stat().st_size > 10_000


def test_dod_check_flags_failures():
    mat, sol = _sol()
    bad = {"n_e": {"l2re": 0.05, "max_rel_pointwise": 0.3, "max_rel_global": 0.1},
           "T_e": {"l2re": 0.05, "max_rel_pointwise": 0.3, "max_rel_global": 0.1}}
    good = {"n_e": {"l2re": 1e-3, "max_rel_pointwise": 0.01, "max_rel_global": 0.01},
            "T_e": {"l2re": 1e-3, "max_rel_pointwise": 0.01, "max_rel_global": 0.01}}
    at_fs = {"50": good, "100": dict(good, n_e=dict(good["n_e"], l2re=2e-3)), "200": bad}   # increasing n_e L2RE, bad final
    profile = profile_comparison(sol, mat, sol.n_e[-1] * 0.5)       # PINN never reaches n_cr → no profile
    checks = {c["name"]: c for c in dod_check(at_fs, profile)}
    assert not checks["L2RE n_e at 200 fs < 0.01"]["ok"]
    assert not checks["width_um within ±10% of FDM"]["ok"]
    assert not checks["L2RE n_e not monotonically increasing in t"]["ok"]
    assert checks["L2RE T_e not monotonically increasing in t"]["ok"]   # 1e-3, 1e-3, 5e-2 is not strictly increasing


def test_pinn_center_series_shapes():
    from fsl.pinn.train import TrainConfig, setup

    mat, sol = _sol()
    cfg = TrainConfig(material="glass", profile="test", layers=2, width=8, num_domain=32, iterations=1,
                      lr=1e-3, loss_weights=(1.0, 1.0, 1.0), display_every=1, ckpt_every=1, dtype="float64",
                      seed=0, distribution="pseudo", activation="silu", initializer="Glorot normal", k=1)
    model, _, sc = setup(cfg, mat)
    s = pinn_center_series(model, sc, mat, sol.t)
    assert s["n_e"].shape == sol.t.shape and np.isfinite(s["R"]).all() and np.isfinite(s["alpha"]).all()
    assert s["n_e"][0] == 0.0 and s["T_e"][0] == 300.0          # hard constraint at t = 0
    assert np.all(s["R"] >= 0) and np.all(s["R"] <= 1)
