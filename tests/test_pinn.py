"""Phase 3: PINN components (GUIDE.md Phase 3 tasks 3.1–3.6; notes/decisions.md I-20 … I-25).

Runs on CPU in float64, a few seconds in total.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import fsl.pinn  # noqa: F401  (sets DDE_BACKEND before deepxde is imported)
import deepxde as dde  # noqa: E402

from fsl import physics as P
from fsl.config import FS, load_material
from fsl.eval.metrics import l2re, max_rel_global, max_rel_pointwise
from fsl.pinn.net import build_net
from fsl.pinn.pde import make_pde, surface_density
from fsl.pinn.train import TrainConfig, load_run, predict_fields, read_history, train
from fsl.scaling import Scales


@pytest.fixture(scope="module", autouse=True)
def _cpu_float64():
    torch.set_default_device("cpu")        # DeepXDE 1.15 defaults to MPS on Apple Silicon (I-26)
    dde.config.set_default_float("float64")
    dde.config.set_random_seed(0)


@pytest.fixture(scope="module")
def glass():
    return load_material("glass")


@pytest.fixture(scope="module")
def sc(glass):
    return Scales.from_material(glass)


def _x(n: int, sc: Scales, seed: int = 1) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    x = rng.random((n, 3))
    x[:, 2] *= sc.t_tilde_max
    return torch.tensor(x, dtype=torch.float64, requires_grad=True)


# ---------------------------------------------------------------- scaling


def test_scaling_roundtrip(glass, sc):
    r, z, t = np.array([-8e-6, 0.0, 3e-6]), np.array([0.0, 0.3e-6, 0.6e-6]), 150e-15
    rt, zt, tt = sc.to_tilde(r, z, t)
    assert rt.min() == 0.0 and rt.max() == pytest.approx(11 / 16) and zt.max() == 1.0 and tt == pytest.approx(1.5)
    r2, z2, t2 = sc.from_tilde(rt, zt, tt)
    np.testing.assert_allclose(r2, r)
    np.testing.assert_allclose(z2, z)
    assert t2 == pytest.approx(t)
    assert sc.n_from_tilde(1.83) == pytest.approx(1.83e27)
    assert sc.T_from_tilde(0.0) == 300.0 and sc.T_to_tilde(10300.0) == pytest.approx(1.0)
    X = sc.grid_inputs(r, z, t)
    assert X.shape == (9, 3) and np.all(X[:, 2] == tt)


# ---------------------------------------------------------------- hard constraints (3.1, D9)


def test_hard_constraints_enforce_ic_and_bc(sc):
    net = build_net(2, 8)
    x = _x(50, sc)
    with torch.no_grad():
        # IC: t̃ = 0 → ñ = T̃ = 0 exactly
        x0 = x.clone(); x0[:, 2] = 0.0
        y = net(x0)
        assert torch.all(y[:, 0] == 0) and torch.all(y[:, 1] == 0)
        # BC: r̃ ∈ {0, 1} → ñ = T̃ = 0 exactly
        for rb in (0.0, 1.0):
            xb = x.clone(); xb[:, 0] = rb
            yb = net(xb)
            assert torch.all(yb[:, 0] == 0) and torch.all(yb[:, 1] == 0)
        # surface: z̃ = 0 → φ = 0 exactly; elsewhere φ ≥ 0 (I-29)
        xs = x.clone(); xs[:, 1] = 0.0
        assert torch.all(net(xs)[:, 2] == 0)
        assert torch.all(net(x)[:, 2] >= 0)
        # interior is generally non-zero
        assert torch.any(net(x)[:, 0] != 0)


def test_pfnn_head_freezing(sc):
    from fsl.pinn.net import head_parameters, set_trainable_heads
    net = build_net(2, 8, kind="pfnn")
    total = sum(p.numel() for p in net.parameters())
    n_ne = set_trainable_heads(net, ("ne",))
    assert n_ne == sum(p.numel() for p in head_parameters(net, 0)) and 0 < n_ne < total
    assert all(not p.requires_grad for p in head_parameters(net, 1))
    assert all(not p.requires_grad for p in head_parameters(net, 2))
    assert set_trainable_heads(net, None) == total
    # the three heads are independent: changing head 2's weights leaves outputs 0 and 1 unchanged
    x = _x(10, sc)
    with torch.no_grad():
        y0 = net(x).clone()
        for p in head_parameters(net, 2):
            p.add_(0.5)
        y1 = net(x)
    assert torch.equal(y0[:, :2], y1[:, :2]) and not torch.equal(y0[:, 2], y1[:, 2])


def test_frozen_heads_do_not_change_during_training(tmp_path, glass):
    """Regression (I-29): DeepXDE re-enables requires_grad on every _test(); HeadFreezer must
    keep the non-trained heads bit-identical through a real model.train() with display_every < iterations."""
    from fsl.pinn.net import head_parameters
    from fsl.pinn.train import parse_stages
    stages = parse_stages("30:1,0,0:ne")
    cfg = TrainConfig(material="glass", profile="t", layers=2, width=8, num_domain=64, iterations=0,
                      lr=1e-3, loss_weights=(1.0, 1.0, 1.0), display_every=10, ckpt_every=10,
                      dtype="float64", seed=0, distribution="pseudo", activation="silu",
                      initializer="Glorot normal", k=1, net="pfnn", stages=stages)
    cfg = TrainConfig.from_dict(cfg.__dict__)
    from fsl.pinn import train as T
    captured = {}
    orig_setup = T.setup

    def spy_setup(c, m):
        model, net, sc = orig_setup(c, m)
        captured["net"] = net
        captured["before"] = {h: [p.detach().clone() for p in head_parameters(net, h)] for h in (0, 1, 2)}
        return model, net, sc

    T.setup = spy_setup
    try:
        m = T.train(cfg, tmp_path / "run", mat=glass)
    finally:
        T.setup = orig_setup
    net = captured["net"]
    after = {h: [p.detach() for p in head_parameters(net, h)] for h in (0, 1, 2)}
    assert all(torch.equal(a, b) for a, b in zip(captured["before"][1], after[1]))   # T head untouched
    assert all(torch.equal(a, b) for a, b in zip(captured["before"][2], after[2]))   # φ head untouched
    assert any(not torch.equal(a, b) for a, b in zip(captured["before"][0], after[0]))  # n head trained
    assert all(p.requires_grad for p in net.parameters())   # restored to all-trainable at the end
    assert not m["nan"]


def test_gauss_seidel_schedule_and_cli_parse():
    from fsl.pinn.train import parse_stages
    s = parse_stages("1000:1,0,0:ne;500:0,0,1:phi;500:0,1,0:Te")
    assert s == ((1000, (1.0, 0.0, 0.0), ("ne",)), (500, (0.0, 0.0, 1.0), ("phi",)), (500, (0.0, 1.0, 0.0), ("Te",)))
    cfg = TrainConfig(material="glass", profile="t", layers=2, width=8, num_domain=64, iterations=0,
                      lr=1e-3, loss_weights=(1.0, 1.0, 1.0), display_every=10, ckpt_every=10,
                      dtype="float64", seed=0, distribution="pseudo", activation="silu",
                      initializer="Glorot normal", k=1, net="pfnn", stages=s, rounds=2)
    cfg = TrainConfig.from_dict(cfg.__dict__)
    assert cfg.iterations == 4000 and cfg.loss_weights == (0.0, 1.0, 0.0)
    sched = cfg.schedule()
    assert len(sched) == 6 and sched[0][0] == 0 and sched[-1][0] == 1 and sched[-1][3] == ("Te",)


# ---------------------------------------------------------------- surface query (3.3)


def test_surface_query_matches_direct_evaluation_and_has_gradient(sc):
    net = build_net(2, 8)
    x = _x(40, sc)
    n_s = surface_density(net, x, sc.n_ref)
    x_direct = x.detach().clone(); x_direct[:, 1] = 0.0
    with torch.no_grad():
        expect = net(x_direct)[:, 0:1] * sc.n_ref
    torch.testing.assert_close(n_s, expect)
    g = torch.autograd.grad(n_s.sum(), x)[0]
    assert torch.all(g[:, 1] == 0)             # the query does not depend on the point's own z̃
    assert torch.any(g[:, 2] != 0) and torch.any(g[:, 0] != 0)   # but it does on t̃ and r̃


# ---------------------------------------------------------------- residuals (3.2)


def test_residuals_shape_and_finite(glass, sc):
    net = build_net(2, 8)
    pde = make_pde(net, glass, sc)
    x = _x(64, sc)
    y = net(x)
    res = pde(x, y)
    assert len(res) == 3
    for r in res:
        assert r.shape == (64, 1) and torch.isfinite(r).all()
    dde.grad.clear()


def test_residuals_finite_in_float32(glass, sc):
    """I-32 (Colab smoke, 2026-10-05): in float32 the SI ionization rates (~4e40 m⁻³ s⁻¹) overflow
    and loss_ne was inf at iteration 0. The nondimensional sources must be finite in float32 and
    agree with float64 to ~1e-4 (points near the pulse peak at r = 0, where I is largest)."""
    from fsl.pinn.pde import residual_terms
    rng = np.random.default_rng(7)
    xs = rng.random((64, 3))
    xs[:, 0] = 0.5 + 0.02 * (xs[:, 0] - 0.5)                      # r ≈ 0
    xs[:, 2] = glass.tc / sc.t_ref + 0.2 * (xs[:, 2] - 0.5)       # t ≈ t_c (peak)
    xs[:, 1] *= 0.05                                              # shallow z → little attenuation
    out, nets = {}, {}
    keys = ("src_photo", "src_impact", "heat", "alpha_h", "R1", "R2", "Rphi")
    try:
        for dt_name, dt in (("float64", torch.float64), ("float32", torch.float32)):
            dde.config.set_default_float(dt_name)
            net = build_net(2, 8)
            if dt_name == "float32":   # same weights as the float64 net, so the two evaluate the same function
                net.load_state_dict({k: v.to(dt) for k, v in nets["float64"].state_dict().items()})
            nets[dt_name] = net
            x = torch.tensor(xs, dtype=dt, requires_grad=True)
            q = residual_terms(net, glass, sc, x, net(x))
            out[dt_name] = {k: q[k].detach().double() for k in keys}
            dde.grad.clear()
    finally:
        dde.config.set_default_float("float64")
    for k, v in out["float32"].items():
        assert torch.isfinite(v).all(), k
    assert out["float32"]["src_photo"].max() > 0.5            # the source is O(1) near the peak, not tiny
    # float32 eps 6e-8 amplified by the cube, exp and the Drude chain → 1e-4-ish; autograd terms a bit worse
    for k in ("src_photo", "src_impact", "heat"):
        torch.testing.assert_close(out["float32"][k], out["float64"][k], rtol=1e-3, atol=1e-6, msg=k)
    torch.testing.assert_close(out["float32"]["alpha_h"], out["float64"]["alpha_h"], rtol=1e-3, atol=1e-2)
    for k in ("R1", "R2", "Rphi"):
        torch.testing.assert_close(out["float32"][k], out["float64"][k], rtol=1e-2, atol=1e-4, msg=k)


def test_residual_matches_manual_formula(glass, sc):
    """R₁ from pde() equals ∂ñ/∂t̃ − t_ref·(α_i I ñ + δ_N I^N / n_ref) computed by hand."""
    net = build_net(2, 8)
    pde = make_pde(net, glass, sc)
    x = _x(32, sc, seed=3)
    y = net(x)
    R1 = pde(x, y)[0]
    dde.grad.clear()
    # manual
    n_t, phi = y[:, 0:1], y[:, 2:3]
    dn_dt = torch.autograd.grad(y[:, 0].sum(), x, create_graph=True)[0][:, 2:3]
    r = sc.r_min + x[:, 0:1] * (sc.r_max - sc.r_min)
    t = x[:, 2:3] * sc.t_ref
    R_s, _ = P.surface_optics(surface_density(net, x, sc.n_ref), glass)
    I = P.intensity(t, r, phi, R_s, glass)
    n_e = n_t * sc.n_ref
    manual = dn_dt - sc.t_ref * (glass.alpha_i * I * n_e + P.photoionization_rate(I, glass)) / sc.n_ref
    torch.testing.assert_close(R1, manual)


# ---------------------------------------------------------------- training loop (3.5, 3.6)


def test_tiny_training_runs_and_reloads(tmp_path, glass):
    cfg = TrainConfig(material="glass", profile="test", layers=2, width=8, num_domain=64, iterations=20,
                      lr=1e-3, loss_weights=(1.0, 1.0, 1.0), display_every=10, ckpt_every=10,
                      dtype="float64", seed=0, distribution="pseudo", activation="silu",
                      initializer="Glorot normal", k=1)
    run = tmp_path / "run"
    m = train(cfg, run, mat=glass)
    assert not m["nan"] and m["iterations_total"] == 20
    assert (run / "history.csv").exists() and (run / "metrics.json").exists() and (run / "figs" / "fig9.png").exists()
    assert m["ckpt_latest"] is not None and m["ckpt_best"] is not None
    # reload + predict on a small grid
    model, mat, sc, cfg2 = load_run(run, "latest")
    assert cfg2.width == 8
    r = np.linspace(sc.r_min, sc.r_max, 5); z = np.linspace(0, sc.z_max, 4)   # r[0] = −8 µm is the PINN boundary
    n_e, T_e, phi = predict_fields(model, sc, r, z, 100e-15)
    assert n_e.shape == (5, 4) and np.isfinite(n_e).all() and np.isfinite(T_e).all()
    assert np.all(T_e[0, :] == 300.0) and np.all(T_e[-1, :] == 300.0)   # r̃ ∈ {0, 1} → T = 300 K exactly
    assert np.all(n_e[0, :] == 0.0) and np.all(n_e[-1, :] == 0.0)       # and n_e = 0
    assert np.all(phi[:, 0] == 0.0)            # z = 0 → φ = 0
    assert np.any(T_e[2, :] != 300.0)          # interior (r = 0) is not pinned
    # resume for 10 more iterations
    cfg3 = TrainConfig.from_dict({**cfg.__dict__, "iterations": 30})
    m2 = train(cfg3, run, resume=run, mat=glass)
    assert m2["iterations_total"] == 30 and m2["resume_round"] == 1 and not m2["nan"]
    assert m2["resume_from_step"] == 20 and m2["iterations_this_run"] == 10


def test_surface_anchors_and_positive_n(glass, sc):
    """D21 / D10 change (I-34): surface anchors land on z̃ = 0 and are part of the training set;
    n_positive keeps ñ ≥ 0 while the IC/BC hard constraints still hold."""
    from fsl.pinn.data import make_data, surface_points
    from fsl.pinn.pde import make_pde

    pts = surface_points(sc, 200, seed=3)
    assert pts.shape == (200, 3) and np.all(pts[:, 1] == 0) and pts[:, 2].max() <= sc.t_tilde_max
    from fsl.pinn.pde import residual_terms

    net = build_net(2, 8)
    data = make_data(make_pde(net, glass, sc, n_positive=True), sc, 300, "pseudo", num_surface=200, seed=3)
    X = data.train_points()
    assert X.shape[0] == 500 and int((X[:, 1] == 0).sum()) >= 200
    x = _x(256, sc)
    y = net(x)
    assert torch.any(y[:, 0] < 0)                       # the untrained paper-form net does go negative
    q_pos = residual_terms(net, glass, sc, x, y, n_positive=True)
    dde.grad.clear()
    q_raw = residual_terms(net, glass, sc, x, y, n_positive=False)
    dde.grad.clear()
    neg = (y[:, 0:1] < 0)
    assert torch.all(q_pos["src_impact"][neg] == 0) and torch.any(q_raw["src_impact"][neg] < 0)
    assert torch.all(q_pos["alpha"] >= 0) and torch.any(q_raw["alpha"] < 0)
    # where ñ < 0 the clamped residual is ∂ñ/∂t̃ − photo: a decaying negative ñ is no longer a solution
    torch.testing.assert_close(q_pos["R1"][neg], (q_pos["dn_dt"] - q_pos["src_photo"])[neg])
    assert torch.all(q_pos["n_t"] == q_raw["n_t"])      # the network output itself is untouched


def test_phi_ref_scales_only_phi(glass, sc):
    """I-35: phi_ref multiplies the optical-depth output only; ñ and T̃ are untouched."""
    net1 = build_net(2, 8, phi_ref=1.0)
    net3 = build_net(2, 8, phi_ref=3.0)
    net3.load_state_dict(net1.state_dict())
    x = _x(64, sc)
    y1, y3 = net1(x), net3(x)
    torch.testing.assert_close(y3[:, 2], 3.0 * y1[:, 2])
    torch.testing.assert_close(y3[:, :2], y1[:, :2])
    assert torch.all(y3[:, 2] >= 0)


def test_tiny_training_with_surface_anchors(tmp_path, glass):
    cfg = TrainConfig(material="glass", profile="test", layers=2, width=8, num_domain=64, iterations=10,
                      lr=1e-3, loss_weights=(1.0, 0.0, 0.0), display_every=5, ckpt_every=5,
                      dtype="float64", seed=0, distribution="pseudo", activation="silu",
                      initializer="Glorot normal", k=1, num_surface=32, n_positive=True)
    m = train(cfg, tmp_path / "run", mat=glass)
    assert not m["nan"] and m["iterations_total"] == 10
    model, _, sc2, cfg2 = load_run(tmp_path / "run", "latest")
    assert cfg2.num_surface == 32 and cfg2.n_positive is True
    assert model.data.train_x_all.shape[0] == 64 + 32


def test_staged_resume_continues_inside_the_schedule(tmp_path, glass):
    """I-24 addendum: a Gauss–Seidel run interrupted mid-stage resumes that stage for its remaining
    iterations, then runs the rest of the schedule; a finished run can be extended by raising rounds."""
    stages = ((10, (1.0, 0.0, 0.0), ("ne",)), (5, (0.0, 0.0, 1.0), ("phi",)), (5, (0.0, 1.0, 0.0), ("Te",)))
    cfg = TrainConfig(material="glass", profile="test", layers=2, width=8, num_domain=64, iterations=40,
                      lr=1e-3, loss_weights=(0.0, 1.0, 0.0), display_every=5, ckpt_every=5,
                      dtype="float64", seed=0, distribution="pseudo", activation="silu",
                      initializer="Glorot normal", k=1, net="pfnn", stages=stages, rounds=2)
    run = tmp_path / "run"
    m = train(cfg, run, mat=glass)
    assert m["iterations_total"] == 40 and len(m["stages"]) == 6 and not m["nan"]
    # emulate a crash at global iteration 25 = inside the n_e stage of round 1: drop later checkpoints
    for p in (run / "ckpt").glob("periodic-*.pt"):
        if int(p.stem.split("-")[1]) > 25:
            p.unlink()
    m2 = train(cfg, run, resume=run, mat=glass)
    assert m2["resume_from_step"] == 25 and m2["iterations_this_run"] == 15 and m2["iterations_total"] == 40
    assert [(s["round"], s["heads"], s["iterations"]) for s in m2["stages"]] == \
        [(1, ["ne"], 5), (1, ["phi"], 5), (1, ["Te"], 5)]
    hist = read_history(run / "history.csv")
    assert np.all(np.diff(hist["step"]) >= 0) and hist["step"][-1] == 40   # rows after 25 were dropped, then redone
    assert (run / "ckpt" / "offsets.json").exists()
    # extend a finished run by one more round
    cfg3 = TrainConfig.from_dict({**cfg.__dict__, "rounds": 3})
    m3 = train(cfg3, run, resume=run, mat=glass)
    assert m3["resume_round"] == 2 and m3["resume_from_step"] == 40 and m3["iterations_total"] == 60
    assert [s["round"] for s in m3["stages"]] == [2, 2, 2]
    model, _, _, _ = load_run(run, "latest")   # newest periodic = periodic_r2-20.pt
    assert model is not None


# ---------------------------------------------------------------- metrics (3.8)


def test_metrics_definitions():
    ref = np.array([1.0, 2.0, 4.0]); nn = np.array([1.1, 2.0, 3.6])
    assert l2re(nn, ref) == pytest.approx(np.sqrt((0.01 + 0.16) / 21))
    assert max_rel_pointwise(nn, ref) == pytest.approx(0.1)
    assert max_rel_global(nn, ref) == pytest.approx(0.4 / 4)
    assert max_rel_pointwise(np.array([1.0, 0.0]), np.array([1.0, 1e-9])) == pytest.approx(0.0)  # tiny ref masked
