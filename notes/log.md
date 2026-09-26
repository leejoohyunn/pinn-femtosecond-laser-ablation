# Experiment log

Format (GUIDE.md §0 ⑦, §6): one entry per run. Newest at the bottom.

```
## YYYY-MM-DD  <phase>/<run_name>
- Setup:   configs/xxx.yaml (profile=...), device (Mac CPU / Colab <GPU>), wall time
- Result:  key numbers from metrics.json, figure paths
- Judge:   DoD pass / fail, what was learned
- Next:    decisions affected (D-ids), follow-up
```

---

## 2026-09-25  phase0/skeleton
- Setup:   package skeleton, configs/materials/{glass,sic,gan}.yaml, `scripts/check_env.py`
- Result:  Mac (conda .venv, py 3.11.16, torch 2.14.0, deepxde 1.15.0, cpu): glass n_cr 1.8324e27 m⁻³ (0.13% off), 7 tests pass.
           Colab (Tesla T4, 2026-09-26): clone via Secret token OK, Drive symlink OK, check_env OK, 7 tests pass.
           Fix on the way: PyYAML reads `1.0e21` as a string → config.py casts every scalar with float().
- Judge:   DoD PASS (both machines load config, n_cr = 1.83e27, Colab has CUDA). First push to GitHub done.
- Next:    Phase 1 (physics.py + tests/test_physics.py)

## 2026-09-26  phase1/physics
- Setup:   src/fsl/physics.py + tests/test_physics.py (Mac CPU, float64 for parity tests)
- Result:  21 tests pass (7 config + 14 physics). D1 hand table, D6 numbers (I₀ 1.69e13 W/cm², δ₃I₀³ 3.39e39 m⁻³s⁻¹, photo-only 50 fs ≈ 1.4e20 cm⁻³) reproduced by code.
- Judge:   DoD PASS. Pushed.
- Next:    Phase 2 (fdm.py + run_fdm.py, τ × t_c scan, D6 memo)
