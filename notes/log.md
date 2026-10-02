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

## 2026-09-26  phase2/fdm-as-written
- Setup:   paper equations + Table 1 central values, τ=1 fs, t_c=0, Euler Δt=1 fs (Mac CPU, 0.02 s/run). Sweep τ×t_c (12 runs): outputs/fdm/glass_sweep_20260926-1501
- Result:  r=z=0 @200 fs: photo-only 2.1e20 cm⁻³ (paper 1.55e21, ×7.4 short), full 9.2e20 (paper 2.07e21, ×2.2 short). n_cr never reached in any of the 12 combos → R ≤ 0.03, no ablation profile. max n_e independent of τ; t_c=t_p worse (4.7e20). Euler vs RK4 < 1% (test passes).
- Judge:   Code verified (35 tests, analytic photo-only integral within 0.5%). D6 mismatch is real and in the paper's stated equations/values. Memo: notes/phase2_d6_memo.md — I must be ~1.95× larger; candidates H-A (prefactor ×2) and H-B (effective t_p ≈ 73 fs, also explains the 50 fs flattening).
- Next:    user decides on the H-A / H-B scan; then τ re-scan, H1 check, SiC reference.
