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

## 2026-10-02  phase3/smoke (series)
- Setup:   configs/forward_glass.yaml profiles, Mac CPU float64, glass.yaml as is (τ=1 fs placeholder, t_c=0). FDM reference `outputs/fdm/smoke_ref_20261002-1119` (same YAML).
- Result:
  - `glass_smoke_…1039` (FNN 4×32, 5k pts, λ=1,1,1): NaN at step ≤100 → real-form sqrt in nk_from_eps has an infinite gradient at n_e→0; fixed with complex sqrt (I-27). DeepXDE picks MPS on this Mac → forced CPU (I-26).
  - `glass_smoke_…1045` (same, after fix): no NaN, but loss_ne 5.5e-3 → 5.0e-3 only; loss_Te → 1e-6. ñ ≈ 0 trivial solution (5.6e-3 = ⟨S²⟩ by hand).
  - A `smoke_A_ne_only` (1,0,0): loss_ne 5.5e-3 → 3.5e-5 (2.2 orders) — the n_e equation alone trains fine.
  - B `smoke_B_w100` (100,1,1): loss_ne raw → 1.1e-3 then stalls, loss_phi 3e-4 → 6e-2 (φ starved).
  - D `smoke_D_curriculum` (1,0,0 → 1,0,1 → 1,1,1): stage 2 starts at loss_phi 1679 — untrained φ head went negative (amplification), n_e overshot, then collapsed back to ñ≈0.
  - E `smoke_E_pfnn` (PFNN, 1,1,1): same collapse as the FNN run (0.12 orders).
  - F `smoke_F_gs` (PFNN, head-wise 3×[1000/500/500]): looked like it worked (n raw 8e-5) but diagnose_run showed φ≈200 and I/I₀~1e-89 after the first n-stage — freezing was undone by DeepXDE's `_test()` (`net.requires_grad_()`), so the "n_e-only" stage trained φ and killed the source. Fixed with the HeadFreezer callback (I-29).
  - G `smoke_G_gs_20261002-1124` (same, fixed): round restarts shrink (φ 3.3e-2 → 6e-4 → 3.9e-4; n 3.7e-4 → 1.8e-5); final raw residuals n 7.2e-6, φ 2.5e-4, T 0.136. evaluate vs FDM: L2RE n_e 4.4e-2/3.3e-2/3.5e-2/3.6e-2 at 50/100/150/200 fs, T_e 0.16–0.14; max n_e 8.7e20 vs 9.2e20 cm⁻³; no profile (paper values never reach n_cr, see Phase 2). Wall 490 s on CPU.
- Judge:   DoD PASS on G (no NaN, 2.8 orders on raw loss_ne, L2RE(n_e) < 0.1 recorded). D2, D7, D9, D10, D15, D18 confirmed; D8 changed (head-wise Gauss–Seidel), D20 added (PFNN). T_e error is the T̃≈100 scale of the τ=1 fs physics, expected to vanish at τ=100 fs.
- Next:    Phase 2 close-out (2.9 scan with τ=100 fs, D1/D3/D6, SiC reference) → Phase 4 full run on Colab with the `full` profile (5 rounds × 5000/2500/2500). Add resume for staged runs before Phase 4.
