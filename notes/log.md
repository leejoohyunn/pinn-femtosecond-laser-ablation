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

## 2026-10-02  phase2/D6-scans (2.9) → D1·D3·D6 confirmed
- Setup:   τ = 100 fs fixed coarse scan `outputs/fdm/D6_scan_20261002-1320` + `D6_scan_photo_20261002-1320` (t_p,eff {73,100,150,200} × t_c {0,35,tp/2} × F {3.6,7.2}, 24+24 runs); fine scan `D6_fine_20261002-1411` + `D6_fine_photo_20261002-1411` (t_p,eff {90,100,110,120} × t_c {25,35,45} × τ {50,100,200}, F = 3.6, 36+36 runs). Euler Δt = 1 fs, Mac CPU, 0.02 s/run. Scored with `scripts/judge_d6.py` (8 criteria of GUIDE 2.9; new metrics t90_fs, ne_ratio_25_50 in fdm.metrics).
- Result:  coarse: paper-as-written 0/8; only (100, 35, 3.6) 8/8; all F = 7.2 rows ≤ 6/8 (flatten too late / cross n_cr too late). Fine: two 8/8 families for every τ — (90, t_c = t_p/2) and (100, 35). (90, t_p/2, τ 100): photo plateau 1.66e21, t90 67 fs, ratio 0.29, n_cr 46.9 fs, n_e(200) 2.00e21, R(200) 0.97, width 6.58 µm, depth 251 nm, max T_e 7.5e4 K, α(50/200) 2.2/5.0e6 /m. τ moves only T_e (1.4e5/7.5e4/3.8e4 K), α and depth (271/251/210 nm).
- Judge:   D1 = 100 fs confirmed (paper p.8; n_e side τ-independent; τ = 200 would match T_e 3.5e4 K — sensitivity note). D3 = t_p/2 confirmed/changed ([27] convention, no extra parameter). D6 = t_p,eff 90 fs, F unchanged, confirmed/changed. Rejected: (100, 35) ad-hoc t_c; F = 7.2; t_c = 0; τ = 200. Remaining gaps vs paper: α ×2–3, T_e ×2.1, depth 0.56, width 0.8 — one root (constant-τ Drude k above n_cr). Full record GUIDE §3.3.
- Next:    YAML defaults → calibrated set (paper values in comments; tests pin the paper set via overrides). 2.10: `scripts/check_dt.py --material glass`, `glass_ref` + `glass_ref_photo`, `sic_ref`. Then Phase 4 on Colab.

## 2026-10-02  phase2/2.10 + references → Phase 2 DoD PASS
- Setup:   YAML defaults = calibrated set (glass tp 90, tc tp/2, tau 100; sic/gan tau 100, tc tp/2). `scripts/check_dt.py --material glass|sic` (Euler 1 / 0.5 fs, RK4 1 / 0.1 fs, last = reference); `run_fdm.py` glass_ref, glass_ref_photo, sic_ref (Euler 1 fs). 46 tests pass (paper-set hand calculations pinned via overrides).
- Result:  glass Euler-1 vs RK4-0.1 (r=z=0, 200 fs): n_e 1.9e-3, T_e 8.1e-3; L2RE final field 6.1e-3 / 1.3e-2; depth 5.9 % (251 vs 237 nm); Euler-0.5 halves everything; RK4-1 ≤ 8e-4 everywhere. SiC: 6.0e-4 / 5.6e-3, depth 2.1 % (490 vs 480 nm). `outputs/fdm/dt_check_glass_20261002-1459`, `dt_check_sic_20261002-1459`.
           glass_ref_20261002-1459: t_ncr 46.89 fs, t90 46 fs, n_e(200) 2.00e21, R 0.936→0.973, α 2.15e6→5.01e6 /m, max T_e 7.47e4 K, width 6.585 µm, depth 251.4 nm. glass_ref_photo_20261002-1459: plateau 1.66e21, t90 66.9 fs, n_e(25)/n_e(50) 0.29.
           sic_ref_20261002-1500 (paper t_p 285, τ 100, t_c 142.5, D5 borrowed): t_ncr 130.6 fs, max n_e 1.20e21, R(200/285) 0.964/0.971, α(285) 4.65e6, width 2.676 µm, depth 490.3 nm, max T_e 1.82e5 K (paper 2.4 µm / 290 nm / 0.95e4 K).
- Judge:   **Phase 2 DoD PASS** — Δt < 1 % at r=z=0 (both materials), calibration table complete, D1·D3·D6 confirmed, glass profile within order of magnitude (documented gaps §3.3), SiC reference saved. Reading: SiC needs no D6-type t_p correction (profile +11 % / ×1.7 with the paper's t_p); its T_e ×19 points at D5. Euler reference error (0.6 % / 1.3 % L2RE, 6 % depth) is a big share of the Phase 4 DoD budget → I-31 proposed (RK4 Δt = 1 fs references).
- Next:    user decision on I-31 (then `--integrator rk4` reruns of the three references), commit + push, `--resume` for staged runs (I-24), Phase 4 full on Colab.

## 2026-10-05  phase2/RK4 references (I-31) + staged resume test (I-24)
- Setup:   `configs/fdm.yaml` integrator rk4 (Δt 1 fs). `run_fdm.py` glass_ref_rk4, glass_ref_rk4_photo, sic_ref_rk4 (Mac CPU, 0.05 s each). pytest 47 pass (new `test_staged_resume_continues_inside_the_schedule`).
- Result:  glass_ref_rk4_20261005-1330: t_ncr 46.94 fs, t90 45.5 fs, n_e(200) 1.99e21, R 0.931→0.973, α 2.02e6→4.95e6 /m, max T_e 7.52e4 K, width 6.604 µm, depth 237.3 nm (Euler 251.4). glass_ref_rk4_photo_20261005-1330: plateau 1.66e21, t90 66.5 fs, ratio 0.296. sic_ref_rk4_20261005-1330: t_ncr 130.3 fs, max n_e 1.20e21, R(285) 0.971, α(285) 4.64e6, width 2.692 µm, depth 480.1 nm, max T_e 1.83e5 K.
- Judge:   matches check_dt's rk4 columns exactly; these are the Phase 4/6 evaluation references (GUIDE §3.3, §4.8). Resume mechanics verified on a mid-stage crash emulation.
- Next:    Phase 4 on Colab — regenerate glass_ref_rk4(+photo) there, smoke_gs (float32) as a GPU sanity check, then `full` (5 rounds × 5000/2500/2500), evaluate vs glass_ref_rk4.
