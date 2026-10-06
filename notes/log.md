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

## 2026-10-05  phase4/colab smoke — float32 overflow (I-32)
- Setup:   Colab T4, cells 1–5 OK (cuda, deepxde 1.15.0, 47 tests after the .cpu() fix). Cell 7: FDM refs OK, `smoke_gs --set dtype=float32` exit 1 (output swallowed by subprocess.run).
- Result:  reproduced on the Mac (float32, 4×32 PFNN, 2k points): loss_ne = inf at step 0 → NaN. Cause: SI ionization rates (δ_N I³ ≈ 4e40, α_i I n_e ≈ 5e40 m⁻³ s⁻¹) exceed float32 before the t_ref/n_ref scaling. Fixed by folding the scale into the constants (physics.py `scale=`, pde.py), reordering impact_rate (α_i·scale = 1e-44 denormal gave 5 %), folding e²/(m_e ε₀). After the fix: float32 400-iter staged run finite (n raw 0.40→0.22, φ 1.6e-2→6.6e-5, T 3.7e-2→2.0e-3); float32 vs float64 residual terms agree to ≤ 1e-3; 48 tests pass.
- Judge:   float32 path was never exercised before (all Phase 3 smoke runs were float64, I-21). Not a physics change: float64 results identical.
- Next:    push; Colab cell 2 (pull) → cell 7 (now streams output; skips finished steps).

## 2026-10-05  phase4/colab_smoke_gs_20261005-1855 (float32, T4) + glass_full start
- Setup:   Colab T4, I-32 fix in place. `smoke_gs --set dtype=float32` (PFNN 4×32, 5k points, 3 × [1000/500/500]); FDM refs regenerated on Colab (`glass_ref_rk4_20261005-1842`). Then `full` (PFNN 3×{8,64}, 50k points, 5 × [5000/2500/2500]) `glass_full_20261005-1859`.
- Result:  smoke: no NaN, 223 s. Raw residuals n 0.44 → 0.032 (1.1 orders; Phase 3 smoke at τ = 1 fs physics reached 7e-6), φ 23 → 0.93, T 2.4e3 → 1.3e3. evaluate (old format) vs glass_ref_rk4: L2RE n_e 0.42 / 0.89 / 0.94 / 0.93 at 50/100/150/200 fs, T_e 14 / 10 / 10 / 11; PINN max n_e 5.5e21 (FDM 2.0e21), PINN profile 2.6 µm / 553 nm (FDM 6.6 µm / 237 nm) → qualitatively wrong: n_e overshoots n_cr by 2.7× instead of flattening at the R jump.
           full (in progress): 0.27 s/iteration on T4 → 10k iterations (one round) = 44 min, 50k ≈ 3.7 h. Round 0: n 0.42 → 1.3e-2 (best 4900), φ 78 → 1.04, T 1.3e4 → 79. Round 1 n-stage reached 5e-3 by 14.3k.
- Judge:   the calibrated physics is a much harder PINN problem than the Phase 3 placeholder physics: the surface R jumps 0.05 → 0.93 within ~5 fs when n_e crosses n_cr, and a smooth network that lags the jump keeps I high and overshoots (positive feedback n_e → R → I → n_e). A 5 % residual RMS gave a 200 % solution error in the smoke. Wait for the full run (4× lower residual so far); mitigations if the DoD fails: time-curriculum (causal) stages, denser/adaptive collocation near the transition (PDEPointResampler, D18 option), more n-head iterations, lr decay — each a recorded deviation from the paper.
- Next:    full evaluate with the new evaluate.py (fig3 shows the overshoot shape directly); optional Mac float64 smoke + evaluate for an early look.

## 2026-10-05  phase4/glass_full_20261005-1859 — first full run, DoD FAIL
- Setup:   Colab T4, `full` profile (PFNN 3×{8,64}, 50k points, 5 × [5000/2500/2500], float32, I-32 fix). Interrupted at 15k, resumed (`periodic-15000.pt`, resume round 1) and finished: 50,000 iterations, 2670 s per round, wall 1340 + 9334 s ≈ 3.0 h of GPU time (3.7 h incl. the first session). Evaluated with the new evaluate.py vs `glass_ref_rk4_20261005-1842` (Colab copy), checkpoint `best_r1-34900.pt`.
- Result:  raw residuals n 0.42 → 3.2e-4 (3.1 orders; round 3 best 3.4e-4, round 4 3.2e-4), φ 78 → 4.9e-3, T 1.3e4 → 4.4 (best) / 7.9 (last). evaluate: L2RE n_e 0.28 / 0.66 / 0.76 / 0.76 at 50/100/150/200 fs, T_e 4.0 / 16.6 / 17.1 / 17.1; max n_e PINN 1.08e22 (FDM 2.0e21); width None (surface never ≥ n_cr), depth 600 nm (= z_max, r = 0 column above n_cr at the bottom); all 6 DoD checks FAIL. fig3a: PINN surface n_e(r=z=0) goes to −0.6e21 for 0–40 fs, then rises to 1.25e21 at 200 fs, never crossing n_cr; R(PINN) ≈ 0 → 0.08, α(PINN) ≈ 0. fig4b: n_e maximal (1e22) in a 20–60 nm layer below the surface, T_e 2e6 K there; z = 0 row dark.
- Judge:   the network found a cheaper "solution" than the physical one: with the surface n_e low, R stays ≈ 0.05, the intensity stays maximal everywhere and the avalanche term grows n_e in a thin sub-surface layer. The residual on the surface plane is never sampled (zero measure in 3-D), and the negative surface n_e is self-consistent with the impact term (α_i I n_e < 0 → ∂n_e/∂t < 0). Loss curves cannot reveal this; fig3a/fig4b can. Mitigation implemented (I-34): D21 surface anchors (`num_surface`), D10 revised ñ ≥ 0 (`n_positive`), profiles `smoke_surf` (5k + 1k surface, 3 rounds) and `full_surf` (50k + 10k surface, 5 rounds). 53 tests.
- Next:    validate on `smoke_surf` (Mac ~10 min or Colab ~4 min): the PINN surface curve in fig3a must cross n_cr near 47 fs and R(PINN) jump; then `full_surf` on Colab (~4.4 h).

## 2026-10-05  phase4/I-34 checks (Mac CPU, float64, smoke_surf 1 round = 2000 it, PFNN 4×32, 5k + 1k points)
- Setup:   (a) `chk_anchors_only_20261005-2032` (num_surface 1000, n_positive false), (b) `chk_anchors_clamp_20261005-2034` (num_surface 1000, n_positive true = physics sees max(n_e, 0)), (c) `chk_clamp_only` (num_surface 0, n_positive true; running). Each ≈ 170 s. Earlier `mac_smoke_surf_20261005-2020` (softplus output form, 3 rounds) collapsed to ñ ≡ 0 (loss_ne stuck at ⟨S²⟩ = 0.474, max n_e 3.6e5 m⁻³) → softplus form rejected (dead gradient), replaced by the clamp.
- Result:  (a) n 0.475 → 0.085 but evaluate still wrong: max n_e 6.4e21, width None, depth 600 nm, L2RE n_e 1.28 — the surface cheat survives the anchors alone.
           (b) n 0.478 → 0.107, φ 0.82 → 0.079, T 4.2 → 0.55; evaluate: L2RE n_e 0.49 / 0.33 / 0.33 / 0.33, T_e 0.53 / 0.44 / 0.44 / 0.45; max n_e 2.35e21 (FDM 2.0e21); **width 6.65 µm (FDM 6.60, +0.7 %)**, depth 71 nm (FDM 237). fig3a: PINN surface n_e crosses n_cr at ≈ 55 fs (FDM 47) and overshoots to 2.3e21; R(PINN) jumps to 0.97 at ≈ 58 fs; α(PINN) rises (overshoots α(FDM) later). fig4b: surface row correct, n_e decays too fast with depth then rises near z_max (φ head barely trained: 500 it).
- Judge:   the physics clamp (D10 rev.) is the essential fix: with it the surface dynamics are qualitatively right after 2000 iterations of a tiny net; anchors alone are not enough. Depth and the overshoot are training-budget effects (1 round, φ 500 it) to be judged on the 3-round smoke (`mac_smoke_surf2`) and then on `full_surf`.
- Next:    (c) tells whether D21 anchors matter on top of the clamp; user runs `smoke_surf` 3 rounds; then `full_surf` on Colab (cell 7 now defaults to the surf profiles).
- (c) `chk_clamp_only_20261005-2038` (clamp, no anchors): bulk loss lowest of the three (n 0.446 → 0.032, φ 0.072 → 0.026, T 0.19 → 0.052) but the solution is the worst physically: width 9.2 µm (FDM 6.6), **depth 9.9 nm** (a surface-skin solution), max n_e 2.9e21, L2RE n_e 0.83. Without surface anchors the network again buys a small bulk residual with a wrong surface. → **D21 and D10-rev. are both necessary**: anchors pin the surface where R is decided, the clamp removes the negative branch; each alone fails in its own way.

## 2026-10-05  phase4/mac_smoke_surf2 (3 rounds) and chk_phiref3 — I-34 + I-35 on the Mac
- Setup:   `smoke_surf` 3 × [1000/500/500], PFNN 4×32, 5k + 1k surface points, clamp on, float64 CPU (≈ 10 min each). `mac_smoke_surf2_20261005-2036` phi_ref 1; `chk_phiref3_20261005-2050` phi_ref 3.
- Result:  phi_ref 1: n 0.48 → 9.3e-3, φ 0.82 → 0.079 then stuck 0.49 / 0.39, T 4.2 → 1.95. L2RE n_e 0.099 / 0.135 / 0.148 / 0.148, T_e ≈ 1.0; width 6.74 µm (+2 %), **depth 600 nm**, max n_e 2.25e21. diagnose: φ max 0.19 (needs ≈ 3) → φ head at its initial value, no attenuation → n_e ≥ n_cr down to z_max.
           phi_ref 3: φ 0.53 → 0.035 / 0.024 / 0.018, T 2.9 → 0.28; L2RE n_e 0.097 / 0.126 / 0.130 / 0.131, T_e 0.41 / 0.47 / 0.48 / 0.48; width 6.72 µm (+1.8 %), depth 78 nm, max n_e 2.03e21 (FDM 2.00). Depth profile at 200 fs, r = 0: n_e PINN 2.03 / 1.79 / 1.64 / 1.55 / 1.64e21 at z = 0 / 100 / 200 / 400 / 600 nm vs FDM 1.99 / 1.91 / 1.85 / 1.76 / 1.68 — a 10–15 % sag at mid-depth; φ PINN 0.26 / 0.29 / 0.31 / 0.35 vs FDM 0.42 / 0.68 / 0.80 / 0.85 (attenuation still under-estimated, not over).
- Judge:   the surface physics is right with anchors + clamp, φ learns once its output scale fits (I-35), and the remaining errors are capacity/iteration effects of the 4×32 smoke net. The depth metric is the hardest one at this physics (n_e(z) nearly flat, 1 % n_e error ≈ 40 nm of depth). Go to `full_surf` on Colab (50k + 10k points, 8×64, phi_ref 3): expected ≈ 4.5 h on T4.
- Next:    commit (I-34, I-35, D21, D10 rev.), push, Colab cell 2 → cell 7 (defaults to smoke_surf / full_surf).
