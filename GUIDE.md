# Femtosecond Laser Ablation PINN — Reproduction Guidebook

> Target paper: Y. Gao et al., *PINN-based computational framework for femtosecond laser ablation: multi-material modeling and parameter estimation*, **Optics & Laser Technology 199 (2026) 115023**
> Paper PDF: `./PINN-based computational framework for femtosecond laser ablation.pdf`
> This document is the **single source of truth** for the implementation. If the code and this document disagree, fix this document first, then bring the code in line.

---

## 0. Workflow (Protocol)

Every Phase **must** follow the steps below.

```
① Review task pack ─ Re-read this Phase's section and settle the open decisions (§3)
② Confirm          ─ No code is written before the user says OK
③ Implement        ─ Claude writes the code (following §4)
④ Run              ─ The user runs it on their Mac or on Google Colab (Claude does not run it)
⑤ Report           ─ Share the metrics.json summary + figure paths in the chat (§6)
⑥ Judge            ─ Check the Phase's Definition of Done (DoD) → move on only if it passes
⑦ Record           ─ Decisions/changes go to the §3 decision log; experiment results go to notes/log.md
```

- Work on **one Phase at a time**, with one exception (agreed 2026-10-02): **Phase 3 (PINN smoke) may start while Phase 2 is still open**, because its DoD (pipeline runs, loss drops, no NaN) does not depend on calibrated physics. **Phase 4 (full training) does not start until Phase 2's DoD passes** — the physics must be frozen before spending GPU hours. No other Phase skips the gate.
- Every deviation from the paper is recorded in the §3 decision log **with its reason**.
- Implementation-level choices (code structure, tooling, why a function has the signature it has) go to `notes/decisions.md` (I-1, I-2, …), not to §3. §3 is only for how the paper's gaps and contradictions were resolved.

---

## 1. Goals and Scope

### 1.1 Reproduction targets (by priority)

| Priority | Part | Material | Outputs to reproduce |
|---|---|---|---|
| ★★★ | Reference FDM | Borosilicate glass | FDM curves corresponding to Fig.3 (time evolution, including the photoionization-only curve), ablation profile width/depth. Note: Fig.3 in the paper is itself a PINN result. We first build the same curves with FDM and use them for calibration |
| ★★★ | Basic training (forward problem) | Borosilicate glass | Fig.3, 4, 5a, 9, 10, Table 2 |
| ★★ | Transfer learning | SiC | Fig.5b, 6, 7, 11, 12, Table 3 |
| ★★ | Inverse solving | GaN | Fig.8, 13 (inverse estimation of fluence F) |
| ★ | Architecture search | Borosilicate glass | Compare 5 settings of hidden layers/neurons {L,n} (L2 error vs training time) |

### 1.2 Paper target values (reference)

| Item | Paper value |
|---|---|
| Glass n_cr (780 nm) | 1.83×10²¹ cm⁻³ |
| Glass max n_e / max T_e | 1.95×10²¹ cm⁻³ / 3.5×10⁴ K (text and Fig.4 colorbar). ⚠️ The r=z=0 curve in Fig.3a rises to about **2.07×10²¹** at 200 fs. This is likely not a real contradiction: the Fig.4a colorbar tops out at 1.95 and the center of the map is saturated (uniform dark red), so 1.95e21 is probably the colorbar cap rather than the true maximum. Record both and compare |
| Glass Fig.3a (r=z=0) | Photo+impact ionization: about 1.83e21 (= n_cr) at 50 fs, about 2.07e21 at 200 fs. **Photoionization only** (α_i=0): flat at about 1.55e21 from around 50 fs |
| Glass z=200 nm slice (Fig.10, t=200 fs) | max n_e about 1.9e21, max T_e about 2.85e4 K |
| Glass time to reach n_e = n_cr (r=z=0) | ≈ 45–50 fs |
| Glass R (r=z=0, Fig.3b) | Jumps from 0 at about 50 fs (text: "to 0.9"; in the figure it goes straight past 0.9 to about 0.93–0.95), about 0.97 at 200 fs |
| Glass α (r=z=0, Fig.3b) | About 0.7e6 /m at 50 fs, rising to about 2.4e6 /m at 200 fs. The text says "approximately linearly", but the curve in the figure is concave (steep at first, then flattening). The "cm⁻¹" in the text is a typo; the axis unit /m is correct |
| Glass ablation profile | Width 8 µm, depth 450 nm |
| Glass L2RE (Table 2) | t=50/100/150/200 fs → n_e 1.03e-2 / 3.6e-3 / 2.20e-3 / 2.24e-3, T_e 7.41e-2 / 1.81e-2 / 8.33e-3 / 4.69e-3 |
| Glass max relative error (Fig.4c/f, t=200 fs) | n_e 2.5%, T_e 9% (presumed pointwise relative error, §4.5). The text literally says "maximum **absolute** error for free electron temperature is 9%", but the Fig.4f colorbar is dimensionless (max 0.09), so this is treated as a typo for relative error |
| SiC ablation profile | Width 2.4 µm, depth ≈ 0.29 µm (text says "about 300 nm") |
| SiC L2RE (Table 3) | t=50/100/200/285 fs → basic n_e 2.49e-2 / 1.44e-2 / 3.04e-3 / 2.35e-3, transfer n_e 2.45e-2 / 1.39e-2 / 2.89e-3 / 2.16e-3; basic T_e 3.86e-2 / 2.13e-2 / 5.64e-3 / 4.61e-3, transfer T_e 3.67e-2 / 1.83e-2 / 1.06e-2 / 3.09e-3. ⚠️ The text claims "the T_e error of transfer is smaller than basic at all times", but **at t=200 fs transfer (1.06e-2) is larger than basic (5.64e-3)** |
| SiC max relative error (Fig.6·7, t=285 fs) | basic n_e 1.1% / T_e 2.5%, transfer (25k) n_e 1.2% / T_e 1.8%. The paper does not say which basic checkpoint Fig.6·7 show; **60k is our inference** (Table 3 uses 60k) |
| SiC z=200 nm max relative error of T_e (Fig.12) | basic 25k 9.6%, basic 60k 2.6%, transfer 25k below 1% (n_e is at most 1.2% for all three) |
| Architecture search (Sec.3.3) | {8,64} chosen. Training time {8,64} about 3,886 s vs {8,128} about 9,035 s, with negligible accuracy gain |
| GaN ablation profile (Fig.8a) | Width 1.2 µm, depth 0.22 µm, 197 experimental data points. Note: the 1.2 µm arrow in Fig.8a is drawn about 40 nm below the surface, not at z=0. With the §2.6 definition (width at z=0) the same profile gives about 1.25 µm |
| GaN estimated F | ≈ 0.6 J/cm² (converges from both F₀=1 and F₀=10) |
| GaN F convergence (Fig.8b) | Stable at about 7,000 epochs for F₀=1 and about 35,000 epochs for F₀=10 |
| Final loss levels | Glass total: text says about 1e-2, but ⚠️ **the Fig.9 curve ends at about 7e-2** (dominated by Loss_Te). SiC at 25k: basic about 1e-2 (text and Fig.11 agree); transfer: text says about 1e-3, but ⚠️ **Fig.11b/c show about 3e-3** (Fig.11b also jumps to about 1e-2 at the last point). GaN data loss about 1e-4, total about 1e-2 (Fig.13, text and figure agree). Use these only as order-of-magnitude references, since the loss scale depends on our own n_ref/T_ref (D15) |
| SiC max n_e / max T_e (t=285 fs) | About 1.07e21 cm⁻³ (top of the Fig.6a/c colorbar; only slightly above n_cr = 1.05e21) / about 0.95e4 K (Fig.7a/c colorbar). Not stated in the text; read from the figures |
| **Our glass FDM at the calibrated set (§3.3), RK4 Δt = 1 fs (I-31) — the reference the PINN is judged against** | photo-only plateau 1.66e21; n_cr at 46.9 fs; n_e(200 fs) 1.99e21; R(50/200 fs) 0.94/0.97; α(50/200 fs) 2.2/5.0e6 /m; max T_e 7.5e4 K; width 6.6 µm; depth 237 nm (`outputs/fdm/glass_ref_rk4_*`, photo-only `glass_ref_rk4_photo_*`; the Euler run `glass_ref_20261002-1459` gives depth 251 nm, otherwise the same) |

> Note on loss figures: the **captions of Fig.9 and Fig.11 swap the colors of Loss_ne/Loss_Te relative to the legend**. When reproducing, follow the legend (Loss_ne magenta, Loss_Te orange), not the caption.

> **Caution:** Some paper parameters are missing and some order-of-magnitude checks do not match (§3 D1, D6). Therefore **the primary criterion is "PINN error relative to our own FDM"**. Paper values are secondary references only.

---

## 2. Physical Model (reference equations for implementation)

Paper typos are written here **in corrected form**.

### 2.1 Governing equations

$$\frac{\partial n_e}{\partial t} = \alpha_i\, I\, n_e + \delta_N I^N \tag{2.1}$$

$$c_e\, n_e \frac{\partial T_e}{\partial t} = \alpha_h I \tag{2.2}$$

> **Typos in the paper's residual equations (follow governing equations (2.1)(2.2))**
> - n_e residual: Eq. (3.1) reads `∂n_e/∂t − α_i − δ_N I^N`, **missing both I and n_e**. L_pde1 in Fig.2 reads `∂n_e/∂t − α_i I − δ_N I^N`, **missing only n_e**.
> - T_e residual: both Eq. (3.2) and L_pde2 in Fig.2 read `c_e ∂T_e/∂t − α_h I`, **missing n_e**. This form is dimensionally inconsistent (LHS in W, RHS in W/m³). Page 10 of the text also says "n_e enters the calculation of T", so this is clearly a typo. Follow Eq. (2.2): `c_e n_e ∂T_e/∂t = α_h I`.
> - IC residual: Eq. (3.6) reads `R_IC2 = I(x,y,t; n_e) − 300`; the variable should be **T_e** (the 300 K initial value belongs to T_e, cf. Eq. 3.4). Irrelevant to the implementation because IC/BC are hard constraints, recorded for completeness.

### 2.2 Laser intensity

$$I(t,r,z) = \frac{2F}{\sqrt{\pi/\ln 2}\; t_p}\,\bigl(1-R(t,r)\bigr)\exp\!\Bigl(-\frac{r^2}{r_0^2} - 4\ln 2\,\Bigl(\frac{t-t_c}{t_p}\Bigr)^2 - \varphi(t,r,z)\Bigr)$$

$$\varphi(t,r,z) = \int_0^z \alpha(t,r,z')\,dz'$$

- Pulse center `t_c = t_p/2` (decision D3, confirmed 2026-10-02): the pulse peaks in the middle of the window, as in ref. [27]. The paper's equation has t_c = 0, which gives a linear rise at t = 0 and never reaches n_cr with the paper's values; the Fig.3a curve starts convex.
- Pulse duration for glass: `t_p = 90 fs` (decision D6, confirmed 2026-10-02) in place of Table 1's 200 fs, with F = 3.6 J/cm² unchanged. This is a reproduction calibration (§3.3): with 200 fs the equations as written give n_e(200 fs) ≈ 2.1e20 cm⁻³ photo-only, 7.4× below Fig.3a.
- R is computed from **n_e at the surface z=0**.
- **Caution on the definition of F (factor of 2):** `F = 2E_pulse/(π r₀²)` in Sec.4.3 is the peak-fluence formula for a beam with spatial profile exp(−2r²/r₀²). Eq. (2.5), however, uses exp(−r²/r₀²). This implementation **takes Eq. (2.5) as the reference** and treats F only as "the parameter that goes into Eq. (2.5)". If F ever needs to be converted to pulse energy, note separately that `E = F·π r₀²/2` for exp(−r²/r₀²) can differ from the paper's formula by a factor of 2.

### 2.3 Optical property chain (Drude)

```
ω    = 2πc/λ
ω_p² = n_e e² / (m_e ε₀)
τ    = 1/(ν_ei + ν_ep)                       ← decision D1 (paper text: function of n_e, T_e / implementation: constant 100 fs, paper p.8)
ε_r  = 1 − ω_p² τ² / (1 + ω² τ²)
ε_i  = ω_p² τ / (ω (1 + ω² τ²))
n    = sqrt((ε_r + |ε|)/2),  k = sqrt((−ε_r + |ε|)/2),   |ε| = sqrt(ε_r² + ε_i²)
R    = ((n−1)² + k²) / ((n+1)² + k²)
α_h  = 2 ω k / c
α    = α_h + α_i · n_e · U₁
c_e  = 3/2 · k_B
n_cr = 4π² c² m_e ε₀ / (λ² e²)
```

### 2.4 Material parameters

| Material | λ (nm) | t_p (fs) | r₀ (µm) | F (J/cm²) | U₁ (eV) | α_i (cm²/J) | δ_N | N |
|---|---|---|---|---|---|---|---|---|
| Borosilicate glass | 780 | 200 → **90 (D6, calibrated)** | 5 | 3.6 | 4.0 | 1.2 (±0.4) | 7×10^(17±0.5) cm⁻³ps⁻¹(cm²/TW)³ | 3 |
| SiC | 1030 | 285 | 2 | 6.0 | 3.26 | **D5** | **D5** | 3 |
| GaN | 1030 | 200 | 1 | 0.6 (inverse target) | 3.4 | **D5** | **D5** | 3 |

Model-wide parameters not in Table 1: **τ = 100 fs constant (D1)**, **t_c = t_p/2 (D3)** for every material. The glass t_p calibration (D6) is material-specific and is not carried over to SiC/GaN; whether SiC needs the same correction is checked against Fig.5b / Table 3 in Phase 6. `configs/materials/*.yaml` hold the calibrated values with the paper's values in comments.

- Why N=3: the paper gives N=3 only for the glass set (4 eV / 1.59 eV). For SiC·GaN the paper gives no N; **N=3 is our inference** from U₁ / photon energy (3.26 or 3.4 eV / 1.20 eV → three-photon absorption), handled together with D5.
- The set α_i=4, δ₆=6e8, N=6 also listed in the paper is the **fused silica** value in the original source (Lenzner 1998). It is not used in this implementation.
- For glass α_i and δ₃, **the central values (1.2, 7e17) are the default**. The uncertainty ranges in the paper (α_i 0.8–1.6, δ₃ 2.2e17–2.2e18) are used only as the candidate range for D6 calibration.

### 2.5 Initial and boundary conditions

- At t=0: n_e=0, T_e=300 K.
- At r=±8 µm (PINN domain boundary): n_e=0, T_e=300 K. Enforced by hard constraints.
- No boundary condition in z, because the PDEs contain no spatial derivatives.

### 2.5.1 Computational domains (paper Sec.2.2 / Sec.4)

| Material | FDM (reference solution) | PINN training domain | Evaluation/figure domain |
|---|---|---|---|
| Borosilicate glass | r ∈ [−5, 5] µm, z ∈ [0, 0.6] µm, t ∈ [0, 200] fs | r ∈ [−8, 8] µm, z ∈ [0, 0.6] µm, t ∈ [0, 200] fs | FDM domain |
| SiC | r ∈ [−2, 2] µm, z ∈ [0, 0.5] µm, t ∈ [0, 285] fs | r ∈ [−8, 8] µm, z ∈ [0, 0.6] µm, t ∈ [0, 285] fs (**D16**) | FDM domain |
| GaN | (for 7.1 synthetic check) r ∈ [−1, 1] µm, z ∈ [0, 1] µm, t ∈ [0, 200] fs | r ∈ [−1, 1] µm, z ∈ [0, 1] µm, t ∈ [0, 200] fs | PINN domain |

- Grid: Δr = 0.25 µm, Δz = 0.02 µm, Δt = 1 fs (paper). GaN FDM uses Δr = 0.05 µm — **our choice, not the paper's** (D19).
- The PDEs have no r or z derivatives, so the FDM solution is independent at each point. A narrower FDM domain than the PINN domain is therefore fine for comparison. **The φ integral accumulates from z = 0, so the z grid must start at 0.**

### 2.6 Ablation profile definition

The contour **n_e(r,z) = n_cr** at the final time is the ablation boundary. Width is the length of the r interval where n_e ≥ n_cr at z=0; depth is the maximum z where n_e ≥ n_cr at r=0.

---

## 3. Decision Log

Status: `proposed` → `confirmed` / `changed`. **Items in `proposed` status must be confirmed before the corresponding Phase starts.**

### 3.1 Decisions at a glance (what we chose and why)

One line per decision; the full reasoning, evidence and alternatives are in the §3.2 table and, for the Phase 2 calibration, in §3.3. "changed" means the adopted value differs from what the paper states or implies.

| ID | Topic | What we chose | Why (one line) | Status |
|---|---|---|---|---|
| D1 | Drude relaxation time τ | **Constant 100 fs** | The only numeric τ in the paper (p.8); the paper's own τ(n_e, T_e) references [27][31] give 0.07–0.4 fs → R ≤ 0.05, which cannot reproduce Fig.3b; every n_e metric is independent of τ ∈ {50, 100, 200} fs (2.9 scan), and 100 fs is also the inverse-problem value (D14), so the physics stays identical through Phase 7 | **confirmed** 2026-10-02 |
| D2 | Nonlocal ∫α dz | **Auxiliary output φ**, φ = z̃·softplus(NN_φ − 2) ≥ 0, residual ∂φ/∂z − α | A PINN needs a pointwise residual; an unconstrained φ went negative in the smoke runs (amplification with depth) and blew n_e up | **confirmed** 2026-10-02 |
| D3 | Pulse center t_c | **t_p/2** | Convention of ref. [27] (window centered on the peak); gives the convex start of Fig.3a and n_cr at 45–55 fs. t_c = 0 (paper equation) gives a linear start and never reaches n_cr | **confirmed (changed)** 2026-10-02 |
| D4 | Framework | **DeepXDE 1.15 + PyTorch** | Same as the paper; user's choice. Only non-standard piece: surface query of the net inside the PDE | **confirmed** |
| D5 | α_i, δ_N for SiC/GaN | **Borrow glass values** (first pass) | Paper gives none. Reference check: GaN δ₃ ≈ 0.011 cm³/GW² (Cai group) ≈ 1.9e22 in paper units, 27,000× glass; SiC at 1030 nm is 4-photon in Ali et al.; the Phase 2 SiC reference with the borrowed values gives max T_e 19× the paper's while the profile is within 11 % / 2× (§3.3) → D5 will be revised in Phase 6 | proposed |
| D6 | Order-of-magnitude mismatch | **t_p,eff = 90 fs, F = 3.6 unchanged** (I₀ ×2.2) | Paper values give 7.4× too little photoionization; the integral needs I ×~2. Only t_p,eff ≈ 90–100 fs with F = 3.6 satisfies all 8 acceptance criteria; F = 7.2 J/cm² fails the 50 fs timing | **confirmed (changed)** 2026-10-02 |
| D7 | Nondimensionalization | r̃, z̃ ∈ [0,1], t̃ = t/100 fs, ñ = n_e/1e21, T̃ = (T−300)/1e4 | The paper's "0 ≤ t ≤ 2 fs" only makes sense as 2 × 100 fs; O(1) outputs for training | **confirmed** 2026-10-02 |
| D8 | Loss weights λ | **Head-wise block Gauss–Seidel**: stages (1,0,0) n → (0,0,1) φ → (0,1,0) T, repeated `rounds` times, other heads frozen | Every joint run with static weights collapsed to ñ → 0 (R₂ and R_φ are ∝ ñ and vanish for free); n_e-only training drops 2 orders in 2k iterations | **confirmed (changed)** 2026-10-02 |
| D9 | Hard-constraint order k | **k = 1** | Paper p.5: "T(t) = t" | **confirmed** 2026-10-02 |
| D10 | Positivity of n_e | **No constraint** | As the paper; |ε_i| in the Drude chain (I-27) makes negative transients harmless (smoke: min ñ −2e-3) | **confirmed** 2026-10-02 |
| D11 | Inverse-problem data | Digitize Fig.8a (197 points) | Data only "available on request" | proposed |
| D12 | Parameterization of F | F = exp(logF) | Guarantees F > 0 | proposed |
| D13 | Execution environment | Mac CPU for smoke, **Colab GPU** for full runs | No local GPU; Apple MPS has no float64 and DeepXDE picks it by default (forced to CPU, I-26) | **confirmed** |
| D14 | Inverse problem τ, t_data | τ = 100 fs, t_data = 200 fs | Paper p.8 verbatim ("τ = 100fs") and Eq. (3.16) under the D7 reading | proposed |
| D15 | Output scales n_ref, T_ref | **1e21 cm⁻³, 1e4 K shared by all materials** | Paper Sec.5.2: same scaling across materials for transfer learning; at the calibrated physics max T̃ ≈ 7.5 | **confirmed** 2026-10-02 |
| D16 | SiC PINN time domain | t ≤ 285 fs (t̃ ≤ 2.85) | Results are reported at 285 fs although Sec.4 writes 200 fs | proposed |
| D17 | GaN input normalization | Same r̃, z̃ rule as other materials | Paper's "no transformation needed" contradicts its own domain; one code path | proposed |
| D18 | Collocation sampling | `pseudo`, 50,000 points, **no resampling** | Paper: "uniform sampling, 50,000 points"; resampling not stated | **confirmed** 2026-10-02 |
| D19 | GaN FDM grid | Δr = 0.05 µm | Paper's 0.25 µm gives 9 points across ±1 µm | proposed |
| D20 | Network architecture | **PFNN: 3 independent sub-nets {8,64}** (ñ, T̃, φ) | Head-wise training (D8) needs separable parameters; a single shared FNN collapsed in every joint smoke run | **confirmed** 2026-10-02 |

### 3.2 Full log

| ID | Item | Paper status | Proposal (default) | Decide at | Status |
|---|---|---|---|---|---|
| D1 | Collision relaxation time τ | No values for ν_ei, ν_ep. The paper describes τ as **a function of n_e and T_e** (Sec.2.1). **However, page 8 (Sec.4.3) fixes "τ = 100fs"** to simplify the equations for the inverse problem (see D14) — the only numeric τ in the paper | Constant parameter `tau_fs`. **Scan {1, 5, 10, 20, 50, 100} fs in Phase 2** and adopt the value closest to the Fig.3b R curve (about 0.9–0.95 right after 50 fs, about 0.97 at 200 fs). **τ = 100 fs is the leading candidate**: it is the paper's own value, and by hand calculation it gives R = 0.977 at 1.13 n_cr, matching Fig.3b at 200 fs. ⚠️ Structural change: with constant τ, T_e does not affect n_e, R, or α, so the coupling becomes **one-way n_e → T_e** (two-way in the paper's Sec.2.1 description). By hand calculation, τ = 1–5 fs gives R ≤ 0.64 and cannot reach Fig.3b; R≈0.9 needs τ ≳ 20 fs (table below). ⚠️ **α cannot be matched at the same time**: at 1.13 n_cr every τ in the table gives α_h ≈ 5.8–6.9e6 /m (α_i n_e U₁ adds only ≈1.6e5 /m), while Fig.3b shows α ≈ 2.4e6 /m at 200 fs — about 2.4× lower. At n_e = n_cr, τ = 100 fs gives α_h ≈ 0.73e6 /m, which does match the 0.7e6 at 50 fs. So with constant τ, R and α at 200 fs cannot both be reproduced; R stays the primary criterion (2.6) and the α gap is recorded, not tuned away. A τ(n_e, T_e) model is kept as an option. **Update 2026-09-25 (refs [27] Jiang & Tsai 2005 and [31] Lee & More 1984 read in full):** the paper's τ(n_e, T_e) is [27] Eqs. (18)–(22): Spitzer τ_ei = T_ev^{3/2} / (3×10⁻⁶ ln Λ n_e Z) (n_e in cm⁻³, T_ev = mean kinetic energy per electron in eV), ln Λ = ½ ln(1 + (b_max/b_min)²) with the [31] floor ln Λ ≥ 2, and 1/ν_ep = (M/m_e)^{1/2} (ħ/U_IP)(T_D/T_l)(n_e/n_cr)^{1/3} (fused silica: U_IP = 13.6 eV, T_D = 290 K, T_l = 300 K). At the paper's glass conditions (n_e ≤ 1.13 n_cr, T_e ≤ 3.5e4 K) this gives τ_e = 0.07–0.4 fs and R ≤ 0.05, so the reference formulas cannot reproduce Fig.3b; R ≈ 0.9 at 1.13 n_cr needs τ ≈ 100 fs. **Revised proposal: default `tau_fs: 100` (constant); the Phase 2 scan becomes a sensitivity check over {20, 50, 100, 200} fs; Eqs. (18)–(22) implemented as option `tau_model: jiang_tsai` for one comparison curve (U_IP, T_D of borosilicate unknown → fused-silica placeholders).** FDM 2026-09-26 confirmed that τ has no effect until n_e crosses n_cr (D6 first). Details: Notion Decision Log, "D1 수정 제안". **Decision 2026-10-02 (2.9 scans, §3.3): τ = 100 fs constant, confirmed.** In the fine scan every n_e-side criterion (plateau, flattening time, convex start, n_cr crossing, n_e(200 fs), R(200 fs), width) is identical for τ ∈ {50, 100, 200} fs; τ only moves max T_e (1.4e5 / 7.5e4 / 3.8e4 K), α(200 fs) (6.3 / 5.0 / 4.0e6 /m) and depth (271 / 251 / 210 nm). τ = 200 fs would reproduce the paper's max T_e (3.5e4 K) but is a second departure from the text and leaves the α and depth gaps in place, so it is recorded as a sensitivity note only. The `tau_model: jiang_tsai` option is dropped from the plan (the reference formulas are 2–3 orders of magnitude away) | Phase 2 | **confirmed** 2026-10-02 |
| D2 | Nonlocal ∫α dz term | Treatment not described | Add an **auxiliary output φ**. Residual ∂φ/∂z − α = 0, hard constraint **φ = z̃·softplus(NN_φ − 2) ≥ 0** (optical depth cannot be negative; an unconstrained φ let the untrained head amplify I with depth and blew n_e up in the smoke runs, I-29). Numerical integration along z kept as an alternative option | Phase 3 | **confirmed** 2026-10-02 |
| D3 | Pulse center t_c | t=0 per the equation. However, **the photoionization-only curve in Fig.3a has near-zero slope at t=0 and steepens (convex shape)**. With t_c=0 the intensity peaks at t=0, so the curve should rise linearly right away — **the equation and the figure disagree**. In addition, the photoionization-only curve **goes flat abruptly at about 50 fs**, which no t_c explains: with t_c=0 the photoionization rate at 100 fs is still 12% of its peak, and with t_c=t_p the pulse is still rising at 50 fs. Both curves in Fig.3a bend at the same time (~48 fs), exactly when the full run reaches n_cr and R jumps. **Hypothesis H1: the paper computed the photoionization-only curve with R (and α) taken from the full run**, not from its own n_e (its own n_e stays below n_cr, so its own R would stay small) | Default is `t_c = 0`, as in the paper's equation. Options `t_c = t_p` and `t_c = t_p/2` are also provided. Decided from the shape of the photoionization-only curve in Phase 2.8, where H1 is also tested. **Update 2026-09-25:** [27] centers its computational window on the pulse peak (Fig. 2–3: 0–50 fs window, peak at 25 fs = t_p/2) while writing the same exp(−4 ln2 (t/t_p)²); if Gao Fig.3a follows that convention the peak sits at 100 fs = t_p/2, which explains the convex start. **Proposed new default `t_c = t_p/2`**, to be confirmed by the Phase 2.9 scan. [27] Fig. 2 also shows the photoionization-only curve flattening when the full run reaches n_cr, so H1 is a trait of this model family. **FDM 2026-09-26:** with the paper's values neither t_c = 0 nor t_c = t_p reaches n_cr (max n_e 9.2e20 / 4.7e20 cm⁻³); t_c = t_p/2 not yet run. **Decision 2026-10-02 (2.9 scans, §3.3): t_c = t_p/2, confirmed (changed from the paper's t_c = 0).** With τ = 100 fs and t_p,eff = 90 fs, t_c = t_p/2 = 45 fs gives photo-only n_e(25)/n_e(50) = 0.29 (convex), flattening at 67 fs, and n_cr at 46.9 fs — all three Fig.3a shape criteria — using the ref. [27] convention with no extra free parameter. t_c = 0 fails the convex start in every combination (ratio ≥ 0.50) and with the paper's t_p never reaches n_cr; the ad-hoc t_c = 35 fs also scores 8/8 (with t_p,eff = 100 fs) but has no basis in a reference. H1 (photo-only curve flattens because R jumps in the full run) was not needed: with the calibrated values the photo-only curve flattens on its own because the pulse is over by ~90 fs after the peak | Phase 2 | **confirmed (changed)** 2026-10-02 |
| D4 | Framework | DeepXDE + PyTorch | **DeepXDE with the PyTorch backend**, same as the paper. Mapping of each paper component to DeepXDE is in §4.7. The only non-standard piece is the surface query for R, done by calling the network inside the PDE function | Phase 0 | **confirmed** |
| D5 | α_i, δ_N for SiC·GaN | No values | First pass borrows the glass values (same N=3). In Phase 6, decide whether to replace them after a literature search. **First sources to check are the paper's own Table 1 references**: [33] M. Yan et al., APL 125, 242110 (2024) for SiC, and [34] X. Cai et al., Comput. Mater. Sci. 214, 111627 (2022) for GaN — [34] is a simulation of fs irradiation of GaN and probably states α_i, δ_N | Phase 6 | proposed |
| D6 | Order-of-magnitude mismatch | 1.5e21 at 50 fs from photoionization alone (hand calculation gives ~1.4e20: I₀ = 16.9 TW/cm², δ₃I₀³ = 3.39e21 cm⁻³ps⁻¹, ∫₀^50fs exp(−12 ln2 (t/t_p)²) dt = 42.6 fs, R≈0, t_c=0) | In Phase 2, compute with the paper's equations as written → if they disagree, check candidate causes one by one (units, definition of I, τ, t_c, δ₃ uncertainty, D3 hypothesis H1) and report. Note: raising δ₃ to its upper bound 7×10^17.5 ≈ 2.2e18 still gives only about 4.5e20 at 50 fs (plateau about 6.6e20 for t→∞), so **the uncertainty alone cannot explain 1.5e21**. **Calibrated values are adopted only after user confirmation**. **FDM result 2026-09-26 (notes/phase2_d6_memo.md):** with the paper's equations and Table 1 values, photo-only n_e(200 fs) = 2.1e20 (paper 1.55e21, ×7.4 short), full = 9.2e20 (paper 2.07e21), n_cr never reached, no ablation profile. The photoionization integral matches the analytic value within 0.5%, so this is the equations/values, not the code. ∫δ₃I³dt must be ×7.4 → I ×1.95. Tested at τ = 1 fs: H-A (I₀×2, F = 7.2 J/cm²) matches the plateau value but flattens at ~100 fs; H-B (t_p,eff = 73 fs) matches value and the ~50 fs flattening. Both overshoot n_e in the full run (8.4e21 / 4.2e21) because τ = 1 fs keeps R low — **D6 cannot be settled independently of D1**; under t_c = t_p/2 the gap widens to ~30× (photo-only ≈ 5e19). Next: Phase 2.9 scan with τ = 100 fs fixed. **Decision 2026-10-02 (2.9 scans, §3.3): t_p,eff = 90 fs with F = 3.6 J/cm² unchanged, confirmed (changed from Table 1's 200 fs).** Coarse scan (`outputs/fdm/D6_scan_20261002-1320`, 24 combos): the paper-as-written set scores 0/8; the only 8/8 is t_p 100 / t_c 35 / F 3.6. Fine scan (`outputs/fdm/D6_fine_20261002-1411`, 36 combos): two 8/8 families, (t_p 90, t_c = t_p/2) and (t_p 100, t_c 35), for every τ. Adopted (90, t_p/2) because it needs only one non-paper assumption (D3 is a reference convention), has n_cr at 46.9 fs (center of the 45–55 window), photo-only plateau 1.66e21 (+7 %), n_e(200 fs) 2.00e21, R(200 fs) 0.97, width 6.6 µm, depth 251 nm. F = 7.2 J/cm² (H-A) fails the ≤ 70 fs flattening and the 45–55 fs crossing at every t_p; t_p,eff = 73 fs (H-B) overshoots the plateau. Reading: with F fixed, halving t_p doubles I₀ — the ×1.95 the memo derived — consistent with the authors using a t_p that is not the FWHM of Eq. (2.5) | Phase 2 | **confirmed (changed)** 2026-10-02 |
| D7 | Nondimensionalization | Only spatial inputs normalized to [0,1] (Eq. 3.17); **t is kept as t' = t**, and the paper literally writes the domain as `0 ≤ t ≤ 2fs` (Eq. 3.13, Sec.4) and the data time as `t = 2fs` (Eq. 3.16). Taken literally, 2 fs contradicts every reported time (50–285 fs); **our interpretation** is that the value 2 is in units of 100 fs (2 → 200 fs = t_p) and the "fs" label is a typo. Output scaling not described | Use the §4.3 scales (ñ = n_e/n_ref, T̃ = (T−300)/T_ref, t̃ = t/100 fs). n_ref and T_ref follow D15 | Phase 3 | **confirmed** 2026-10-02 |
| D8 | Loss weights λ | No values (Eq. 3.14 has λ_pde1, λ_pde2; the paper trains all equations jointly) | **Changed (Phase 3 smoke, 2026-10-02): static weights do not work in our scaling — every joint run collapsed to the trivial solution.** (1,1,1): n_e → 0 because R₂ and R_φ are ∝ ñ and vanish for free (loss_ne stuck at 5.5e-3 = ⟨S²⟩ of the source alone); (100,1,1): φ starved, loss_ne stuck at 1e-3; n_e-only (1,0,0): 2.2 orders in 2k iterations. **Adopted: block Gauss–Seidel over heads** — PFNN (D20), stages [n head on R₁ with (1,0,0)] → [φ head on R_φ with (0,0,1)] → [T head on R₂ with (0,1,0)], repeated `rounds` times, other heads frozen in each stage (`HeadFreezer`, re-applied every iteration because DeepXDE's `_test` re-enables all parameters). Smoke (4×32, 5k points, 3 rounds × 1000/500/500): restart losses shrink every round (φ 3.3e-2 → 6e-4 → 3.9e-4; n 3.7e-4 → 1.8e-5), final raw residuals n 7e-6, φ 2.5e-4, T 0.14; L2RE(n_e, 200 fs) vs FDM = 3.6e-2. Full run: 5 rounds × (5000/2500/2500) = 50k iterations (I-28, I-29) | Phase 3 | **confirmed (changed)** 2026-10-02 |
| D9 | Hard-constraint order k | **Stated in the paper**: page 5, "the transition function T(t) = t can satisfy", i.e., k=1. Eq. (3.13) `t*[(x−a)(x−b)] + I` has a **typo: the network output û_NN is not multiplied in**. The intended form is `t·(x−a)(x−b)·û_NN + I` | k=1 (same as the paper) | Phase 3 | **confirmed** 2026-10-02 |
| D10 | Positivity of n_e | Not mentioned | No constraint on ñ, as in the paper. Negative transients are harmless since the Drude chain uses |ε_i| (I-27); at the end of the smoke run the most negative ñ was −2e-3 (max 0.83). softplus kept as an option | Phase 3 | **confirmed** 2026-10-02 |
| D11 | Inverse-problem data | "Available on request" | Digitize Fig.8a with WebPlotDigitizer to approximate the 197 points → `data/gan_profile.csv` | Phase 7 | proposed |
| D12 | Parameterization of F (inverse) | F learned directly | `F = exp(logF)` to guarantee positivity. Plots show F | Phase 7 | proposed |
| D13 | Execution environment | GPU assumed | Smoke configs on the Mac (CPU); full configs on **Google Colab GPU** (CUDA, float32). Code is written on the Mac and moved to Colab via a git repository (§4.6) | Phase 0 | **confirmed** |
| D14 | Inverse problem: τ and data time t_data | Page 8 (Sec.4.3), verbatim: "We need to appropriately simplify the equations to consider **τ = 100fs**, allowing us to perform inverse calculation of the laser fluence based solely on the free electron density equation." The symbol is **τ (relaxation time, Eq. 2.9), not t**. Reading: fixing τ as a constant makes R and α functions of n_e only, so the n_e equation closes on its own and the T_e equation can be dropped — which is exactly the "based solely on the n_e equation" in the same sentence. The data time is given only by Eq. (3.16): `t = 2fs` → 200 fs (= t_p) under the D7 interpretation. So there is **no conflict** between the text and Eq. (3.16) | **τ = 100 fs** for the GaN inverse problem (paper value; ties in with D1). **t_data = 200 fs (= t_p, Eq. 3.16)**, which is also the final state after the pulse. Low-priority alternative reading (the "τ" is a typo for t, i.e., t_data = 100 fs) is kept only as an option for the 7.1 synthetic check | Phase 7 | proposed |
| D15 | Output scales n_ref, T_ref (across materials) | Sec.5.2: "for the sake of facilitating transfer learning, the **scaling factors for the boundary conditions of silicon carbide and borosilicate glass were set to be the same**". Sec.5.1 likewise mentions "scaling coefficients in the boundary condition constraints". The paper cites this as the reason for the large share of the T loss in SiC. The paper ties the scaling to the boundary-condition (hard-constraint) term; **reading it as output scaling is our interpretation** (in practice the scale sits in the same output transform). Sec.3.2 gives the magnitudes: n_e 0 → 1e21, T_e 1e2 → 1e4 | **For reproduction, use fixed scales shared by all materials**: n_ref = 1e21 cm⁻³, T_ref = 1e4 K (glass, SiC, GaN). Per-material `n_ref = n_cr` is kept as an option; if time permits, compare its effect on transfer learning in Phase 6. Note from the smoke run: with the uncalibrated τ = 1 fs physics T_e reaches ~1e6 K, so T̃ ≈ 100 and the T head converges slowly (L2RE(T_e) 0.14); with the calibrated physics of §3.3 (τ = 100 fs, t_p,eff = 90 fs) the FDM gives max T_e ≈ 7.5e4 K, i.e. T̃ ≈ 7.5 — the O(1–10) range T_ref = 1e4 K was chosen for | Phase 3 | **confirmed** 2026-10-02 |
| D16 | SiC PINN time domain | **Internal inconsistency**: Sec.4 gives `0 ≤ t ≤ 2fs` (= 200 fs under the D7 interpretation) for both glass and SiC, but SiC results are reported at t = 285 fs (= t_p) | SiC uses **t̃ ∈ [0, 2.85]** (t ≤ 285 fs), since the reporting time must be covered. Different t̃ ranges for glass and SiC are accepted for transfer learning | Phase 6 | proposed |
| D17 | GaN input normalization | Says "GaN inputs are already in [0,1], so no transformation is needed", but the same paper's GaN domain is r ∈ [−1, 1] µm — **internally inconsistent** | Apply the same rule `r̃ = (r−a)/(b−a)`, `z̃ = z/z_max` to all materials, to keep a single code path | Phase 7 | proposed |
| D18 | Collocation sampling | "uniform sampling … 50,000 sampling points" (Sec.3.3). Whether the points are fixed or resampled during training is **not stated** | DeepXDE `train_distribution="pseudo"` (uniform random), 50,000 points, **no resampling (our assumption)**. `"uniform"` (equispaced grid) and `PDEPointResampler` kept as options | Phase 3 | **confirmed** 2026-10-02 |
| D19 | GaN FDM grid | Paper gives only one grid (Δr = 0.25 µm), which would put just 9 points across GaN's r ∈ [−1, 1] µm | Δr = 0.05 µm for GaN FDM (7.1 synthetic check); Δz, Δt unchanged | Phase 7 | proposed |
| D20 | Network architecture | One FNN {8,64} with two outputs (n_e, T_e), all equations trained jointly (Sec.3.3) | **PFNN: three independent sub-networks {8,64}, one per output (ñ, T̃, φ)** (`dde.nn.PFNN`), required by the head-wise training of D8 (parameters must be separable to freeze them). A single FNN with shared hidden layers cannot be trained head-wise and, trained jointly, collapsed to the trivial solution in every smoke run. Parameter count is 3× the paper's net. Depth/width per sub-net follow the paper | Phase 3 | **confirmed** 2026-10-02 |

**D1 hand-calculation table** (the paper's Drude equations as written, λ = 780 nm, surface at r = z = 0)

| τ | ωτ | R (n_e = n_cr) | R (n_e = 1.13 n_cr) | α_h (1.13 n_cr) |
|---|---|---|---|---|
| 1 fs | 2.4 | 0.15 | 0.20 | 6.9e6 /m |
| 5 fs | 12.1 | 0.44 | 0.64 | 6.0e6 /m |
| 10 fs | 24.1 | 0.56 | 0.80 | 5.9e6 /m |
| 20 fs | 48.3 | 0.66 | 0.89 | 5.8e6 /m |
| 50 fs | 120.7 | 0.77 | 0.96 | 5.8e6 /m |
| **100 fs** (paper, p.8) | 241.5 | 0.83 | 0.977 | 5.8e6 /m |

> 1.13 n_cr ≈ 2.07e21 cm⁻³ is the Fig.3a value at 200 fs. This table is a static check of R only, without FDM; the final τ was decided by the Phase 2.9 scans (§3.3).

### 3.3 Phase 2 calibration record (D1 · D3 · D6, confirmed 2026-10-02)

**Problem.** With the paper's equations and Table 1 values as written (t_p = 200 fs, t_c = 0, F = 3.6 J/cm², any τ) the glass FDM gives n_e(200 fs) = 2.1e20 cm⁻³ photo-only and 9.2e20 full at r = z = 0; n_cr is never reached, so there is no R jump and no ablation profile (Fig.3 and Fig.5a cannot exist). The photoionization integral matches the analytic value within 0.5 %, so the gap is in the stated equations/values, not in the code (`notes/phase2_d6_memo.md`). D1 (τ), D3 (t_c) and D6 (magnitude) could not be settled one at a time: τ has no effect until n_cr is crossed, and whether n_cr is crossed depends on D6 and D3.

**Method.** Two FDM scans with τ, t_p,eff, t_c and F as free parameters, full and photoionization-only, scored by `scripts/judge_d6.py` against eight acceptance criteria fixed in task 2.9 **before** the scans were run:

| Code | Criterion (paper Fig.3 / Fig.5a) | Pass range |
|---|---|---|
| P1 | photo-only n_e(200 fs) plateau 1.55e21 | ±15 % |
| P2 | photo-only flattening time (90 % of final) ≈ 50 fs | ≤ 70 fs |
| P3 | photo-only convex start | n_e(25 fs)/n_e(50 fs) < 0.5 |
| F1 | full run reaches n_cr at 45–50 fs | 45–55 fs |
| F2 | full n_e(200 fs) 2.07e21 | 1.9–2.2e21 |
| F3 | R(200 fs) 0.97 | ≥ 0.9 |
| F4 | width 8 µm | 4–16 µm |
| F5 | depth 450 nm | 150–1500 nm |

- Coarse scan, τ = 100 fs fixed: t_p,eff ∈ {73, 100, 150, 200} × t_c ∈ {0, 35, t_p/2} × F ∈ {3.6, 7.2} (24 combos; `outputs/fdm/D6_scan_20261002-1320`, `…_photo_…`). Paper-as-written 0/8. Single 8/8: (100, 35, 3.6). Six 6/8.
- Fine scan, F = 3.6 fixed: t_p,eff ∈ {90, 100, 110, 120} × t_c ∈ {25, 35, 45} × τ ∈ {50, 100, 200} (36 combos; `outputs/fdm/D6_fine_20261002-1411`, `…_photo_…`).

**Result of the fine scan** (τ = 100 fs rows; the other τ give the same n_e values):

| t_p,eff | t_c | score | photo n_e(200) | t90 | n_e(25)/n_e(50) | t(n_cr) | n_e(200) | R(200) | width | depth |
|---|---|---|---|---|---|---|---|---|---|---|
| **90** | **45 = t_p/2** | **8/8** | 1.66e21 | 67 fs | 0.29 | 46.9 fs | 2.00e21 | 0.97 | 6.6 µm | 251 nm |
| 100 | 35 | 8/8 | 1.41e21 | 65 fs | 0.41 | 45.5 fs | 1.98e21 | 0.97 | 6.05 µm | 235 nm |
| 90 | 35 | 7/8 | 1.63e21 | 58 fs | 0.41 | 39.5 fs (F1) | 1.99e21 | 0.97 | 6.5 µm | 246 nm |
| 100 | 45 | 7/8 | 1.45e21 | 74 fs (P2) | 0.32 | 52.0 fs | 1.99e21 | 0.97 | 6.2 µm | 242 nm |
| 110 | 35 | 7/8 | 1.18e21 (P1) | 70 fs | 0.42 | 52.4 fs | 1.97e21 | 0.97 | 5.6 µm | 223 nm |
| any | 25 | ≤ 6/8 | — | — | 0.50–0.53 (P3) | — | — | — | — | — |
| 120 | any | 5/8 | ≤ 1.05e21 (P1) | — | — | 58–65 fs (F1) | — | — | — | — |
| paper 200 | 0 | 0/8 | 2.1e20 | — | — | never | 9.2e20 | 0.03 | — | — |

τ dependence at (90, t_p/2): only T_e, α and depth move.

| τ | max T_e | α(50 fs) | α(200 fs) | R(50 fs) | R(200 fs) | depth |
|---|---|---|---|---|---|---|
| 50 fs | 1.4e5 K | 2.6e6 /m | 6.3e6 /m | 0.90 | 0.96 | 271 nm |
| **100 fs** | **7.5e4 K** | **2.2e6 /m** | **5.0e6 /m** | **0.94** | **0.97** | **251 nm** |
| 200 fs | 3.8e4 K | 1.8e6 /m | 4.0e6 /m | 0.96 | 0.98 | 210 nm |
| paper | 3.5e4 K | 0.7e6 /m | 2.4e6 /m | 0.9 | 0.97 | 450 nm |

**Adopted set (glass):** `tp_fs: 90`, `tc_fs: "tp/2"`, `tau_fs: 100`, `F_Jcm2: 3.6` (everything else Table 1).

**Why this set and not the alternatives**
1. **(90, t_p/2) over (100, 35):** both are 8/8, but t_c = t_p/2 is the convention of the paper's own reference [27], so the calibration introduces exactly one non-paper number (t_p,eff). t_c = 35 fs is a fitted number with no source. (90, t_p/2) also sits in the middle of the n_cr window (46.9 fs vs 45.5 at the edge) and is closer to the paper's width and depth.
2. **F = 3.6 kept, not 7.2 (H-A):** doubling F flattens the photo-only curve too late (≥ 100 fs) and shifts the n_cr crossing outside 45–55 fs for every t_p tested; halving t_p at fixed F produces the same I₀ ×2 with the right timing. Reading: the authors' "200 fs" is most likely not the FWHM that Eq. (2.5) assumes (e.g. a full 1/e² or 2σ width ≈ 2 × FWHM would make the FWHM ≈ 85–120 fs).
3. **τ = 100 fs, not 200 fs:** 100 fs is the paper's stated value and the D14 value; the n_e side is indifferent. τ = 200 fs would match max T_e (3.8e4 vs 3.5e4 K) but is a second departure from the text and does not fix α or depth. Sensitivity recorded, not adopted.
4. **t_c = 0 (paper equation) rejected:** linear start (P3 fails everywhere) and, with the paper's t_p, n_cr is never reached.

**Remaining gaps with the adopted set (recorded, not tuned away).** All of them have one root: with constant τ, once n_e > n_cr the Drude k is larger than the paper's, so α is ~2× too high, the light is absorbed in a thinner layer (depth ½), and fewer electrons carry the same absorbed power (T_e ×2). The n_e dynamics at the surface, which drive the ablation width, are reproduced.

| Quantity | Ours (FDM, RK4 Δt = 1 fs; Euler value in brackets where different) | Paper | Ratio |
|---|---|---|---|
| photo-only plateau | 1.66e21 | 1.55e21 | +7 % |
| n_e(200 fs), r = z = 0 | 1.99e21 (2.00e21) | 2.07e21 | −4 % |
| t(n_cr) | 46.9 fs | 45–50 fs | ✓ |
| R(200 fs) | 0.97 | 0.97 | ✓ |
| α(50 → 200 fs) | 2.2 → 5.0e6 /m | 0.7 → 2.4e6 /m | ×2–3 |
| max T_e | 7.5e4 K | 3.5e4 K | ×2.1 |
| width | 6.6 µm | 8 µm | 0.8 |
| depth | 237 nm (251 nm) | 450 nm | 0.53 |

Consequence for later phases: **PINN accuracy is judged against our FDM at this calibrated set** (§1.2 caution). Paper figures are compared qualitatively; the α/T_e/depth factor-of-2 is expected and is not a PINN error. Physics is frozen from here (Phase 2 rule); the Phase 3 smoke runs used the pre-calibration YAML (τ = 1 fs, t_c = 0) and their FDM reference `smoke_ref_20261002-1119` is **stale** — Phase 4 uses a new reference (`glass_ref`, task 2.10).

**2.10 — Δt convergence with the adopted set (2026-10-02, `scripts/check_dt.py`).** Relative difference to RK4 Δt = 0.1 fs; "end" = 200 fs (glass) / 285 fs (SiC) at r = z = 0; L2RE over the final (r, z) field.

| Material | Case | n_e(end) | T_e(end) | L2RE n_e | L2RE T_e | t(n_cr) | Width | Depth |
|---|---|---|---|---|---|---|---|---|
| glass | Euler 1 fs (paper) | 1.9e-3 | 8.1e-3 | 6.1e-3 | 1.3e-2 | 1.1e-3 | 2.9e-3 | **5.9e-2** (251 vs 237 nm) |
| glass | Euler 0.5 fs ([27]) | 9.2e-4 | 4.0e-3 | 3.0e-3 | 6.1e-3 | 7.5e-4 | 1.6e-3 | 2.9e-2 |
| glass | RK4 1 fs | 4.3e-6 | 3.3e-4 | 1.8e-6 | 8.3e-4 | 7.2e-5 | 3.9e-8 | 3.2e-5 |
| SiC | Euler 1 fs (paper) | 6.0e-4 | 5.6e-3 | 4.5e-3 | 5.3e-3 | 2.4e-3 | 5.7e-3 | 2.1e-2 (490 vs 480 nm) |
| SiC | Euler 0.5 fs | 2.9e-4 | 2.8e-3 | 2.3e-3 | 2.6e-3 | 1.2e-3 | 3.0e-3 | 1.1e-2 |
| SiC | RK4 1 fs | 3.1e-7 | 1.2e-4 | 1.3e-7 | 2.5e-4 | 4.4e-6 | 5.0e-8 | 1.1e-6 |

The Phase 2 DoD (< 1 % at r = z = 0) passes for both materials. The Euler error is first order (halving Δt halves it), RK4 at Δt = 1 fs is converged to < 1e-3 at 0.05 s per run.

**I-31 — reference integrator (confirmed by the user 2026-10-02): RK4, Δt = 1 fs.** Reasoning: the PINN approximates the exact ODE, not the paper's time stepping, so the reference it is judged against must be a converged solution; otherwise the reference's own error (Euler: 0.6 % / 1.3 % L2RE, 6 % depth) is counted as PINN error and eats a large share of the Phase 4 DoD budget (L2RE n_e < 1e-2, T_e < 2e-2, depth ±10 %). RK4 at Δt = 1 fs removes that floor (≤ 8e-4 everywhere) at no practical cost (0.05 s per run). The paper's scheme (explicit Euler, Δt = 1 fs) stays available as `--integrator euler` for "as in the paper" comparisons; I-14's original argument ("our FDM ≈ the paper's FDM") no longer holds anyway, because the calibrated t_p already differs from the paper. `configs/fdm.yaml` now has `integrator: rk4`; the Euler references of 2.10 (`glass_ref_20261002-1459`, `glass_ref_photo_20261002-1459`, `sic_ref_20261002-1500`) are superseded by the RK4 references **`outputs/fdm/glass_ref_rk4_20261005-1330`, `glass_ref_rk4_photo_20261005-1330`, `sic_ref_rk4_20261005-1330`** (same physics, same grid; generated 2026-10-05, 0.05 s each). With RK4 the glass reference numbers are: n_cr at 46.94 fs, n_e(200 fs) 1.99e21, R(50/200 fs) 0.931/0.973, α(50/200 fs) 2.02/4.95e6 /m, width 6.60 µm, depth **237.3 nm**, max T_e 7.52e4 K — only the depth moves visibly (251 → 237 nm) relative to the Euler run. Photo-only: plateau 1.66e21, t90 66.5 fs, n_e(25)/n_e(50) 0.30. SiC: n_cr at 130.3 fs, width 2.69 µm, depth 480.1 nm, max T_e 1.83e5 K. On Colab the references are regenerated with the same commands (FDM is deterministic), so `evaluate.py` never depends on files copied from the Mac.

**SiC reference (Phase 2 DoD item, 2026-10-02)** — `outputs/fdm/sic_ref_20261002-1500` (Euler 1 fs) and its RK4 successor `sic_ref_rk4_*` (I-31): paper Table 1 values (t_p = 285 fs, F = 6 J/cm², r₀ = 2 µm) with the model-wide τ = 100 fs and t_c = t_p/2 = 142.5 fs, glass ionization coefficients borrowed (D5).

| Quantity | Ours (FDM, RK4; Euler in brackets) | Paper |
|---|---|---|
| t(n_cr), r = z = 0 | 130.3 fs (130.6) | not stated |
| max n_e (285 fs) | 1.20e21 | ≈ 1.07e21 (Fig.6 colorbar) |
| R(200 / 285 fs) | 0.964 / 0.971 | not stated |
| α(285 fs) | 4.7e6 /m | not stated |
| Width | 2.69 µm (2.68) | 2.4 µm (+12 %) |
| Depth | 480 nm (490) | ≈ 290 nm (×1.7) |
| max T_e | 1.8e5 K | ≈ 0.95e4 K (×19) |

Reading: SiC needs **no D6-type pulse correction** — with the paper's own t_p = 285 fs the profile lands within 11 % in width and within 2× in depth, which supports treating the glass t_p as a glass-specific Table 1 inconsistency rather than a model error. The factor of 19 in T_e is far beyond the glass gap (×2) and points at the borrowed coefficients (D5) on top of the constant-τ α issue — the first thing task 6.1 has to look at. Nothing is adopted for SiC now; its YAML keeps the paper values.

---

## 4. Implementation Rules (for consistency)

### 4.1 Directory layout

```
femtosecond/
├── GUIDE.md                  ← this document
├── PINN-based ....pdf
├── pyproject.toml / requirements.txt
├── configs/
│   ├── materials/
│   │   ├── glass.yaml
│   │   ├── sic.yaml
│   │   └── gan.yaml
│   ├── fdm.yaml
│   ├── paper_reference.yaml   ← values read from the paper's figures (§1.2), used as plot overlays only
│   ├── forward_glass.yaml     ← includes full / smoke profiles
│   ├── transfer_sic.yaml
│   └── inverse_gan.yaml
├── src/fsl/                  ← femtosecond laser package
│   ├── constants.py          ← physical constants (SI)
│   ├── config.py             ← YAML loading + unit conversion → SI dataclasses
│   ├── physics.py            ← Drude chain, R, α, I, ionization rates (numpy/torch compatible)
│   ├── fdm.py                ← reference solver
│   ├── scaling.py            ← nondimensionalization / inverse transform
│   ├── pinn/
│   │   ├── net.py            ← dde.nn.FNN (L, n, SiLU, Glorot normal) + hard-constraint output transform
│   │   ├── pde.py            ← PDE residual function pde(x, y) for DeepXDE (R₁, R₂, R_φ, surface query)
│   │   ├── data.py           ← geometry (dde.geometry) + dde.data.PDE, collocation points, PointSetBC for inverse data
│   │   └── train.py          ← dde.Model compile/train, callbacks, restore (forward / transfer / inverse)
│   └── eval/
│       ├── metrics.py        ← L2RE (Eq. 5.1), max relative error
│       ├── profile.py        ← n_cr contour → width/depth
│       └── plots.py          ← functions reproducing the paper's figures
├── scripts/
│   ├── check_env.py          ← versions, device, glass n_cr (Phase 0 DoD)
│   ├── run_fdm.py
│   ├── train_forward.py
│   ├── train_transfer.py
│   ├── train_inverse.py
│   ├── evaluate.py           ← PINN vs FDM on the FDM grid (L2RE, max rel. error, width/depth)
│   └── diagnose_run.py       ← intermediate quantities at the collocation points (debugging)
├── notebooks/
│   └── colab_runner.ipynb    ← thin Colab launcher (§4.6); no logic lives here
├── tests/                    ← pytest
├── data/                     ← external data such as gan_profile.csv
├── outputs/                  ← run outputs (excluded from git)
│   └── <phase>/<run_name>_<YYYYmmdd-HHMM>/
│       ├── config.yaml       ← copy of the config used for the run
│       ├── ckpt/             ← DeepXDE ModelCheckpoint files (model-<step>.pt, model + optimizer)
│       ├── history.csv       ← per-term losses from DeepXDE loss history
│       ├── metrics.json
│       └── figs/
└── notes/
    ├── log.md                ← experiment log (date, run, result summary, judgment)
    ├── decisions.md          ← implementation decision records I-1, I-2, … (§0)
    └── phase2_d6_memo.md     ← Phase 2.7 memo on the D6 mismatch and the H-A / H-B tests
```

### 4.2 Unit rules

1. **Everything inside `physics.py` and `fdm.py` is in SI units** (m, s, K, J, W/m², m⁻³).
2. YAML uses human-friendly units, but **the unit is appended to the key name**. Examples: `lambda_nm`, `tp_fs`, `r0_um`, `F_Jcm2`, `U1_eV`, `alpha_i_cm2J`, `delta_N_cm3ps_cm2TW`.
3. Unit conversion happens **only in `config.py`**. No other file hard-codes conversion constants like 1e-4 or 1e-15.
4. δ_N has unusual units (cm⁻³ps⁻¹(cm²/TW)^N). Its conversion therefore lives only inside `physics.photoionization_rate(I_SI)` and is pinned down by unit tests.
5. Figure axes use the same units as the paper (µm, nm, fs, cm⁻³, K).

### 4.3 Nondimensionalization (PINN only, D7)

| Quantity | Definition | Reference value |
|---|---|---|
| r̃ | (r − a)/(b − a) ∈ [0,1] | a=−8, b=8 µm (glass/SiC), ±1 µm (GaN) |
| z̃ | z / z_max ∈ [0,1] | z_max = 0.6 µm (glass/SiC), 1 µm (GaN) |
| t̃ | t / t_ref | t_ref = 100 fs → glass·GaN t̃∈[0,2], SiC t̃∈[0,2.85] (D16) |
| ñ | n_e / n_ref | n_ref = 1e21 cm⁻³, shared by all materials (D15). Option: n_ref = per-material n_cr |
| T̃ | (T_e − 300 K) / T_ref | T_ref = 1e4 K, shared by all materials (D15) |
| φ | Optical depth (already dimensionless) | — |

> Paper Eq. (3.17) normalizes with a flipped coordinate x' = (b−x)/(b−a), and also has a variable typo y' = (d−x)/(d−c) (should be y). This implementation uses the unflipped (x−a)/(b−a); the results are the same. GaN uses the same rule (D17).

Nondimensional residuals:

$$R_1 = \frac{\partial \tilde n}{\partial \tilde t} - t_{ref}\Bigl(\alpha_i I \tilde n + \frac{\delta_N I^N}{n_{ref}}\Bigr)$$

$$R_2 = \tilde n\,\frac{\partial \tilde T}{\partial \tilde t} - \frac{t_{ref}\,\alpha_h I}{c_e\, n_{ref}\, T_{ref}}$$

$$R_\varphi = \frac{\partial \varphi}{\partial \tilde z} - z_{max}\,\alpha$$

- The ablation boundary (n_e = n_cr) is **ñ = n_cr / n_ref** in nondimensional form (glass 1.83, SiC·GaN 1.05).

Hard constraints (D9, k=1):

```
ñ  = t̃^k · r̃(1−r̃) · NN_n
T̃  = t̃^k · r̃(1−r̃) · NN_T
φ  = z̃ · softplus(NN_φ − 2)      ← φ ≥ 0 (D2, I-29); softplus(−2) ≈ 0.13 keeps the initial attenuation small
```

- R(t,r) needs the surface value. So for each collocation point (r,z,t), **call the network once more at (r,0,t)** to get the surface n_e.

### 4.4 Code rules

- Config values flow only through YAML → `config.py` → dataclass. No magic numbers inside scripts.
- Functions in `physics.py` accept both numpy and torch inputs, so FDM and PINN **share the same physics functions** and cannot drift apart.
- Fix the random seed with `dde.config.set_random_seed(seed)`. Default dtype is float32 (`dde.config.set_default_float("float32")`). The backend is fixed to PyTorch via the environment variable `DDE_BACKEND=pytorch`, set at the top of every training script before importing deepxde (done once in `fsl/pinn/__init__.py`). DeepXDE picks CUDA automatically when available; **on Apple Silicon DeepXDE 1.15 picks MPS at import time, which has no float64 — `train.setup()` therefore forces CPU unless CUDA is present** (`device: auto`, I-26).
- Every run writes a config copy, loss history, and metrics.json to `outputs/`.
- Figures are drawn only through functions in `eval/plots.py`. Function names follow the paper's numbering (`fig3`, `fig4`, ...).
- Training scripts save checkpoints every `ckpt_every` epochs with `dde.callbacks.ModelCheckpoint` and accept `--resume <run_dir>` (loads the latest file with `model.restore`), because Colab sessions can disconnect (§4.6). Resume mechanics (I-24, §4.8): DeepXDE restarts its iteration counter after `restore`, so `ckpt/offsets.json` records the global iteration at which each resume round started; a checkpoint's global step is `offsets[suffix] + step_in_name`. History rows after the restored step are dropped (that stretch is redone). Staged runs resume **inside** the stage schedule: finished stages are skipped, the interrupted stage runs its remaining iterations with its own weights/heads, later stages follow. `train_forward.py --resume` takes config and material from the run's own `config.yaml` (so a changed YAML cannot silently change the physics mid-run); `--set rounds=N` extends a finished run.
- Reference FDM solutions for PINN evaluation are computed with **RK4, Δt = 1 fs** (`configs/fdm.yaml`, I-31, §3.3); the paper's explicit Euler is `--integrator euler` and is used only for "as in the paper" comparisons.
- Pin the DeepXDE version installed in Phase 0 in `requirements.txt` (its API changes between releases).

### 4.5 Common evaluation metrics

- **L2RE(t)** (paper Eq. 5.1): computed over all FDM grid points (Δr=0.25 µm, Δz=0.02 µm).
  $\sqrt{\sum (u_{NN}-u_M)^2 / \sum u_M^2}$
- **Max relative error**: the paper does not define it. However, in Fig.4c/f the error concentrates at the edges where values are small (r = ±5 µm), so it appears to be a **pointwise relative error**. We therefore report both:
  - `max_rel_pointwise`: max |u_NN − u_M| / |u_M|. To avoid a blow-up of the denominator, only points with |u_M| ≥ 1e-3·max|u_M| are used. For T_e, use T itself, not (T − 300 K). **Use this value when comparing against the paper's numbers (2.5%/9% etc.).**
  - `max_rel_global`: max |u_NN − u_M| / max|u_M|. Normalized by the global maximum; for internal judgment.
  - Figures (fig4c/f, fig6b/d, fig7b/d, fig12) are drawn with the pointwise relative error.
- **Ablation profile**: width (µm), depth (nm). Computed with the §2.6 definition for both FDM and PINN.

### 4.6 Mac ↔ Colab workflow (D13)

Code is written on the Mac and run on Colab. The git repository is the only channel between them.

```
Mac (~/lab/femtosecond)                    Google Colab (GPU runtime)
  Claude writes code                          notebooks/colab_runner.ipynb
  user: git commit && git push  ───────────►  git clone / git pull
                                              pip install -e .
                                              python scripts/train_*.py ...
  outputs pulled back via Drive  ◄──────────  outputs/ written to Google Drive
```

- **Repository:** `github.com/leejoohyunn/pinn-femtosecond-laser-ablation` (private; already created, local `git init` + `origin` done, first push happens in Phase 0.4). The PDF and `outputs/` are excluded via `.gitignore`.
- **Colab notebook:** `notebooks/colab_runner.ipynb` only does setup and launching: mount Drive, clone/pull the repo, `pip install -e .`, check `nvidia-smi`, then call the same `scripts/*.py` commands used on the Mac. **No model or physics code goes in the notebook.**
- **Outputs:** on Colab, `outputs/` is symlinked to a Google Drive folder (e.g., `MyDrive/femtosecond/outputs`) so results survive disconnects. Download or sync the relevant run folders back to the Mac when reporting.
- **Disconnects:** Colab sessions have time limits and can drop. Long runs (Phase 4, 6, 7) must use checkpoints and `--resume` (§4.4).
- **Private repo access:** clone with a GitHub personal access token stored in Colab Secrets, never pasted into the notebook.

### 4.7 DeepXDE mapping (D4)

| Paper component | DeepXDE implementation |
|---|---|
| Domain Ω (nondimensional r̃, z̃, t̃) | `dde.geometry.Cuboid([0,0,0], [1,1,t̃_max])` treated as a plain 3D geometry. No `GeometryXTime`, because IC/BC are enforced by hard constraints |
| Collocation points | `dde.data.PDE(geom, pde, bcs=[], num_domain=50000, train_distribution="pseudo")` (D18) |
| Network {8,64}, SiLU, Glorot normal | `dde.nn.PFNN([3] + [[64]*3]*8 + [3], "silu", "Glorot normal")` — three independent {8,64} sub-nets, outputs (NN_n, NN_T, NN_φ) (D20). `dde.nn.FNN` kept as `net: fnn` for comparison runs |
| Hard constraints (§4.3) | `net.apply_output_transform(transform)` |
| Head-wise training (D8) | `stages` in the training config: (iterations, loss_weights, heads) repeated `rounds` times; `set_trainable_heads` freezes the other sub-nets and the `HeadFreezer` callback re-applies it on every `on_epoch_begin` (DeepXDE's `_outputs_losses` calls `net.requires_grad_()` on the whole net at every evaluation). One `model.compile` per stage |
| PDE residuals R₁, R₂, R_φ | `pde(x, y)` returns `[R1, R2, Rphi]`, derivatives via `dde.grad.jacobian(y, x, i=…, j=…)` |
| **Surface query for R(t,r)** | Inside `pde`, build `x_s = x` with the z̃ column set to 0 and call `net(x_s)` (the net is captured by closure; the output transform is applied automatically). Not a built-in DeepXDE feature; unit-tested in Phase 3 |
| Loss weights λ (D8) | `model.compile("adam", lr=1e-3, loss_weights=[λ1, λ2, λφ])` per stage: (1,0,0) / (0,0,1) / (0,1,0). `history.csv` stores the weights and the trainable-head mask next to the (weighted) losses so raw residuals can be recovered |
| Per-term loss curves (Fig.9, 11, 13) | `losshistory.loss_train` (one column per residual/BC term) → `history.csv` |
| Best-loss model | `model.train(..., callbacks=[ModelCheckpoint(..., save_better_only=True)])` |
| Transfer learning (Phase 6) | Build the SiC model with the same net, `model.restore(<glass checkpoint>)`, re-compile, train 25k |
| Trainable F (Phase 7) | `logF = dde.Variable(log F₀)`, used as `F = exp(logF)` inside `pde`; `model.compile(..., external_trainable_variables=[logF])`; track with `dde.callbacks.VariableValue` |
| Inverse data term (Eq. 3.16) | `dde.icbc.PointSetBC(points, values, component=0)` with values ñ = n_cr/n_ref at the digitized profile points |

---

### 4.8 Implementation decisions that affect results (summary of `notes/decisions.md`)

§3 records how the paper's gaps were resolved; the I-items below are *our* engineering choices that a reader needs in order to interpret the numbers. Full reasoning per item is in `notes/decisions.md` (Korean).

| ID | Choice | Why | Effect on results |
|---|---|---|---|
| I-14 → **I-31** | Reference FDM integrator: **RK4, Δt = 1 fs** (`configs/fdm.yaml`); explicit Euler Δt = 1 fs (paper) kept as `--integrator euler` | The PINN approximates the exact ODE; the Euler reference's own error (0.6 % / 1.3 % L2RE, 6 % depth at the calibrated physics) would be booked as PINN error and eat the Phase 4 DoD budget. RK4 at 1 fs is converged to < 1e-3 at 0.05 s/run. Confirmed by the user 2026-10-02 | Glass depth 237 nm (Euler: 251); everything else changes by < 0.3 % |
| I-15 | T_e not updated where n_e < 1e6 m⁻³ | The T_e equation divides by n_e | None in practice (n_e > 1e6 wherever I > 0) |
| I-21 | Smoke runs in float64, full runs in float32 | Remove precision from the debugging equation on the CPU; GPU speed for full runs | Smoke losses reach 1e-6 raw; float32 floor is ~1e-7 |
| I-24 | Checkpoints: `best-<step>.pt` (lowest weighted loss of the current stage) + `periodic-<step>.pt`; resume via `ckpt/offsets.json`, history truncated to the restored step, staged runs resume inside the schedule | DeepXDE restarts its step counter after `restore`; Colab disconnects are certain for 50k-iteration runs | Resumed runs are equivalent to uninterrupted ones up to the Adam state (which restarts at every stage boundary anyway) |
| I-26 | Device `auto` = CUDA if available, else CPU; **MPS never used** | DeepXDE 1.15 picks MPS on Apple Silicon at import; MPS has no float64 and double backward is unverified | Mac runs are CPU-only |
| I-27 | n, k from the **complex** square root sqrt(ε_r + i·|ε_i|) | The real-form k = sqrt((−ε_r + |ε|)/2) has an infinite gradient at n_e → 0 and produced NaN within 100 iterations | Identical values for n_e ≥ 0; finite gradients; k ≥ 0 even for transient n_e < 0 |
| I-28 / I-29 | Stage-wise training (`stages`, `rounds`), head-wise freezing with the `HeadFreezer` callback, φ = z̃·softplus(NN_φ − 2) | D8/D20: joint training with static weights collapses to ñ → 0; DeepXDE's `_test()` re-enables all parameters every evaluation, so freezing must be re-applied each iteration | Loss curves are per stage (weights and trainable heads are logged in `history.csv`); compare raw residuals, not weighted totals |
| I-30 | Material YAMLs carry the **calibrated** values; the paper's values live in comments and are pinned in tests through overrides (`tp_fs=200, tc_fs=0`) | Every script reads the YAML, so the YAML must be the frozen physics; the paper-as-written case remains reproducible with `--set` | Any run after 2026-10-02 uses t_p,eff = 90 fs, t_c = t_p/2, τ = 100 fs unless overridden |
| I-32 | Rate constants are folded with the nondimensional scale **before** touching tensors: `photoionization_rate(I, mat, scale=t_ref/n_ref)`, `impact_rate` ordered as (α_i I)(n_e·scale), `plasma_freq_sq` with e²/(m_e ε₀) pre-folded | The SI rates (δ_N I³ ≈ 4e40, α_i I n_e ≈ 5e40 m⁻³ s⁻¹ at the glass peak) overflow float32 (max 3.4e38): the first Colab float32 run had loss_ne = inf at iteration 0 (2026-10-05). α_i·scale = 1e-44 and m_e ε₀ = 8e-42 are float32 denormals (5 % error). Phase 3 smoke ran in float64 and never hit this | None in float64 (identical algebra); float32 residual terms now agree with float64 to ≤ 1e-3 (test `test_residuals_finite_in_float32`) |

## 5. Task Packs by Phase

### Phase 0 — Environment and skeleton

**Goal:** Create the empty package skeleton and the Mac + Colab execution environments.

| # | Task | Output |
|---|---|---|
| 0.1 | Python 3.11 venv and dependency spec (deepxde, torch, numpy, scipy, matplotlib, pyyaml, pytest; DeepXDE version pinned) | `requirements.txt`, `pyproject.toml` (installable with `pip install -e .`) |
| 0.2 | Create the §4.1 directory skeleton | Folders, `__init__.py` |
| 0.3 | `constants.py`, `config.py`, three material YAML files | Config can be loaded |
| 0.4 | `.gitignore` (outputs/, .venv/, *.pdf) and first push. Repository already created: `github.com/leejoohyunn/pinn-femtosecond-laser-ablation` (private), local `git init` + `origin` done | Repo pushed |
| 0.5 | `notebooks/colab_runner.ipynb` (§4.6) | Colab can clone, install, and run the smoke check |
| 0.6 | `notes/log.md` template | — |

**User runs (Mac)** — `.venv/` already exists (conda env, Python 3.11.16, created 2026-09-22; `python3.11` is not on PATH, so do not recreate it with `venv`)
```bash
cd ~/lab/femtosecond
source .venv/bin/activate
pip install -e . -r requirements.txt
python scripts/check_env.py      # versions, device, glass n_cr
pytest -q                        # tests/test_config.py
```

**User runs (Colab):** open `notebooks/colab_runner.ipynb`, select a GPU runtime, and run all cells.

**DoD:** The config loads and glass n_cr prints as 1.83e27 m⁻³ (= 1.83e21 cm⁻³) on both the Mac and Colab, and Colab reports a CUDA device.

---

### Phase 1 — Physics module + unit tests

**Goal:** Port the §2 equations into `physics.py` and pin them down with tests.

| # | Task |
|---|---|
| 1.1 | `plasma_freq`, `drude_eps`, `nk_from_eps`, `reflectivity`, `alpha_h`, `alpha_total` |
| 1.2 | `photoionization_rate(I)`, `impact_rate(I)`, `intensity(t, r, phi, R, params)` |
| 1.3 | `critical_density(λ)` |
| 1.4 | Write tests (below) |

**Test list (`tests/test_physics.py`)**
- n_cr(780 nm) = 1.83e21 cm⁻³ ± 1%, n_cr(1030 nm) ≈ 1.05e21 cm⁻³
- At n_e=0: ε=1, n=1, k=0, R=0, α_h=0
- As n_e → ∞: R → 1
- Peak intensity I₀ (glass) ≈ 1.69e13 W/cm² (= 16.9 TW/cm²)
- Unit-converted value of δ₃·I₀³ matches a hand calculation
- numpy and torch inputs give the same results

**User runs:** `pytest -q`
**DoD:** All tests pass.

---

### Phase 2 — FDM reference solution + parameter calibration ⚠️ key gate

**Goal:** Implement the numerical solution of paper Sec.2.2 and settle the missing parameters (D1, D3, D6).

**Status (2026-10-02, night): Phase 2 DoD passed.** 2.6 (12 runs) was inconclusive because n_cr is never reached with the paper's values; 2.7 memo (`notes/phase2_d6_memo.md`); 2.8 photo-only runs are part of the 2.9 scans; **2.9 coarse + fine scans scored by `scripts/judge_d6.py` → D1·D3·D6 `confirmed` (§3.3): τ = 100 fs, t_c = t_p/2, t_p,eff = 90 fs, F = 3.6.** YAML defaults updated (paper values in comments). **2.10:** `scripts/check_dt.py` PASS for glass and SiC (Euler 1 fs vs RK4 0.1 fs at r = z = 0, 200 fs: glass 0.19 % n_e / 0.81 % T_e, SiC 0.06 % / 0.56 %; full tables in §3.3). References saved with the paper's scheme (Euler 1 fs): `outputs/fdm/glass_ref_20261002-1459`, `glass_ref_photo_20261002-1459`, `sic_ref_20261002-1500`. Because the Euler reference carries 0.6 % / 1.3 % L2RE and 6 % depth discretization error, **I-31 (confirmed 2026-10-02) makes RK4 Δt = 1 fs the reference integrator**; the three references were regenerated 2026-10-05 as `glass_ref_rk4_20261005-1330`, `glass_ref_rk4_photo_20261005-1330`, `sic_ref_rk4_20261005-1330` (§3.3; 47 tests pass incl. the staged-resume test). Phase 3 ran in parallel (done); `--resume` for staged runs is implemented (I-24); Phase 4 can start.

| # | Task |
|---|---|
| 2.1 | `fdm.py`: time-march the whole (r,z) grid in vectorized form. Each step: ① R from surface n_e ② α(z) ③ φ = cumulative trapezoidal integral ④ I ⑤ update n_e, T_e |
| 2.2 | Time integration: explicit Euler, Δt=1 fs (paper). Check convergence against RK4 with Δt=0.1 fs. **Outcome (2.10, I-31):** Euler passes the < 1 % DoD at r = z = 0 but carries 0.6 % / 1.3 % L2RE and 6 % depth error; the reference solutions used for PINN evaluation are therefore **RK4, Δt = 1 fs** (default in `configs/fdm.yaml`), Euler stays as `--integrator euler` |
| 2.3 | n_e floor when updating T_e (e.g., dT=0 if n_e < 1e6 m⁻³) |
| 2.4 | `run_fdm.py`: save results to `outputs/fdm/…/solution.npz` (r, z, t snapshots, n_e, T_e, R, α) |
| 2.5 | Run the paper's equations as written (τ=1 fs placeholder, t_c=0) → Fig.3a/b and ablation profile. Domain is the FDM domain in §2.5.1 |
| 2.6 | **Calibration scan**: τ ∈ {1, 5, 10, 20, 50, 100} fs (100 fs = paper value, D1/D14) × t_c ∈ {0, t_p}. Tabulate each combination (time to reach n_cr, max n_e, max T, R(50 fs), R(200 fs), α(50 fs), α(200 fs), width, depth). **The primary criterion is the shape of the R curve** (Fig.3b values in §1.2). fig3b overlays representative values of the paper's curve for comparison |
| 2.7 | Memo analyzing the cause of D6 (order-of-magnitude mismatch) → **settle calibrated values with the user** |
| 2.8 | **Photoionization-only run** (`alpha_i_cm2J: 0`, `--ionization photo_only`) → corresponds to the dashed curve in Fig.3a. Check three things: ① the plateau value (paper about 1.55e21) → D6, ② the curve shape near t=0 (convex or linear) → D3, ③ hypothesis H1 (D3): a second variant (`--ionization photo_only_shared_R`) that uses R(t,r) and α from the full run instead of its own n_e — does it reproduce the abrupt flattening at ~50 fs? |
| 2.9 | **D6 scan with τ fixed at 100 fs** (D1 revised default; added 2026-10-02 after 2.7 showed D6 and D1 cannot be settled separately). Grid: t_p,eff ∈ {73, 100, 150, 200} fs × t_c ∈ {0, 35 fs, t_p/2} × F ∈ {3.6, 7.2} J/cm², full and photo-only. Acceptance (all required): photo-only n_e(200 fs) = 1.55e21 ± 15% with flattening ≤ 70 fs **and a convex start**; full run reaches n_cr at 45–55 fs, n_e(200 fs) = 1.9–2.2e21, R(200 fs) ≥ 0.9, width/depth within an order of magnitude of 8 µm / 450 nm. Any adopted value that differs from Table 1 is recorded as a reproduction calibration, with the paper's value kept in the YAML comment. **Done 2026-10-02:** coarse scan (24) + fine scan (36: t_p,eff {90,100,110,120} × t_c {25,35,45} × τ {50,100,200}, F = 3.6) → adopted (90 fs, t_p/2, 100 fs), full record in §3.3 |
| 2.10 | Re-check Δt convergence with the adopted settings (the R jump makes the ODE stiffer than the uncalibrated case); add Δt = 0.5 fs (the step used in [27]) as an extra point → `scripts/check_dt.py` (Euler 1 / 0.5 fs, RK4 1 / 0.1 fs; last case is the reference). Then save the **calibrated glass reference** (`glass_ref`, full + photo-only, used by Phase 4 evaluate/fig3a) and the **SiC reference** (`sic_ref`, same τ/t_c; its t_p stays 285 fs until Phase 6 compares with Fig.5b). **Done 2026-10-02: PASS for glass and SiC, three references saved (§3.3); follow-up I-31 (RK4 reference) proposed** |

**User runs (Mac; FDM is cheap on CPU)**
```bash
python scripts/run_fdm.py --material glass --config configs/fdm.yaml
python scripts/run_fdm.py --material glass --sweep tau_fs=1,5,10,20,50,100 tc_fs=0,tp
python scripts/run_fdm.py --material glass --config configs/fdm.yaml --ionization photo_only
python scripts/run_fdm.py --material glass --config configs/fdm.yaml --ionization photo_only_shared_R --shared_from outputs/fdm/<full_run>
# 2.9 coarse (τ fixed): 24 full + 24 photo-only runs, then the fine scan (36 + 36), scored by judge_d6
python scripts/run_fdm.py --material glass --set tau_fs=100 --sweep tp_fs=73,100,150,200 tc_fs=0,35,tp/2 F_Jcm2=3.6,7.2 --name D6_scan
python scripts/run_fdm.py --material glass --set tau_fs=100 --sweep tp_fs=73,100,150,200 tc_fs=0,35,tp/2 F_Jcm2=3.6,7.2 --ionization photo_only --name D6_scan_photo
python scripts/run_fdm.py --material glass --set F_Jcm2=3.6 --sweep tp_fs=90,100,110,120 tc_fs=25,35,45 tau_fs=50,100,200 --name D6_fine
python scripts/run_fdm.py --material glass --set F_Jcm2=3.6 --sweep tp_fs=90,100,110,120 tc_fs=25,35,45 tau_fs=50,100,200 --ionization photo_only --name D6_fine_photo
python scripts/judge_d6.py --full "$(ls -d outputs/fdm/D6_fine_2* | tail -1)" --photo "$(ls -d outputs/fdm/D6_fine_photo_* | tail -1)"
# 2.10 (YAML defaults = adopted set): Δt check, calibrated references (Euler, the paper's scheme)
python scripts/check_dt.py --material glass
python scripts/run_fdm.py --material glass --integrator euler --name glass_ref
python scripts/run_fdm.py --material glass --integrator euler --ionization photo_only --name glass_ref_photo
python scripts/check_dt.py --material sic
python scripts/run_fdm.py --material sic --integrator euler --name sic_ref
# I-31: the references the PINN is judged against (RK4 Δt = 1 fs is now the fdm.yaml default)
python scripts/run_fdm.py --material glass --name glass_ref_rk4
python scripts/run_fdm.py --material glass --ionization photo_only --name glass_ref_rk4_photo
python scripts/run_fdm.py --material sic --name sic_ref_rk4
```

> The pre-2.9 commands above assume the YAML still carried the paper's values; since 2026-10-02 the defaults are the calibrated set, so reproducing the paper-as-written run needs `--set tp_fs=200 tc_fs=0`.

**DoD**
- Results converge as the time step shrinks (difference between Δt=1 fs and 0.1 fs < 1%).
- The calibration table is complete, and D1·D3·D6 are `confirmed`.
- With the confirmed settings, the glass profile **qualitatively matches** the paper (8 µm / 450 nm) in order of magnitude and shape. If the numbers do not match, document the difference and agree on it with the user.
- Run SiC with the same settings and save its reference solution (used in Phase 6).

> Physics settings confirmed in this Phase **do not change** in any later Phase. If they change, rebuild from the FDM onward.

---

### Phase 3 — PINN forward problem (glass, smoke)

**Goal:** Confirm that the whole pipeline runs end to end with a small configuration.

**May start before Phase 2 closes** (§0 exception, 2026-10-02). The smoke run uses the material YAML as it is; which τ / t_c / F values are in it does not matter for this DoD. What matters is that the **same** YAML is used for the FDM comparison in 3.8, so that "PINN ≈ FDM for identical physics" can be checked without waiting for the calibration.

**Result (2026-10-02): DoD passed with profile `smoke_gs`** (runs `smoke_*` in `outputs/forward/`, details in `notes/log.md`). What it took: (1) complex-sqrt Drude chain to remove a NaN at n_e → 0 (I-27); (2) forcing CPU on Apple Silicon (I-26); (3) the joint-training collapse (D8) and its fix — φ ≥ 0 (D2), PFNN (D20), block Gauss–Seidel with `HeadFreezer` (I-28, I-29). Final smoke numbers: raw residuals n 7e-6 / φ 2.5e-4 / T 0.14, L2RE vs FDM (same physics, τ = 1 fs placeholder) n_e 3.6e-2 and T_e 0.14 at 200 fs, max n_e 8.7e20 vs 9.2e20 cm⁻³. `scripts/diagnose_run.py` prints every intermediate quantity at the collocation points and was what exposed both failure modes.

| # | Task |
|---|---|
| 3.1 | `net.py`: `dde.nn.FNN` (3→[n]×L→3), SiLU, Glorot normal, outputs NN_n, NN_T, NN_φ; §4.3 hard constraints via `apply_output_transform` |
| 3.2 | `pde.py`: `pde(x, y)` returning R₁, R₂, R_φ with `dde.grad.jacobian`; R(t,r) via the surface query (§4.7) |
| 3.3 | Test: the surface query returns the same n_e as evaluating the net directly at z̃=0, and gradients flow through it |
| 3.4 | `data.py`: Cuboid geometry + `dde.data.PDE`, N collocation points (paper: 50,000; no resampling is our assumption, D18) |
| 3.5 | `train.py`: `dde.Model`, Adam lr=1e-3, `loss_weights` (D8), ModelCheckpoint (best + periodic), loss history → `history.csv`, `--resume` |
| 3.6 | Smoke config: {4,32}, 5,000 points, 2,000 epochs → check that the loss decreases and no NaN appears |
| 3.7 | Settle D8 (λ) from the loss-term ratios over the first 1k epochs |
| 3.8 | **Pipeline check against FDM with identical physics:** run `run_fdm.py` with the same material YAML, evaluate the smoke PINN on the FDM grid with `evaluate.py` (L2RE, §4.5). Plumbing target only: L2RE(n_e, t = t_max) < 0.1. This is not an accuracy claim; Phase 4 makes that claim with the frozen physics |

**User runs (Mac or Colab)**
```bash
python scripts/train_forward.py --config configs/forward_glass.yaml --profile smoke
python scripts/run_fdm.py --material glass --name smoke_ref          # same YAML as the smoke run
python scripts/evaluate.py --run outputs/forward/<smoke_run> --fdm outputs/fdm/<smoke_ref>
```
**DoD:** Training finishes without NaN, the loss drops by at least two orders of magnitude, and the 3.8 comparison number is recorded in `notes/log.md`. D2, D7, D8, D9, D10, D15, D18 become `confirmed`.

---

### Phase 4 — PINN forward problem (glass, full) + evaluation

**Goal:** Train with the paper's settings and reproduce the paper's figures.

**Precondition: met 2026-10-02** — Phase 2 DoD passed (§3.3): D1·D3·D6 `confirmed`, 2.10 Δt check PASS, physics frozen, I-31 confirmed (RK4 references `glass_ref_rk4_*`, `glass_ref_rk4_photo_*`, `sic_ref_rk4_*`), `--resume` for staged runs implemented (I-24). `evaluate.py --fdm` must point at the RK4 glass reference, not at the stale `smoke_ref` (pre-calibration physics) or the Euler `glass_ref` (the script warns on a material mismatch, not on an integrator mismatch — check the name).

| # | Task |
|---|---|
| 4.1 | Full config (`full` profile): PFNN 3×{8,64} (D20), 50,000 points, 50,000 iterations as 5 rounds × [n 5000 / φ 2500 / T 2500] head-wise stages (D8), float32, Colab GPU (D13). **`--resume <run_dir>` works for staged runs (I-24 addendum, 2026-10-02):** the newest periodic checkpoint (every `ckpt_every` = 1000 iterations) is restored, finished stages are skipped, the interrupted stage runs its remaining iterations, history rows after the checkpoint are dropped; config and material come from the run's own `config.yaml`. `--set rounds=6` on a resume extends a finished run by one round |
| 4.2 | `evaluate.py`: PINN inference on the FDM grid → L2RE(t=50/100/150/200 fs), max relative error (both pointwise and global, §4.5) |
| 4.3 | Figures: fig3 (time evolution at r=z=0; the "photoionization only" curve in 3a overlays the FDM result from Phase 2.8), fig4 (2D maps and pointwise relative error), fig5a (profile), fig9 (loss, colors per legend), fig10 (z=200 nm slices at 4 times) |
| 4.4 | L2RE table in Table 2 format |

**User runs (Colab)**
```bash
python scripts/train_forward.py --config configs/forward_glass.yaml --profile full --name glass_full
python scripts/train_forward.py --resume "$(ls -d outputs/forward/glass_full_* | tail -1)"
python scripts/evaluate.py --run "$(ls -d outputs/forward/glass_full_* | tail -1)" --fdm "$(ls -d outputs/fdm/glass_ref_rk4_2* | tail -1)"
```
(the second line only after a Colab disconnect; it continues inside the stage schedule)

**DoD**
- L2RE(t=200 fs): n_e < 1e-2, T_e < 2e-2 (paper: 2.2e-3 / 4.7e-3)
- Ablation width and depth within ±10% of our FDM
- Errors do not accumulate over time (L2RE is not monotonically increasing in t)

---

### Phase 5 — Architecture search (optional)

| # | Task |
|---|---|
| 5.1 | Train each of {4,32}, {4,64}, {8,32}, {8,64}, {8,128} (fewer epochs are fine, but keep conditions identical) |
| 5.2 | Table and scatter plot of L2RE(t_p) vs training time |

**DoD:** Table complete; the choice of {8,64} is confirmed or refuted.

---

### Phase 6 — Transfer learning (SiC)

| # | Task |
|---|---|
| 6.1 | Settle D5: report literature values for SiC α_i, δ_N and decide whether to adopt them or keep the borrowed glass values |
| 6.2 | SiC FDM reference solution (reuse Phase 2 settings, SiC domain in §2.5.1). **Already saved in Phase 2** (`sic_ref_20261002-1500`, §3.3): width 2.68 µm, depth 490 nm, max T_e 1.8e5 K vs paper 2.4 µm / 290 nm / 0.95e4 K. RK4 version `sic_ref_rk4_*` (I-31). Regenerate after 6.1 if D5 changes the coefficients |
| 6.3 | SiC basic: random initialization, 60,000 epochs (also save a checkpoint at 25,000 epochs). t̃ ∈ [0, 2.85] (D16) |
| 6.4 | SiC transfer: initialize from the full glass model weights (`model.restore`), 25,000 epochs, lr=1e-3. Architecture and scales (n_ref, T_ref, D15) use **the same values** as glass; only material parameters change |
| 6.5 | Figures: fig5b, fig6, fig7, fig11 (loss comparison), fig12 (three z=200 nm relative-error curves). Table 3 |

**DoD**
- Transfer at 25k beats basic at 25k in both loss and L2RE(T_e) (the paper's main claim).
- Transfer at 25k reaches the same L2RE level as basic at 60k. Since paper Table 3 also shows transfer T_e worse at t=200 fs, this is judged **at t = 285 fs (final time)**.

---

### Phase 7 — Inverse problem (GaN, fluence estimation)

| # | Task |
|---|---|
| 7.1 | **Synthetic check first:** run GaN FDM with F=0.6 → extract about 200 points on the n_cr contour → inverse-estimate F from this data and check that 0.6 is recovered |
| 7.2 | D11: digitize Fig.8a → `data/gan_profile.csv` (r_um, depth_nm) |
| 7.3 | Loss (paper 3.16), data term via `PointSetBC`: λ_t⟨(ñ(r_i,z_i,t_data) − n_cr/n_ref)²⟩ + λ⟨R₁²⟩ (+ R_φ). For GaN, n_cr/n_ref ≈ 1.05 (D15). As in the paper, **use only the n_e equation**. τ and t_data follow D14 (τ = 100 fs, t_data = 200 fs; t_data = 100 fs only as an option) |
| 7.4 | Trainable parameter `F = exp(logF)` via `dde.Variable` + `external_trainable_variables` (D12, §4.7), logged with `VariableValue`. Two runs with F₀=1 / F₀=10 (30k / 70k epochs) |
| 7.5 | Figures: fig8a (profile comparison), fig8b (F convergence), fig13 (loss) |

**DoD**
- F recovery error < 5% in the synthetic check. Also record whether the synthetic profile matches the order of magnitude of paper Fig.8a (width 1.2 µm, depth 0.22 µm).
- On the real data, F₀=1 and F₀=10 converge to the same value (difference < 5%).
- Compare the converged F with the paper's 0.6 J/cm² and record it. If they differ, analyze the effect of D5 (borrowed coefficients).

---

### Phase 8 — Wrap-up

| # | Task |
|---|---|
| 8.1 | `notes/report.md`: reproduction results table (paper vs ours), figures, decision-log summary, items not reproduced and why |
| 8.2 | README: installation and per-Phase re-run commands (Mac and Colab) |

---

## 6. Result Report Template (user → Claude)

After each run, sharing in the format below speeds up judgment.

```
[Phase X / run name]
- Setup: configs/xxx.yaml (profile=...), device (Mac CPU / Colab GPU model), wall time
- Final loss: total / ne / Te (/ phi / data)
- Key metrics.json values: L2RE, width/depth, (F for the inverse problem)
- Anomalies: NaN, divergence, odd-looking figures, Colab disconnects/resumes
- Figures: outputs/.../figs/fig4.png etc. (attach images if needed)
```

---

## 7. Known Risks and Mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| Paper values cannot be reproduced due to τ/unit mismatch | FDM profile differs greatly from the paper | **Realized and resolved in Phase 2 (§3.3):** the paper's values never reach n_cr; calibrated to t_p,eff = 90 fs, t_c = t_p/2, τ = 100 fs. Residual factor-of-2 gaps in α, T_e and depth are documented and expected. The primary criterion is error relative to our FDM |
| Error in the fast n_e rise (0–50 fs) | Large L2RE at t=50 fs | The paper shows the same (1e-2). Denser sampling along t is an option |
| Intensity is not zero at the r=±8 µm boundary | Errors concentrate at the edges | The paper shows the same. Enlarging the domain (±10 µm) is an option |
| Singularity of the T equation at n_e≈0 | Unstable early T residuals | Keep the residual in division-free form (ñ·∂T̃/∂t̃ = …) |
| Full training is too slow on the Mac | 50k epochs × 50k points | Smoke configs locally, full configs on Colab GPU |
| Colab session limits / disconnects | Long runs stop midway | Periodic checkpoints, `--resume`, outputs on Google Drive (§4.6) |
| Identifiability in the inverse problem (F correlated with α_i, δ_N) | F converges to different values per initial guess | Pass the synthetic check (7.1) first and keep the coefficients fixed |
| Trivial-solution attractor of joint training (seen in Phase 3) | loss_ne stalls at ≈ 5e-3 while loss_Te, loss_phi → 0; or φ → large and I → 0 | R₂ and R_φ are ∝ ñ; train head-wise (D8, D20), keep φ ≥ 0 (D2), check with `scripts/diagnose_run.py` (ñ max, I/I₀ median, z_max·α) before trusting a loss curve |
| Unphysical φ from an untrained head | n_e overshoots n_cr by 10×, loss_phi starts at 1e3–1e5 | φ = z̃·softplus(NN_φ − 2) (D2); never train n_e with a free, unconstrained φ |
| float32 range (full runs on GPU) | loss = inf/NaN from iteration 0 although float64 smoke runs are fine | SI intermediates must stay inside 1e±38: fold unit constants with the nondimensional scale before multiplying tensors (I-32, §4.8); `test_residuals_finite_in_float32` guards it. Never add a new SI product to `pde.py` without checking its magnitude at the pulse peak |

---

## 8. Progress

| Phase | Status | Notes |
|---|---|---|
| 0 Environment | ✅ Done 2026-09-26 | Mac + Colab (T4) DoD passed, 7 tests |
| 1 Physics module | ✅ Done 2026-09-26 | 21 tests; D1/D6 hand calculations reproduced by code |
| 2 FDM calibration | ✅ Done 2026-10-02 | **DoD passed.** D1·D3·D6 confirmed (τ 100 fs, t_c = t_p/2, t_p,eff 90 fs; §3.3); Δt check PASS for glass and SiC; I-31 confirmed → PINN references are RK4 Δt = 1 fs (`glass_ref_rk4_*`, `glass_ref_rk4_photo_*`, `sic_ref_rk4_*`); Euler references of 2.10 kept for "as in the paper" comparison |
| 3 PINN smoke | ✅ Done 2026-10-02 | DoD passed (`smoke_gs`): L2RE n_e 3.6e-2 vs FDM, no NaN, 2.8 orders. D2, D7–D10, D15, D18, D20 confirmed; D8 changed to head-wise training |
| 4 PINN full | ⬜ Ready | Phase 2 DoD passed, I-31 confirmed, `--resume` for staged runs implemented 2026-10-02 (I-24). Next: `full` profile on Colab, evaluate against `glass_ref_rk4_*` |
| 5 Architecture search | ⬜ | Optional |
| 6 Transfer learning | ⬜ | Colab |
| 7 Inverse problem | ⬜ | Colab |
| 8 Wrap-up | ⬜ | |
