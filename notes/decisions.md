# 구현 결정 기록 (Implementation Decision Records)

GUIDE.md §3의 D1–D19는 **"논문과 다르게 한 것"**(논문 빈칸·모순을 어떻게 메웠나)이다.
이 파일은 그와 별개로 **"코드를 왜 이렇게 짰나"**를 남긴다 — 나중에 "왜 이 구조지?"가 궁금해질 때 보는 곳.

번호는 `I-n` (Implementation). 한 항목 = 무엇을 / 왜 / 대안은 뭐였고 왜 버렸나 / 상태.
논문 해석과 관련되는 항목은 해당 D-id를 같이 적는다.

---

## Phase 0 (2026-09-25 ~ 09-26)

### I-1. DeepXDE 1.15.0 고정, torch는 고정 안 함
- **결정:** `requirements.txt`에 `deepxde==1.15.0` (작성 시점 PyPI 최신). `torch>=2.2`로만.
- **왜:** DeepXDE는 릴리스마다 API가 바뀌어서(GUIDE §4.4) 맥과 Colab이 다른 버전을 깔면 조용히 깨진다. torch는 반대로 플랫폼마다 휠이 다르고(맥 CPU / Colab CUDA), Colab에 이미 CUDA 빌드가 깔려 있어서 버전을 박으면 오히려 재설치 충돌이 난다.
- **대안:** torch까지 고정 → Colab에서 매번 CUDA torch 재설치(수 GB). 버림.
- **상태:** 확정. 실제 설치 결과 맥 torch 2.14.0, deepxde 1.15.0.

### I-2. `.venv`는 conda 환경 그대로 사용
- **결정:** 2026-09-22에 만들어진 `.venv/`(miniforge conda env, Python 3.11.16)를 그대로 쓴다. 활성화는 `conda activate ./.venv`.
- **왜:** GUIDE 원문은 `python3.11 -m venv`였는데 `python3.11`이 PATH에 없고, 이미 3.11 conda env가 있었다. 새로 만들 이유가 없다.
- **상태:** 확정. GUIDE Phase 0 "User runs" 블록을 이에 맞게 수정.

### I-3. 단위 변환은 `config.py` 한 곳, δ_N만 예외
- **결정:** YAML 키에 단위를 붙이고(`lambda_nm`, `F_Jcm2`, …) `config.py`가 SI dataclass로 바꾼다. 변환 계수(`NM=1e-9` 등)도 이 파일에만 있다. 단 δ_N은 논문 단위(cm⁻³ps⁻¹(cm²/TW)^N) 그대로 들고 다니고 `physics.photoionization_rate` 안에서만 변환한다.
- **왜:** GUIDE §4.2 규칙 3·4. δ_N은 단위가 I^N에 얽혀 있어서 "SI δ_N" 하나로 미리 바꿔 두면 N이 바뀔 때(D5) 같이 틀어진다. 식이 있는 자리에서 변환하는 게 안전하고, 손계산(D6, δ₃I₀³=3.39e21 cm⁻³ps⁻¹)과 바로 대조된다.
- **상태:** 확정. `tests/test_physics.py::test_photoionization_units`로 고정.

### I-4. YAML 숫자는 전부 `float()`로 캐스팅
- **결정:** `config.py`의 `_f()`가 모든 스칼라를 `float()`로 바꾼 뒤 단위를 곱한다.
- **왜:** PyYAML은 YAML 1.1이라 지수에 부호가 없는 `1.0e21`을 **문자열**로 읽는다(`1.0e+21`이어야 float). 09-26 맥에서 `can't multiply sequence by non-int`로 실제 터짐. YAML 쪽을 `1.0e+21`로 고치는 대신 코드에서 막아서, 앞으로 누가 어떤 표기로 써도 안전하게 했다.
- **대안:** YAML 표기 규칙을 강제 → 사람이 잊는다. 버림.
- **상태:** 확정.

### I-5. 재료 YAML에 도메인·grid·scaling까지 넣음
- **결정:** `configs/materials/*.yaml` 하나에 물성(§2.4) + FDM/PINN 도메인(§2.5.1) + grid + 무차원화 기준값(§4.3)을 모두 둔다. `fdm.yaml`, `forward_glass.yaml` 등은 **솔버/학습 옵션**만 갖는다(Phase 2·3에서 생성).
- **왜:** 도메인(GaN ±1 µm), Δr(GaN 0.05 µm, D19), t_max(SiC 285 fs, D16)가 전부 재료마다 다르다. 재료를 바꾸면 이것들이 같이 따라가야 하므로 한 파일이 맞다. scaling은 재료 공통(D15)이지만 "공통이어야 한다"를 코드로 강제하지 않고 각 파일에 같은 값을 적어 두었다 — 전이학습 실험(D15 옵션 `n_ref=n_cr`)에서 재료별로 바꿔 볼 여지를 남기기 위해.
- **상태:** 확정.

### I-6. `t_c`는 숫자 또는 `"tp"`, `"tp/2"`
- **결정:** `tc_fs: 0` / `tc_fs: tp` / `tc_fs: tp/2` 세 표기를 `config.py._parse_tc`가 해석한다.
- **왜:** D3 스캔 후보가 정확히 이 셋이고, `tp`는 재료마다 다르니 숫자로 적으면 SiC(285)·GaN(200)에서 실수한다.
- **상태:** 확정 (D3 값 자체는 Phase 2에서).

### I-7. `load_material(name, **overrides)`로 스캔
- **결정:** 오버라이드는 **YAML 키 이름**으로 받는다(`tau_fs=100`, `alpha_i_cm2J=0`). 없는 키는 `KeyError`.
- **왜:** Phase 2 스캔(τ × t_c, photo_only)이 YAML 파일을 6×2개 만들지 않고 코드에서 돌 수 있게. 키 이름을 YAML과 같게 해서 "SI인지 논문 단위인지" 헷갈릴 일이 없다.
- **상태:** 확정.

### I-8. SiC·GaN의 α_i, δ_N은 glass 값을 YAML에 직접 적음 (D5)
- **결정:** `sic.yaml`, `gan.yaml`에 `1.2`, `7e17`을 그대로 적고 주석으로 "D5: borrowed from glass"라고 표시.
- **왜:** "glass에서 상속" 같은 로직을 config에 넣으면 값이 어디서 왔는지 파일만 봐서는 안 보인다. Phase 6에서 문헌값으로 바꿀 때도 숫자만 고치면 된다.
- **상태:** 확정 (값은 D5, Phase 6).

### I-9. Colab 노트북은 런처만, 토큰은 Secrets
- **결정:** `colab_runner.ipynb`는 Drive 마운트 → Secret `GITHUB_TOKEN`으로 clone → `pip install -e .` → `outputs/`를 Drive에 심링크 → `scripts/*.py` 호출. clone 직후 `remote set-url`로 URL에서 토큰을 지운다.
- **왜:** GUIDE §4.6. 물리/모델 코드가 노트북에 들어가면 맥 코드와 갈라진다. 토큰을 셀에 붙이면 git에 올라간다.
- **상태:** 확정. 09-26 T4에서 검증.

---

## Phase 1 (2026-09-26)

### I-10. numpy/torch 겸용은 얇은 디스패치 `_xp()` 한 벌 (결정 1)
- **결정:** `physics.py`의 각 함수가 `xp = _xp(*inputs)`로 `numpy` 또는 `torch` 모듈을 고른 뒤 `xp.sqrt`, `xp.exp`, `xp.maximum`만 그걸로 부른다. 사칙연산·거듭제곱은 양쪽 공통이라 그대로.
- **왜:** FDM(numpy)과 PINN(torch, autograd 필수)이 **같은 물리 코드**를 실행해야 한다(GUIDE §4.4). Phase 2에서 τ, t_c, δ₃를 여러 번 바꿀 텐데 코드가 두 벌이면 한쪽만 고치는 사고가 난다. `np.sqrt(tensor)`는 autograd가 끊기거나 GPU에서 죽으므로 진짜 디스패치가 필요하다.
- **대안 B:** `physics_np.py` + `physics_torch.py` 두 벌 → drift 위험. 버림.
- **대안 C:** FDM도 torch로 → 자동미분 필요 없는 곳에서 텐서/npz 변환만 늘어남. 버림.
- **검증:** `test_numpy_torch_parity` (float64, rtol 1e-12), `test_torch_gradients_flow`.
- **상태:** 확정.

### I-11. τ는 상수, `surface_optics(n_e, mat, tau=None)` (결정 2, D1)
- **결정:** Drude 체인은 `mat.tau`(YAML `tau_fs`) 상수를 쓴다. `tau=` 인자를 열어 두어 Phase 2 스캔이나 훗날 τ(n_e,T_e) 모델을 호출부에서 넘길 수 있게 했다. **T_e는 인자로 받지 않는다.**
- **왜:** 논문은 τ를 n_e·T_e 함수라 하지만 값이 없고(D1), 유일한 숫자가 τ=100 fs(p.8). 상수 τ면 T_e가 광학 특성에 영향을 주지 않으므로(결합이 n_e→T_e 일방향) T_e를 받는 건 안 쓰는 인자를 PINN pde까지 끌고 다니는 셈이다. τ가 배열이어도 브로드캐스팅으로 그냥 동작하므로 확장 비용이 0이다.
- **오해 방지:** 이 결정은 "τ 값을 100으로 박는다"가 아니다. **값은 Phase 2.6 스캔 {1,5,10,20,50,100} fs에서 R 곡선(Fig.3b)으로 정한다.** 현재 glass/sic YAML의 `tau_fs: 1`은 Phase 2.5 "논문 식 그대로" 자리표시, gan의 `100`은 논문 명시값(D14).
- **대안:** `surface_optics(n_e, T_e, mat)` 논문 원형 → 안 쓰는 인자, D1과 코드 모양 불일치. 버림.
- **상태:** 확정 (구조). 값은 D1.

### I-12. `nk_from_eps`의 sqrt 앞에 `max(·,0)` 가드
- **결정:** `(±ε_r + |ε|)/2`에 `_relu`를 씌운 뒤 sqrt.
- **왜:** 수학적으로 |ε| ≥ |ε_r|이지만 ε_i=0, ε_r>0(n_e≈0)에서 부동소수 반올림이 −1e-17을 만들어 `nan`이 된다. 학습 중 nan 하나가 전체를 죽인다.
- **열린 문제 (Phase 3로 이월):** n_e가 정확히 0이면 k=sqrt(0)이고 torch의 sqrt 기울기가 0에서 inf다. 하드 제약 때문에 t̃=0에서 ñ=0이 정확히 나오는데, collocation point가 t̃=0에 정확히 찍힐 확률은 낮지만 0은 아니다. Phase 3에서 NaN이 보이면 sqrt 안에 ε(1e-30 등)을 더하거나 D10(softplus)과 함께 처리한다.
- **상태:** 확정 (가드), 이월 (기울기).

### I-13. `intensity()`는 R과 φ를 **입력**으로 받는다
- **결정:** 식 (2.5)의 R(t,r), φ(t,r,z)를 함수가 계산하지 않고 인자로 받는다.
- **왜:** R은 표면 n_e에서, φ는 ∫α dz에서 오는데 이 둘을 구하는 방식이 FDM(누적 사다리꼴, 격자)과 PINN(surface query + 보조 출력 φ, D2)에서 완전히 다르다. `intensity`가 그걸 알면 두 경로가 갈라진다. 식 자체만 담고 "어디서 R·φ를 가져오나"는 호출부(fdm.py / pde.py) 책임.
- **상태:** 확정.

---

## Phase 2 (2026-09-26 제안 → 같은 날 확정)

> 상태 `제안` = 사용자 컨펌 전. 컨펌되면 `확정`으로, 바뀌면 바뀐 내용과 이유를 덧붙인다.
> I-14~I-19: 2026-09-26 사용자 컨펌("웅 진행해줘"), 변경 없이 확정.

### I-14. 시간 적분기: explicit Euler Δt=1 fs 기본, RK4는 검증용
- **결정:** `fdm.py`의 기본 적분기는 논문과 같은 explicit Euler, Δt=1 fs. `--integrator rk4 --dt_fs 0.1`로 RK4를 돌려 수렴을 확인한다(GUIDE 2.2, DoD <1%).
- **왜:** 식 (2.1)(2.2)에 공간 미분이 없어 격자 각 점이 독립 ODE이고, 200 fs 동안 n_e가 3자릿수 커지는 정도라 stiff하지 않다. 논문 방법을 기본으로 두어야 "우리 FDM ≈ 논문 FDM"이 성립한다.
- **대안:** scipy `solve_ivp`(적응 스텝) → 격자 전체를 한 벡터로 넘겨야 하고 스냅샷 시각 맞추기가 번거로움. RK4로 충분.
- **상태:** 확정 (2026-09-26) → **2026-10-02 I-31로 대체**: 기준해는 RK4 Δt = 1 fs, Euler는 논문 방식 비교용 옵션.

### I-15. T_e 갱신에 n_e 바닥값
- **결정:** `dT_e/dt = α_h I / (c_e n_e)`에서 `n_e < ne_floor`(기본 1e6 m⁻³, `fdm.yaml`)이면 dT_e = 0.
- **왜:** t=0에 n_e=0이라 0/0. 물리적으로는 α_h ∝ n_e(작은 n_e 극한)라 비율이 유한하지만 부동소수로는 nan. GUIDE 2.3.
- **대안:** `n_e + ε`로 나누기 → ε 크기에 따라 초기 T_e가 미세하게 달라짐. 바닥값 방식이 "전자가 없으면 가열 없음"이라 해석이 깨끗하다.
- **상태:** 확정 (2026-09-26).

### I-16. φ = ∫₀ᶻ α dz' 는 z 격자에서 누적 사다리꼴
- **결정:** `scipy.integrate.cumulative_trapezoid(alpha, z, initial=0)`. z 격자는 반드시 0에서 시작(GUIDE 2.5.1). R은 z=0 행의 n_e에서만 계산.
- **왜:** 논문 Sec.2.2도 격자 적분. Δz=0.02 µm에 α~1e6/m이면 한 칸당 φ 증가 0.02 — 사다리꼴로 충분.
- **상태:** 확정 (2026-09-26).

### I-17. 전체 시간 이력 + R, α 저장
- **결정:** `solution.npz`에 n_e(t,r,z), T_e(t,r,z), R(t,r), α(t,r,z), I(t,r,z)를 매 스텝 저장. 스냅샷만 따로 뽑지 않는다.
- **왜:** glass 201×41×31 ≈ 2.6e5 개 × 5 배열 × float64 ≈ 10 MB. 작다. Phase 4 평가(L2RE at 50/100/150/200 fs)와 fig3(r=z=0 시계열)이 같은 파일을 읽고, `photo_only_shared_R`(I-19)가 R·α를 읽는다.
- **상태:** 확정 (2026-09-26).

### I-18. width/depth는 격자 사이 선형 보간
- **결정:** `eval/profile.py`: z=0 행에서 n_e−n_cr의 부호가 바뀌는 두 격자점 사이를 선형 보간해 폭, r=0 열에서 같은 방식으로 깊이. FDM·PINN 공통(§2.6).
- **왜:** Δr=0.25 µm 격자만 쓰면 폭 8 µm의 해상도가 ±0.25 µm(3%). Phase 4 DoD가 "FDM 대비 ±10%"라 보간 없이는 격자 오차가 판정을 흐린다.
- **대안:** `contour` 함수로 등고선 추출 → 의존성만 늘고 1D 보간과 같은 결과.
- **상태:** 확정 (2026-09-26).

### I-19. `photo_only_shared_R`: full run의 R·α를 파일에서 읽어 씀 (D3 H1 검증)
- **결정:** `--ionization photo_only_shared_R --shared_from outputs/fdm/<full_run>`. 그 run의 R(t,r), α(t,r,z)를 자기 n_e 대신 써서 광이온화 항만 적분.
- **왜:** GUIDE D3 가설 H1 — 논문 Fig.3a 점선(광이온화만)이 ~50 fs에서 갑자기 평탄해지는 건, 자기 n_e(n_cr 미달)로는 R이 안 튀므로 설명이 안 되고, full run의 R을 공유했을 때만 재현된다는 가설. 같은 프로세스 안에서 두 번 돌리는 것보다 파일로 분리하는 게 "어느 run의 R을 썼나"가 남는다.
- **상태:** 확정 (2026-09-26).

---

## Phase 3 (2026-10-02, 사용자 승인 "내가 실행할게" → 구현)

### I-20. PINN 입력·출력은 전부 무차원, 변환은 `scaling.py` 한 곳
- **결정:** `Scales.from_material(mat)`가 YAML의 `domain.pinn`·`scaling`에서 r̃, z̃, t̃, ñ, T̃ 변환을 만든다(GUIDE §4.3). numpy/torch 공용(사칙연산만).
- **왜:** 변환식이 pde.py·evaluate.py·train.py 세 곳에 흩어지면 하나만 바뀌는 사고가 난다. D7/D15 값이 YAML에 있으니 코드에 숫자가 없다.
- **상태:** 확정.

### I-21. smoke는 float64, full은 float32 (`dtype` 프로파일 값)
- **결정:** `forward_glass.yaml`의 프로파일마다 `dtype`. `train.setup()`이 `dde.config.set_default_float(dtype)`를 네트워크 생성 **전에** 호출.
- **왜:** 물리를 SI로 계산하면 n_e~1e27, I~1e17이라 float32 반올림과 모델 버그가 섞인다. 맥 CPU smoke는 float64가 비용이 없고, Colab full은 GUIDE §4.4대로 float32. 둘의 차이는 Phase 4 전에 기록한다.
- **상태:** 확정.

### I-22. 표면 조회는 `net`을 캡처한 클로저 + z̃=0 복사본 재호출
- **결정:** `pde.make_pde(net, mat, sc)`가 `pde(x, y)`를 돌려주고, 안에서 `surface_density(net, x, n_ref)`가 x의 z̃ 열을 0으로 바꾼 복사본으로 `net`을 한 번 더 호출해 n_e(z=0)→R(t,r)을 얻는다.
- **왜:** DeepXDE는 다른 점의 출력을 손실에 쓰는 기능이 없다(GUIDE §4.7). 클로저가 가장 짧고, 출력 변환이 자동 적용되며, 기울기가 x로 흐른다. `tests/test_pinn.py::test_surface_query_*`로 검증(3.3).
- **대안:** auxiliary_var_function으로 R을 미리 계산 → R이 학습 중 갱신되지 않아 틀림. 버림.
- **상태:** 확정.

### I-23. 잔차는 SI 물리 → 무차원 환산, `physics.py` 함수만 호출
- **결정:** pde 안에서 x를 SI(r, t)로 되돌리고 `P.surface_optics`, `P.intensity`, `P.impact_rate`, `P.photoionization_rate`, `P.alpha_total`을 그대로 호출한 뒤 R₁·R₂·R_φ를 §4.3 식으로 만든다. R₂는 나눗셈 없는 형태(ñ ∂T̃/∂t̃ = …, GUIDE §7).
- **왜:** FDM(`fdm._rhs`)과 같은 함수를 쓰므로 물리가 갈라질 수 없다(GUIDE §4.4). `test_residual_matches_manual_formula`가 R₁을 손계산과 대조.
- **상태:** 확정.

### I-24. 체크포인트 두 종류, 재개 시 접두사 `_r<k>`
- **결정:** `ckpt/best-<step>.pt`(최저 학습 손실, `save_better_only`, 검사 주기 = `display_every`)와 `ckpt/periodic-<step>.pt`(재개용, `ckpt_every`). `--resume`은 최신 periodic을 복원하고 남은 반복만 돌리며, 새 파일은 `best_r1-…`, `periodic_r1-…`로 쓴다. history.csv는 step 오프셋을 더해 이어 쓴다.
- **왜:** DeepXDE `restore`는 가중치·옵티마이저만 되돌리고 step 카운터는 0부터라 같은 접두사면 파일이 덮어써진다. evaluate는 best, resume은 periodic을 쓰도록 분리.
- **상태:** 확정.

### I-25. `evaluate.py`는 FDM 격자에서 PINN 추론 (물리 설정 일치 검사 포함)
- **결정:** FDM `solution.npz`의 (r, z) 격자와 report time에서 `predict_fields` → L2RE(식 5.1)·최대 상대오차(점별/전역)·폭/깊이. FDM meta의 `material_yaml`과 PINN run의 material을 비교해 물리 키가 다르면 경고.
- **왜:** 3.8의 전제는 "같은 물리"다. 경고가 없으면 Phase 2 보정 전후 run을 섞어 비교하는 실수를 못 잡는다.
- **상태:** 확정.

### I-26. 디바이스 정책: `auto` = CUDA 있으면 CUDA, 아니면 CPU. MPS는 쓰지 않음
- **결정:** `train.select_device()`가 `torch.set_default_device`를 다시 설정한다. `forward_glass.yaml`의 `device: auto` (cpu/cuda로 고정 가능).
- **왜:** DeepXDE 1.15의 `backend/pytorch/tensor.py`는 import 시점에 Apple Silicon에서 **기본 디바이스를 MPS로** 바꾼다(Phase 0 `check_env.py`의 "DeepXDE uses cpu on Mac" 문구는 틀렸음, 같이 수정). MPS는 float64를 지원하지 않아 smoke(I-21)가 깨지고, PINN의 이중 역전파 연산이 MPS에서 검증돼 있지 않다. GUIDE D13도 맥은 CPU다.
- **상태:** 확정.

### I-27. n, k는 복소수 sqrt로 계산 (PINN NaN 수정)
- **결정:** `physics.nk_from_eps`를 `n + ik = sqrt(ε_r + i|ε_i|)`(주값 복소 제곱근)로 바꿈. numpy·torch 공용.
- **왜:** 첫 smoke(2026-10-02, glass_smoke_20261002-1039)가 100 iter 안에 NaN. 실수식 k = sqrt((−ε_r+|ε|)/2)는 n_e→0에서 |ε|−ε_r이 0으로 반올림되고 sqrt(0)의 기울기가 ∞라 역전파가 NaN이 된다(I-12에서 Phase 3로 이월한 그 지점). 복소 sqrt는 ε≈1에서 기울기가 유한하고 cancellation이 없다. |ε_i|는 네트워크가 잠시 n_e<0을 내도 k≥0(α_h≥0)을 보장. FDM(n_e≥0)에서는 값이 동일하다.
- **검증:** `tests/test_physics.py` 극한·numpy/torch 동등성 테스트가 그대로 통과해야 함. smoke 재실행으로 NaN 소멸 확인.
- **상태:** 확정 (수정 후 smoke 결과 대기).

### I-28. 손실 균형(D8) 수단: 커리큘럼(단계별 가중치)과 출력별 독립 서브넷(PFNN)을 둘 다 옵션으로
- **배경 (2026-10-02 smoke):** λ=(1,1,1)이면 T·φ 잔차가 ñ에 비례해 ñ→0인 자명해로 끌려가 loss_ne가 5.6e-3(ñ≡0일 때의 소스항 제곱 평균, 손계산과 일치)에서 멈춤. λ=(1,0,0)이면 2000 iter에 2.2자릿수 감소. λ=(100,1,1)이면 이번엔 φ 식이 굶어 loss_phi 6e-2, loss_ne 1e-3에서 정체 — 세 식이 은닉층을 공유해 서로 당김.
- **결정:** `TrainConfig.stages`(iterations, loss_weights 목록)로 단계별 compile/train. 단계마다 ModelCheckpoint를 새로 만들어 'best'가 마지막 단계 기준이 되게 함. history.csv에 가중치 열(w_ne, w_Te, w_phi)을 추가해 비가중 잔차를 복원 가능. `net: pfnn`은 `dde.nn.PFNN`으로 출력마다 독립 서브넷(논문은 단일 FNN이므로 채택 시 D-로그에 편차 기록).
- **상태:** 구현 완료, D8 값은 smoke_cl / smoke_pfnn 결과로 결정.

### I-29. 블록 가우스–자이델 학습 + φ ≥ 0 하드제약
- **배경 (2026-10-02 smoke D/E):** 커리큘럼(D)은 1단계(n_e만) 뒤 2단계에서 loss_phi가 1679로 시작 — 학습 안 된 φ 헤드가 음수 φ(깊이 방향 증폭)를 내서 n_e가 n_cr의 10배까지 부풀었고, φ 식을 켜자 R_φ를 줄이는 최단 경로(ñ→0)로 붕괴. PFNN(E)만으로는 끌림이 안 사라짐. 구조적 원인: R_φ, R₂가 ñ에 비례해 ñ을 줄이면 자기 손실이 공짜로 내려감.
- **결정:** (1) φ = z̃·softplus(NN_φ − 2) — 광학깊이는 음수가 될 수 없고, 초기 감쇠를 작게(≈0.13 z̃) 둠. `phi_positive: true` 기본, false면 논문형 z̃·NN_φ. (2) `stages`에 `heads`를 추가해 **한 단계에 한 헤드만 학습**(PFNN 필요): n 헤드 ↔ R₁, φ 헤드 ↔ R_φ, T 헤드 ↔ R₂를 `rounds`번 반복. 각 부분문제는 다른 헤드가 고정돼 있어 ñ을 눌러서 손실을 낮출 수 없고, φ·T 단계는 사실상 적분/선형 적합이라 수렴이 쉬움. 바깥 반복은 결합(감쇠 I ← φ ← α(n_e))에 대한 피카드 반복.
- **논문과의 차이:** 논문은 단일 FNN·동시 학습. 채택되면 D8 결정에 "손실 가중치 = 단계별 (1,0,0)/(0,0,1)/(0,1,0), 헤드별 교대"로 기록하고 GUIDE §4.7 표를 갱신.
- **상태:** 구현 완료 (`smoke_gs` 프로파일), 결과 대기.
- **I-29 추가 (2026-10-02 저녁, smoke_F_gs 진단):** 첫 구현에서는 동결이 안 먹었다. DeepXDE `Model._outputs_losses`(`_test`에서 호출, 학습 시작 시와 `display_every`마다)가 `net.requires_grad_(False)` 뒤 `net.requires_grad_()`로 **모든 파라미터를 다시 켠다.** 그래서 n 단계에서 φ 헤드가 같이 학습돼 φ≈200으로 I를 0으로 만들고(ñ≈2e-3, I/I₀~1e-89) R₁=1.7e-5의 자명해가 나왔다. 수정: `HeadFreezer` 콜백이 `on_train_begin`·`on_epoch_begin`마다 동결을 다시 건다. 회귀 테스트 `test_frozen_heads_do_not_change_during_training`(실제 `model.train`에서 비학습 헤드가 비트 단위로 불변인지 확인). 진단 도구 `scripts/diagnose_run.py` 추가(콜로케이션 점에서 중간량 백분위수, FDM 대조).

### I-30. Phase 2 보정 결과를 YAML 기본값으로, 논문 값은 주석·테스트 override로
- **결정:** `configs/materials/glass.yaml` 기본값을 보정 세트(tp_fs 90, tc_fs "tp/2", tau_fs 100)로 바꿈. sic/gan은 모델 공통 결정(D1 τ=100, D3 tc=tp/2)만 반영하고 t_p는 논문 값 유지. 논문 Table 1 그대로의 손계산을 검증하는 테스트(`test_physics` glass fixture, `test_fdm.test_photo_only_matches_analytic_integral`)는 `load_material("glass", tp_fs=200, tc_fs=0)`으로 논문 세트를 명시. `test_config.test_glass_calibrated_defaults`가 기본값이 보정 세트임을 고정.
- **왜:** GUIDE §0 "Phase 2에서 확정한 물리는 이후 바뀌지 않는다" — 모든 스크립트(run_fdm, train_forward, evaluate)가 YAML 기본값을 읽으므로 기본값이 곧 동결된 물리다. 논문 값은 재현 편차를 추적할 수 있게 주석과 `--set tp_fs=200 tc_fs=0`로 남긴다. `evaluate.py`의 물리 불일치 경고(I-25)가 보정 전 FDM(`smoke_ref`)과 섞이는 실수를 잡는다.
- **판정 도구:** `fdm.metrics`에 `t90_fs`(최종값 90% 도달 시각 = 평탄화 시각), `ne_ratio_25_50`(0.5 = 선형 시작, <0.5 볼록) 추가; `scripts/judge_d6.py`가 full/photo sweep.csv를 조합 키로 join해 GUIDE 2.9의 8개 기준을 점수화(채택은 사용자 확정, 자동 아님). 기준을 스캔 전에 고정한 이유: 결과를 보고 기준을 고르는 걸 막기 위해.
- **2.10 도구:** `scripts/check_dt.py` — (적분기, Δt) 케이스를 돌려 마지막 케이스를 기준으로 r=z=0 값·L2RE·t(n_cr)·폭/깊이의 상대차를 표로 출력, DoD(<1%) 판정, summary.json 저장. [27]의 Δt = 0.5 fs 포함.
- **상태:** 확정 (2026-10-02, 사용자 컨펌 "가족 A, τ=100").

### I-31. (제안) PINN 평가용 기준해는 RK4 Δt = 1 fs, Euler 1 fs는 "논문 방식" 옵션으로
- **배경 (2026-10-02, 2.10):** 보정 세트에서 Euler 1 fs vs RK4 0.1 fs 차이는 r=z=0 끝값으로 0.19 % / 0.81 %(Phase 2 DoD 통과)지만, 최종 장 전체 L2RE로는 n_e 0.6 %, T_e 1.3 %, 깊이는 5.9 %(251 vs 237 nm). Phase 4 DoD(L2RE n_e < 1e-2, T_e < 2e-2, 폭·깊이 ±10 %)의 상당 부분을 기준해 자체의 이산화 오차가 차지한다. RK4 Δt = 1 fs는 0.1 fs 대비 ≤ 8e-4로 수렴했고 비용은 0.05 s/run.
- **제안:** `configs/fdm.yaml` `integrator: rk4`(Δt 1 fs)를 기본으로 바꾸고 `glass_ref`·`glass_ref_photo`·`sic_ref`를 재생성해 Phase 4 `evaluate.py` 기준으로 쓴다. Euler 1 fs는 `--integrator euler`로 유지(논문 방식 비교, 2.5 as-written 재현).
- **왜 I-14를 뒤집나:** I-14의 근거 "우리 FDM ≈ 논문 FDM"은 D6 보정으로 이미 성립하지 않는다(t_p가 다름). PINN은 연속 ODE를 근사하므로 기준해는 수렴한 해여야 하고, 그래야 L2RE가 PINN 오차만 재게 된다. 테스트 `test_euler_vs_rk4_convergence`(<1 %)는 그대로 둔다.
- **상태:** 확정 (2026-10-02, 사용자 "ㅇㅇ RK4로 가자"). `configs/fdm.yaml` `integrator: rk4`; 기준해 `glass_ref_rk4`·`glass_ref_rk4_photo`·`sic_ref_rk4` 재생성. I-14는 "Euler = 논문 방식 옵션(`--integrator euler`)"으로 축소. GUIDE §3.3·§4.8에 기록.

### I-24 추가 (2026-10-02): 단계형(stages/rounds) run의 `--resume`
- **결정:** `train()`이 재시작 시 (1) 최신 periodic 체크포인트의 **전역 반복 수**를 `ckpt/offsets.json`(suffix → 그 라운드가 시작한 전역 step)으로 복원하고, (2) `history.csv`에서 그 step 이후 행을 지우고(체크포인트 뒤에 돌았다가 끊긴 구간은 다시 돈다), (3) `plan_stages(schedule, done)`으로 끝난 단계는 건너뛰고 중단된 단계는 남은 반복만, 그 뒤 단계는 그대로 이어서 돈다. `scripts/train_forward.py --resume`은 run의 `config.yaml`에서 설정·재료를 읽고(`--config/--profile` 무시) `--set`만 덧씌움 — `--set rounds=6`으로 끝난 run을 한 라운드 연장 가능.
- **왜:** DeepXDE는 `restore` 뒤 `train_state.iteration`을 0부터 세므로 파일명의 step만으로는 전역 위치를 모른다(이전엔 history 마지막 step을 썼는데, 그러면 마지막 체크포인트 뒤에 돈 구간을 "끝난 것"으로 잘못 친다). Phase 4 full(50k iter, Colab)은 세션 끊김이 거의 확실하므로 단계 안에서 이어 붙일 수 있어야 한다. Adam 상태는 단계 compile마다 어차피 초기화되므로 재시작이 단계 중간에 떨어져도 추가 손실은 없다. 콜로케이션 점은 같은 seed로 다시 뽑혀 동일(D18).
- **검증:** `tests/test_pinn.py::test_staged_resume_continues_inside_the_schedule` — 2라운드×[ne 10/φ 5/T 5] run을 step 25(라운드 1 ne 단계 중간)에서 끊은 것으로 만들어 재시작: (1, ne, 5)→(1, φ, 5)→(1, T, 5), history 단조·40에서 끝남, rounds=3으로 연장 시 60.
- **상태:** 구현 완료, 테스트 결과 대기.

### I-32. float32 오버플로/언더플로: 비율 상수를 무차원 스케일과 먼저 묶는다
- **배경 (2026-10-05, Colab smoke_gs float32):** 첫 반복부터 loss_ne = inf → exit 1. 맥 CPU float32로 재현. 원인: `pde.py`가 SI 비율을 만든 뒤 나누는 `t_ref * photoionization_rate(I) / n_ref` 구조라, 중간값 δ_N I_tw^N × 1e18 ≈ 4e40 m⁻³s⁻¹(유리 피크)와 α_i I n_e ≈ 5e40이 float32 최대 3.4e38을 넘는다. 논문 그대로의 값(3e39)도 넘친다. Phase 3 smoke는 전부 float64(I-21)라 걸리지 않았다.
- **결정:** (1) `physics.photoionization_rate(I, mat, scale=1)`·`impact_rate(I, n_e, mat, scale=1)`에 `scale` 인자를 두고 상수(1e18·scale 등)를 파이썬 float64에서 먼저 곱한 뒤 텐서에 적용. PINN은 `scale = t_ref/n_ref`로 호출해 무차원 소스(≈4)를 바로 받는다. FDM(float64)은 기본값 그대로. (2) `impact_rate`는 `(α_i·I)·(n_e·scale)` 순서 — `α_i·scale = 1.2e-44`는 float32 비정규수라 5 % 오차가 났다. (3) `plasma_freq_sq`는 `e²/(m_e ε₀)`를 먼저 접음(`/ 8e-42` 비정규수 나눗셈 회피). (4) `heat`도 상수 묶음.
- **검증:** `tests/test_pinn.py::test_residuals_finite_in_float32` — 같은 가중치의 float32/float64 네트로 피크 근처 점에서 src_photo·src_impact·heat·α_h가 1e-3 이내, R1·R2·Rphi 1e-2 이내, 전부 유한. 맥 CPU float32 400-iter 단계 학습 NaN 없음. 수학적으로 동일하므로 float64 결과(Phase 3 smoke, FDM)는 변하지 않는다(`test_residual_matches_manual_formula` 그대로 통과).
- **교훈:** SI 단위의 중간값은 float32 범위(1e±38)를 쉽게 넘는다. 무차원 잔차를 쓰는 코드에서는 상수를 스케일과 함께 먼저 접고, 텐서에는 O(1)~O(1e15) 범위 값만 곱한다. 노트북 런처는 `subprocess.run` 대신 출력을 스트리밍하는 `Popen`으로 바꿈(실패 원인이 셀에 보이지 않았음).
- **상태:** 확정 (2026-10-05).

### I-33. Phase 4 그림·표: 그림 함수는 배열만 받고, `evaluate.py` 하나가 보고서를 만든다
- **결정:** `eval/compare.py`(PINN 평가 헬퍼: `pinn_fields`, `pinn_center_series`, `compare_at_times`, `profile_comparison`, `dod_check`, `table2_markdown`)와 `eval/plots.py`의 `fig3`·`fig4`·`fig5a`·`fig10`. 그림 함수는 PINN 쪽을 numpy 배열로 받는다(네트 호출은 compare.py에만). `scripts/evaluate.py`가 지표·Table 2(`eval/table2.md`)·DoD 판정·그림(`eval/figs/`)을 한 번에 쓴다 — 태스크팩의 4.4 `report_phase4.py`는 별도 스크립트 대신 evaluate.py에 합침(명령 하나로 끝나는 쪽이 Colab 셀에 맞음).
- **왜 배열 인터페이스인가:** FDM 장을 PINN 자리에 넣으면 오차 0인 "정답" 케이스가 되어 그림·표·DoD 코드를 학습 없이 테스트할 수 있다(`tests/test_eval_figs.py`). PINN이 실제로 들어와도 코드 경로가 같다.
- **Fig.3의 R·α(PINN):** PINN의 표면 n_e를 FDM과 같은 `physics.surface_optics`에 넣어 계산. 네트는 n_e·T_e·φ만 내므로 R·α는 유도량이다.
- **Fig.4 오차 맵:** §4.5의 점별 상대오차(분모 하한 1e-3·max, 그 밖은 0으로 표시)로 그리고 제목에 점별 최대와 전역 최대를 함께 적는다(논문 2.5 %/9 %와 비교).
- **Fig.10:** z = 200 nm 단면 4시각의 n_e·T_e(FDM 선, PINN 마커) + (c)(d) 절대오차 패널 추가(9/23 제안; 논문 본문이 n_e 오차 피크 50 fs, T_e 피크 150 fs라고 적어 둔 것을 확인하기 위해).
- **광이온화-only 곡선:** `--photo` 없으면 `--fdm` 폴더 옆의 `<stem>_photo_<stamp>`를 자동 탐색.
- **DoD 판정:** metrics.json의 `dod`(6개 체크: L2RE n_e/T_e 최종 시각, 폭·깊이 ±10 %, L2RE 단조증가 아님)와 `dod_pass`.
- **상태:** 구현 완료 (2026-10-05), 사용자 컨펌 "ㅇㅇ 그림 코드 짜줘".
