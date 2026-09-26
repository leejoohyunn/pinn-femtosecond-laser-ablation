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

## Phase 2 (2026-09-26 제안, 컨펌 대기)

> 상태 `제안` = 사용자 컨펌 전. 컨펌되면 `확정`으로, 바뀌면 바뀐 내용과 이유를 덧붙인다.

### I-14. 시간 적분기: explicit Euler Δt=1 fs 기본, RK4는 검증용
- **결정:** `fdm.py`의 기본 적분기는 논문과 같은 explicit Euler, Δt=1 fs. `--integrator rk4 --dt_fs 0.1`로 RK4를 돌려 수렴을 확인한다(GUIDE 2.2, DoD <1%).
- **왜:** 식 (2.1)(2.2)에 공간 미분이 없어 격자 각 점이 독립 ODE이고, 200 fs 동안 n_e가 3자릿수 커지는 정도라 stiff하지 않다. 논문 방법을 기본으로 두어야 "우리 FDM ≈ 논문 FDM"이 성립한다.
- **대안:** scipy `solve_ivp`(적응 스텝) → 격자 전체를 한 벡터로 넘겨야 하고 스냅샷 시각 맞추기가 번거로움. RK4로 충분.
- **상태:** 제안.

### I-15. T_e 갱신에 n_e 바닥값
- **결정:** `dT_e/dt = α_h I / (c_e n_e)`에서 `n_e < ne_floor`(기본 1e6 m⁻³, `fdm.yaml`)이면 dT_e = 0.
- **왜:** t=0에 n_e=0이라 0/0. 물리적으로는 α_h ∝ n_e(작은 n_e 극한)라 비율이 유한하지만 부동소수로는 nan. GUIDE 2.3.
- **대안:** `n_e + ε`로 나누기 → ε 크기에 따라 초기 T_e가 미세하게 달라짐. 바닥값 방식이 "전자가 없으면 가열 없음"이라 해석이 깨끗하다.
- **상태:** 제안.

### I-16. φ = ∫₀ᶻ α dz' 는 z 격자에서 누적 사다리꼴
- **결정:** `scipy.integrate.cumulative_trapezoid(alpha, z, initial=0)`. z 격자는 반드시 0에서 시작(GUIDE 2.5.1). R은 z=0 행의 n_e에서만 계산.
- **왜:** 논문 Sec.2.2도 격자 적분. Δz=0.02 µm에 α~1e6/m이면 한 칸당 φ 증가 0.02 — 사다리꼴로 충분.
- **상태:** 제안.

### I-17. 전체 시간 이력 + R, α 저장
- **결정:** `solution.npz`에 n_e(t,r,z), T_e(t,r,z), R(t,r), α(t,r,z), I(t,r,z)를 매 스텝 저장. 스냅샷만 따로 뽑지 않는다.
- **왜:** glass 201×41×31 ≈ 2.6e5 개 × 5 배열 × float64 ≈ 10 MB. 작다. Phase 4 평가(L2RE at 50/100/150/200 fs)와 fig3(r=z=0 시계열)이 같은 파일을 읽고, `photo_only_shared_R`(I-19)가 R·α를 읽는다.
- **상태:** 제안.

### I-18. width/depth는 격자 사이 선형 보간
- **결정:** `eval/profile.py`: z=0 행에서 n_e−n_cr의 부호가 바뀌는 두 격자점 사이를 선형 보간해 폭, r=0 열에서 같은 방식으로 깊이. FDM·PINN 공통(§2.6).
- **왜:** Δr=0.25 µm 격자만 쓰면 폭 8 µm의 해상도가 ±0.25 µm(3%). Phase 4 DoD가 "FDM 대비 ±10%"라 보간 없이는 격자 오차가 판정을 흐린다.
- **대안:** `contour` 함수로 등고선 추출 → 의존성만 늘고 1D 보간과 같은 결과.
- **상태:** 제안.

### I-19. `photo_only_shared_R`: full run의 R·α를 파일에서 읽어 씀 (D3 H1 검증)
- **결정:** `--ionization photo_only_shared_R --shared_from outputs/fdm/<full_run>`. 그 run의 R(t,r), α(t,r,z)를 자기 n_e 대신 써서 광이온화 항만 적분.
- **왜:** GUIDE D3 가설 H1 — 논문 Fig.3a 점선(광이온화만)이 ~50 fs에서 갑자기 평탄해지는 건, 자기 n_e(n_cr 미달)로는 R이 안 튀므로 설명이 안 되고, full run의 R을 공유했을 때만 재현된다는 가설. 같은 프로세스 안에서 두 번 돌리는 것보다 파일로 분리하는 게 "어느 run의 R을 썼나"가 남는다.
- **상태:** 제안.
