# pinn-femtosecond-laser-ablation

Reproduction of Y. Gao et al., *PINN-based computational framework for femtosecond laser
ablation: multi-material modeling and parameter estimation*, Optics & Laser Technology 199
(2026) 115023, with DeepXDE (PyTorch backend).

**`GUIDE.md` is the single source of truth** — physics, decision log (D1–D19), task packs per
Phase, and code rules. Read it before touching anything.

## Setup (Mac)

```bash
cd ~/lab/femtosecond
source .venv/bin/activate          # Python 3.11
pip install -e . -r requirements.txt
python scripts/check_env.py        # Phase 0 smoke check
pytest -q
```

## Setup (Colab GPU)

Open `notebooks/colab_runner.ipynb`, select a GPU runtime, run all cells. The notebook only
clones this repo, installs it, and calls the same `scripts/*.py` used on the Mac.

## Layout

See GUIDE.md §4.1. Physics/config code lives in `src/fsl/`, run configs in `configs/`,
entry points in `scripts/`, run outputs in `outputs/` (git-ignored, synced via Drive).
