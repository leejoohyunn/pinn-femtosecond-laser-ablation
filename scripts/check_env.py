"""Phase 0 smoke check — run on both the Mac and Colab (GUIDE.md Phase 0 DoD).

    python scripts/check_env.py

Prints the installed versions, the compute device DeepXDE will use, and the glass
critical density (expected 1.83e27 m⁻³ = 1.83e21 cm⁻³).
"""

import os
import platform
import sys

os.environ.setdefault("DDE_BACKEND", "pytorch")  # GUIDE §4.4: backend fixed before importing deepxde

import numpy  # noqa: E402
import torch  # noqa: E402
import deepxde as dde  # noqa: E402

from fsl.config import load_material  # noqa: E402


def main() -> int:
    print(f"python   {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")
    print(f"numpy    {numpy.__version__}")
    print(f"torch    {torch.__version__}")
    print(f"deepxde  {dde.__version__}  backend={dde.backend.backend_name}")

    dde_default = torch.get_default_device().type   # what DeepXDE chose at import (cuda / mps / cpu)
    if torch.cuda.is_available():
        dev = f"cuda ({torch.cuda.get_device_name(0)})"
    elif dde_default == "mps":
        dev = "mps chosen by DeepXDE → training forces cpu (no float64 on MPS, I-26)"
    else:
        dev = "cpu"
    print(f"device   {dev}")

    ok = True
    for name in ("glass", "sic", "gan"):
        m = load_material(name)
        print(m)
    glass = load_material("glass")
    target = 1.83e27
    rel = abs(glass.n_cr - target) / target
    print(f"\nglass n_cr = {glass.n_cr:.4e} m⁻³  (target 1.83e27, rel. diff {rel:.2%})")
    if rel > 0.01:
        print("FAIL: glass n_cr off by more than 1%")
        ok = False
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
