"""Physics functions (GUIDE.md §2, §4.2). All inputs/outputs in SI.

Phase 0 only provides ``critical_density`` (needed for the config DoD).
The Drude chain, ionization rates and intensity are added in Phase 1.
Functions must accept both numpy and torch inputs (§4.4).
"""

import math

from fsl import constants as C


def critical_density(lambda_m: float) -> float:
    """Critical electron density n_cr = 4π² c² m_e ε₀ / (λ² e²)  [m⁻³] (GUIDE §2.3)."""
    return 4.0 * math.pi**2 * C.c**2 * C.m_e * C.eps0 / (lambda_m**2 * C.e**2)
