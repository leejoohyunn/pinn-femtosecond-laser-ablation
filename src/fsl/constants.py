"""Physical constants in SI units (GUIDE.md §4.2 rule 1).

Only fundamental constants live here. Unit-conversion factors (nm → m, fs → s, ...)
belong to ``config.py`` (rule 3), and the δ_N conversion to ``physics.photoionization_rate``
(rule 4).
"""

# CODATA 2018
c = 2.99792458e8          # speed of light            [m/s]
e = 1.602176634e-19       # elementary charge         [C]
m_e = 9.1093837015e-31    # electron mass             [kg]
eps0 = 8.8541878128e-12   # vacuum permittivity       [F/m]
k_B = 1.380649e-23        # Boltzmann constant        [J/K]
h = 6.62607015e-34        # Planck constant           [J s]
eV = 1.602176634e-19      # 1 eV in joules            [J]

T_ROOM = 300.0            # initial / boundary electron temperature (paper §2.4) [K]
