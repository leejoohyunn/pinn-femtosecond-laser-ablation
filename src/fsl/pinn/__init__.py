"""PINN components (GUIDE.md §4.7): net, pde, data, train.

The DeepXDE backend must be chosen before ``deepxde`` is imported anywhere
(GUIDE §4.4). Importing any ``fsl.pinn`` module runs this file first, so the
environment variable is set here once.
"""

import os

os.environ.setdefault("DDE_BACKEND", "pytorch")
