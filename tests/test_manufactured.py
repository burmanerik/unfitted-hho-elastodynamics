"""The manufactured solution satisfies the linear slip interface conditions.

The construction of manufactured.py is symbolic; this checks numerically that

    [sigma(u) n] = 0    and    [u] + K sigma(u_1) n = 0     on Gamma

hold to machine precision over the range of compliancies used in the
manuscript, including the degenerate cases alpha = 0 and beta = 0.  The
residuals are normalised by the size of the quantities they compare, since the
displacement jump itself grows linearly with the compliancy.
"""
import numpy as np
import pytest

from hho import Material, compliance
from manufactured import Manufactured

NRM = np.array([2.0, 1.0]) / np.sqrt(5.0)
TAN = np.array([-NRM[1], NRM[0]])
CC = float(NRM @ np.array([0.5, 0.5]))
MAT = Material.from_E_nu([1.0, 10.0], [0.25, 0.3])

CASES = [(0.0, 0.0), (1e-4, 1e-4), (5e-2, 5e-2), (1.0, 1.0), (1e4, 1e4),
         (0.0, 1e-2), (1e-2, 0.0), (0.0, 1.0), (1.0, 0.0)]


@pytest.mark.parametrize('alpha,beta', CASES)
def test_interface_conditions(alpha, beta):
    ex = Manufactured(NRM, CC, MAT.mu, MAT.lam, alpha, beta)
    res_d, res_n = ex.check(compliance(alpha, beta, NRM))
    s = np.linspace(-0.6, 0.6, 7)
    pts = CC * NRM[None, :] + s[:, None] * TAN[None, :]
    scale_d = max(np.abs(ex.u(0, pts) - ex.u(1, pts)).max(), 1.0)
    scale_n = max(np.abs(ex.sigma_n(0, pts)).max(), 1.0)
    assert res_d < 1e-10 * scale_d
    assert res_n < 1e-10 * scale_n
