"""Convergence of the stationary scheme, in three increasingly demanding settings.

1. no interface (plain HHO for elasticity), to check the core discretisation;
2. an interface cut by the mesh, same material on both sides and K = 0, so that
   the exact solution is smooth across Gamma;
3. the manufactured linear slip solution of the manuscript (test case 1).

In each case the observed rates must reach the h^{k+1} / h^{k+2} predicted by
Theorems 5.6 and 5.9, up to the usual preasymptotic slack.
"""
import numpy as np
import pytest

import mesh as M
from case1 import run as run_slip
from hho import Material, Space, solve_static

MAT = Material.from_E_nu([1.0, 1.0], [0.25, 0.25])
NS = (4, 8, 16)
ESLACK, L2SLACK = 0.3, 0.4


def uex(i, p):
    x, y = p[:, 0], p[:, 1]
    return np.column_stack([np.sin(np.pi * x) * np.sin(np.pi * y),
                            np.sin(2 * np.pi * x) * np.sin(np.pi * y)])


def fex(i, p):
    x, y = p[:, 0], p[:, 1]
    pi = np.pi
    mu, lam = MAT.mu[i], MAT.lam[i]
    u1 = np.sin(pi * x) * np.sin(pi * y)
    u2 = np.sin(2 * pi * x) * np.sin(pi * y)
    u1xx = -pi ** 2 * u1
    u1yy = -pi ** 2 * u1
    u1xy = pi ** 2 * np.cos(pi * x) * np.cos(pi * y)
    u2xx = -4 * pi ** 2 * u2
    u2yy = -pi ** 2 * u2
    u2xy = 2 * pi ** 2 * np.cos(2 * pi * x) * np.cos(pi * y)
    f1 = -(mu * (u1xx + u1yy) + (mu + lam) * (u1xx + u2xy))
    f2 = -(mu * (u2xx + u2yy) + (mu + lam) * (u1xy + u2yy))
    return np.column_stack([f1, f2])


def _rate(errs):
    """Rate observed on the last mesh refinement."""
    return float(np.log(errs[-2] / errs[-1]) / np.log(2.0))


def _smooth_errors(k, iface):
    ee, e0 = [], []
    for n in NS:
        m = M.Mesh((0, 1, 0, 1), n, n, iface)
        S = Space(m, k, MAT, np.zeros((2, 2)))
        e = solve_static(S, S.load(fex), ud=S.interpolate(uex)) - S.interpolate(uex)
        ee.append(S.energy_norm(e))
        e0.append(S.l2_cell_norm(e))
    return ee, e0


@pytest.mark.parametrize('k', [1, 2])
def test_no_interface(k):
    """All cells on one side of Gamma: plain HHO for linear elasticity."""
    ee, e0 = _smooth_errors(k, M.Interface([0.0, 1.0], 10.0))
    assert _rate(ee) > k + 1 - ESLACK
    assert _rate(e0) > k + 2 - L2SLACK


@pytest.mark.parametrize('k', [1, 2])
def test_cut_mesh_continuous_solution(k):
    """Interface cutting the mesh, identical materials and K = 0."""
    nrm = np.array([2.0, 1.0]) / np.sqrt(5.0)
    ee, e0 = _smooth_errors(k, M.Interface(nrm, float(nrm @ np.array([0.5, 0.5]))))
    assert _rate(ee) > k + 1 - ESLACK
    assert _rate(e0) > k + 2 - L2SLACK


@pytest.mark.parametrize('k', [1, 2])
def test_slip_interface(k):
    """Test case 1 of the manuscript, on the three coarsest meshes."""
    ee, e0 = [], []
    for n in NS:
        r = run_slip(k, n, 5e-2, 5e-2)
        ee.append(r['Ee'])
        e0.append(r['E0'])
    assert _rate(ee) > k + 1 - ESLACK
    assert _rate(e0) > k + 2 - L2SLACK
