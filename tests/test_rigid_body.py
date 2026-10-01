"""The discrete bilinear form annihilates the rigid-body motions.

For a perfectly bonded interface (K = 0) the kernel of a_h contains the global
rigid-body motions; in the decoupled limit (K -> infinity) it contains the
rigid-body motions of the two subdomains taken independently.  Both are checked
on a mesh that the interface cuts, for every polynomial degree used in the
manuscript.  The energy is compared with that of a generic smooth field, so the
test measures a relative quantity.
"""
import numpy as np
import pytest

import mesh as M
from hho import Material, Space, compliance

NRM = np.array([2.0, 1.0]) / np.sqrt(5.0)
CC = float(NRM @ np.array([0.5, 0.5]))
MAT = Material.from_E_nu([1.0, 10.0], [0.25, 0.3])


def _space(k, alpha, beta, n=8):
    m = M.Mesh((0, 1, 0, 1), n, n, M.Interface(NRM, CC))
    return Space(m, k, MAT, compliance(alpha, beta, NRM))


def _rigid(c1, c2):
    """Rigid-body motion with coefficients c_i = (a, b, omega) on subdomain i."""
    def fun(i, p):
        a, b, om = (c1, c2)[i]
        return np.column_stack([a - om * p[:, 1], b + om * p[:, 0]])
    return fun


def _generic(i, p):
    return np.column_stack([np.sin(p[:, 0]), np.cos(p[:, 1])])


def _relative_energy(S, v):
    g = S.interpolate(_generic)
    return abs(float(v @ (S.A @ v))) / float(g @ (S.A @ g))


@pytest.mark.parametrize('k', [1, 2, 3])
def test_bonded_interface_kernel(k):
    S = _space(k, 0.0, 0.0)
    c = (0.7, -0.3, 1.1)
    assert _relative_energy(S, S.interpolate(_rigid(c, c))) < 1e-11


@pytest.mark.parametrize('k', [1, 2, 3])
def test_decoupled_interface_kernel(k):
    S = _space(k, 1e12, 1e12)
    v = S.interpolate(_rigid((0.7, -0.3, 1.1), (-0.2, 0.9, -0.4)))
    assert _relative_energy(S, v) < 1e-11
