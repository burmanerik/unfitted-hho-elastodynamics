"""Convergence and energy behaviour of the fully discrete schemes.

The space rates and the Newmark order are measured on test case 1 in the time
domain, whose exact solution is u(x, t) = cos(omega t) u_s(x) with u_s the
manufactured slip solution.  The temporal orders of the Runge-Kutta schemes are
measured on test case 2 (homogeneous boundary data, free vibration), as in
Table 9 of the manuscript: with the time-dependent Dirichlet data of test case 1
the SDIRK stages suffer the usual order reduction and only order 2.4 is seen.
"""
import numpy as np
import pytest

import case1_dyn as td
import case23 as c23

SLACK = 0.3


def _rate(errs, fac=2.0):
    return float(np.log(errs[-2] / errs[-1]) / np.log(fac))


@pytest.mark.parametrize('k', [1, 2])
def test_newmark_space_convergence(k):
    """With dt small enough, the error is dominated by the space discretisation."""
    ee, e0 = [], []
    for n in (4, 8, 16):
        S, ex = td.setup(k, n)
        E0, Ee = td.integrate(S, ex, 1.0 / 2000, T=0.5, ret='max')
        ee.append(Ee)
        e0.append(E0)
    assert _rate(ee) > k + 1 - SLACK
    assert _rate(e0) > k + 2 - SLACK - 0.1


def test_newmark_temporal_order():
    """Newmark with beta = 1/4, gamma = 1/2 is second-order accurate."""
    S, ex = td.setup(2, 8)
    ref = td.integrate(S, ex, 1.0 / 3200, T=0.5, ret='raw')
    errs = [S.l2_cell_norm(td.integrate(S, ex, 1.0 / nt, T=0.5, ret='raw') - ref)
            for nt in (50, 100, 200)]
    assert _rate(errs) > 2.0 - SLACK


@pytest.fixture(scope='module')
def wave_reference():
    """Test case 2 on a coarse mesh, integrated with a very small time step."""
    r = c23.run(1, 12, 0.0, 0.0, dt=1.0 / 2560, T=0.25, scheme=('sdirk', 3))
    return r['S'], r['u'].copy()


@pytest.mark.parametrize('scheme,order', [('newmark', 2), (('sdirk', 2), 3),
                                          (('sdirk', 3), 4)])
def test_temporal_order(wave_reference, scheme, order):
    """Newmark is of order 2, SDIRK(s, s+1) of Crouzeix of order s + 1."""
    S, uref = wave_reference
    errs = [S.l2_cell_norm(c23.run(1, 12, 0.0, 0.0, dt=1.0 / nt, T=0.25,
                                   scheme=scheme)['u'] - uref)
            for nt in (80, 160, 320)]
    assert _rate(errs) > order - SLACK


def test_newmark_conserves_energy():
    """The Newmark scheme conserves the discrete energy (63) exactly."""
    e = np.array(c23.run(1, 16, dt=1.0 / 200, T=0.5, scheme='newmark')['energy'])
    assert (e.max() - e.min()) / e[0] < 1e-10


def test_sdirk_dissipates_energy():
    """SDIRK(3,4) is dissipative: the discrete energy decreases monotonically."""
    e = np.array(c23.run(1, 16, dt=1.0 / 200, T=0.5, scheme=('sdirk', 3))['energy'])
    assert np.all(np.diff(e) <= 1e-14 * e[0])
    assert 0.0 < (e[0] - e[-1]) / e[0] < 1e-1
