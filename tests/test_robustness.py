"""Robustness of the method with respect to the two sources of degeneracy.

* the compliancy: one single formulation must cover the whole range from the
  perfectly bonded to the traction-free interface (Table 2);
* the position of the interface: without agglomeration the conditioning of the
  stiffness matrix degenerates as the cut becomes small, with agglomeration it
  does not (Table 3).
"""
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spl

import mesh as M
from case1 import run as run_slip
from hho import Material, Space, compliance, solve_static
from manufactured import Manufactured

MAT = Material.from_E_nu([1.0, 10.0], [0.25, 0.3])
ALPHA = BETA = 5e-2


@pytest.mark.parametrize('alpha,beta', [(0.0, 0.0), (1e-6, 1e-6), (1.0, 1.0),
                                        (1e6, 1e6), (1e12, 1e12),
                                        (0.0, 1.0), (1.0, 0.0)])
def test_compliancy_robustness(alpha, beta):
    """The relative errors stay at the same level over the whole range.

    On this very coarse mesh they sit between 5e-2 and 9e-2 in the energy norm
    and between 9e-3 and 2.2e-2 in the L2-norm, bonded and traction-free limits
    included; the bounds below leave a factor of about two.
    """
    r = run_slip(1, 8, alpha, beta)
    assert r['Ee'] / r['nrm_e'] < 0.15
    assert r['E0'] / r['norm_i'] < 0.05


def _small_cut(xi, agglo_tol, n=8, k=1):
    """Horizontal interface at distance xi*h above a mesh line."""
    nrm = np.array([0.0, 1.0])
    cc = 0.5 + xi / n
    ex = Manufactured(nrm, cc, MAT.mu, MAT.lam, ALPHA, BETA)
    m = M.Mesh((0, 1, 0, 1), n, n, M.Interface(nrm, cc), agglo_tol=agglo_tol)
    S = Space(m, k, MAT, compliance(ALPHA, BETA, nrm))
    Aff = S.A[S.free][:, S.free].tocsc()
    D = sp.diags(1.0 / np.sqrt(Aff.diagonal()))
    Asc = (D @ Aff @ D).tocsc()
    lmax = spl.eigsh(Asc, k=1, which='LA', tol=1e-4, return_eigenvectors=False)[0]
    lmin = spl.eigsh(Asc, k=1, sigma=0.0, which='LM', tol=1e-4,
                     return_eigenvectors=False)[0]
    ui = S.interpolate(ex.u)
    e = solve_static(S, S.load(ex.f), ud=ui) - ui
    return abs(float(lmax / lmin)), S.energy_norm(e) / S.energy_norm(ui), m.n_agglo


def test_small_cut_without_agglomeration_degenerates():
    c_fat, _, _ = _small_cut(0.5, 0.0)
    c_thin, _, nagg = _small_cut(1e-4, 0.0)
    assert nagg == 0
    assert c_thin / c_fat > 20.0


def test_agglomeration_cures_the_small_cut():
    c_fat, e_fat, _ = _small_cut(0.5, 0.3)
    c_thin, e_thin, nagg = _small_cut(1e-4, 0.3)
    assert nagg > 0
    assert c_thin / c_fat < 3.0
    assert e_thin / e_fat < 3.0
