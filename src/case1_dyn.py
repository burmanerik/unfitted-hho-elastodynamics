"""Test case 1 in the time domain.

Manufactured solution u(x, t) = cos(omega t) u_s(x), where u_s is the static
manufactured solution of manufactured.py, so that the linear slip interface
conditions hold exactly at every time.  Used by test case 1 of the manuscript
(Tables 5 and 6) and by the temporal-order tests.
"""
import numpy as np
import mesh as M
from hho import Material, Space, compliance
from manufactured import Manufactured
from dynamics import Newmark, SDIRK, FaceSolver, initial_acceleration
from case1 import NRM, CC

OM = 2.0


def setup(k, n, alpha=5e-2, beta=5e-2):
    mat = Material.from_E_nu([1.0, 10.0], [0.25, 0.3])
    K = compliance(alpha, beta, NRM)
    ex = Manufactured(NRM, CC, mat.mu, mat.lam, alpha, beta)
    m = M.Mesh((0, 1, 0, 1), n, n, M.Interface(NRM, CC))
    S = Space(m, k, mat, K)
    return S, ex


def integrate(S, ex, dt, T=1.0, ret='u'):
    us = S.interpolate(ex.u)
    fs = S.load(ex.f)
    Mu = S.M @ us
    fsv = FaceSolver(S)
    uD = us.copy(); uD[S.free] = 0.0
    base = us.copy(); base[S.free] = 0.0
    base[:S.ncell_dofs] = us[:S.ncell_dofs]
    base = fsv.fill(base)                      # consistent initial face values
    u0 = base.copy(); u0[S.bnd] = 0.0
    a0 = initial_acceleration(S, fsv, u0, uD, fs - OM ** 2 * Mu, aD0=-OM ** 2 * uD)
    nm = Newmark(S, dt)
    nm.start(u0, np.zeros_like(u0), a0)
    mE0 = mEe = 0.0
    for s in range(int(round(T / dt))):
        t = (s + 1) * dt
        chi = np.cos(OM * t)
        nm.step(chi * (fs - OM ** 2 * Mu), uD_next=chi * uD)
        if ret == 'max':
            e = nm.u + chi * uD - chi * us
            mE0 = max(mE0, S.l2_cell_norm(e))
            mEe = max(mEe, S.energy_norm(e))
    if ret == 'max':
        return mE0, mEe
    if ret == 'u':
        return nm.u + np.cos(OM * T) * uD, np.cos(OM * T) * us
    return nm.u.copy()


def integrate_sdirk(S, ex, dt, s=3, T=0.5):
    us = S.interpolate(ex.u)
    fs = S.load(ex.f)
    Mu = S.M @ us
    fsv = FaceSolver(S)
    uD = us.copy(); uD[S.free] = 0.0
    base = us.copy(); base[S.free] = 0.0
    base[:S.ncell_dofs] = us[:S.ncell_dofs]
    base = fsv.fill(base)
    u0 = base.copy(); u0[S.bnd] = 0.0
    AuD = S.A @ uD

    def load(t):
        return np.cos(OM * t) * (fs - OM ** 2 * Mu - AuD)

    nm = SDIRK(S, dt, s=s)
    nm.start(u0, np.zeros_like(u0))
    n = int(round(T / dt))
    for m in range(n):
        nm.step(m * dt, load,
                uD=lambda t: np.cos(OM * t) * uD,
                duD=lambda t: -OM * np.sin(OM * t) * uD)
    return nm.u.copy()
