"""Test cases 2 and 3: elastic wave propagation across an unfitted interface.

Configuration of [BDES21, Sect. 6.2] with the material interface rotated so that
it is not resolved by the Cartesian background mesh.
"""
import numpy as np
import mesh as M
from hho import Material, Space, compliance
from dynamics import (Newmark, SDIRK, FaceSolver, point_eval, sample_grid,
                      initial_acceleration)

PHI = np.pi / 12.0
NRM = np.array([-np.sin(PHI), np.cos(PHI)])
CC = 0.0
BOX = (-1.5, 1.5, -1.5, 1.5)
THETA_END = 1.0

MAT = Material.from_speeds(cs=[1.0, 2.0], cp=[np.sqrt(3.0), 2 * np.sqrt(3.0)], rho=[1.0, 1.0])
XC, YC = 0.0, 2.0 / 3.0
FC = 5.0
LAMW = 2 * np.sqrt(3.0) / FC
AMP = 1e-2

S1 = np.array([1.0 / 3.0, -1.0 / 3.0])
S2 = np.array([1.0 / 3.0, 1.0 / 3.0])


def v0(i, p):
    x, y = p[:, 0] - XC, p[:, 1] - YC
    r2 = x ** 2 + y ** 2
    g = AMP * np.exp(-np.pi ** 2 * r2 / LAMW ** 2)
    return np.column_stack([g * x, g * y])


def zero(i, p):
    return np.zeros((len(p), 2))


def run(k, n, alpha=0.0, beta=0.0, dt=1.0 / 400, T=THETA_END, sensors=(S1, S2),
        snap_times=(), nsnap=161, verbose=True, scheme='newmark'):
    """`scheme` is 'newmark' or ('sdirk', s) with s in {1, 2, 3}."""
    K = compliance(alpha, beta, NRM)
    m = M.Mesh(BOX, n, n, M.Interface(NRM, CC))
    S = Space(m, k, MAT, K)
    fsv = FaceSolver(S)
    u0 = np.zeros(S.ndof)
    vv = S.interpolate(v0)
    v0h = np.zeros(S.ndof)
    v0h[:S.ncell_dofs] = vv[:S.ncell_dofs]
    v0h = fsv.fill(v0h)
    F = np.zeros(S.ndof)
    if scheme == 'newmark':
        a0 = initial_acceleration(S, fsv, u0, np.zeros(S.ndof), np.zeros(S.ndof))
        nm = Newmark(S, dt)
        nm.start(u0, v0h, a0)
        advance = lambda step: nm.step(F)
    else:
        nm = SDIRK(S, dt, s=scheme[1])
        nm.start(u0, v0h)
        advance = lambda step: nm.step(step * dt, lambda t: F)
    nsteps = int(round(T / dt))
    ts = [0.0]
    vel = [[point_eval(S, v0h, s) for s in sensors]]
    en = [nm.energy()]
    snaps = {}
    for st in snap_times:
        if abs(st) < 1e-14:
            snaps[st] = sample_grid(S, v0h, nsnap, nsnap)
    for s in range(nsteps):
        advance(s)
        t = (s + 1) * dt
        ts.append(t)
        vel.append([point_eval(S, nm.v, x) for x in sensors])
        en.append(nm.energy())
        for st in snap_times:
            if abs(t - st) < 0.5 * dt:
                snaps[st] = sample_grid(S, nm.v, nsnap, nsnap)
    return dict(S=S, mesh=m, t=np.array(ts), vel=np.array(vel), energy=np.array(en),
                u=nm.u.copy(), v=nm.v.copy(), snaps=snaps, nm=nm)


if __name__ == '__main__':
    import time
    t0 = time.time()
    r = run(1, 16, dt=1 / 50, T=0.2)
    print('cells', r['mesh'].stats(), 'ndof', r['S'].ndof, 'time %.1fs' % (time.time() - t0))
    e = r['energy']
    print('energy drift', (e.max() - e.min()) / e[0])
    print('sensor v at final', r['vel'][-1])
