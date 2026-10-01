"""Static condensation and the HHO-Newmark scheme for unfitted elastodynamics."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

import geom
from basis import vec_eval


class Condensed:
    """Solver for a system whose cell-cell block is block diagonal."""

    def __init__(self, S, A):
        self.S = S
        nc = S.ncell_dofs
        self.nc = nc
        ff = S.free.copy()
        ff[:nc] = False                     # face dofs that are free
        self.fidx = np.where(ff)[0]
        A = A.tocsr()
        cidx = np.arange(nc)
        Acc = A[:nc, :][:, cidx].tocsr()
        self.Acf = A[:nc, :][:, self.fidx].tocsr()
        self.Afc = A[self.fidx, :][:, cidx].tocsr()
        Aff = A[self.fidx, :][:, self.fidx].tocsr()
        data, ri, ci = [], [], []
        for k in range(len(S.mesh.cells)):
            rows = np.concatenate([S.cdof[(k, i)] + np.arange(S.nC)
                                   for i in sorted(S.mesh.cells[k]['sub'])])
            inv = np.linalg.inv(Acc[rows, :][:, rows].toarray())
            ri.append(np.repeat(rows, len(rows)))
            ci.append(np.tile(rows, len(rows)))
            data.append(inv.ravel())
        self.Ainv = sp.csr_matrix((np.concatenate(data),
                                   (np.concatenate(ri), np.concatenate(ci))), shape=(nc, nc))
        Sch = (Aff - self.Afc @ (self.Ainv @ self.Acf)).tocsc()
        self.lu = spl.splu(Sch)

    def solve(self, b):
        """b: full-length rhs (boundary entries ignored). Returns full-length x."""
        bc = b[:self.nc]
        bf = b[self.fidx]
        y = self.Ainv @ bc
        xf = self.lu.solve(bf - self.Afc @ y)
        xc = y - self.Ainv @ (self.Acf @ xf)
        x = np.zeros(len(b))
        x[:self.nc] = xc
        x[self.fidx] = xf
        return x


class FaceSolver:
    """Determines the face unknowns from the cell ones through the face equations
    a_h((v_T, v_F), (0, w_F)) = 0, as required by the initial conditions."""

    def __init__(self, S):
        self.S = S
        nc = S.ncell_dofs
        ff = S.free.copy()
        ff[:nc] = False
        self.fidx = np.where(ff)[0]
        self.Arow = S.A[self.fidx].tocsr()
        self.lu = spl.splu(self.Arow[:, self.fidx].tocsc())

    def fill(self, v):
        v = v.copy()
        v[self.fidx] = 0.0
        v[self.fidx] = self.lu.solve(-(self.Arow @ v))
        return v


def initial_acceleration(S, fsv, u_free0, uD0, F0, aD0=None):
    """Initial acceleration consistent with the semi-discrete equations."""
    r = F0 - S.A @ (u_free0 + uD0)
    nc = S.ncell_dofs
    z = np.zeros(S.ndof)
    z[:nc] = r[:nc] / S.M.diagonal()[:nc]
    if aD0 is not None:
        z[S.bnd] = aD0[S.bnd]
    z = fsv.fill(z)
    z[S.bnd] = 0.0
    return z


class Newmark:
    """Second-order Newmark scheme (beta, gamma) for M u'' + A u = F."""

    def __init__(self, S, dt, beta=0.25, gamma=0.5):
        self.S, self.dt, self.beta, self.gamma = S, dt, beta, gamma
        Aeff = (S.M + beta * dt ** 2 * S.A).tocsr()
        self.solver = Condensed(S, Aeff)

    def start(self, u0, v0, a0):
        self.u, self.v, self.a = u0.copy(), v0.copy(), a0.copy()

    def step(self, Fnext, uD_next=None):
        dt, b, g = self.dt, self.beta, self.gamma
        S = self.S
        us = self.u + dt * self.v + 0.5 * dt ** 2 * (1 - 2 * b) * self.a
        vs = self.v + dt * (1 - g) * self.a
        rhs = Fnext - S.A @ us
        if uD_next is not None:
            rhs = rhs - S.A @ uD_next
        rhs[S.bnd] = 0.0
        a = self.solver.solve(rhs)
        a[S.bnd] = 0.0
        self.u = us + b * dt ** 2 * a
        self.v = vs + g * dt * a
        self.a = a
        return self.u

    def energy(self, uD=None):
        """Discrete energy 1/2 ||v_T||^2_rho + 1/2 a_h(u,u)  (delta = 0)."""
        S = self.S
        u = self.u if uD is None else self.u + uD
        return 0.5 * (self.v @ (S.M @ self.v)) + 0.5 * (u @ (S.A @ u))


#  --------------------------------------------- singly diagonally implicit RK
def sdirk_tableau(s):
    """Butcher tableau of the SDIRK(s, s+1) schemes of [BDE22, Sect. 5]."""
    if s == 1:
        return (np.array([[0.5]]), np.array([1.0]), np.array([0.5]))
    if s == 2:
        g = (3.0 + np.sqrt(3.0)) / 6.0            # = 1/2 + 1/(2 sqrt 3)
        A = np.array([[g, 0.0], [1.0 - 2.0 * g, g]])
        return (A, np.array([0.5, 0.5]), np.array([g, 1.0 - g]))
    if s == 3:
        g = np.cos(np.pi / 18.0) / np.sqrt(3.0) + 0.5
        d = 1.0 / (6.0 * (2.0 * g - 1.0) ** 2)
        A = np.array([[g, 0.0, 0.0],
                      [0.5 - g, g, 0.0],
                      [2.0 * g, 1.0 - 4.0 * g, g]])
        return (A, np.array([d, 1.0 - 2.0 * d, d]), np.array([g, 0.5, 1.0 - g]))
    raise ValueError('s must be 1, 2 or 3')


class SDIRK:
    """SDIRK(s, s+1) applied to the first-order-in-time reformulation

        d_t u = v,     M d_t v + A u = F(t),

    of the space semi-discrete problem.  Since all the diagonal entries of the
    Butcher matrix are equal, every stage involves the same matrix
    M + (a_* dt)^2 A, which is factorised once after static condensation.
    """

    def __init__(self, S, dt, s=3):
        self.S, self.dt = S, dt
        self.A, self.b, self.c = sdirk_tableau(s)
        self.astar = self.A[0, 0]
        Aeff = (S.M + (self.astar * dt) ** 2 * S.A).tocsr()
        self.solver = Condensed(S, Aeff)
        self.faces = FaceSolver(S)

    def start(self, u0, v0):
        self.u, self.v = u0.copy(), v0.copy()

    def step(self, t0, load, uD=None, duD=None):
        """One step from t0 to t0+dt.  `load(t)` returns the effective load
        (including -A uD(t) when a Dirichlet lifting uD is used)."""
        S, dt, ast = self.S, self.dt, self.astar
        s = len(self.b)
        Vs, Ks = [], []
        for i in range(s):
            PU, PV = self.u.copy(), self.v.copy()
            for j in range(i):
                PU += dt * self.A[i, j] * Vs[j]
                PV += dt * self.A[i, j] * Ks[j]
            P = PU + ast * dt * PV
            rhs = load(t0 + self.c[i] * dt) - S.A @ P
            rhs[S.bnd] = 0.0
            K = self.solver.solve(rhs)
            K[S.bnd] = 0.0
            Vs.append(PV + ast * dt * K)
            Ks.append(K)
        for i in range(s):
            self.u = self.u + dt * self.b[i] * Vs[i]
            self.v = self.v + dt * self.b[i] * Ks[i]
        # re-impose the algebraic face equations on the updated state
        t1 = t0 + dt
        self.u = self._project(self.u, uD, t1)
        self.v = self._project(self.v, duD, t1)
        return self.u

    def _project(self, w, lift, t):
        S = self.S
        if lift is None:
            return self.faces.fill(w)
        g = lift(t)
        return self.faces.fill(w + g) - g

    def energy(self, uD=None):
        S = self.S
        u = self.u if uD is None else self.u + uD
        return 0.5 * (self.v @ (S.M @ self.v)) + 0.5 * (u @ (S.A @ u))


# --------------------------------------------------------------- point values
def locate(mesh, x):
    for ci, c in enumerate(mesh.cells):
        P = c['poly']
        if (P[:, 0].min() - 1e-12 <= x[0] <= P[:, 0].max() + 1e-12 and
                P[:, 1].min() - 1e-12 <= x[1] <= P[:, 1].max() + 1e-12):
            return ci
    return -1


def point_eval(S, v, x, side=None):
    ci = locate(S.mesh, x)
    if ci < 0:
        return np.zeros(2)
    L = S.locs[ci]
    lev = S.mesh.iface.level(np.atleast_2d(x))[0]
    i = (0 if lev < 0 else 1) if side is None else side
    if i not in L['sides']:
        i = L['sides'][0]
    V = vec_eval(L['sb'][i], np.atleast_2d(x))[0]     # (nC, 2)
    o = S.cdof[(ci, i)]
    return V.T @ v[o:o + S.nC]


def sample_grid(S, v, nx=200, ny=200, box=None):
    """Sample a component field on a Cartesian grid (nan outside)."""
    box = box or S.mesh.box
    xs = np.linspace(box[0], box[1], nx)
    ys = np.linspace(box[2], box[3], ny)
    X, Y = np.meshgrid(xs, ys)
    out = np.full((ny, nx, 2), np.nan)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    lev = S.mesh.iface.level(pts)
    # bucket the points by cell using the base Cartesian structure
    for ci, c in enumerate(S.mesh.cells):
        P = c['poly']
        sel = ((pts[:, 0] >= P[:, 0].min()) & (pts[:, 0] <= P[:, 0].max()) &
               (pts[:, 1] >= P[:, 1].min()) & (pts[:, 1] <= P[:, 1].max()))
        idx = np.where(sel)[0]
        if len(idx) == 0:
            continue
        L = S.locs[ci]
        for i in L['sides']:
            sub = idx[(lev[idx] < 0) == (i == 0)]
            if len(sub) == 0:
                continue
            V = vec_eval(L['sb'][i], pts[sub])
            o = S.cdof[(ci, i)]
            out.reshape(-1, 2)[sub] = np.einsum('qax,a->qx', V, v[o:o + S.nC])
    return xs, ys, out
