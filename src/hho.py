"""Unfitted HHO discretisation of linear elasticity / elastodynamics with a
linear-slip (imperfect) interface.  Two space dimensions, straight interface."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

import geom
import basis as B
from basis import CellBasis, FaceBasis, vec_eval, vec_strain, sym_eval, deg_dim, E_SYM, TR_SYM


class Material:
    def __init__(self, mu, lam, rho=(1.0, 1.0)):
        self.mu = np.asarray(mu, float)
        self.lam = np.asarray(lam, float)
        self.rho = np.asarray(rho, float)

    @staticmethod
    def from_E_nu(E, nu, rho=(1.0, 1.0)):
        E, nu = np.asarray(E, float), np.asarray(nu, float)
        lam = E * nu / ((1 - 2 * nu) * (1 + nu))
        mu = E / (2 * (1 + nu))
        return Material(mu, lam, rho)

    @staticmethod
    def from_speeds(cs, cp, rho):
        cs, cp, rho = np.asarray(cs, float), np.asarray(cp, float), np.asarray(rho, float)
        mu = rho * cs ** 2
        lam = rho * cp ** 2 - 2 * mu
        return Material(mu, lam, rho)

    def C_apply(self, i, tau):
        """C_i : tau  for a stack of 2x2 tensors."""
        tr = tau[..., 0, 0] + tau[..., 1, 1]
        out = 2 * self.mu[i] * tau.copy()
        out[..., 0, 0] += self.lam[i] * tr
        out[..., 1, 1] += self.lam[i] * tr
        return out


def compliance(alpha, beta, n):
    """K = alpha I + (beta - alpha) n (x) n."""
    return alpha * np.eye(2) + (beta - alpha) * np.outer(n, n)


class Space:
    """Discrete space, local operators and global matrices."""

    def __init__(self, mesh, k, mat, K, qbump=4, aniso_stab=False):
        self.mesh, self.k, self.mat, self.K = mesh, k, mat, np.asarray(K, float)
        self.aniso_stab = aniso_stab
        self.nk1 = deg_dim(k + 1)
        self.nk = deg_dim(k)
        self.nC = 2 * self.nk1
        self.nF = 2 * (k + 1)
        self.qdeg = 2 * k + qbump
        self.delta = 2 * mat.mu[0] + 2 * mat.lam[0]
        self._number()
        self._build()

    # ------------------------------------------------------------- numbering
    def _number(self):
        m = self.mesh
        self.cdof = {}
        n = 0
        for ci, c in enumerate(m.cells):
            for i in sorted(c['sub']):
                self.cdof[(ci, i)] = n
                n += self.nC
        self.ncell_dofs = n
        self.fdof = {}
        for fi, f in enumerate(m.faces):
            for i in sorted(f['sub']):
                self.fdof[(fi, i)] = n
                n += self.nF
        self.ndof = n
        self.bnd = np.zeros(n, bool)
        for fi, f in enumerate(m.faces):
            if f['bnd']:
                for i in sorted(f['sub']):
                    o = self.fdof[(fi, i)]
                    self.bnd[o:o + self.nF] = True
        self.free = ~self.bnd

    def Sh(self, h):
        return np.linalg.inv(h / self.delta * np.eye(2) + self.K)

    # --------------------------------------------------------- local operators
    def local(self, ci):
        m, k = self.mesh, self.k
        c = m.cells[ci]
        hT = c['h']
        nG = m.iface.n
        Sh = self.Sh(hT)
        ImSK = np.eye(2) - Sh @ self.K
        MK = ImSK @ self.K

        sides = sorted(c['sub'])
        loc = {}                       # local offsets
        nloc = 0
        for i in sides:
            loc[('c', i)] = nloc
            nloc += self.nC
        pieces = []                    # (fid, side, seg, n_out, local offset)
        xcT = geom.poly_centroid(c['poly'])
        for fid in c['faces']:
            f = m.faces[fid]
            sgn = 1.0 if np.dot(f['n'], 0.5 * (f['a'] + f['b']) - xcT) > 0 else -1.0
            for i in sorted(f['sub']):
                if i not in c['sub']:
                    continue
                loc[('f', fid, i)] = nloc
                pieces.append((fid, i, f['sub'][i], sgn * f['n'], nloc))
                nloc += self.nF

        # bases and quadratures
        sb, qp, qw = {}, {}, {}
        for i in sides:
            P = c['sub'][i]
            p, w = geom.poly_quad(P, self.qdeg + 2)
            xc = geom.poly_centroid(P)
            sb[i] = CellBasis(p, w, xc, hT, k + 1)
            qp[i], qw[i] = p, w

        cut = c['gam'] is not None
        if cut:
            ga, gb = c['gam']
            gp, gw = geom.seg_quad(ga, gb, self.qdeg)

        # ---- reconstruction operators
        ER = {}
        nk3 = 3 * self.nk
        for i in sides:
            LHS = np.eye(nk3)
            RHS = np.zeros((nk3, nloc))
            Phi = sym_eval(sb[i], qp[i], k)                      # (nq, nk3, 2,2)
            Eps = vec_strain(sb[i], qp[i])                       # (nq, nC, 2,2)
            RHS[:, loc[('c', i)]:loc[('c', i)] + self.nC] += \
                np.einsum('q,qaxy,qbxy->ab', qw[i], Phi, Eps)
            for fid, j, seg, nout, off in pieces:
                if j != i:
                    continue
                fp, fw = geom.seg_quad(seg[0], seg[1], self.qdeg)
                Phif = sym_eval(sb[i], fp, k)
                Phin = np.einsum('qaxy,y->qax', Phif, nout)      # (nq, nk3, 2)
                fb = FaceBasis(seg[0], seg[1], k)
                Psi = vec_eval(fb, fp)                           # (nq, nF, 2)
                RHS[:, off:off + self.nF] += np.einsum('q,qax,qbx->ab', fw, Phin, Psi)
                Vc = vec_eval(sb[i], fp)
                RHS[:, loc[('c', i)]:loc[('c', i)] + self.nC] -= \
                    np.einsum('q,qax,qbx->ab', fw, Phin, Vc)
            if cut and i == 0:
                Phig = sym_eval(sb[0], gp, k)
                Phign = np.einsum('qaxy,y->qax', Phig, nG)
                CPhi = self.mat.C_apply(0, Phig)
                CPhin = np.einsum('qaxy,y->qax', CPhi, nG)
                LHS += np.einsum('q,qax,xy,qby->ab', gw, Phign, MK, CPhin)
                V0 = vec_eval(sb[0], gp)
                V1 = vec_eval(sb[1], gp)
                T0 = np.einsum('q,qax,xy,qby->ab', gw, Phign, ImSK, V0)
                T1 = np.einsum('q,qax,xy,qby->ab', gw, Phign, ImSK, V1)
                RHS[:, loc[('c', 0)]:loc[('c', 0)] + self.nC] -= T0
                RHS[:, loc[('c', 1)]:loc[('c', 1)] + self.nC] += T1
            ER[i] = np.linalg.solve(LHS, RHS)

        # ---- consistency term a_T
        aT = np.zeros((nloc, nloc))
        tr = np.tile(TR_SYM, self.nk)
        for i in sides:
            A = 2 * self.mat.mu[i] * np.eye(nk3)
            blk = np.zeros((nk3, nk3))
            for j in range(self.nk):
                s = slice(3 * j, 3 * j + 3)
                blk[s, s] = self.mat.lam[i] * np.outer(TR_SYM, TR_SYM)
            A = A + blk
            aT += ER[i].T @ A @ ER[i]
        if cut:
            Phig = sym_eval(sb[0], gp, k)
            CPhin = np.einsum('qaxy,y->qax', self.mat.C_apply(0, Phig), nG)
            Bm = np.einsum('q,qax,xy,qby->ab', gw, CPhin, MK, CPhin)
            aT += ER[0].T @ Bm @ ER[0]

        # ---- stabilisation
        sT = np.zeros((nloc, nloc))
        for fid, i, seg, nout, off in pieces:
            fp, fw = geom.seg_quad(seg[0], seg[1], self.qdeg)
            fb = FaceBasis(seg[0], seg[1], k)
            Psi = vec_eval(fb, fp)
            Vc = vec_eval(sb[i], fp)
            P = np.einsum('q,qax,qbx->ab', fw, Psi, Vc)          # (nF, nC)
            D = np.zeros((self.nF, nloc))
            D[:, off:off + self.nF] = np.eye(self.nF)
            D[:, loc[('c', i)]:loc[('c', i)] + self.nC] -= P
            if self.aniso_stab:
                W = (2 * self.mat.mu[i] * np.eye(2)
                     + 2 * self.mat.lam[i] * np.outer(nout, nout)) / hT
                G = np.kron(np.eye(self.k + 1), W)
                sT += D.T @ G @ D
            else:
                w = (2 * self.mat.mu[i] + 2 * self.mat.lam[i]) / hT
                sT += w * (D.T @ D)
        if cut:
            V0 = vec_eval(sb[0], gp)
            V1 = vec_eval(sb[1], gp)
            J = np.zeros((len(gp), 2, nloc))
            J[:, :, loc[('c', 0)]:loc[('c', 0)] + self.nC] = np.transpose(V0, (0, 2, 1))
            J[:, :, loc[('c', 1)]:loc[('c', 1)] + self.nC] -= np.transpose(V1, (0, 2, 1))
            sT += np.einsum('q,qxa,xy,qyb->ab', gw, J, Sh, J)

        # ---- mass
        MT = np.zeros((nloc, nloc))
        for i in sides:
            o = loc[('c', i)]
            MT[o:o + self.nC, o:o + self.nC] = self.mat.rho[i] * np.eye(self.nC)

        return dict(loc=loc, nloc=nloc, sides=sides, pieces=pieces, sb=sb,
                    qp=qp, qw=qw, ER=ER, a=aT + sT, M=MT, cut=cut,
                    gp=(gp if cut else None), gw=(gw if cut else None))

    def gidx(self, L, ci):
        g = np.zeros(L['nloc'], int)
        for key, off in L['loc'].items():
            if key[0] == 'c':
                g[off:off + self.nC] = self.cdof[(ci, key[1])] + np.arange(self.nC)
            else:
                g[off:off + self.nF] = self.fdof[(key[1], key[2])] + np.arange(self.nF)
        return g

    # ------------------------------------------------------------- assembling
    def _build(self):
        rows, cols, vals = [], [], []
        mrows, mcols, mvals = [], [], []
        self.locs = []
        self.gidxs = []
        for ci in range(len(self.mesh.cells)):
            L = self.local(ci)
            g = self.gidx(L, ci)
            self.locs.append(L)
            self.gidxs.append(g)
            R = np.repeat(g, L['nloc'])
            C = np.tile(g, L['nloc'])
            rows.append(R); cols.append(C); vals.append(L['a'].ravel())
            mrows.append(R); mcols.append(C); mvals.append(L['M'].ravel())
            del L['a'], L['M'], L['ER']       # keep memory for the large runs
        n = self.ndof
        self.A = sp.csr_matrix((np.concatenate(vals),
                                (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        self.M = sp.csr_matrix((np.concatenate(mvals),
                                (np.concatenate(mrows), np.concatenate(mcols))), shape=(n, n))

    # -------------------------------------------------------------- load / data
    def load(self, ffun, t=None):
        """ffun(i, pts[, t]) -> (nq,2) body force in Omega_{i+1}."""
        b = np.zeros(self.ndof)
        for ci, L in enumerate(self.locs):
            for i in L['sides']:
                V = vec_eval(L['sb'][i], L['qp'][i])
                fv = ffun(i, L['qp'][i]) if t is None else ffun(i, L['qp'][i], t)
                o = self.cdof[(ci, i)]
                b[o:o + self.nC] += np.einsum('q,qax,qx->a', L['qw'][i], V, fv)
        return b

    def interpolate(self, ufun, t=None):
        """L2 interpolate: P^{k+1} on subcells, P^k on subfaces."""
        v = np.zeros(self.ndof)
        for ci, L in enumerate(self.locs):
            for i in L['sides']:
                V = vec_eval(L['sb'][i], L['qp'][i])
                uv = ufun(i, L['qp'][i]) if t is None else ufun(i, L['qp'][i], t)
                o = self.cdof[(ci, i)]
                v[o:o + self.nC] = np.einsum('q,qax,qx->a', L['qw'][i], V, uv)
        for fi, f in enumerate(self.mesh.faces):
            for i in sorted(f['sub']):
                a, b_ = f['sub'][i]
                fp, fw = geom.seg_quad(a, b_, self.qdeg)
                fb = FaceBasis(a, b_, self.k)
                Psi = vec_eval(fb, fp)
                uv = ufun(i, fp) if t is None else ufun(i, fp, t)
                o = self.fdof[(fi, i)]
                v[o:o + self.nF] = np.einsum('q,qax,qx->a', fw, Psi, uv)
        return v

    def l2_cell_norm(self, v):
        return np.linalg.norm(v[:self.ncell_dofs])

    def energy_norm(self, v):
        return np.sqrt(max(v @ (self.A @ v), 0.0))

    # ---------------------------------------------------- solve (static, cond.)
    def static_condensation(self, A):
        """Eliminate the cell unknowns of A (block diagonal on cells)."""
        nc = self.ncell_dofs
        Acc = A[:nc, :nc].tocsc()
        Acf = A[:nc, nc:].tocsr()
        Afc = A[nc:, :nc].tocsr()
        Aff = A[nc:, nc:].tocsr()
        blocks = []
        for ci in range(len(self.mesh.cells)):
            idx = [self.cdof[(ci, i)] for i in sorted(self.mesh.cells[ci]['sub'])]
            sz = self.nC * len(idx)
            rows = np.concatenate([o + np.arange(self.nC) for o in idx])
            blocks.append((rows, np.linalg.inv(Acc[np.ix_(rows, rows)].toarray())))
        data, ri, ci_ = [], [], []
        for rows, inv in blocks:
            ri.append(np.repeat(rows, len(rows)))
            ci_.append(np.tile(rows, len(rows)))
            data.append(inv.ravel())
        Ainv = sp.csr_matrix((np.concatenate(data),
                              (np.concatenate(ri), np.concatenate(ci_))), shape=(nc, nc))
        S = (Aff - Afc @ (Ainv @ Acf)).tocsc()
        return Ainv, Acf, Afc, S


def solve_static(sp_, b, ud=None):
    """Solve A u = b with Dirichlet face values ud on the boundary dofs."""
    n = sp_.ndof
    u = np.zeros(n)
    if ud is not None:
        u[sp_.bnd] = ud[sp_.bnd]
    r = b - sp_.A @ u
    fr = sp_.free
    Aff = sp_.A[fr][:, fr].tocsc()
    u[fr] = spl.spsolve(Aff, r[fr])
    return u
