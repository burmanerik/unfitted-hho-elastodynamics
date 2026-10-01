"""L2-orthonormal hierarchical polynomial bases on cut subcells and subfaces."""
import numpy as np

# canonical orthonormal basis of the symmetric 2x2 matrices
E_SYM = np.array([
    [[1.0, 0.0], [0.0, 0.0]],
    [[0.0, 0.0], [0.0, 1.0]],
    [[0.0, 1.0 / np.sqrt(2.0)], [1.0 / np.sqrt(2.0), 0.0]],
])
TR_SYM = np.array([1.0, 1.0, 0.0])


def mono_exponents(deg):
    """Exponents ordered by total degree (hierarchical)."""
    out = []
    for d in range(deg + 1):
        for a in range(d, -1, -1):
            out.append((a, d - a))
    return out


class CellBasis:
    """Scalar orthonormal basis of P^deg on a (cut) subcell, from a quadrature."""

    def __init__(self, qp, qw, xc, h, deg):
        self.xc, self.h, self.deg = np.asarray(xc, float), float(h), deg
        self.exp = mono_exponents(deg)
        self.n = len(self.exp)
        M = self._mono(qp)
        mass = (M * qw[:, None]).T @ M
        L = np.linalg.cholesky(mass)
        self.C = np.linalg.inv(L)          # phi = C @ m

    def _mono(self, pts):
        z = (np.asarray(pts, float) - self.xc) / self.h
        return np.column_stack([z[:, 0] ** a * z[:, 1] ** b for a, b in self.exp])

    def _dmono(self, pts):
        z = (np.asarray(pts, float) - self.xc) / self.h
        g = np.zeros((len(z), self.n, 2))
        for k, (a, b) in enumerate(self.exp):
            if a > 0:
                g[:, k, 0] = a * z[:, 0] ** (a - 1) * z[:, 1] ** b / self.h
            if b > 0:
                g[:, k, 1] = b * z[:, 0] ** a * z[:, 1] ** (b - 1) / self.h
        return g

    def eval(self, pts):
        return self._mono(pts) @ self.C.T

    def grad(self, pts):
        return np.einsum('ij,qjd->qid', self.C, self._dmono(pts))

    def sub(self, deg):
        """Coefficient slice selecting the hierarchical P^deg subspace."""
        return len(mono_exponents(deg))


class FaceBasis:
    """Scalar orthonormal basis of P^deg on a (cut) subface, arclength based."""

    def __init__(self, a, b, deg):
        self.a, self.b = np.asarray(a, float), np.asarray(b, float)
        self.L = np.linalg.norm(self.b - self.a)
        self.t = (self.b - self.a) / self.L
        self.mid = 0.5 * (self.a + self.b)
        self.deg = deg
        self.n = deg + 1
        # shifted Legendre on [-1,1], orthonormal for the L2(F) inner product
        self.scal = np.sqrt((2 * np.arange(self.n) + 1.0) / self.L)

    def _s(self, pts):
        return 2.0 * ((np.asarray(pts, float) - self.mid) @ self.t) / self.L

    def eval(self, pts):
        s = self._s(pts)
        out = np.zeros((len(s), self.n))
        for k in range(self.n):
            cc = np.zeros(k + 1)
            cc[k] = 1.0
            out[:, k] = np.polynomial.legendre.legval(s, cc) * self.scal[k]
        return out


def vec_eval(scal, pts):
    """(nq, 2*n, 2) array of the vector basis phi_j e_d."""
    v = scal.eval(pts)
    nq, n = v.shape
    out = np.zeros((nq, 2 * n, 2))
    out[:, 0::2, 0] = v
    out[:, 1::2, 1] = v
    return out


def vec_strain(scal, pts):
    """(nq, 2*n, 2, 2) symmetric gradients of the vector basis."""
    g = scal.grad(pts)
    nq, n, _ = g.shape
    out = np.zeros((nq, 2 * n, 2, 2))
    for d in range(2):
        E = np.zeros((nq, n, 2, 2))
        for bq in range(2):
            E[:, :, d, bq] += 0.5 * g[:, :, bq]
            E[:, :, bq, d] += 0.5 * g[:, :, bq]
        out[:, d::2] = E
    return out


def sym_eval(scal, pts, deg):
    """(nq, 3*n, 2, 2) array of the symmetric-matrix-valued basis phi_j E_m."""
    v = scal.eval(pts)[:, :deg_dim(deg)]
    nq, n = v.shape
    return np.einsum('qj,mab->qjmab', v, E_SYM).reshape(nq, 3 * n, 2, 2)


def deg_dim(deg):
    return (deg + 1) * (deg + 2) // 2
