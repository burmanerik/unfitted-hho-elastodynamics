"""Manufactured solution of the elasticity problem with a linear-slip interface
across a straight oblique interface (symbolic construction, cf. the fitted case).

Given a smooth field u_2 defined on the whole of Omega, we build u_1 so that
    [sigma(u) n] = 0     and     [u] + K sigma(u_1) n = 0      on Gamma,
by a Taylor expansion in the normal direction; f_i := -div sigma_i(u_i).
"""
import numpy as np
import sympy as sy


class Manufactured:
    def __init__(self, n, c, mu, lam, alpha, beta, psi=None, lam2_div=True):
        n = np.asarray(n, float) / np.linalg.norm(np.asarray(n, float))
        t = np.array([-n[1], n[0]])
        self.n, self.t, self.c = n, t, float(c)
        x, y, tau = sy.symbols('x y tau', real=True)
        ns = sy.Matrix(2, 1, [sy.Float(n[0]), sy.Float(n[1])])
        ts = sy.Matrix(2, 1, [sy.Float(t[0]), sy.Float(t[1])])
        X = sy.Matrix(2, 1, [x, y])
        mu1, mu2 = sy.Float(mu[0]), sy.Float(mu[1])
        lm1, lm2 = sy.Float(lam[0]), sy.Float(lam[1])
        Ks = sy.eye(2) * sy.Float(alpha) + (sy.Float(beta) - sy.Float(alpha)) * (ns * ns.T)

        if psi is None:
            psi = sy.sin(sy.pi * x + sy.pi / 4) * sy.sin(2 * sy.pi * y)
        u2 = sy.Matrix(2, 1, [sy.diff(psi, y), -sy.diff(psi, x)])
        if lam2_div:
            u2 = u2 + X / (2 * lm2)          # keeps lam_2 * div u_2 = 1

        def eps(u):
            G = u.jacobian(X)
            return (G + G.T) / 2

        def sig(u, mu_, lm_):
            e = eps(u)
            return 2 * mu_ * e + lm_ * sy.trace(e) * sy.eye(2)

        # traction and mismatch evaluated on Gamma, as functions of tau
        xg = self.c * ns + tau * ts
        sub = {x: xg[0], y: xg[1]}
        T = sy.expand((sig(u2, mu2, lm2) * ns).subs(sub))
        R = sy.expand(T - (sig(u2, mu1, lm1) * ns).subs(sub))
        V = -Ks * T
        Vn, Vt = (V.T * ns)[0], (V.T * ts)[0]
        Rn, Rt = (R.T * ns)[0], (R.T * ts)[0]
        Wn = (Rn - lm1 * sy.diff(Vt, tau)) / (2 * mu1 + lm1)
        Wt = Rt / mu1 - sy.diff(Vn, tau)
        W = Wn * ns + Wt * ts
        g = sy.diff(Wt, tau)

        s_ = (ns.T * X)[0] - self.c
        tau_ = (ts.T * X)[0]
        rep = {tau: tau_}
        u1 = u2 + V.subs(rep) + s_ * W.subs(rep) - sy.Rational(1, 2) * s_**2 * g.subs(rep) * ns

        self.sym = dict(x=x, y=y, u=[u1, u2])
        f = []
        for i, (u, mu_, lm_) in enumerate(((u1, mu1, lm1), (u2, mu2, lm2))):
            S = sig(u, mu_, lm_)
            div = sy.Matrix(2, 1, [sy.diff(S[0, 0], x) + sy.diff(S[0, 1], y),
                                   sy.diff(S[1, 0], x) + sy.diff(S[1, 1], y)])
            f.append(-div)
        def lam2(expr_list):
            return [sy.lambdify((x, y), e, "numpy") for e in expr_list]

        self._u = [lam2([u1[0], u1[1]]), lam2([u2[0], u2[1]])]
        self._f = [lam2([f[0][0], f[0][1]]), lam2([f[1][0], f[1][1]])]
        sg = [sig(u1, mu1, lm1), sig(u2, mu2, lm2)]
        self._sn = [lam2([(sg[0] * ns)[0], (sg[0] * ns)[1]]),
                    lam2([(sg[1] * ns)[0], (sg[1] * ns)[1]])]
        self._div = [lam2([sy.trace(eps(u1))]), lam2([sy.trace(eps(u2))])]

    @staticmethod
    def _ev(funs, p):
        p = np.atleast_2d(np.asarray(p, float))
        cols = [np.broadcast_to(np.asarray(g(p[:, 0], p[:, 1]), float), (len(p),))
                for g in funs]
        return np.ascontiguousarray(np.column_stack(cols))

    def u(self, i, p):
        return self._ev(self._u[i], p)

    def f(self, i, p):
        return self._ev(self._f[i], p)

    def sigma_n(self, i, p):
        return self._ev(self._sn[i], p)

    def div(self, i, p):
        return self._ev(self._div[i], p)[:, 0]

    # ------------------------------------------------------------- validation
    def check(self, K, npts=7):
        """Max residual of the two interface conditions along Gamma."""
        s = np.linspace(-0.6, 0.6, npts)
        pts = self.c * self.n[None, :] + s[:, None] * self.t[None, :]
        jump = self.u(0, pts) - self.u(1, pts)
        t1 = self.sigma_n(0, pts)
        t2 = self.sigma_n(1, pts)
        return (np.abs(jump + (K @ t1.T).T).max(), np.abs(t1 - t2).max())
