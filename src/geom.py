"""Geometry helpers: convex-polygon clipping by a half-plane, quadratures."""
import numpy as np


def cross2(a, b):
    """Scalar cross product of two planar vectors."""
    return a[0] * b[1] - a[1] * b[0]


TOL = 1e-13


def poly_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def poly_centroid(P):
    x, y = P[:, 0], P[:, 1]
    cr = x * np.roll(y, -1) - np.roll(x, -1) * y
    A = 0.5 * np.sum(cr)
    cx = np.sum((x + np.roll(x, -1)) * cr) / (6 * A)
    cy = np.sum((y + np.roll(y, -1)) * cr) / (6 * A)
    return np.array([cx, cy])


def poly_diam(P):
    d = 0.0
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            d = max(d, np.linalg.norm(P[i] - P[j]))
    return d


def clip_halfplane(P, n, c, keep_neg=True, tol=1e-13):
    """Clip convex polygon P by {x : n.x <= c} (keep_neg) or {n.x >= c}.

    Returns an (m,2) array, possibly empty.
    """
    if P is None or len(P) == 0:
        return np.zeros((0, 2))
    s = P @ n - c
    if not keep_neg:
        s = -s
    # scale tolerance
    sc = max(np.max(np.abs(s)), 1.0)
    s = np.where(np.abs(s) < tol * sc, 0.0, s)
    if np.all(s <= 0):
        return P.copy()
    if np.all(s >= 0):
        return np.zeros((0, 2))
    out = []
    m = len(P)
    for i in range(m):
        j = (i + 1) % m
        si, sj = s[i], s[j]
        if si <= 0:
            out.append(P[i])
        if (si < 0 and sj > 0) or (si > 0 and sj < 0):
            t = si / (si - sj)
            out.append(P[i] + t * (P[j] - P[i]))
    if len(out) < 3:
        return np.zeros((0, 2))
    return np.array(out)


def poly_line_segment(P, n, c, tol=1e-13):
    """Intersection of the line {n.x = c} with the convex polygon P.

    Returns (a, b) or None.
    """
    if P is None or len(P) == 0:
        return None
    s = P @ n - c
    sc = max(np.max(np.abs(s)), 1.0)
    s = np.where(np.abs(s) < tol * sc, 0.0, s)
    pts = []
    m = len(P)
    for i in range(m):
        j = (i + 1) % m
        si, sj = s[i], s[j]
        if si == 0.0:
            pts.append(P[i])
        if (si < 0 and sj > 0) or (si > 0 and sj < 0):
            t = si / (si - sj)
            pts.append(P[i] + t * (P[j] - P[i]))
    if len(pts) < 2:
        return None
    pts = np.array(pts)
    # keep the two extreme points along the line direction
    d = np.array([-n[1], n[0]])
    t = pts @ d
    i0, i1 = int(np.argmin(t)), int(np.argmax(t))
    a, b = pts[i0], pts[i1]
    if np.linalg.norm(b - a) < tol * max(1.0, np.max(np.abs(P))):
        return None
    return a, b


def clip_segment(a, b, n, c, keep_neg=True, tol=1e-13):
    """Clip the segment [a,b] by the half-plane. Returns (a',b') or None."""
    sa = a @ n - c
    sb = b @ n - c
    if not keep_neg:
        sa, sb = -sa, -sb
    L = np.linalg.norm(b - a)
    eps = tol * max(L, 1.0)
    if sa <= eps and sb <= eps:
        return a, b
    if sa >= -eps and sb >= -eps:
        return None
    t = sa / (sa - sb)
    p = a + t * (b - a)
    if sa <= 0:
        out = (a, p)
    else:
        out = (p, b)
    if np.linalg.norm(out[1] - out[0]) < 1e-14 * max(L, 1.0):
        return None
    return out


# ---------------------------------------------------------------- quadratures
_GL_CACHE = {}


def gauss_legendre(npts):
    if npts not in _GL_CACHE:
        _GL_CACHE[npts] = np.polynomial.legendre.leggauss(npts)
    return _GL_CACHE[npts]


def seg_quad(a, b, deg):
    """Gauss points/weights on the segment [a,b] exact for degree `deg`."""
    npts = max(1, (deg + 2) // 2)
    x, w = gauss_legendre(npts)
    t = 0.5 * (x + 1.0)
    L = np.linalg.norm(b - a)
    pts = a[None, :] + t[:, None] * (b - a)[None, :]
    return pts, 0.5 * L * w


def tri_quad(p0, p1, p2, deg):
    """Collapsed (Duffy) tensor-product Gauss rule on a triangle, exact for `deg`."""
    n1 = max(1, (deg + 2) // 2)
    n2 = max(1, (deg + 3) // 2)   # extra degree from the Jacobian (1-u)
    x1, w1 = gauss_legendre(n1)
    x2, w2 = gauss_legendre(n2)
    u = 0.5 * (x2 + 1.0)          # u in [0,1], carries the (1-u) factor
    v = 0.5 * (x1 + 1.0)
    U, V = np.meshgrid(u, v, indexing="ij")
    W = np.outer(0.25 * w2, w1) * (1.0 - U)
    xi = U
    eta = V * (1.0 - U)
    J = abs(cross2(p1 - p0, p2 - p0))
    pts = p0[None, :] + xi.ravel()[:, None] * (p1 - p0)[None, :] \
        + eta.ravel()[:, None] * (p2 - p0)[None, :]
    return pts, W.ravel() * J


def poly_quad(P, deg):
    """Quadrature on a convex polygon by fan triangulation from the centroid."""
    if P is None or len(P) < 3:
        return np.zeros((0, 2)), np.zeros(0)
    c = poly_centroid(P)
    allp, allw = [], []
    m = len(P)
    for i in range(m):
        j = (i + 1) % m
        if abs(cross2(P[i] - c, P[j] - c)) < 1e-16:
            continue
        p, w = tri_quad(c, P[i], P[j], deg)
        allp.append(p)
        allw.append(w)
    if not allp:
        return np.zeros((0, 2)), np.zeros(0)
    return np.vstack(allp), np.concatenate(allw)
