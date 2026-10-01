"""Cartesian background meshes, straight interfaces, and pairwise agglomeration."""
import numpy as np
import geom


class Interface:
    """Straight interface Gamma = {x : n.x = c}; Omega_1 = {n.x < c}."""

    def __init__(self, n, c):
        n = np.asarray(n, float)
        self.n = n / np.linalg.norm(n)
        self.c = float(c)

    def level(self, pts):
        return np.asarray(pts, float) @ self.n - self.c


class Mesh:
    """Agglomerated Cartesian mesh cut by a straight interface.

    Attributes
    ----------
    cells : list of dicts with keys
        'poly' (vertices, ccw), 'h', 'faces' (list of (face_id, sign)),
        'sub'  : {i: polygon of T^i} for the non-empty sides (i = 0, 1 <-> Omega_1, Omega_2)
        'gam'  : (a, b) interface segment or None
    faces : list of dicts with keys 'a','b','n' (unit normal a->b rotated),
        'cells' (list of adjacent cell ids), 'bnd' (bool), 'sub' {i: (a,b)}
    """

    def __init__(self, box, nx, ny, iface, agglo_tol=0.3, drop_tol=1e-10):
        self.box, self.nx, self.ny = box, nx, ny
        self.iface = iface
        x0, x1, y0, y1 = box
        hx, hy = (x1 - x0) / nx, (y1 - y0) / ny
        self.h = max(hx, hy)

        base_poly = []
        for j in range(ny):
            for i in range(nx):
                a, b = x0 + i * hx, y0 + j * hy
                base_poly.append(np.array([[a, b], [a + hx, b],
                                           [a + hx, b + hy], [a, b + hy]]))
        nb = nx * ny

        # base faces: (a, b, cellL, cellR)
        bf = []
        for j in range(ny):
            for i in range(nx + 1):
                a = np.array([x0 + i * hx, y0 + j * hy])
                b = np.array([x0 + i * hx, y0 + (j + 1) * hy])
                left = j * nx + (i - 1) if i > 0 else -1
                right = j * nx + i if i < nx else -1
                bf.append([a, b, left, right])
        for j in range(ny + 1):
            for i in range(nx):
                a = np.array([x0 + i * hx, y0 + j * hy])
                b = np.array([x0 + (i + 1) * hx, y0 + j * hy])
                down = (j - 1) * nx + i if j > 0 else -1
                up = j * nx + i if j < ny else -1
                bf.append([a, b, down, up])

        # ---- cut classification of the base cells, then agglomeration
        frac = np.zeros((nb, 2))
        for t in range(nb):
            P = base_poly[t]
            A = abs(geom.poly_area(P))
            p1 = geom.clip_halfplane(P, iface.n, iface.c, True)
            p2 = geom.clip_halfplane(P, iface.n, iface.c, False)
            frac[t, 0] = abs(geom.poly_area(p1)) / A if len(p1) else 0.0
            frac[t, 1] = abs(geom.poly_area(p2)) / A if len(p2) else 0.0

        nbr = [[] for _ in range(nb)]
        for a, b, cl, cr in bf:
            if cl >= 0 and cr >= 0:
                nbr[cl].append(cr)
                nbr[cr].append(cl)

        group = list(range(nb))
        used = np.zeros(nb, bool)          # already involved in an agglomeration
        small = [t for t in range(nb)
                 if 0 < min(frac[t]) < agglo_tol]
        # stage 1: small cut on the Omega_1 side, then stage 2 for Omega_2
        order = sorted(small, key=lambda t: min(frac[t]))
        for t in order:
            if used[t]:
                continue
            i = int(np.argmin(frac[t]))
            best, bestval = -1, -1.0
            for s in nbr[t]:
                if used[s] or s == t:
                    continue
                if frac[s, i] < agglo_tol:
                    continue
                # prefer the neighbour with the smallest (positive) area on the
                # other side, cf. [BCDE21, Rem. 4.4]
                val = -frac[s, 1 - i]
                if val > bestval:
                    best, bestval = s, val
            if best < 0:
                for s in nbr[t]:
                    if not used[s] and frac[s, i] > frac[t, i]:
                        best = s
                        break
            if best >= 0:
                used[t] = used[best] = True
                group[t] = group[best] = min(t, best)

        # ---- build the agglomerated cells
        gid = {}
        self.cells = []
        for t in range(nb):
            g = group[t]
            if g not in gid:
                gid[g] = len(self.cells)
                self.cells.append({'base': [], 'faces': []})
            self.cells[gid[g]]['base'].append(t)
        self.n_agglo = sum(1 for c in self.cells if len(c['base']) > 1)

        for c in self.cells:
            pts = np.vstack([base_poly[t] for t in c['base']])
            xa, xb = pts[:, 0].min(), pts[:, 0].max()
            ya, yb = pts[:, 1].min(), pts[:, 1].max()
            c['poly'] = np.array([[xa, ya], [xb, ya], [xb, yb], [xa, yb]])
            c['h'] = geom.poly_diam(c['poly'])

        # ---- faces
        self.faces = []
        for a, b, cl, cr in bf:
            gl = gid[group[cl]] if cl >= 0 else -1
            gr = gid[group[cr]] if cr >= 0 else -1
            if gl == gr:
                continue                      # internal to an agglomerated cell
            t = b - a
            nrm = np.array([t[1], -t[0]])
            nrm /= np.linalg.norm(nrm)
            f = {'a': a, 'b': b, 'n': nrm, 'cells': [], 'bnd': (gl < 0 or gr < 0)}
            fid = len(self.faces)
            for g in (gl, gr):
                if g >= 0:
                    f['cells'].append(g)
                    self.cells[g]['faces'].append(fid)
            self.faces.append(f)

        # ---- cut data
        L = max(x1 - x0, y1 - y0)
        for c in self.cells:
            P = c['poly']
            A = abs(geom.poly_area(P))
            sub = {}
            for i, keep in ((0, True), (1, False)):
                p = geom.clip_halfplane(P, iface.n, iface.c, keep)
                if len(p) >= 3 and abs(geom.poly_area(p)) > drop_tol * A:
                    sub[i] = p
            c['sub'] = sub
            c['gam'] = geom.poly_line_segment(P, iface.n, iface.c) if len(sub) == 2 else None
            if c['gam'] is not None:
                a, b = c['gam']
                if np.linalg.norm(b - a) < drop_tol * L:
                    c['gam'] = None
        for f in self.faces:
            sub = {}
            for i, keep in ((0, True), (1, False)):
                s = geom.clip_segment(f['a'], f['b'], iface.n, iface.c, keep)
                if s is not None and np.linalg.norm(s[1] - s[0]) > drop_tol * L:
                    sub[i] = s
            f['sub'] = {i: s for i, s in sub.items()
                        if all(i in self.cells[g]['sub'] for g in f['cells'])}

        self.n_cut = sum(1 for c in self.cells if len(c['sub']) == 2)

    # -------------------------------------------------------------- reporting
    def stats(self):
        return dict(cells=len(self.cells), faces=len(self.faces),
                    cut=self.n_cut, agglomerated=self.n_agglo, h=self.h)
