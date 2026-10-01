"""Production runs for the numerical section of

    P. Huang and E. Burman, An unfitted hybrid high-order method for the
    elastodynamics problem with imperfect interface.

Each stage writes its raw output to results/ as JSON (and NPZ for the field
snapshots).  Run `python produce.py --list` for the stage list, `python
produce.py --all` to recompute everything, or `python produce.py case2` for a
single stage.  make_figures.py turns results/ into the tables and figures.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'src'))
OUT = os.path.join(HERE, 'results')
os.makedirs(OUT, exist_ok=True)


def save(name, obj):
    with open(os.path.join(OUT, name + '.json'), 'w') as fh:
        json.dump(obj, fh, indent=1)
    print('  wrote', name)


def load_json(name):
    with open(os.path.join(OUT, name + '.json')) as fh:
        return json.load(fh)


# time step and snapshot times of the wave-propagation test cases
DT2 = 1.0 / 640
SD = ('sdirk', 3)
SNAPT = (0.125, 0.25, 0.5, 1.0)
SCHEMES = {'nm': 'newmark', 'sd2': ('sdirk', 2), 'sd3': ('sdirk', 3)}


def _traces(r):
    return dict(t=r['t'].tolist(), vel=r['vel'].tolist(),
                energy=r['energy'].tolist())


def stage_conv():
    """Test case 1: convergence, compliancy robustness, quasi-incompressible limit."""
    from case1 import run, NRM, CC
    ALPHA = BETA = 5e-2

    # A1: convergence rates
    tab = []
    for k in (1, 2, 3):
        for n in (4, 8, 16, 32):
            r = run(k, n, ALPHA, BETA)
            st = r['mesh'].stats()
            tab.append(dict(k=k, n=n, h=np.sqrt(2.0) / n, cells=st['cells'],
                            cut=st['cut'], agglo=st['agglomerated'], ndof=r['ndof'],
                            Ee=r['Ee'], E0=r['E0'], nEe=r['nrm_e'], n0=r['norm_i']))
            print('  A1', k, n, '%.3e %.3e' % (r['Ee'], r['E0']))
    save('t1_conv', tab)

    # A2: robustness with respect to the compliancy
    combos = [(0, 0), (1e-4, 1e-4), (1e-2, 1e-2), (1e0, 1e0), (1e4, 1e4), (1e12, 1e12),
              (0, 1e-2), (1e-2, 0), (0, 1e0), (1e0, 0)]
    tab = []
    for (a, b) in combos:
        row = dict(alpha=a, beta=b)
        for k in (1, 2, 3):
            r = run(k, 16, a, b)
            row['Ee%d' % k] = r['Ee'] / r['nrm_e']
            row['E0%d' % k] = r['E0'] / r['norm_i']
        tab.append(row)
        print('  A2', a, b)
    save('t2_robust', tab)

    # A3: quasi-incompressible limit, both stabilisation weights
    import mesh as M
    from hho import Material, Space, solve_static, compliance
    from manufactured import Manufactured
    tab = []
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, (20000, 2))
    lev = NRM @ pts.T - CC
    for nu in (0.3, 0.49, 0.4999, 0.499999):
        mat = Material.from_E_nu([1.0, 10.0], [nu, nu])
        ex = Manufactured(NRM, CC, mat.mu, mat.lam, 5e-2, 5e-2)
        lamdiv = max(np.abs(mat.lam[0] * ex.div(0, pts[lev < 0])).max(),
                     np.abs(mat.lam[1] * ex.div(1, pts[lev > 0])).max())
        for k in (1, 2, 3):
            row = dict(nu=nu, k=k, lam1=mat.lam[0], lam2=mat.lam[1], lamdiv=lamdiv)
            for aniso in (False, True):
                res = []
                for n in (8, 16):
                    m = M.Mesh((0, 1, 0, 1), n, n, M.Interface(NRM, CC))
                    S = Space(m, k, mat, compliance(5e-2, 5e-2, NRM), aniso_stab=aniso)
                    ui = S.interpolate(ex.u)
                    uh = solve_static(S, S.load(ex.f), ud=ui)
                    e = uh - ui
                    res.append((S.energy_norm(e), S.l2_cell_norm(e), S.l2_cell_norm(ui)))
                tag = 'a' if aniso else 'i'
                row['Ee' + tag] = res[1][0]
                row['E0' + tag] = res[1][1]
                row['rEe' + tag] = np.log(res[0][0] / res[1][0]) / np.log(2)
                row['rE0' + tag] = np.log(res[0][1] / res[1][1]) / np.log(2)
                row['nrm'] = res[1][2]
            tab.append(row)
            print('  A3', nu, k)
    save('t3_incomp', tab)

    # A4: influence of the cell agglomeration (errors and conditioning)
    import mesh as M2
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    from hho import Material as Mt, Space as Sp, solve_static as ss, compliance as cp
    mat = Mt.from_E_nu([1.0, 10.0], [0.25, 0.3])
    ex = Manufactured(NRM, CC, mat.mu, mat.lam, ALPHA, BETA)
    tab = []
    for agg in (0.3, 0.0):
        for k in (1, 2):
            for n in (8, 16, 32):
                m = M2.Mesh((0, 1, 0, 1), n, n, M2.Interface(NRM, CC), agglo_tol=agg)
                S = Sp(m, k, mat, cp(ALPHA, BETA, NRM))
                ui = S.interpolate(ex.u)
                uh = ss(S, S.load(ex.f), ud=ui)
                e = uh - ui
                Aff = S.A[S.free][:, S.free].tocsc()
                D = sp.diags(1.0 / np.sqrt(Aff.diagonal()))
                Asc = (D @ Aff @ D).tocsc()
                try:
                    lmax = spl.eigsh(Asc, k=1, which='LA', tol=1e-4,
                                     return_eigenvectors=False)[0]
                    lmin = spl.eigsh(Asc, k=1, sigma=0.0, which='LM', tol=1e-4,
                                     return_eigenvectors=False)[0]
                    cond = float(abs(lmax / lmin))
                except Exception:
                    cond = float('nan')
                tab.append(dict(agglo=agg, k=k, n=n, Ee=S.energy_norm(e),
                                E0=S.l2_cell_norm(e), cond=cond,
                                cells=len(m.cells), cut=m.n_cut))
                print('  A4', agg, k, n, '%.3e %.3e cond %.2e'
                      % (tab[-1]['Ee'], tab[-1]['E0'], cond))
    save('t4_agglo', tab)

def stage_dyn():
    """Test case 1 in the time domain: space and time convergence."""
    import case1_dyn as td
    tab = []
    for k in (1, 2, 3):
        for n in (4, 8, 16, 32):
            if k == 3 and n == 32:
                continue
            S, ex = td.setup(k, n)
            E0, Ee = td.integrate(S, ex, 1.0 / 2000, T=0.5, ret='max')
            tab.append(dict(k=k, n=n, Ee=Ee, E0=E0, ndof=int(S.free.sum())))
            print('  B1', k, n, '%.3e %.3e' % (tab[-1]['Ee'], tab[-1]['E0']))
    save('t5_dyn_space', tab)

    S, ex = td.setup(2, 8)
    ref = td.integrate(S, ex, 1.0 / 3200, T=0.5, ret='raw')
    tab = []
    for nt in (25, 50, 100, 200, 400, 800):
        d = S.l2_cell_norm(td.integrate(S, ex, 1.0 / nt, T=0.5, ret='raw') - ref)
        tab.append(dict(nt=nt, diff=d))
        print('  B2', nt, '%.3e' % d)
    save('t6_dyn_time', tab)

def stage_smallcut():
    """Small-cut study: a horizontal interface at distance xi*h above a mesh line."""
    import mesh as M
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    from hho import Material, Space, solve_static, compliance
    from manufactured import Manufactured
    nrm = np.array([0.0, 1.0])
    mat = Material.from_E_nu([1.0, 10.0], [0.25, 0.3])
    ALPHA = BETA = 5e-2
    n = 16
    tab = []
    for agg in (0.3, 0.0):
        for k in (1, 2):
            for xi in (0.5, 1e-1, 1e-2, 1e-3, 1e-4):
                cc = 0.5 + xi / n
                exx = Manufactured(nrm, cc, mat.mu, mat.lam, ALPHA, BETA)
                m = M.Mesh((0, 1, 0, 1), n, n, M.Interface(nrm, cc), agglo_tol=agg)
                S = Space(m, k, mat, compliance(ALPHA, BETA, nrm))
                ui = S.interpolate(exx.u)
                uh = solve_static(S, S.load(exx.f), ud=ui)
                e = uh - ui
                Aff = S.A[S.free][:, S.free].tocsc()
                D = sp.diags(1.0 / np.sqrt(Aff.diagonal()))
                Asc = (D @ Aff @ D).tocsc()
                try:
                    lmax = spl.eigsh(Asc, k=1, which='LA', tol=1e-4,
                                     return_eigenvectors=False)[0]
                    lmin = spl.eigsh(Asc, k=1, sigma=0.0, which='LM', tol=1e-4,
                                     return_eigenvectors=False)[0]
                    cond = float(abs(lmax / lmin))
                except Exception:
                    cond = float('nan')
                tab.append(dict(agglo=agg, k=k, xi=xi, cond=cond,
                                E0=S.l2_cell_norm(e) / S.l2_cell_norm(ui),
                                Ee=S.energy_norm(e) / S.energy_norm(ui),
                                agglomerated=m.n_agglo))
                print('  E', agg, k, xi, '%.3e %.3e cond %.3e agg %d'
                      % (tab[-1]['Ee'], tab[-1]['E0'], cond, m.n_agglo))
    save('t9_smallcut', tab)

def _sensor_err(ref, d, tstar=(0.9, 0.6)):
    rt, vref = np.array(ref['t']), np.array(ref['vel'])
    t, v = np.array(d['t']), np.array(d['vel'])
    out = []
    for si in range(2):
        m_ = t <= tstar[si]
        mr = rt <= tstar[si]
        for c_ in range(2):
            vr = np.interp(t[m_], rt, vref[:, si, c_])
            out.append(100 * np.abs(v[m_, si, c_] - vr).max()
                       / np.abs(vref[mr, si, c_]).max())
    return out

def stage_case2():
    """Test case 2: wave across a perfectly bonded unresolved interface."""
    import case23 as c
    t0 = time.time()
    ref = c.run(4, 64, 0.0, 0.0, dt=DT2, T=1.0, scheme=SD, snap_times=SNAPT)
    print('  K reference done %.0fs' % (time.time() - t0))
    np.savez_compressed(os.path.join(OUT, 'k_snaps.npz'),
                        **{('s%d' % i): ref['snaps'][t][2] for i, t in enumerate(SNAPT)},
                        xs=ref['snaps'][SNAPT[0]][0], ys=ref['snaps'][SNAPT[0]][1])
    save('k_ref', _traces(ref))
    save('k_ref_stats', dict(ref['mesh'].stats(), ndof=int(ref['S'].ndof)))
    refd = _traces(ref)
    del ref
    tab = []
    for k in (1, 2, 3):
        for n in (16, 32, 64):
            t1 = time.time()
            r = c.run(k, n, 0.0, 0.0, dt=DT2, T=1.0, scheme=SD)
            d = _traces(r)
            save('k_k%dn%d' % (k, n), d)
            e = r['energy']
            tab.append(dict(k=k, n=n, ndof=int(r['S'].ndof),
                            err=_sensor_err(refd, d),
                            loss=float((e[0] - e[-1]) / e[0])))
            print('  K', k, n, ['%.2f' % x for x in tab[-1]['err']],
                  '%.0fs' % (time.time() - t1))
            del r
    save('t7_case2', tab)

def stage_case3():
    """Test case 3: wave across a linear slip interface."""
    import case23 as c
    from dynamics import point_eval
    A = 2e-2
    t0 = time.time()
    ref = c.run(4, 64, A, A, dt=DT2, T=1.0, scheme=SD)
    print('  L reference done %.0fs' % (time.time() - t0))
    refd = _traces(ref)
    save('l_ref', refd)
    save('l_ref_stats', dict(ref['mesh'].stats(), ndof=int(ref['S'].ndof)))
    del ref
    tab = []
    for k in (1, 2, 3):
        for n in (16, 32, 64):
            r = c.run(k, n, A, A, dt=DT2, T=1.0, scheme=SD)
            d = _traces(r)
            save('l_k%dn%d' % (k, n), d)
            tab.append(dict(k=k, n=n, ndof=int(r['S'].ndof),
                            err=_sensor_err(refd, d)))
            print('  L', k, n, ['%.2f' % x for x in tab[-1]['err']])
            del r
    save('t8_case3', tab)

    # compliancy sweep: traces, slip profile and snapshots
    tt = np.linspace(-1.0, 1.0, 161)
    tang = np.array([-c.NRM[1], c.NRM[0]])
    pts = c.CC * c.NRM[None, :] + tt[:, None] * tang[None, :]
    prof = {}
    for a in (0.0, 5e-3, 2e-2, 5e-2):
        t1 = time.time()
        r = c.run(3, 64, a, a, dt=DT2, T=1.0, scheme=SD,
                  snap_times=(0.25,) if a in (0.0, 5e-2) else ())
        save('l_traces_a%g' % a, _traces(r))
        if r['snaps']:
            np.savez_compressed(os.path.join(OUT, 'l_snap_a%g.npz' % a),
                                s=r['snaps'][0.25][2], xs=r['snaps'][0.25][0],
                                ys=r['snaps'][0.25][1])
        S = r['S']
        prof['%g' % a] = [((point_eval(S, r['u'], p + 1e-9 * c.NRM, side=0)
                            - point_eval(S, r['u'], p - 1e-9 * c.NRM, side=1)).tolist())
                          for p in pts]
        print('  L slip', a, '%.0fs' % (time.time() - t1))
        del r, S
    save('l_slip', dict(s=tt.tolist(), jump=prof))

def stage_timestep():
    """Time-step study and temporal convergence (test case 2 setting)."""
    import case23 as c
    ref = load_json('k_ref')
    tab = []
    for tag in ('nm', 'sd3'):
        for nt in (80, 160, 320, 640):
            r = c.run(3, 32, 0.0, 0.0, dt=1.0 / nt, T=1.0, scheme=SCHEMES[tag])
            d = _traces(r)
            save('m_%s_nt%d' % (tag, nt), d)
            e = r['energy']
            tab.append(dict(scheme=tag, nt=nt, err=_sensor_err(ref, d),
                            loss=float((e[0] - e[-1]) / e[0])))
            print('  M', tag, nt, ['%.2f' % x for x in tab[-1]['err']])
            del r
    save('t11_dtstudy', tab)

    T = 0.25
    rr = c.run(2, 16, 0.0, 0.0, dt=1.0 / 5120, T=T, scheme=SD)
    S, uref = rr['S'], rr['u'].copy()
    tab = []
    for tag in ('nm', 'sd2', 'sd3'):
        for nt in (40, 80, 160, 320, 640):
            r = c.run(2, 16, 0.0, 0.0, dt=1.0 / nt, T=T, scheme=SCHEMES[tag])
            tab.append(dict(scheme=tag, nt=nt,
                            diff=float(S.l2_cell_norm(r['u'] - uref))))
            print('  M2', tag, nt, '%.3e' % tab[-1]['diff'])
    save('t10_time', tab)

# name -> (function, prerequisite stages, approximate single-core runtime,
#          what it feeds in the manuscript)
STAGES = {
    'conv':     (stage_conv,     (),         '14 min',
                 'Tables 1, 2, 4 and Figure 3'),
    'dyn':      (stage_dyn,      (),         '17 min',
                 'Tables 5 and 6'),
    'smallcut': (stage_smallcut, (),         '3 min',
                 'Table 3'),
    'case2':    (stage_case2,    (),         '30 min',
                 'Table 7 and Figures 4, 5'),
    'case3':    (stage_case3,    (),         '62 min',
                 'Table 10 and Figures 7, 8'),
    'timestep': (stage_timestep, ('case2',), '6 min',
                 'Tables 8, 9 and Figure 6'),
}
ORDER = ['conv', 'dyn', 'smallcut', 'case2', 'case3', 'timestep']
# a file written by the stage, used to check that a prerequisite has been run
SENTINEL = {'case2': 'k_ref.json'}


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('stages', nargs='*', help='stages to run (see --list)')
    p.add_argument('--all', action='store_true', help='run every stage')
    p.add_argument('--list', action='store_true', help='list the stages and exit')
    a = p.parse_args()
    if a.list or not (a.stages or a.all):
        print('stage       runtime   produces')
        for s in ORDER:
            _, dep, rt, what = STAGES[s]
            dep = (' [needs %s]' % ', '.join(dep)) if dep else ''
            print('  %-10s %-8s %s%s' % (s, rt, what, dep))
        return
    todo = ORDER if a.all else list(a.stages)
    for s in todo:
        if s not in STAGES:
            raise SystemExit('unknown stage %r (see --list)' % s)
    for s in todo:
        fn, deps, _, _ = STAGES[s]
        for d in deps:
            if d not in todo and not os.path.exists(os.path.join(OUT, SENTINEL[d])):
                raise SystemExit('stage %r needs stage %r to have been run first' % (s, d))
        print('=== stage', s)
        t0 = time.time()
        fn()
        print('=== stage %s done in %.0fs' % (s, time.time() - t0))


if __name__ == '__main__':
    main()
