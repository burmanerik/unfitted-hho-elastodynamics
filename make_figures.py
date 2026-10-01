"""Turns results/ into the LaTeX tables and the figures of the manuscript.

Run `python make_figures.py` to rebuild everything, or name individual targets,
e.g. `python make_figures.py conv case2`.  Tables are written to paper/tab/ and
figures to paper/fig/; `--list` prints the target list with the number of the
corresponding float in the manuscript.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.rcParams.update({'font.size': 9, 'font.family': 'serif',
                     'axes.grid': True, 'grid.alpha': 0.3,
                     'figure.constrained_layout.use': True})

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'src'))
OUT = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'paper', 'fig')
TAB = os.path.join(HERE, 'paper', 'tab')
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)


def load(name):
    with open(os.path.join(OUT, name + '.json')) as fh:
        return json.load(fh)

def w(name, txt):
    with open(os.path.join(TAB, name + '.tex'), 'w') as fh:
        fh.write(txt)
    print('  wrote tab/' + name + '.tex')

def sci(x, nd=2):
    if x == 0:
        return '0'
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return r'$%.*f\cdot10^{%d}$' % (nd, m, e)

def rate(a, b, fa=2.0):
    return '--' if a is None else '%.2f' % (np.log(a / b) / np.log(fa))

def fig_mesh():
    import mesh as M
    from case1 import NRM, CC
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.2))
    for ax, n in zip(axs, (16, 32)):
        m = M.Mesh((0, 1, 0, 1), n, n, M.Interface(NRM, CC))
        for c in m.cells:
            agg = len(c['base']) > 1
            ax.add_patch(Polygon(c['poly'], closed=True, fill=agg,
                                 facecolor='#7fb3d5' if agg else 'none',
                                 edgecolor='0.45', lw=0.4))
        t = np.array([-NRM[1], NRM[0]])
        s = np.linspace(-1.2, 1.2, 2)
        p = CC * NRM[None, :] + s[:, None] * t[None, :]
        ax.plot(p[:, 0], p[:, 1], 'r-', lw=1.6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
        ax.grid(False)
        st = m.stats()
        ax.set_title(r'$n=%d$: %d cells, %d cut, %d agglomerated'
                     % (n, st['cells'], st['cut'], st['agglomerated']), fontsize=8)
    fig.savefig(os.path.join(FIG, 'mesh.pdf'))
    plt.close(fig)
    print('  wrote fig/mesh.pdf')

def tab_conv():
    t = load('t1_conv')
    lines = [r'\begin{tabular}{rrrrrr@{\quad}rr@{\quad}rr}', r'\toprule',
             r'$k$ & $n$ & card$(\mathcal{T}_h)$ & cut & aggl. & $N_{\rm dof}$'
             r' & $E_e$ & rate & $E_0$ & rate\\', r'\midrule']
    prev = {}
    for r in t:
        k = r['k']
        re_ = rate(prev[k][0], r['Ee']) if k in prev else '--'
        r0 = rate(prev[k][1], r['E0']) if k in prev else '--'
        if r['n'] == 4 and prev:
            lines.append(r'\midrule')
        lines.append('%d & %d & %d & %d & %d & %d & %s & %s & %s & %s\\\\'
                     % (k, r['n'], r['cells'], r['cut'], r['agglo'], r['ndof'],
                        sci(r['Ee']), re_, sci(r['E0']), r0))
        prev[k] = (r['Ee'], r['E0'])
    lines += [r'\bottomrule', r'\end{tabular}']
    w('t1_conv', '\n'.join(lines))

    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9))
    mk = {1: 'o', 2: 's', 3: 'D'}
    for k in (1, 2, 3):
        rows = [r for r in t if r['k'] == k]
        h = np.array([r['h'] for r in rows])
        for ax, key in zip(axs, ('Ee', 'E0')):
            ax.loglog(h, [r[key] for r in rows], mk[k] + '-k', ms=4, lw=0.9,
                      mfc='w', label='$k=%d$' % k)
    for ax, key, lab, off in zip(axs, ('Ee', 'E0'),
                                 (r'$E_e$', r'$E_0$'), (1, 2)):
        h = np.array([r['h'] for r in t if r['k'] == 1])
        for k in (1, 2, 3):
            rows = [r for r in t if r['k'] == k]
            y0 = rows[-1][key]
            hh = np.array([h[1], h[-1]])
            ax.loglog(hh, y0 * (hh / h[-1]) ** (k + off), ':', color='0.5', lw=0.8)
        ax.set_xlabel('$h$'); ax.set_ylabel(lab); ax.legend(fontsize=8)
        ax.set_xticks(h)
        ax.set_xticklabels([('%.3f' % v).rstrip('0') for v in h])
        ax.minorticks_off()
    fig.savefig(os.path.join(FIG, 'conv1.pdf'))
    plt.close(fig)
    print('  wrote fig/conv1.pdf')

def tab_robust():
    t = load('t2_robust')
    lines = [r'\begin{tabular}{rr@{\qquad}rr@{\qquad}rr@{\qquad}rr}', r'\toprule',
             r'& & \multicolumn{2}{c}{$k=1$} & \multicolumn{2}{c}{$k=2$}'
             r' & \multicolumn{2}{c}{$k=3$}\\',
             r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
             r'$\alpha$ & $\beta$ & $\tilde E_e$ & $\tilde E_0$ & $\tilde E_e$'
             r' & $\tilde E_0$ & $\tilde E_e$ & $\tilde E_0$\\', r'\midrule']
    for i, r in enumerate(t):
        if i == 6:
            lines.append(r'\midrule')
        lines.append('%s & %s & %s & %s & %s & %s & %s & %s\\\\'
                     % (fmtexp(r['alpha']), fmtexp(r['beta']),
                        sci(r['Ee1']), sci(r['E01']), sci(r['Ee2']), sci(r['E02']),
                        sci(r['Ee3']), sci(r['E03'])))
    lines += [r'\bottomrule', r'\end{tabular}']
    w('t2_robust', '\n'.join(lines))

def fmtexp(x):
    if x == 0:
        return '$0$'
    e = int(round(np.log10(x)))
    return '$10^{%d}$' % e

def tab_incomp():
    t = load('t3_incomp')
    lines = [r'\begin{tabular}{rrrr@{\quad}rrrr@{\quad}rrrr}', r'\toprule',
             r'& & & & \multicolumn{4}{c}{isotropic weight \eqref{eq:stab}}'
             r' & \multicolumn{4}{c}{anisotropic weight}\\',
             r'\cmidrule(lr){5-8}\cmidrule(lr){9-12}',
             r'$k$ & $\nu$ & $\lambda_1$ & $\lambda\|\nabla\!\cdot\!\underline{u}\|_\infty$'
             r' & $E_e$ & rate & $E_0$ & rate & $E_e$ & rate & $E_0$ & rate\\',
             r'\midrule']
    last = None
    for r in sorted(t, key=lambda r: (r['k'], r['nu'])):
        if last is not None and r['k'] != last:
            lines.append(r'\midrule')
        last = r['k']
        lines.append('%d & %.6g & %s & %.1f & %s & %.2f & %s & %.2f & %s & %.2f & %s & %.2f\\\\'
                     % (r['k'], r['nu'], sci(r['lam1'], 2), r['lamdiv'],
                        sci(r['Eei']), r['rEei'], sci(r['E0i']), r['rE0i'],
                        sci(r['Eea']), r['rEea'], sci(r['E0a']), r['rE0a']))
    lines += [r'\bottomrule', r'\end{tabular}']
    w('t3_incomp', '\n'.join(lines))

def tab_smallcut():
    t = load('t9_smallcut')
    lines = [r'\begin{tabular}{r@{\qquad}rrr@{\qquad}rrr}', r'\toprule',
             r'& \multicolumn{3}{c}{with agglomeration}'
             r' & \multicolumn{3}{c}{without agglomeration}\\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
             r'$\xi$ & $\tilde E_e$ & $\tilde E_0$ & $\mathrm{cond}$'
             r' & $\tilde E_e$ & $\tilde E_0$ & $\mathrm{cond}$\\', r'\midrule']
    for k in (1, 2):
        lines.append(r'\multicolumn{7}{l}{\itshape $k=%d$}\\' % k)
        xis = sorted({r['xi'] for r in t}, reverse=True)
        for xi in xis:
            a = [r for r in t if r['k'] == k and r['xi'] == xi and r['agglo'] > 0][0]
            b = [r for r in t if r['k'] == k and r['xi'] == xi and r['agglo'] == 0][0]
            lines.append('%s & %s & %s & %s & %s & %s & %s\\\\'
                         % (fmtexp(xi) if xi != 0.5 else r'$5\cdot10^{-1}$',
                            sci(a['Ee']), sci(a['E0']), sci(a['cond'], 1),
                            sci(b['Ee']), sci(b['E0']), sci(b['cond'], 1)))
        if k == 1:
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    w('t9_smallcut', '\n'.join(lines))

def tab_dyn():
    t = load('t5_dyn_space')
    lines = [r'\begin{tabular}{rrrr@{\quad}rr}', r'\toprule',
             r'$k$ & $n$ & $N_{\rm dof}$ & $E_e^\infty$ & rate & $E_0^\infty$\\',
             r'\midrule']
    prev = {}
    lines = [r'\begin{tabular}{rrrrrrr}', r'\toprule',
             r'$k$ & $n$ & $N_{\rm dof}$ & $E_e^\infty$ & rate & $E_0^\infty$ & rate\\',
             r'\midrule']
    for r in t:
        k = r['k']
        if r['n'] == 4 and prev:
            lines.append(r'\midrule')
        lines.append('%d & %d & %d & %s & %s & %s & %s\\\\'
                     % (k, r['n'], r['ndof'], sci(r['Ee']),
                        rate(prev[k][0], r['Ee']) if k in prev else '--',
                        sci(r['E0']),
                        rate(prev[k][1], r['E0']) if k in prev else '--'))
        prev[k] = (r['Ee'], r['E0'])
    lines += [r'\bottomrule', r'\end{tabular}']
    w('t5_dyn_space', '\n'.join(lines))

    t = [r for r in load('t6_dyn_time') if r['nt'] >= 50]
    cells, prev = [], None
    for r in t:
        cells.append(r'$1/%d$ & %s & %s' % (r['nt'], sci(r['diff']),
                                            rate(prev, r['diff']) if prev else '--'))
        prev = r['diff']
    lines = [r'\begin{tabular}{rrr}', r'\toprule',
             r'$\Delta t$ & $\|\bm{u}_\mathcal{T}^N-\bm{u}^N_{\mathcal{T},\rm ref}\|_\Omega$'
             r' & rate\\', r'\midrule']
    lines += [c + r'\\' for c in cells]
    lines += [r'\bottomrule', r'\end{tabular}']
    w('t6_dyn_time', '\n'.join(lines))

def fig_domain():
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.7))
    ax = axs[0]
    for i in range(7):
        ax.plot([i / 6, i / 6], [0, 1], color='0.6', lw=0.5)
        ax.plot([0, 1], [i / 6, i / 6], color='0.6', lw=0.5)
    th = np.linspace(0, 2 * np.pi, 400)
    r = 0.30 + 0.03 * np.cos(3 * th)
    x, y = 0.5 + r * np.cos(th), 0.5 + r * np.sin(th)
    ax.plot(x, y, 'r-', lw=1.6)
    ax.text(0.5, 0.5, r'$\Omega_1$', ha='center', fontsize=11)
    ax.text(0.10, 0.88, r'$\Omega_2$', fontsize=11)
    ax.text(0.80, 0.66, r'$\Gamma$', color='r', fontsize=11)
    j = 40
    nx, ny = np.cos(th[j]), np.sin(th[j])
    ax.annotate('', xy=(x[j] + 0.13 * nx, y[j] + 0.13 * ny), xytext=(x[j], y[j]),
                arrowprops=dict(arrowstyle='->', lw=1.0))
    ax.text(x[j] + 0.16 * nx, y[j] + 0.13 * ny, r'$\mathbf{n}_\Gamma$', fontsize=10)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal'); ax.axis('off'); ax.grid(False)

    ax = axs[1]
    sx = np.linspace(0, 1, 200)
    gy = 0.30 + 0.30 * sx + 0.25 * sx ** 2          # interface inside the cell
    ax.fill_between(sx, 0.0, gy, color='#bcd6ea', zorder=0)
    ax.plot(sx, gy, 'r-', lw=2.0, zorder=3)
    b1 = dict(color='#1b4f72', lw=3.2, solid_capstyle='butt', zorder=2)
    b2 = dict(color='#af7ac5', lw=3.2, solid_capstyle='butt', zorder=2)
    ax.plot([0, 0], [0, gy[0]], **b1)
    ax.plot([0, 1], [0, 0], **b1)
    ax.plot([1, 1], [0, gy[-1]], **b1)
    ax.plot([0, 0], [gy[0], 1], **b2)
    ax.plot([0, 1], [1, 1], **b2)
    ax.plot([1, 1], [gy[-1], 1], **b2)
    ax.text(0.45, 0.18, r'$T^1$', fontsize=12)
    ax.text(0.25, 0.78, r'$T^2$', fontsize=12)
    ax.text(0.62, 0.68, r'$T^\Gamma$', color='r', fontsize=11)
    ax.text(-0.42, 0.12, r'$(\partial T)^1$', color='#1b4f72', fontsize=10)
    ax.text(-0.42, 0.70, r'$(\partial T)^2$', color='#7d3c98', fontsize=10)
    ax.set_xlim(-0.45, 1.15); ax.set_ylim(-0.12, 1.15)
    ax.set_aspect('equal'); ax.axis('off'); ax.grid(False)
    fig.savefig(os.path.join(FIG, 'domain.pdf'))
    plt.close(fig)
    print('  wrote fig/domain.pdf')

def _tab_sensor(name, out, block_by_k=True, nmin=32):
    t = [r for r in load(name) if r['n'] >= nmin]
    lines = [r'\begin{tabular}{rrrrrrr}', r'\toprule',
             r'$k$ & $n$ & $\lambda_w/h$ & $N_{\rm dof}$ & $v_x(S_1)$ & $v_y(S_1)$'
             r' & $v_x(S_2)$ & $v_y(S_2)$\\', r'\midrule']
    lines[2] = (r'$k$ & $n$ & $\lambda_w/h$ & $N_{\rm dof}$ & $v_x(S_1)$ & $v_y(S_1)$'
                r' & $v_x(S_2)$ & $v_y(S_2)$\\')
    lines[0] = r'\begin{tabular}{rrrrrrrr}'
    last = None
    for r in t:
        if block_by_k and last is not None and r['k'] != last:
            lines.append(r'\midrule')
        last = r['k']
        lines.append('%d & %d & %.1f & %d & %s\\\\'
                     % (r['k'], r['n'], 0.69282 / (3.0 / r['n']), r['ndof'],
                        ' & '.join('%.2f' % x for x in r['err'])))
    lines += [r'\bottomrule', r'\end{tabular}']
    w(out, '\n'.join(lines))

def tab_case2():
    _tab_sensor('t7_case2', 't7_case2')

def tab_case3():
    _tab_sensor('t8_case3', 't8_case3')

def tab_dtstudy():
    t = load('t11_dtstudy')
    names = {'nm': 'Newmark', 'sd3': 'SDIRK(3,4)'}
    lines = [r'\begin{tabular}{llrrrrr}', r'\toprule',
             r'scheme & $\Delta t$ & $v_x(S_1)$ & $v_y(S_1)$ & $v_x(S_2)$ & $v_y(S_2)$'
             r' & energy loss\\', r'\midrule']
    last = None
    for r in t:
        if last is not None and r['scheme'] != last:
            lines.append(r'\midrule')
        first = r['scheme'] != last
        last = r['scheme']
        lines.append('%s & $1/%d$ & %s & %s\\\\'
                     % (names.get(r['scheme'], r['scheme']) if first else '', r['nt'],
                        ' & '.join('%.2f' % x for x in r['err']),
                        sci(max(r['loss'], 0.0), 1) if r['loss'] > 1e-12 else '$<10^{-12}$'))
    lines += [r'\bottomrule', r'\end{tabular}']
    w('t11_dtstudy', '\n'.join(lines))

def tab_timeconv():
    t = load('t10_time')
    nts = sorted({r['nt'] for r in t})
    lines = [r'\begin{tabular}{r' + 'rr' * 3 + '}', r'\toprule',
             r'& \multicolumn{2}{c}{Newmark} & \multicolumn{2}{c}{SDIRK(2,3)}'
             r' & \multicolumn{2}{c}{SDIRK(3,4)}\\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
             r'$\Delta t$ & err & rate & err & rate & err & rate\\', r'\midrule']
    prev = {}
    for nt in nts:
        cells = []
        for tag in ('nm', 'sd2', 'sd3'):
            v = [r['diff'] for r in t if r['scheme'] == tag and r['nt'] == nt][0]
            cells.append(sci(v))
            cells.append(rate(prev.get(tag), v) if tag in prev else '--')
            prev[tag] = v
        lines.append('$1/%d$ & %s\\\\' % (nt, ' & '.join(cells)))
    lines += [r'\bottomrule', r'\end{tabular}']
    w('t10_time', '\n'.join(lines))

def fig_traces():
    ref = load('k_ref')
    t, vref = np.array(ref['t']), np.array(ref['vel'])
    styles = [('k_k1n64', '$k=1$, $n=64$', '-', '#d95f02'),
              ('k_k2n64', '$k=2$, $n=64$', '-', '#1b9e77'),
              ('k_k3n32', '$k=3$, $n=32$', '--', '#7570b3')]
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 3.8), sharex=True)
    for si in range(2):
        for comp in range(2):
            ax = axs[si][comp]
            ax.plot(t, vref[:, si, comp], 'k-', lw=1.2, label='reference')
            for nm, lab, ls, col in styles:
                try:
                    d = load(nm)
                except FileNotFoundError:
                    continue
                v = np.array(d['vel'])
                ax.plot(np.array(d['t']), v[:, si, comp], ls, color=col, lw=0.9, label=lab)
            ax.set_title(r'$v_%s$ at $S_%d$' % ('xy'[comp], si + 1), fontsize=8)
            if si == 1:
                ax.set_xlabel('$t$')
    axs[0][0].legend(fontsize=6.5)
    fig.savefig(os.path.join(FIG, 'traces2.pdf'))
    plt.close(fig)
    print('  wrote fig/traces2.pdf')

def fig_snaps():
    import case23 as c
    d = np.load(os.path.join(OUT, 'k_snaps.npz'))
    xs, ys = d['xs'], d['ys']
    times = (0.125, 0.25, 0.5, 1.0)
    fig, axs = plt.subplots(2, 4, figsize=(6.9, 3.6))
    for j in range(4):
        S = d['s%d' % j]
        for i in range(2):
            ax = axs[i][j]
            v = S[:, :, i]
            mm = np.nanmax(np.abs(v))
            ax.imshow(v, origin='lower', extent=(xs[0], xs[-1], ys[0], ys[-1]),
                      cmap='RdBu_r', vmin=-mm, vmax=mm)
            tt = np.linspace(-2.2, 2.2, 2)
            tv = np.array([-c.NRM[1], c.NRM[0]])
            p = tt[:, None] * tv[None, :]
            ax.plot(p[:, 0], p[:, 1], 'k-', lw=0.7)
            ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1])
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if i == 0:
                ax.set_title('$t=%.3g$' % times[j], fontsize=8)
            if j == 0:
                ax.set_ylabel('$v_%s$' % 'xy'[i], fontsize=9)
    fig.savefig(os.path.join(FIG, 'snaps2.pdf'), dpi=200)
    plt.close(fig)
    print('  wrote fig/snaps2.pdf')

def fig_energy():
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6))
    cols = {1: '#d95f02', 2: '#1b9e77', 3: '#7570b3'}
    for k in (1, 2, 3):
        try:
            d = load('m_nm_nt640') if k == 1 else None
        except FileNotFoundError:
            d = None
    for tag, lab, ls in (('m_nm_nt%d', 'Newmark', '-'), ('m_sd3_nt%d', 'SDIRK(3,4)', '--')):
        for nt, col in zip((160, 320, 640), ('#d95f02', '#1b9e77', '#7570b3')):
            try:
                d = load(tag % nt)
            except FileNotFoundError:
                continue
            e = np.array(d['energy'])
            axs[0].semilogy(np.array(d['t']),
                            np.maximum(np.abs(e - e[0]) / e[0], 1e-17), ls,
                            color=col, lw=0.9, label=r'%s, $\Delta t=1/%d$' % (lab, nt))
    axs[0].set_ylabel(r'$|\hat E^n-\hat E^0|/\hat E^0$')
    axs[0].set_title(r'$k=3$, $n=32$', fontsize=9)
    for k in (1, 2, 3):
        for n, ls in ((32, '--'), (64, '-')):
            try:
                d = load('k_k%dn%d' % (k, n))
            except FileNotFoundError:
                continue
            e = np.array(d['energy'])
            axs[1].semilogy(np.array(d['t']), np.maximum((e[0] - e) / e[0], 1e-14), ls,
                            color=cols[k], lw=0.9, label='$k=%d$, $n=%d$' % (k, n))
    axs[1].set_ylabel(r'$(\hat E^0-\hat E^n)/\hat E^0$')
    axs[1].set_title(r'SDIRK$(3,4)$, $\Delta t=1/640$', fontsize=9)
    for ax in axs:
        ax.set_xlabel('$t$')
        ax.legend(fontsize=6, loc='lower right')
    fig.savefig(os.path.join(FIG, 'energy2.pdf'))
    plt.close(fig)
    print('  wrote fig/energy2.pdf')

def fig_case3():
    comps = ['0', '0.005', '0.02', '0.05']
    cols = ['k', '#d95f02', '#1b9e77', '#7570b3']
    sl = load('l_slip')
    s_ = np.array(sl['s'])
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.7))
    for a, col in zip(comps, cols):
        if a not in sl['jump']:
            continue
        j = np.array(sl['jump'][a])
        axs[0].plot(s_, np.linalg.norm(j, axis=1), color=col, lw=1.0,
                    label=r'$\alpha=\beta=%s$' % a)
    axs[0].set_xlabel(r'arclength along $\Gamma$')
    axs[0].set_ylabel(r'$|[\![\mathbf{u}_h]\!]|$')
    axs[0].set_title(r'slip at $t=T$', fontsize=8)
    axs[0].legend(fontsize=6.5, loc='upper right')
    for idx, (si, comp) in enumerate(((0, 1), (1, 1))):
        ax = axs[idx + 1]
        for a, col in zip(comps, cols):
            try:
                d = load('l_traces_a%s' % a)
            except FileNotFoundError:
                continue
            v = np.array(d['vel'])
            ax.plot(np.array(d['t']), v[:, si, comp], color=col, lw=0.9)
        ax.set_xlabel('$t$')
        ax.set_title(r'$v_y$ at $S_%d$' % (si + 1), fontsize=8)
    fig.savefig(os.path.join(FIG, 'case3.pdf'))
    plt.close(fig)
    print('  wrote fig/case3.pdf')

def fig_case3snap():
    import case23 as c
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 3.0))
    try:
        data = {a: np.load(os.path.join(OUT, 'l_snap_a%s.npz' % a)) for a in ('0', '0.05')}
    except FileNotFoundError:
        return
    m0 = max(np.nanmax(np.abs(d['s'][:, :, 1])) for d in data.values())
    for ax, a, lab in zip(axs, ('0', '0.05'),
                          (r'$\alpha=\beta=0$', r'$\alpha=\beta=5\cdot10^{-2}$')):
        d = data[a]
        xs, ys, S = d['xs'], d['ys'], d['s']
        ax.imshow(S[:, :, 1], origin='lower', extent=(xs[0], xs[-1], ys[0], ys[-1]),
                  cmap='RdBu_r', vmin=-m0, vmax=m0)
        tt = np.linspace(-2.2, 2.2, 2)
        tv = np.array([-c.NRM[1], c.NRM[0]])
        pp = tt[:, None] * tv[None, :]
        ax.plot(pp[:, 0], pp[:, 1], 'k-', lw=0.8)
        ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1])
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(lab, fontsize=9)
    fig.savefig(os.path.join(FIG, 'case3snap.pdf'), dpi=200)
    plt.close(fig)
    print('  wrote fig/case3snap.pdf')

# target -> (function, float in the manuscript, file written)
TARGETS = {
    'domain':   (fig_domain,      'Figure 1',          'fig/domain.pdf'),
    'mesh':     (fig_mesh,        'Figure 2',          'fig/mesh.pdf'),
    'conv':     (tab_conv,        'Table 1, Figure 3', 'tab/t1_conv.tex, fig/conv1.pdf'),
    'robust':   (tab_robust,      'Table 2',           'tab/t2_robust.tex'),
    'smallcut': (tab_smallcut,    'Table 3',           'tab/t9_smallcut.tex'),
    'incomp':   (tab_incomp,      'Table 4',           'tab/t3_incomp.tex'),
    'dyn':      (tab_dyn,         'Tables 5 and 6',    'tab/t5_dyn_space.tex, tab/t6_dyn_time.tex'),
    'case2':    (tab_case2,       'Table 7',           'tab/t7_case2.tex'),
    'traces':   (fig_traces,      'Figure 4',          'fig/traces2.pdf'),
    'snaps':    (fig_snaps,       'Figure 5',          'fig/snaps2.pdf'),
    'energy':   (fig_energy,      'Figure 6',          'fig/energy2.pdf'),
    'dtstudy':  (tab_dtstudy,     'Table 8',           'tab/t11_dtstudy.tex'),
    'timeconv': (tab_timeconv,    'Table 9',           'tab/t10_time.tex'),
    'case3':    (tab_case3,       'Table 10',          'tab/t8_case3.tex'),
    'slip':     (fig_case3,       'Figure 7',          'fig/case3.pdf'),
    'case3snap': (fig_case3snap,  'Figure 8',          'fig/case3snap.pdf'),
}
ORDER = ['domain', 'mesh', 'conv', 'robust', 'smallcut', 'incomp', 'dyn', 'case2',
         'traces', 'snaps', 'energy', 'dtstudy', 'timeconv', 'case3', 'slip',
         'case3snap']


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    if '--list' in sys.argv or '-l' in sys.argv:
        print('target       float in the manuscript   file')
        for t in ORDER:
            _, fl, f = TARGETS[t]
            print('  %-11s %-24s %s' % (t, fl, f))
        return
    todo = args or ORDER
    missing = []
    for t in todo:
        if t not in TARGETS:
            raise SystemExit('unknown target %r (see --list)' % t)
        try:
            TARGETS[t][0]()
        except FileNotFoundError as e:
            missing.append((t, os.path.basename(str(e).split("'")[-2] if "'" in str(e) else str(e))))
            print('  skip %s (missing %s)' % (t, e))
    if missing:
        print('\n%d target(s) skipped: run the corresponding stage of produce.py first.'
              % len(missing))


if __name__ == '__main__':
    main()
