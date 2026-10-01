"""Test case 1: stationary elasticity with an unfitted linear-slip interface."""
import numpy as np
import mesh as M
from hho import Material, Space, solve_static, compliance
from manufactured import Manufactured

NRM = np.array([2.0, 1.0]) / np.sqrt(5.0)
CC = float(NRM @ np.array([0.5, 0.5]))


def run(k, n, alpha, beta, mat=None, Emod=(1.0, 10.0), nu=(0.25, 0.3), verbose=True):
    mat = mat or Material.from_E_nu(Emod, nu)
    K = compliance(alpha, beta, NRM)
    ex = Manufactured(NRM, CC, mat.mu, mat.lam, alpha, beta)
    m = M.Mesh((0, 1, 0, 1), n, n, M.Interface(NRM, CC))
    S = Space(m, k, mat, K)
    b = S.load(ex.f)
    ui = S.interpolate(ex.u)
    uh = solve_static(S, b, ud=ui)
    e = uh - ui
    return dict(mesh=m, S=S, uh=uh, ui=ui, E0=S.l2_cell_norm(e), Ee=S.energy_norm(e),
                ndof=int(S.free.sum()), ex=ex, norm_i=S.l2_cell_norm(ui),
                nrm_e=S.energy_norm(ui))


if __name__ == '__main__':
    alpha = beta = 5e-2
    for k in (1, 2, 3):
        print(f"--- k={k}  alpha=beta={alpha}")
        prev = None
        for n in (4, 8, 16, 32):
            r = run(k, n, alpha, beta)
            st = r['mesh'].stats()
            rr = ("     ", "     ") if prev is None else (
                f"{np.log(prev[0]/r['E0'])/np.log(2):5.2f}",
                f"{np.log(prev[1]/r['Ee'])/np.log(2):5.2f}")
            print(f"  n={n:3d} cells={st['cells']:5d} cut={st['cut']:4d} agg={st['agglomerated']:3d}"
                  f" ndof={r['ndof']:6d}  Ee={r['Ee']:.3e} {rr[1]}   E0={r['E0']:.3e} {rr[0]}")
            prev = (r['E0'], r['Ee'])
