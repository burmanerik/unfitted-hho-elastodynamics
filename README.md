# Unfitted HHO for elastodynamics with an imperfect interface

Verification and reproduction code for

> P. Huang and E. Burman, *An unfitted hybrid high-order method for the
> elastodynamics problem with imperfect interface*.

Every table and every figure of the numerical section of the paper is produced
by the scripts in this repository, from the raw output of the production runs
included in `results/`. There are no hand-entered numbers anywhere.

The method discretises the elastic wave equation in a medium made of two
components separated by an interface of linear slip type,

```
[sigma(u) n] = g_N,      [u] + K sigma(u_1) n = g_D,     K = alpha I + (beta - alpha) n (x) n,
```

on a mesh that does **not** resolve the interface: the unknowns are doubled in
the cut cells, the interface condition is built into the local strain
reconstruction on the softer side through the regularised interface stiffness
`S_h = (h_T/delta I + K)^{-1}`, and the cells carrying a small cut are cured by
agglomeration. No unknown is attached to the interface.

## Quick start

```sh
pip install -r requirements.txt

python make_figures.py        # rebuild every table and figure from results/  (~1 min)
python -m pytest tests -q     # verify the solver itself                      (~1 min)
python produce.py --list      # see what recomputing from scratch involves
```

`make_figures.py` writes the LaTeX tables to `paper/tab/` and the figures to
`paper/fig/`. Running it on a fresh clone reproduces the tables of the paper
byte for byte.

A `Makefile` wraps the same commands (`make figures`, `make test`, `make all`).

## Requirements

Python 3.9 or later with NumPy, SciPy, SymPy and Matplotlib; `pytest` for the
test suite. The exact versions used for the shipped results are pinned in
`requirements.txt` (Python 3.11, NumPy 2.4, SciPy 1.17, SymPy 1.14,
Matplotlib 3.10). Nothing is compiled and there are no other dependencies.

Recomputing the two wave-propagation stages needs about 8 GB of memory: the
reference solution uses `k = 4` on a 64×64 background mesh (256 584 degrees of
freedom before static condensation) and peaks at 5.7 GB.

## Reproducing the numerical section

Computation and post-processing are separate. `produce.py` runs the
computations and writes raw results as JSON (and NPZ for the field snapshots)
into `results/`; `make_figures.py` turns `results/` into the tables and
figures. Since `results/` is shipped, the second step alone suffices to
rebuild the paper.

| Float in the paper | `make_figures.py` target | file written | `produce.py` stage |
|---|---|---|---|
| Figure 1 (domain and notation) | `domain` | `fig/domain.pdf` | — (drawing only) |
| Figure 2 (background meshes, cut and agglomerated cells) | `mesh` | `fig/mesh.pdf` | — (meshing only) |
| Table 1 (convergence, test case 1) | `conv` | `tab/t1_conv.tex` | `conv` |
| Figure 3 (error versus *h*) | `conv` | `fig/conv1.pdf` | `conv` |
| Table 2 (robustness in the compliancy) | `robust` | `tab/t2_robust.tex` | `conv` |
| Table 3 (small cuts and agglomeration) | `smallcut` | `tab/t9_smallcut.tex` | `smallcut` |
| Table 4 (quasi-incompressible limit) | `incomp` | `tab/t3_incomp.tex` | `conv` |
| Table 5 (test case 1 in time, space errors) | `dyn` | `tab/t5_dyn_space.tex` | `dyn` |
| Table 6 (test case 1 in time, time errors) | `dyn` | `tab/t6_dyn_time.tex` | `dyn` |
| Table 7 (test case 2, sensor errors) | `case2` | `tab/t7_case2.tex` | `case2` |
| Figure 4 (velocity traces) | `traces` | `fig/traces2.pdf` | `case2` |
| Figure 5 (snapshots) | `snaps` | `fig/snaps2.pdf` | `case2` |
| Figure 6 (discrete energy) | `energy` | `fig/energy2.pdf` | `case2`, `timestep` |
| Table 8 (sensor errors versus the time step) | `dtstudy` | `tab/t11_dtstudy.tex` | `timestep` |
| Table 9 (temporal convergence) | `timeconv` | `tab/t10_time.tex` | `timestep` |
| Table 10 (test case 3, sensor errors) | `case3` | `tab/t8_case3.tex` | `case3` |
| Figure 7 (slip profile and traces) | `slip` | `fig/case3.pdf` | `case3` |
| Figure 8 (snapshots, bonded versus compliant) | `case3snap` | `fig/case3snap.pdf` | `case3` |

The file names under `tab/` follow the order in which the tables were written
rather than their final numbering in the paper; the mapping above is the
authoritative one, and the `\input` commands of the manuscript use these names.

Individual targets and stages can be named on the command line:

```sh
python make_figures.py conv case2      # just those two
python produce.py case2                # recompute the raw results they rest on
python produce.py --all                # everything, several hours
```

### Stages and their cost

Single core of an x86-64 cloud VM, Python 3.11:

| stage | runtime | memory | produces |
|---|---|---|---|
| `conv` | 3 min | < 1 GB | Tables 1, 2, 4 and Figure 3 |
| `dyn` | 1 min | < 1 GB | Tables 5 and 6 |
| `smallcut` | 30 s | < 1 GB | Table 3 |
| `case2` | 30 min | 5.7 GB | Table 7 and Figures 4, 5 |
| `case3` | ~60 min | 5.7 GB | Table 10 and Figures 7, 8 |
| `timestep` | 6 min | < 1 GB | Tables 8, 9 and Figure 6 (needs `case2` first) |

Of the 30 minutes of `case2`, 13 go into the `k = 4`, `n = 64` reference
solution and 8 into the finest run of the table itself.

`timestep` measures its errors against the reference solution computed by
`case2`, so that stage must have been run before (or in the same invocation).
All other stages are independent and can be run in any order, or in parallel on
separate cores.

## Layout

```
produce.py        the production runs, one function per stage
make_figures.py   turns results/ into paper/tab/*.tex and paper/fig/*.pdf
src/              the solver
results/          raw output of the production runs (shipped)
paper/            where the generated tables and figures land
tests/            verification suite
```

### The solver

| module | contents |
|---|---|
| `src/geom.py` | clipping of a convex polygon by a half-plane; polygon and segment quadratures (fan triangulation, collapsed Gauss rules) |
| `src/basis.py` | hierarchical *L*²-orthonormal bases on cut subcells and subfaces, obtained by a Cholesky factorisation of the monomial mass matrix |
| `src/mesh.py` | Cartesian background meshes, exact cut geometry for a straight interface, classification of cut cells and subfaces, simplified two-stage cell agglomeration |
| `src/hho.py` | material data, compliancy tensor, regularised interface stiffness, local strain and divergence reconstructions, the two stabilisations, assembly of `a_h` and of the mass matrix, static solve |
| `src/dynamics.py` | static condensation, consistent initial data, Newmark (β = 1/4, γ = 1/2) and SDIRK(*s*, *s*+1) for *s* = 1, 2, 3 (Crouzeix), point evaluation and sampling |
| `src/manufactured.py` | symbolic construction of a solution satisfying the linear slip conditions exactly, for an arbitrary straight interface and arbitrary (α, β) |
| `src/case1.py` | driver for test case 1 (stationary) |
| `src/case1_dyn.py` | driver for test case 1 in the time domain |
| `src/case23.py` | driver for test cases 2 and 3 (wave across an unresolved interface) |

Only two space dimensions and straight interfaces are implemented, the cut
geometry being exact in that case. The agglomeration is a simplified version of
the procedure of Burman, Cicuttin, Delay and Ern: each cell whose cut area
falls below `agglo_tol` times the cell area is merged with the neighbour that
shares a face on the same side of Γ and has the smallest area on the other
side, and each cell is merged at most once.

### What is in `results/`

| file | contents |
|---|---|
| `t1_conv.json`, `t2_robust.json`, `t3_incomp.json`, `t4_agglo.json` | one record per `(k, n)` or per compliancy, with the errors, the norms used to make them relative, and the mesh statistics (stage `conv`; `t4_agglo` is reported in the text, not in a table) |
| `t9_smallcut.json` | errors and condition numbers as a function of the relative cut size ξ, with and without agglomeration |
| `t5_dyn_space.json`, `t6_dyn_time.json` | errors of test case 1 in the time domain |
| `k_*.json`, `k_snaps.npz` | test case 2: `k_ref` is the reference solution, `k_k<k>n<n>` the sensor traces and energy history of each run, `k_snaps.npz` the sampled velocity fields; `t7_case2.json` collects the sensor errors |
| `l_*.json`, `l_snap_*.npz` | test case 3, same conventions, plus `l_traces_a<alpha>` for the compliancy sweep and `l_slip` for the slip profile along Γ; `t8_case3.json` collects the sensor errors |
| `m_*.json`, `t11_dtstudy.json`, `t10_time.json` | time-step study and temporal convergence |

Every trace file has the same shape: `t` (times), `vel` (`n_t × 2 × 2`, sensor
by component) and `energy` (the discrete energy at each time). The errors
reported in the tables are maxima over `[0, t_*i]` of the pointwise velocity
error at a sensor, in per cent of the maximum in time of the same component of
the reference solution.

## Verification

```sh
python -m pytest tests -q        # 38 tests, about one minute
```

The suite checks the method rather than the plumbing:

* `test_rigid_body.py` — the rigid-body motions lie in the kernel of `a_h`, globally for a bonded interface and subdomain by subdomain in the decoupled limit, for *k* = 1, 2, 3 on a cut mesh;
* `test_manufactured.py` — the symbolic solution satisfies both interface conditions to machine precision over the whole range of compliancies, degenerate cases α = 0 and β = 0 included;
* `test_static_convergence.py` — the rates *h*<sup>*k*+1</sup> and *h*<sup>*k*+2</sup> of Theorems 5.6 and 5.9, without an interface, with a cut mesh and a continuous solution, and with the slip interface;
* `test_time_integration.py` — the space rates of the Newmark scheme, the temporal orders 2, 3 and 4 of Newmark, SDIRK(2,3) and SDIRK(3,4), exact energy conservation by Newmark and monotone energy decay by SDIRK;
* `test_robustness.py` — the errors stay at the same level over sixteen orders of magnitude of compliancy; the condition number blows up as the cut shrinks without agglomeration and stays bounded with it.

### Determinism

The computations are deterministic: recomputing a stage reproduces the shipped
JSON files byte for byte. This was checked for the stages `conv`, `dyn`,
`smallcut` and `case2` — the last being the one the sensor errors of Table 7
come from — by deleting `results/` and running them again from a clean copy of
the repository. The one exception is the condition numbers of
`t4_agglo.json` and `t9_smallcut.json`, which come from a sparse eigenvalue
solve (`scipy.sparse.linalg.eigsh` with a randomised start and a tolerance of
1e-4) and move by about 1e-5 in relative terms from one run to the next. They
are reported in the paper with two significant digits, so this is invisible in
Table 3.

The temporal orders of the Runge–Kutta schemes are measured on test case 2,
with homogeneous boundary data. With the time-dependent Dirichlet data of test
case 1 the stages suffer the usual order reduction and only order ≈ 2.4 is
observed for SDIRK(2,3) and SDIRK(3,4); this is a property of the test, not of
the implementation.

## Licence and citation

The code is released under the MIT licence (see `LICENSE`). If you use it,
please cite the paper and, if you wish to refer to this exact version, the
archived release; `CITATION.cff` carries the metadata.

## Acknowledgements

This work was supported by the National Natural Science Foundation of China
grant 11301267, the Natural Science Foundation of Jiangsu Province grant
BK20191386, and EPSRC grants EP/P01576X/1 and EP/V050400/1.
