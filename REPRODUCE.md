# Reproducing the Block Method vs hp-FEM comparison

This document lets a reviewer reproduce, on their own machine, the head-to-head
wall-clock comparison between the automated Block Method (this library) and a
properly configured **hp-adaptive finite-element method** (deal.II; UMFPACK
direct solver by default, or matrix-free conjugate gradients with a multigrid
preconditioner via `--solver mf-pmg`). Both solvers are C++ and run on the same machine, and both are
graded against **method-independent** reference solutions:

| Geometry | Reference | Independence |
|----------|-----------|--------------|
| unit square | exact Fourier-series solution | analytical (closed form) |
| L-shape | Lightning rational-approximation solution | non-FEM; cross-validated against deal.II hp-FEM (they agree to RMSE ≈ 4e-9) |
| square with hole | converged deal.II hp-FEM solution | Lightning cannot handle multiply-connected domains; cross-validated against a converged Block-Method solve (they agree to RMSE ≈ 2e-9) |

All references are committed to the repository, so **MATLAB is not required**.
The driver re-verifies each non-analytical reference live against a *different*
solver than the one that produced it (L-shape vs a fresh hp-FEM solve; square
with hole vs a fresh Block-Method solve), so its trustworthiness can be
confirmed without re-running the original tool. Grading the hp-FEM sweep on the
square with hole against a fine hp-FEM reference is conservative: it flatters the
competitor, not the Block Method.

Timing is apples-to-apples and reported **two ways** (addressing reviewer
comment R3.5, which asks that preprocessing be counted):

- **full pipeline** — the Block Method's block-covering generation + solve vs.
  the hp-FEM's mesh generation + adaptation + assembly + solve;
- **solve only** — both methods' preprocessing excluded.

Only deal.II's process startup is excluded (a one-time cost, not per-solve
compute). The Block Method wins under both accountings; its advantage is larger
under the full-pipeline accounting because hp-adaptivity's error-estimation and
mesh-adaptation cost is substantial.

---

## Option A — Docker (recommended, zero setup)

Requires only Docker. Builds both solvers and runs the comparison:

```bash
docker build -t pyblockmethod-repro .
docker run --rm pyblockmethod-repro
```

The image installs deal.II 9.3, Eigen, NumPy ≥ 2.0 and pybind11 ≥ 2.12, builds
both C++ components, runs the test suite, and prints the comparison tables plus
per-target speedups.

To also save the Pareto figure to your host:

```bash
docker run --rm -v "$PWD/out:/pyblockmethod/benchmarks/results" pyblockmethod-repro
# figure at ./out/reproduce_pareto.pdf
```

---

## Option B — Native build (Ubuntu 22.04)

```bash
# 1. Dependencies
sudo apt-get install -y build-essential cmake libdeal.ii-dev libeigen3-dev \
                        liblapack-dev python3 python3-pip
pip3 install "numpy>=2.0" "pybind11>=2.12" matplotlib scipy pytest

# 2. Build the pyBlockGrid C++ backend
cmake -S cpp -B cpp/build \
      -DPYTHON_EXECUTABLE="$(which python3)" \
      -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)"
cmake --build cpp/build -j"$(nproc)"
cp cpp/build/_pyblockgrid_cpp*.so src/pyBlockGrid/

# 3. Build the deal.II hp-FEM benchmark solver
cmake -S benchmarks/dealii -B benchmarks/dealii/build -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/dealii/build -j"$(nproc)"

# 3b. Build the C++ Lightning and AAA-LS spectral solvers (for the full Pareto
#     comparison; addresses reviewer comment R1.4 -- C++ vs C++, not vs MATLAB)
cmake -S benchmarks/spectral -B benchmarks/spectral/build -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/spectral/build -j"$(nproc)"

# 4. (optional) verify the build
python3 -m pytest tests/ -q

# 5. Run the comparison
python3 benchmarks/reproduce_comparison.py --reps 6
```

> **pybind11 ≥ 2.12 is required.** Earlier versions corrupt `py::array_t<bool>`
> outputs under NumPy 2.0, which silently breaks the point-in-polygon masks.

---

## What you should see

For each geometry, an accuracy/time table for both methods and the Block Method
speedup at matched RMSE under both accountings, e.g. on the L-shape:

```
  reference self-check: committed ref vs fresh converged hp-FEM -> RMSE 4.4e-09 (OK)
  ...
  Block-Method speedup vs hp-FEM at matched accuracy:
     target RMSE   full pipeline   solve only
           1e-04           12.1x         2.9x
           1e-05           15.4x         6.5x
           1e-06           16.6x        10.3x
```

The Block Method's advantage **grows with target accuracy**, consistent with
its exponential convergence versus the algebraic convergence of the finite
element method on these singular problems. Absolute timings depend on your CPU;
the *ratios* and the growing trend are the reproducible quantities. Use a
larger `--reps` for more stable timings.

## Validation and robustness experiments

Three additional self-contained scripts reproduce the paper's robustness and
mixed-boundary-condition results (each writes a PDF into `benchmarks/results/`):

```bash
# Re-entrant-corner robustness: Block Method vs hp-FEM on the L-shape's 270-degree
# corner, incl. convergence of the flux-intensity factor c1 -> 0.524.
python3 benchmarks/robustness_lshape.py

# Near-edge external singularity: boundary-data Taylor degree as a precision knob,
# and the accuracy floor as the singularity approaches an edge.
python3 benchmarks/boundary_singularity.py

# Motz's problem (mixed Dirichlet-Neumann): the leading singular coefficient
# converges to the standard benchmark value a0 = 401.1625.
python3 benchmarks/motz.py
```

The pytest suite includes `tests/test_neumann.py`, which checks homogeneous
(zero-flux) Neumann accuracy on manufactured solutions and the Motz coefficient.

## Direct solver comparison (without this harness)

The deal.II solver is a standalone executable and can be driven directly:

```bash
benchmarks/dealii/build/hp_laplace --geometry l_shape --bc hot_bottom \
    --refine-mode adaptive --min-degree 2 --max-degree 6 --cycles 10 \
    --out sol.dat --timing timing.dat
cat timing.dat   # setup / assembly / solve / n_dofs / solver
```

### Linear solver: sparse direct or multigrid-preconditioned CG

`--solver umfpack` (default) uses the UMFPACK sparse-direct solver.
`--solver mf-pmg` is the matrix-free configuration used in the benchmarks: the
hp operator is never assembled (it is applied cell by cell with sum
factorization via deal.II's MatrixFree), and the system is solved by conjugate
gradients with the p-multigrid preconditioner described next.
`--solver cg-pmg` uses the same preconditioner on the assembled matrix:
conjugate gradients preconditioned by a p-multigrid
V-cycle: the hp space is coarsened on the same mesh to linear elements
(Chebyshev-Jacobi smoother, `--pmg-smoother-degree 5`), and algebraic multigrid
(Trilinos ML) solves the linear-element coarse problem. It is warm-started from
the previous refinement cycle; the multigrid setup is counted in the `solve`
time, and the CG iteration counts are written to the timing file.
`--solver cg-amg` applies algebraic multigrid directly to the hp matrix; it is
mesh-independent but needs 20-47 iterations on these high-order problems.

All hp-FEM runs in the benchmark scripts use one thread
(`DEAL_II_NUM_THREADS=1`): the other solvers are single-threaded, and deal.II's
task parallelism is slower than serial execution at these problem sizes. Defaults:
relative residual `--cg-tol 1e-10` on the final cycle and `--cg-tol-intermediate 1e-6` on
earlier cycles (they only steer the adaptation), `--amg-threshold 0.02`, `--amg-sweeps 2`.
The timing file also splits the solve into multigrid setup, CG and warm-start transfer.
Both solvers share the discretization and adaptation strategy, so they reach
the same accuracy at a given cycle count.

```bash
benchmarks/dealii/build/hp_laplace --geometry l_shape --bc hot_bottom \
    --refine-mode adaptive --min-degree 2 --max-degree 6 --cycles 10 \
    --solver mf-pmg --out sol.dat --timing timing.dat
```

The Pareto benchmark treats the two as separate baselines (`dealii` and
`dealii_mg`); to rerun only those and keep the other methods' saved results:

```bash
python benchmarks/benchmark_pareto.py --only dealii,dealii_mg
```

The solver initializes MPI (required by Trilinos). If it appears to hang at
startup on a workstation with a display server, set `HWLOC_COMPONENTS=-gl`
(the benchmark scripts already do this).
