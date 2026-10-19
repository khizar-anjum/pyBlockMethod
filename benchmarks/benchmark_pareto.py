#!/usr/bin/env python3
"""
Benchmark: Pareto curves (wall-clock time vs MSE) for the Block Method, the
deal.II hp-adaptive FEM baseline (with a sparse-direct solve and with a
multigrid-preconditioned CG solve), FreeFEM P2, Lightning, and AAA-LS.

Every method is graded against a *method-independent* reference (analytical
where a closed form exists, otherwise a cross-validated solution) instead of the
earlier circular FreeFEM-P2 reference -- see benchmarks/analytical_reference.py.

Timing is the FULL pipeline for every method (R3.5), not the linear solve alone:
Block Method = block-covering generation + solve; deal.II hp-FEM = mesh
generation + adaptation + assembly + solve; FreeFEM P2 = mesh generation +
assembly + solve; Lightning/AAA-LS = full solve incl. pole/sample preprocessing.

Produces:
  - benchmarks/results/pareto_data.npz  (all raw data)
  - benchmarks/results/pareto_*.pdf     (publication figures)

Usage:
    python benchmarks/benchmark_pareto.py              # run all benchmarks
    python benchmarks/benchmark_pareto.py --plot-only   # just regenerate figures from saved data
    python benchmarks/benchmark_pareto.py --only dealii,dealii_mg   # rerun some methods, keep the rest
"""

import os
import sys
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- paths ---
EXPERIMENTS_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENTS_DIR / 'results'
FREEFEM_DIR = EXPERIMENTS_DIR / 'freefem'
INDEP_REF_DIR = EXPERIMENTS_DIR / 'analytical_reference_data'
DEALII_BIN = EXPERIMENTS_DIR / 'dealii' / 'build' / 'hp_laplace'
LIGHTNING_BIN = EXPERIMENTS_DIR / 'spectral' / 'build' / 'lightning_laplace'
AAALS_BIN = EXPERIMENTS_DIR / 'spectral' / 'build' / 'aaals_laplace'
REPO_ROOT = EXPERIMENTS_DIR.parent

# Add repo to path for pyBlockGrid import
sys.path.insert(0, str(REPO_ROOT / 'src'))
from pyBlockGrid import polygon, volkovSolver

# --- constants ---
DELTA = 0.05
N_TIMING_REPEATS = 5  # mean of 5 for timing, with std for error bars

# --- geometry definitions ---
GEOMETRIES = {
    'unit_square': {
        'vertices': np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
        'bc': [[1.0], [0.0], [0.0], [0.0]],
        'is_dirichlet': [True] * 4,
        'hole': None,
        'lightning_ok': True,
    },
    'l_shape': {
        'vertices': np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float),
        'bc': [[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
        'is_dirichlet': [True] * 6,
        'hole': None,
        'lightning_ok': True,
    },
    'square_with_hole': {
        'vertices': np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
        'bc': [[1.0], [0.0], [0.0], [0.0]],
        'is_dirichlet': [True] * 4,
        'hole': {
            'vertices': np.array([[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]])[::-1],
            'bc': [[0.0]] * 4,
            'is_dirichlet': [True] * 4,
        },
        'lightning_ok': False,  # Lightning doesn't support multiply-connected domains
    },
}

# --- parameter sweeps ---
VOLKOV_N_VALUES = [5, 10, 15, 20, 30, 40, 50]
FREEFEM_MESH_DENSITIES = [5, 10, 20, 50, 100, 150, 200]
DEALII_CYCLES = [3, 4, 6, 8, 10, 12, 14]  # hp-adaptive refinement cycles
METHODS = ['volkov', 'freefem', 'dealii', 'dealii_mg', 'dealii_mg_assembled',
           'lightning', 'aaals']
LIGHTNING_TOLS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]
AAALS_TOLS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]

# Method-independent reference per geometry (replaces the circular FreeFEM-P2
# reference the reviewer flagged): analytical where a closed form exists,
# otherwise a solution cross-validated by a second, independent solver.
INDEP_REF_FILES = {
    'unit_square': 'analytical_unit_square_hot_bottom.npz',
    'l_shape': 'reference_l_shape_hot_bottom.npz',
    'square_with_hole': 'reference_square_with_hole_hot_bottom.npz',
}


def load_reference(geom_name):
    """Load the method-independent reference solution for a geometry."""
    ref_file = INDEP_REF_DIR / INDEP_REF_FILES[geom_name]
    if not ref_file.exists():
        raise FileNotFoundError(f"Reference not found: {ref_file}\n"
                                "Run: python benchmarks/analytical_reference.py")
    data = np.load(ref_file)
    return data['solution'], data['mask']


def compute_grid_and_mask(geom):
    """Compute grid coordinates and mask for a geometry."""
    vertices = geom['vertices']
    poly = polygon(vertices)
    if geom['hole'] is not None:
        h = geom['hole']
        poly.add_hole(h['vertices'], h['bc'], h['is_dirichlet'])

    xmin, ymin = np.min(vertices, axis=0)
    xmax, ymax = np.max(vertices, axis=0)
    nx = int(np.ceil((xmax - xmin) / DELTA))
    ny = int(np.ceil((ymax - ymin) / DELTA))
    x = np.linspace(xmin + 0.5 * DELTA, xmax + 0.5 * DELTA, nx)
    y = np.linspace(ymin + 0.5 * DELTA, ymax + 0.5 * DELTA, ny)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    mask = poly.is_inside(points).reshape(ny, nx)
    return x, y, nx, ny, mask


def compute_mse(sol, ref, mask):
    """MSE between solution and reference on valid (inside) points."""
    valid = mask
    return np.mean((sol[valid] - ref[valid]) ** 2)


# ======================================================================
# Volkov benchmark
# ======================================================================
def benchmark_volkov(geom_name, geom, ref_sol, mask):
    """Run Volkov solver at multiple n values, return list of (n, mean_time, mse, std_time)."""
    results = []
    for n_val in VOLKOV_N_VALUES:
        poly = polygon(geom['vertices'].copy())
        if geom['hole'] is not None:
            h = geom['hole']
            poly.add_hole(h['vertices'].copy(), h['bc'], h['is_dirichlet'])

        solver = volkovSolver(
            poly=poly,
            boundary_conditions=geom['bc'],
            is_dirichlet=geom['is_dirichlet'],
            delta=DELTA,
            n=n_val,
            radial_heuristic=0.85,
            overlap_heuristic=0.1,
            max_iter=10,
        )

        # Time N_TIMING_REPEATS runs, drop first (warmup), report mean + std
        times = []
        for _ in range(N_TIMING_REPEATS):
            t0 = time.perf_counter()
            solution = solver.solve(verbose=False)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        times = times[1:]  # drop warmup run
        mean_time = np.mean(times)
        std_time = np.std(times)
        sol_data = solution.data if hasattr(solution, 'data') else np.array(solution)

        mse = compute_mse(sol_data, ref_sol, mask)
        results.append((n_val, mean_time, mse, std_time))
        print(f"    Volkov n={n_val:3d}: mean={mean_time:.4f}s, std={std_time:.4f}s, MSE={mse:.2e}")

    return results


# ======================================================================
# FreeFEM benchmark
# ======================================================================
def benchmark_freefem(geom_name, geom, ref_sol, mask):
    """Run FreeFEM at multiple mesh densities, return list of (density, mean_time, mse, std_time)."""
    results = []
    _, _, nx, ny, _ = compute_grid_and_mask(geom)

    for density in FREEFEM_MESH_DENSITIES:
        script = FREEFEM_DIR / f'benchmark_{geom_name}.edp'
        if not script.exists():
            print(f"    FreeFEM: skipping {geom_name} (no script)")
            continue

        # Run FreeFEM with timing (N_TIMING_REPEATS runs)
        times = []
        sol = None
        for _ in range(N_TIMING_REPEATS):
            # Clean previous output
            for f in FREEFEM_DIR.glob(f'benchmark_{geom_name}_{density}*'):
                f.unlink()

            result = subprocess.run(
                ['FreeFem++', str(script), '-n', str(density)],
                cwd=str(FREEFEM_DIR),
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"    FreeFEM density={density}: FAILED\n{result.stderr[:200]}")
                break

            # Read timing
            timing_file = FREEFEM_DIR / f'benchmark_{geom_name}_{density}_timing.dat'
            if timing_file.exists():
                t = float(timing_file.read_text().strip())
                times.append(t)

            # Read solution (only need once)
            if sol is None:
                dat_file = FREEFEM_DIR / f'benchmark_{geom_name}_{density}.dat'
                if dat_file.exists():
                    data = np.loadtxt(dat_file)
                    if data.shape[0] == ny * nx:
                        sol = data[:, 2].reshape(ny, nx)

        if not times or sol is None:
            print(f"    FreeFEM density={density}: no results")
            continue

        if len(times) > 1:
            times = times[1:]  # drop warmup run
        mean_time = np.mean(times)
        std_time = np.std(times)
        mse = compute_mse(sol, ref_sol, mask)
        results.append((density, mean_time, mse, std_time))
        print(f"    FreeFEM d={density:4d}: mean={mean_time:.4f}s, std={std_time:.4f}s, MSE={mse:.2e}")

    return results


# ======================================================================
# deal.II hp-FEM benchmark
# ======================================================================
DEALII_PHASES = ('mesh', 'setup', 'assembly', 'solve', 'refine', 'total')


def dealii_env():
    """Environment for the deal.II binary. It initializes MPI (needed by the
    Trilinos AMG preconditioner), and on some machines hwloc's GL component
    blocks MPI startup while probing the X server, so that component is off.

    deal.II runs on one thread: every other solver in the comparison is
    single-threaded, and at these problem sizes deal.II's task parallelism is
    slower than serial execution (measured for 1, 2, 4, 8 and 16 threads)."""
    env = dict(os.environ)
    env.setdefault('HWLOC_COMPONENTS', '-gl')
    env.setdefault('DEAL_II_NUM_THREADS', '1')
    return env


def benchmark_dealii(geom_name, geom, ref_sol, mask, solver='umfpack'):
    """Run the deal.II hp-adaptive FEM baseline over a refinement sweep.

    solver is 'umfpack' (sparse direct), 'mf-pmg' (matrix-free conjugate
    gradients with a p-multigrid preconditioner and an algebraic-multigrid coarse
    solve: the O(N) solver requested in the second review round, with no matrix
    assembly), 'cg-pmg' (the same multigrid on the assembled matrix), or 'cg-amg'
    (algebraic multigrid applied directly to the hp matrix; slower on high-order
    elements). All share the discretization and adaptation strategy.

    Returns (results, breakdown). results is a list of (cycles, mean_time, mse,
    std_time); time is the full pipeline, as for every other method. breakdown
    is a list of dicts with the mean per-phase times, DOF count and (iterative solvers)
    CG iteration counts, one per sweep point.
    """
    if not DEALII_BIN.exists():
        print(f"    deal.II: skipping {geom_name} (binary not built at {DEALII_BIN})")
        return [], []
    _, _, nx, ny, _ = compute_grid_and_mask(geom)
    out_dat, out_t = '/tmp/_pareto_hp.dat', '/tmp/_pareto_hp_t.dat'
    max_deg = 5 if geom_name == 'unit_square' else 6
    results, breakdown = [], []
    for cyc in DEALII_CYCLES:
        args = [str(DEALII_BIN), '--geometry', geom_name, '--bc', 'hot_bottom',
                '--refine-mode', 'adaptive', '--min-degree', '2',
                '--max-degree', str(max_deg), '--init-refine', '2',
                '--cycles', str(cyc), '--solver', solver,
                '--out', out_dat, '--timing', out_t]
        metas, sol = [], None
        for _ in range(N_TIMING_REPEATS):
            subprocess.run(args, check=True, capture_output=True, env=dealii_env())
            metas.append(dict(l.split()[:2] for l in open(out_t) if len(l.split()) >= 2))
            if sol is None:
                sol = np.loadtxt(out_dat)[:, 2].reshape(ny, nx)
        metas = metas[1:]  # drop warmup run
        times = [float(m['total']) for m in metas]
        mean_time, std_time = np.mean(times), np.std(times)
        mse = compute_mse(sol, ref_sol, mask)
        results.append((cyc, mean_time, mse, std_time))
        phases = {p: float(np.mean([float(m[p]) for m in metas])) for p in DEALII_PHASES}
        phases.update(cycles=cyc, n_dofs=int(metas[-1]['n_dofs']))
        for key in ('cg_iters_last', 'cg_iters_total'):
            if key in metas[-1]:
                phases[key] = int(metas[-1][key])
        breakdown.append(phases)
        iters = (f", CG its (last cycle)={phases['cg_iters_last']}"
                 if 'cg_iters_last' in phases else '')
        print(f"    deal.II hp [{solver}] cyc={cyc:3d}: mean={mean_time:.4f}s, "
              f"std={std_time:.4f}s, MSE={mse:.2e}, DOFs={phases['n_dofs']}, "
              f"solve share={100 * phases['solve'] / phases['total']:.0f}%{iters}")
    return results, breakdown


# ======================================================================
# Spectral benchmarks (C++ Lightning and AAA-LS, addressing R1.4)
# ======================================================================
def _benchmark_spectral(name, binary, tols, geom_name, geom, ref_sol, mask):
    """Run a C++ spectral solver over a tolerance sweep, mirroring benchmark_dealii.

    Returns list of (tol, mean_time, mse, std_time). Time is the full solve
    (pole placement + least-squares), the like-for-like C++ replacement for the
    earlier MATLAB timings the reviewer flagged (R1.4).
    """
    if not binary.exists():
        print(f"    {name}: skipping {geom_name} (binary not built at {binary})")
        return []
    _, _, nx, ny, _ = compute_grid_and_mask(geom)
    out_dat, out_t = '/tmp/_pareto_spec.dat', '/tmp/_pareto_spec_t.dat'
    results = []
    for tol in tols:
        args = [str(binary), '--geometry', geom_name, '--bc', 'hot_bottom',
                '--tol', f'{tol:.0e}', '--out', out_dat, '--timing', out_t]
        times, sol = [], None
        for _ in range(N_TIMING_REPEATS):
            subprocess.run(args, check=True, capture_output=True)
            meta = dict(l.split()[:2] for l in open(out_t) if len(l.split()) >= 2)
            times.append(float(meta['total']))
            if sol is None:
                sol = np.loadtxt(out_dat)[:, 2].reshape(ny, nx)
        times = times[1:]  # drop warmup run
        mean_time, std_time = np.mean(times), np.std(times)
        mse = compute_mse(sol, ref_sol, mask)
        results.append((tol, mean_time, mse, std_time))
        print(f"    {name} tol={tol:.0e}: mean={mean_time:.4f}s, "
              f"std={std_time:.4f}s, MSE={mse:.2e}")
    return results


def benchmark_lightning(geom_name, geom, ref_sol, mask):
    """C++ Lightning solver (simply-connected geometries only)."""
    if not geom['lightning_ok']:
        print(f"    Lightning: skipping {geom_name} (not supported)")
        return []
    return _benchmark_spectral('Lightning', LIGHTNING_BIN, LIGHTNING_TOLS,
                               geom_name, geom, ref_sol, mask)


def benchmark_aaals(geom_name, geom, ref_sol, mask):
    """C++ AAA-LS solver (all geometries, including multiply-connected)."""
    return _benchmark_spectral('AAA-LS', AAALS_BIN, AAALS_TOLS,
                               geom_name, geom, ref_sol, mask)


# ======================================================================
# Plotting
# ======================================================================
def _unpack_results(results):
    """Unpack results that may be 3-tuples or 4-tuples (with std_time).

    Returns asymmetric error bars [lower, upper] clipped so that
    time - lower > 0 (avoids negative values on log-scale x-axis).
    """
    params, times, mses = [], [], []
    lower_err, upper_err = [], []
    for r in results:
        params.append(r[0])
        t = r[1]
        times.append(t)
        mses.append(r[2])
        s = r[3] if len(r) > 3 else 0.0
        upper_err.append(s)
        lower_err.append(min(s, t * 0.9))  # clip so bar doesn't go below 0.1*t
    return params, times, mses, [lower_err, upper_err]


def _setup_ax(ax, all_data, geom_name, refs):
    """Plot one geometry's Pareto data on the given axes with error bars."""
    import matplotlib.ticker as ticker
    REF_FREEFEM, REF_LIGHTNING, REF_AAALS = refs
    data = all_data[geom_name]

    # Volkov
    if 'volkov' in data and data['volkov']:
        params, times, mses, stds = _unpack_results(data['volkov'])
        ax.errorbar(times, mses, xerr=stds, fmt='o', color='#1565C0',
                    label='Ours', markersize=7, capsize=3, capthick=1,
                    elinewidth=1, zorder=3)
        label_indices = {0, 2, 3, len(params) - 1}
        for i, (p, t, m) in enumerate(zip(params, times, mses)):
            if i in label_indices:
                ax.annotate(f'$n$={p}', (t, m), fontsize=9,
                            textcoords='offset points', xytext=(6, 6),
                            color='#1565C0')

    # FreeFEM
    if 'freefem' in data and data['freefem']:
        params, times, mses, stds = _unpack_results(data['freefem'])
        ax.errorbar(times, mses, xerr=stds, fmt='s', color='#D84315',
                    label=f'FreeFEM P2 [{REF_FREEFEM}]', markersize=7,
                    capsize=3, capthick=1, elinewidth=1, zorder=2)
        label_indices = {0, 2, len(params) - 1}
        for i, (p, t, m) in enumerate(zip(params, times, mses)):
            if i in label_indices:
                ax.annotate(f'$d$={p}', (t, m), fontsize=9,
                            textcoords='offset points', xytext=(6, -14),
                            color='#D84315')

    # deal.II hp-FEM (the real high-order FEM baseline)
    if 'dealii' in data and data['dealii']:
        params, times, mses, stds = _unpack_results(data['dealii'])
        ax.errorbar(times, mses, xerr=stds, fmt='v', color='#00838F',
                    label='hp-FEM, direct (deal.II)', markersize=7,
                    capsize=3, capthick=1, elinewidth=1, zorder=2)
        for i, (p, t, m) in enumerate(zip(params, times, mses)):
            if i in {0, len(params) - 1}:
                ax.annotate(f'cyc={p}', (t, m), fontsize=9,
                            textcoords='offset points', xytext=(6, 6),
                            color='#00838F')

    # deal.II hp-FEM with the O(N) solver: matrix-free CG + p-multigrid
    if 'dealii_mg' in data and data['dealii_mg']:
        params, times, mses, stds = _unpack_results(data['dealii_mg'])
        ax.errorbar(times, mses, xerr=stds, fmt='P', color='#5D4037',
                    label='hp-FEM, matrix-free MG (deal.II)', markersize=7,
                    capsize=3, capthick=1, elinewidth=1, zorder=2)

    # Lightning
    if 'lightning' in data and data['lightning']:
        params, times, mses, stds = _unpack_results(data['lightning'])
        ax.errorbar(times, mses, xerr=stds, fmt='D', color='#2E7D32',
                    label=f'Lightning [{REF_LIGHTNING}]', markersize=6,
                    capsize=3, capthick=1, elinewidth=1, zorder=2)
        if len(params) >= 2:
            tmin, tmax = f'{params[0]:.0e}', f'{params[-1]:.0e}'
            mid_idx = len(params) // 2
            ax.annotate(f'tol={tmin}..{tmax}', (times[mid_idx], mses[mid_idx]),
                        fontsize=8, textcoords='offset points', xytext=(6, -12),
                        color='#2E7D32')

    # AAA-LS
    if 'aaals' in data and data['aaals']:
        params, times, mses, stds = _unpack_results(data['aaals'])
        ax.errorbar(times, mses, xerr=stds, fmt='^', color='#7B1FA2',
                    label=f'AAA-LS [{REF_AAALS}]', markersize=6,
                    capsize=3, capthick=1, elinewidth=1, zorder=2)
        if len(params) >= 2:
            tmin, tmax = f'{params[0]:.0e}', f'{params[-1]:.0e}'
            mid_idx = len(params) // 2
            ax.annotate(f'tol={tmin}..{tmax}', (times[mid_idx], mses[mid_idx]),
                        fontsize=8, textcoords='offset points', xytext=(6, 6),
                        color='#7B1FA2')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Wall-clock Time (s)')
    ax.legend(loc='lower left', framealpha=0.9, fontsize=9.5)
    # Force minor ticks on both axes for consistent grid
    ax.xaxis.set_minor_locator(ticker.LogLocator(subs='all', numticks=20))
    ax.yaxis.set_minor_locator(ticker.LogLocator(subs='all', numticks=20))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(True, which='major', alpha=0.4, linewidth=0.7)
    ax.grid(True, which='minor', alpha=0.15, linewidth=0.4)


def _get_global_ylim(all_data):
    """Compute global y-axis limits across all geometries."""
    all_mses = []
    for geom_data in all_data.values():
        for method_results in geom_data.values():
            if method_results:
                all_mses.extend([r[2] for r in method_results])
    if not all_mses:
        return (1e-14, 1)
    ymin = min(all_mses) / 5
    ymax = max(all_mses) * 5
    return (ymin, ymax)


def plot_pareto(all_data):
    """Generate publication-quality Pareto curve figures."""
    plt.rcParams.update({
        'font.size': 13,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    # Reference numbers from compiled paper (update if bibliography changes)
    refs = (13, 14, 15)  # FreeFEM, Lightning, AAA-LS

    geom_names = list(all_data.keys())
    n_geoms = len(geom_names)
    ylim = _get_global_ylim(all_data)

    # Combined figure
    fig, axes = plt.subplots(1, n_geoms, figsize=(5.5 * n_geoms, 4.5), squeeze=False)
    for gi, geom_name in enumerate(geom_names):
        ax = axes[0, gi]
        _setup_ax(ax, all_data, geom_name, refs)
        ax.set_ylim(ylim)
        if gi == 0:
            ax.set_ylabel('MSE vs Reference')

    plt.tight_layout(w_pad=2.0)
    fname = RESULTS_DIR / 'pareto_curves.pdf'
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"\nSaved {fname}")

    # Individual subplots for LaTeX subfloat
    for gi, geom_name in enumerate(geom_names):
        fig_single, ax_single = plt.subplots(1, 1, figsize=(5.5, 4.5))
        _setup_ax(ax_single, all_data, geom_name, refs)
        ax_single.set_ylim(ylim)
        ax_single.set_ylabel('MSE vs Reference')
        plt.tight_layout()
        fname_single = RESULTS_DIR / f'pareto_{geom_name}.pdf'
        fig_single.savefig(fname_single, bbox_inches='tight', dpi=300)
        plt.close(fig_single)
        print(f"Saved {fname_single}")


def plot_summary_table(all_data):
    """Print a summary table for the paper."""
    print("\n" + "=" * 100)
    print(f"{'Geometry':<20} {'Method':<25} {'Time (s)':>12} {'Std (s)':>12} {'MSE':>12} {'Param':>10}")
    print("=" * 100)

    for geom_name, data in all_data.items():
        for method_name, results in data.items():
            if not results:
                continue
            # Pick a representative point (best MSE for each method)
            best = min(results, key=lambda x: x[2])
            param, t, mse = best[0], best[1], best[2]
            std_t = best[3] if len(best) > 3 else 0.0
            label = {'volkov': 'Ours (C++)', 'freefem': 'FreeFEM P2',
                     'dealii': 'hp-FEM direct (deal.II)',
                     'dealii_mg': 'hp-FEM matrix-free MG',
                     'dealii_mg_assembled': 'hp-FEM assembled MG',
                     'lightning': 'Lightning', 'aaals': 'AAA-LS'}[method_name]
            param_str = {
                'volkov': f'n={param}',
                'freefem': f'd={param}',
                'dealii': f'cyc={param}',
                'dealii_mg': f'cyc={param}',
                'dealii_mg_assembled': f'cyc={param}',
                'lightning': f'tol={param:.0e}',
                'aaals': f'tol={param:.0e}',
            }[method_name]
            print(f"{geom_name:<20} {label:<25} {t:12.4f} {std_t:12.4f} {mse:12.2e} {param_str:>10}")

    print("=" * 100)


# ======================================================================
# Main
# ======================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    plot_only = '--plot-only' in sys.argv

    if plot_only:
        data_file = RESULTS_DIR / 'pareto_data.npz'
        if not data_file.exists():
            print("No saved data found. Run without --plot-only first.")
            return
        saved = np.load(data_file, allow_pickle=True)
        all_data = saved['all_data'].item()
        plot_pareto(all_data)
        plot_summary_table(all_data)
        return

    # --only m1,m2 reruns just those methods and merges them into the saved data.
    only = None
    if '--only' in sys.argv:
        only = set(sys.argv[sys.argv.index('--only') + 1].split(','))
        unknown = only - set(METHODS)
        if unknown:
            sys.exit(f"unknown method(s) for --only: {sorted(unknown)}")

    data_file = RESULTS_DIR / 'pareto_data.npz'
    all_data, breakdown = {}, {}
    if only is not None:
        if not data_file.exists():
            sys.exit("--only needs saved data to merge into; run the full benchmark first.")
        saved = np.load(data_file, allow_pickle=True)
        all_data = saved['all_data'].item()
        if 'breakdown' in saved.files:
            breakdown = saved['breakdown'].item()

    def wanted(method):
        return only is None or method in only

    for geom_name, geom in GEOMETRIES.items():
        print(f"\n{'=' * 60}")
        print(f"Geometry: {geom_name}")
        print(f"{'=' * 60}")

        # Load reference
        ref_sol, ref_mask = load_reference(geom_name)
        _, _, nx, ny, mask = compute_grid_and_mask(geom)

        # Use mask intersection (both reference and our mask must agree)
        valid_mask = mask & ref_mask

        geom_data = all_data.get(geom_name, {})
        geom_breakdown = breakdown.setdefault(geom_name, {})

        # --- Volkov ---
        if wanted('volkov'):
            print(f"\n  Volkov C++ solver:")
            geom_data['volkov'] = benchmark_volkov(geom_name, geom, ref_sol, valid_mask)

        # --- FreeFEM ---
        if wanted('freefem'):
            print(f"\n  FreeFEM P2:")
            geom_data['freefem'] = benchmark_freefem(geom_name, geom, ref_sol, valid_mask)

        # --- deal.II hp-FEM, sparse-direct solve ---
        if wanted('dealii'):
            print(f"\n  deal.II hp-FEM (UMFPACK sparse direct):")
            geom_data['dealii'], geom_breakdown['dealii'] = benchmark_dealii(
                geom_name, geom, ref_sol, valid_mask, solver='umfpack')

        # --- deal.II hp-FEM, matrix-free CG + p-multigrid ---
        if wanted('dealii_mg'):
            print(f"\n  deal.II hp-FEM (matrix-free CG + p-multigrid):")
            geom_data['dealii_mg'], geom_breakdown['dealii_mg'] = benchmark_dealii(
                geom_name, geom, ref_sol, valid_mask, solver='mf-pmg')

        # --- same multigrid on the assembled matrix (recorded, not plotted) ---
        if wanted('dealii_mg_assembled'):
            print(f"\n  deal.II hp-FEM (assembled CG + p-multigrid):")
            (geom_data['dealii_mg_assembled'],
             geom_breakdown['dealii_mg_assembled']) = benchmark_dealii(
                geom_name, geom, ref_sol, valid_mask, solver='cg-pmg')

        # --- Lightning (C++) ---
        if wanted('lightning'):
            print(f"\n  Lightning solver (C++):")
            geom_data['lightning'] = benchmark_lightning(geom_name, geom, ref_sol, valid_mask)

        # --- AAA-LS (C++) ---
        if wanted('aaals'):
            print(f"\n  AAA-LS solver (C++):")
            geom_data['aaals'] = benchmark_aaals(geom_name, geom, ref_sol, valid_mask)

        # Keep a fixed method order so legends and tables are stable.
        all_data[geom_name] = {m: geom_data[m] for m in METHODS if m in geom_data}

    # Save all data
    np.savez(data_file, all_data=all_data, breakdown=breakdown)
    print(f"\nData saved to {data_file}")

    # Generate figures
    plot_pareto(all_data)
    plot_summary_table(all_data)


if __name__ == '__main__':
    main()
