#!/usr/bin/env python3
"""
Benchmark: Crowd dynamics case study — MSE vs FreeFEM reference.

Runs the Volkov solver on the crowd dynamics geometry (unit square with 3
horizontal walls creating 2 narrow passages) at various (n, nu_max) settings,
and computes MSE against a converged FreeFEM P2 reference solution.

Produces:
  - benchmarks/results/crowd_mse_vs_freefem.npz   (raw data)
  - <paper_figs>/mse_vs_iterations.pdf             (replacement figure)

Usage:
    python benchmarks/benchmark_crowd_dynamics.py
    python benchmarks/benchmark_crowd_dynamics.py --plot-only
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- paths ---
BENCHMARKS_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARKS_DIR / 'results'
FREEFEM_DIR = BENCHMARKS_DIR / 'freefem'
REPO_ROOT = BENCHMARKS_DIR.parent
PAPER_FIGS = Path('/home/khizar/Documents/PerCom_Jext_OL/figs/toolpaper')

sys.path.insert(0, str(REPO_ROOT / 'src'))
from pyBlockGrid import polygon, volkovSolver

# --- Geometry definition ---
# 16-vertex non-convex polygon: unit square with 3 horizontal walls
# creating 2 narrow passages (exits) on the right side.
# From FreeFEM script: desvel.2narrowpass_obst.edp
VERTICES = np.array([
    [0,    0],       # v0
    [1,    0],       # v1
    [1,    0.3],     # v2
    [0.35, 0.3],     # v3
    [0.35, 0.35],    # v4
    [1,    0.35],    # v5
    [1,    0.415],   # v6
    [0.35, 0.415],   # v7
    [0.35, 0.435],   # v8
    [1,    0.435],   # v9
    [1,    0.5],     # v10
    [0.35, 0.5],     # v11
    [0.35, 0.55],    # v12
    [1,    0.55],    # v13
    [1,    1],       # v14
    [0,    1],       # v15
], dtype=float)

# All-Dirichlet boundary conditions (matching FreeFEM all-Dirichlet reference):
# u=0 on all edges except the two passage exits (edges 5, 9) where u=1.
BOUNDARY_CONDITIONS = [
    [0.0],  # Edge 0  (label 1):  bottom
    [0.0],  # Edge 1  (label 2):  right lower
    [0.0],  # Edge 2  (label 3):  wall 1 bottom
    [0.0],  # Edge 3  (label 4):  wall 1 left
    [0.0],  # Edge 4  (label 5):  wall 1 top
    [1.0],  # Edge 5  (label 6):  passage 1 EXIT
    [0.0],  # Edge 6  (label 7):  wall 2 bottom
    [0.0],  # Edge 7  (label 8):  wall 2 left
    [0.0],  # Edge 8  (label 9):  wall 2 top
    [1.0],  # Edge 9  (label 10): passage 2 EXIT
    [0.0],  # Edge 10 (label 11): wall 3 bottom
    [0.0],  # Edge 11 (label 12): wall 3 left
    [0.0],  # Edge 12 (label 13): wall 3 top
    [0.0],  # Edge 13 (label 14): right upper
    [0.0],  # Edge 14 (label 15): top
    [0.0],  # Edge 15 (label 16): left side
]

IS_DIRICHLET = [True] * 16

# --- Parameter sweeps ---
N_VALUES = [5, 10, 15, 20, 25, 30, 35, 40]
NU_MAX_VALUES = [5, 10, 15, 20]
DELTA = 0.01  # Grid spacing
RHO = 0.85
OMEGA = 0.1


def load_freefem_reference(ref_path=None):
    """Load the FreeFEM reference solution from the .dat file."""
    if ref_path is None:
        ref_path = FREEFEM_DIR / 'crowd_dynamics_alldirichlet.dat'

    data = np.loadtxt(ref_path, comments='#')
    x_ref = data[:, 0]
    y_ref = data[:, 1]
    u_ref = data[:, 2]

    delta_ref = x_ref[1] - x_ref[0]
    Nx = int(round(1.0 / delta_ref)) + 1
    Ny = Nx

    X_ref = x_ref.reshape(Ny, Nx)
    Y_ref = y_ref.reshape(Ny, Nx)
    U_ref = u_ref.reshape(Ny, Nx)

    wall_mask = U_ref < -900
    U_ref_masked = np.ma.array(U_ref, mask=wall_mask)

    return X_ref, Y_ref, U_ref_masked, delta_ref


def run_volkov(n, nu_max, delta=DELTA, verbose=False):
    """Run the Volkov solver for the crowd dynamics geometry."""
    poly = polygon(VERTICES)
    solver = volkovSolver(
        poly=poly,
        boundary_conditions=BOUNDARY_CONDITIONS,
        is_dirichlet=IS_DIRICHLET,
        delta=delta,
        n=n,
        max_iter=nu_max,
        radial_heuristic=RHO,
        overlap_heuristic=OMEGA,
    )
    solution = solver.solve(verbose=verbose)
    X = solver._X
    Y = solver._Y
    return X, Y, solution


def compute_mse(X_vol, Y_vol, U_vol, X_ref, Y_ref, U_ref, delta_ref):
    """Compute MSE between Volkov solution and FreeFEM reference."""
    x0_ref = X_ref[0, 0]
    y0_ref = Y_ref[0, 0]

    ny_vol, nx_vol = U_vol.shape
    mse_sum = 0.0
    count = 0

    for j in range(ny_vol):
        for i in range(nx_vol):
            if U_vol.mask[j, i]:
                continue

            xv = X_vol[j, i]
            yv = Y_vol[j, i]
            ri = int(round((xv - x0_ref) / delta_ref))
            rj = int(round((yv - y0_ref) / delta_ref))

            if ri < 0 or ri >= X_ref.shape[1] or rj < 0 or rj >= X_ref.shape[0]:
                continue
            if U_ref.mask[rj, ri]:
                continue

            diff = float(U_vol[j, i]) - float(U_ref[rj, ri])
            mse_sum += diff * diff
            count += 1

    if count == 0:
        return float('nan')
    return mse_sum / count


def run_benchmarks():
    """Run all (n, nu_max) combinations and compute MSE."""
    print("Loading FreeFEM reference solution...")
    X_ref, Y_ref, U_ref, delta_ref = load_freefem_reference()
    print(f"  Reference grid: {U_ref.shape}, delta={delta_ref}")
    print(f"  Valid reference points: {np.sum(~U_ref.mask)}")

    results = {}

    for n in N_VALUES:
        for nu_max in NU_MAX_VALUES:
            key = (n, nu_max)
            print(f"\nRunning Volkov: n={n}, nu_max={nu_max}, delta={DELTA}")
            try:
                t0 = time.perf_counter()
                X_vol, Y_vol, U_vol = run_volkov(n, nu_max, DELTA)
                elapsed = time.perf_counter() - t0
                print(f"  Solve time: {elapsed:.2f}s")
                print(f"  Solution range: [{U_vol.min():.6f}, {U_vol.max():.6f}]")
                print(f"  Valid points: {np.sum(~U_vol.mask)}")

                mse = compute_mse(X_vol, Y_vol, U_vol, X_ref, Y_ref, U_ref, delta_ref)
                results[key] = mse
                print(f"  MSE vs FreeFEM: {mse:.6e} (log10={np.log10(mse):.2f})")
            except Exception as e:
                print(f"  FAILED: {e}")
                results[key] = float('nan')

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        RESULTS_DIR / 'crowd_mse_vs_freefem.npz',
        n_values=np.array(N_VALUES),
        nu_max_values=np.array(NU_MAX_VALUES),
        mse_matrix=np.array([[results.get((n, nu), float('nan'))
                              for nu in NU_MAX_VALUES] for n in N_VALUES]),
    )
    print(f"\nResults saved to {RESULTS_DIR / 'crowd_mse_vs_freefem.npz'}")
    return results


def plot_results(results=None):
    """Generate MSE vs n figure (replacement for mse_vs_iterations.pdf).

    Plot shows MSE vs n_mu for multiple nu_max values, demonstrating
    that the solver converges quickly in n and stabilizes by nu_max=10.
    """
    if results is None:
        data = np.load(RESULTS_DIR / 'crowd_mse_vs_freefem.npz')
        n_values = data['n_values']
        nu_max_values = data['nu_max_values']
        mse_matrix = data['mse_matrix']
    else:
        n_values = np.array(N_VALUES)
        nu_max_values = np.array(NU_MAX_VALUES)
        mse_matrix = np.array([[results.get((n, nu), float('nan'))
                                for nu in nu_max_values] for n in n_values])

    # Plot MSE vs n for each nu_max
    markers = ['o', 's', 'D', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, ax = plt.subplots(figsize=(7, 5))

    for nu_idx, nu_max in enumerate(nu_max_values):
        mse_col = mse_matrix[:, nu_idx]
        valid = ~np.isnan(mse_col) & (mse_col > 0)
        if not np.any(valid):
            continue
        log_mse = np.log10(mse_col[valid])
        ax.plot(
            n_values[valid], log_mse,
            marker=markers[nu_idx],
            color=colors[nu_idx],
            label=fr'$\nu_{{max}}={nu_max}$',
            linewidth=1.5,
            markersize=7,
        )

    ax.set_xlabel(r'$n_\mu$ (Angular Discretization)', fontsize=16)
    ax.set_ylabel(r'$\log_{10}$ (MSE vs FreeFEM)', fontsize=16)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)

    fig.tight_layout()

    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    out_path = PAPER_FIGS / 'mse_vs_iterations.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Figure saved to {out_path}")

    fig.savefig(RESULTS_DIR / 'crowd_mse_vs_freefem.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    if '--plot-only' in sys.argv:
        plot_results()
    else:
        results = run_benchmarks()
        plot_results(results)
