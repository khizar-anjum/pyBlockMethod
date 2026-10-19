#!/usr/bin/env python3
"""Generate Volkov vs FreeFEM comparison figures for the README."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from pyBlockGrid import polygon, volkovSolver

ROOT = Path(__file__).resolve().parent.parent
FREEFEM_REF_DIR = ROOT / 'tests' / 'freefem_reference_data'
FREEFEM_DIR = ROOT / 'tests' / 'freefem'
OUT_DIR = ROOT / 'docs' / 'images'
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEOMETRIES = {
    'unit_square': {
        'vertices': np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
        'label': 'Unit Square',
        'nx': 20, 'ny': 20,
    },
    'l_shape': {
        'vertices': np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float),
        'label': 'L-Shape',
        'nx': 40, 'ny': 40,
    },
    'square_with_hole': {
        'vertices': np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
        'hole': np.array([[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]])[::-1],
        'hole_bc': [[0.0], [0.0], [0.0], [0.0]],
        'hole_is_dirichlet': [True, True, True, True],
        'label': 'Square with Hole',
        'nx': 20, 'ny': 20,
    },
}

ALL_BCS = {
    'unit_square': {
        'hot_bottom': ([[1.0], [0.0], [0.0], [0.0]], [True] * 4),
        'hot_right':  ([[0.0], [1.0], [0.0], [0.0]], [True] * 4),
        'mixed':      ([[1.0], [0.5], [0.0], [0.0]], [True] * 4),
    },
    'l_shape': {
        'hot_bottom': ([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [True] * 6),
        'hot_left':   ([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]], [True] * 6),
        'mixed':      ([[1.0], [0.0], [0.0], [0.5], [0.0], [0.0]], [True] * 6),
    },
    'square_with_hole': {
        'hot_bottom': ([[1.0], [0.0], [0.0], [0.0]], [True] * 4),
        'hot_right':  ([[0.0], [1.0], [0.0], [0.0]], [True] * 4),
        'mixed':      ([[1.0], [0.5], [0.0], [0.0]], [True] * 4),
    },
}


def build_polygon(geom_name):
    geom = GEOMETRIES[geom_name]
    poly = polygon(geom['vertices'])
    if 'hole' in geom:
        poly.add_hole(geom['hole'], geom['hole_bc'], geom['hole_is_dirichlet'])
    return poly


def plot_masked(ax, X, Y, data, mask, title, vmin=None, vmax=None, cmap='viridis'):
    """Plot a masked solution on the given axes."""
    masked = np.ma.array(data, mask=~mask)
    im = ax.pcolormesh(X, Y, masked, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)
    return im


def generate_apples_to_apples():
    """Generate the main comparison: Volkov vs FreeFEM at same grid density.

    3 rows (geometries) x 4 cols (Volkov | FreeFEM n=20 | Volkov error | FreeFEM n=20 error).
    Both errors are measured against the converged FreeFEM n=200 reference.
    """
    geom_names = ['unit_square', 'l_shape', 'square_with_hole']
    bc_name = 'hot_bottom'

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))

    for row, geom_name in enumerate(geom_names):
        geom = GEOMETRIES[geom_name]
        label = geom['label']
        nx, ny = geom['nx'], geom['ny']

        # Load FreeFEM n=200 reference (ground truth)
        ref = np.load(FREEFEM_REF_DIR / f'freefem_{geom_name}_{bc_name}.npz')
        ref_solution = ref['solution']
        ref_mask = ref['mask']

        # Load FreeFEM n=20 solution
        dat = np.loadtxt(FREEFEM_DIR / f'{geom_name}_n20_{bc_name}.dat')
        n20_solution = dat[:, 2].reshape(ny, nx)

        # Run Volkov solver
        poly = build_polygon(geom_name)
        bc_values, is_dirichlet = ALL_BCS[geom_name][bc_name]
        solver = volkovSolver(
            poly=poly, boundary_conditions=bc_values, is_dirichlet=is_dirichlet,
            delta=0.05, n=20, radial_heuristic=0.85, overlap_heuristic=0.1, max_iter=10,
        )
        volkov_data = solver.solve(verbose=False).data
        X, Y = solver.state.X, solver.state.Y

        # Common colorscale for solutions
        valid_vals = np.concatenate([volkov_data[ref_mask], n20_solution[ref_mask]])
        vmin, vmax = valid_vals.min(), valid_vals.max()

        # Volkov solution
        plot_masked(axes[row, 0], X, Y, volkov_data, ref_mask,
                    f'{label} — Volkov (n=20)', vmin=vmin, vmax=vmax)

        # FreeFEM n=20 solution
        plot_masked(axes[row, 1], X, Y, n20_solution, ref_mask,
                    f'{label} — FreeFEM P2 (n=20)', vmin=vmin, vmax=vmax)

        # Errors vs ground truth (n=200), shared error colorscale
        volkov_err = np.abs(volkov_data - ref_solution)
        fem_err = np.abs(n20_solution - ref_solution)
        err_max = max(volkov_err[ref_mask].max(), fem_err[ref_mask].max())

        mse_v = np.mean((volkov_data[ref_mask] - ref_solution[ref_mask]) ** 2)
        mse_f = np.mean((n20_solution[ref_mask] - ref_solution[ref_mask]) ** 2)

        plot_masked(axes[row, 2], X, Y, volkov_err, ref_mask,
                    f'Volkov |error| (MSE={mse_v:.1e})', cmap='hot', vmin=0, vmax=err_max)
        plot_masked(axes[row, 3], X, Y, fem_err, ref_mask,
                    f'FreeFEM |error| (MSE={mse_f:.1e})', cmap='hot', vmin=0, vmax=err_max)

    fig.suptitle('Same grid density (20 pts/unit): Volkov vs FreeFEM P2 — errors vs converged reference',
                 fontsize=13, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / 'volkov_vs_freefem_same_grid.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def generate_all_bc_comparison():
    """Generate per-geometry figures showing all 3 BCs: rows=BCs, cols=Volkov/FreeFEM/Error."""
    for geom_name, bcs in ALL_BCS.items():
        geom = GEOMETRIES[geom_name]
        label = geom['label']
        bc_names = list(bcs.keys())

        fig, axes = plt.subplots(3, 3, figsize=(14, 12))

        for row, bc_name in enumerate(bc_names):
            ref = np.load(FREEFEM_REF_DIR / f'freefem_{geom_name}_{bc_name}.npz')
            ref_solution = ref['solution']
            ref_mask = ref['mask']

            poly = build_polygon(geom_name)
            bc_values, is_dirichlet = bcs[bc_name]
            solver = volkovSolver(
                poly=poly, boundary_conditions=bc_values, is_dirichlet=is_dirichlet,
                delta=0.05, n=20, radial_heuristic=0.85, overlap_heuristic=0.1, max_iter=10,
            )
            volkov_data = solver.solve(verbose=False).data

            valid_vals = np.concatenate([volkov_data[ref_mask], ref_solution[ref_mask]])
            vmin, vmax = valid_vals.min(), valid_vals.max()
            X, Y = solver.state.X, solver.state.Y

            bc_label = bc_name.replace('_', ' ').title()
            plot_masked(axes[row, 0], X, Y, volkov_data, ref_mask,
                        f'{bc_label} — Volkov (n=20)', vmin=vmin, vmax=vmax)
            plot_masked(axes[row, 1], X, Y, ref_solution, ref_mask,
                        f'{bc_label} — FreeFEM (P2, n=200)', vmin=vmin, vmax=vmax)

            error = np.abs(volkov_data - ref_solution)
            mse = np.mean((volkov_data[ref_mask] - ref_solution[ref_mask]) ** 2)
            plot_masked(axes[row, 2], X, Y, error, ref_mask,
                        f'|Error|  (MSE={mse:.2e})', cmap='hot')

        fig.suptitle(f'{label} — Volkov vs FreeFEM across BCs (n=20, rad=0.85, delta=0.05)',
                     fontsize=13, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = OUT_DIR / f'comparison_{geom_name}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out_path}')


if __name__ == '__main__':
    generate_apples_to_apples()
    generate_all_bc_comparison()
