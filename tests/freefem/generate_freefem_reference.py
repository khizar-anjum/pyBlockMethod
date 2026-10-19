#!/usr/bin/env python3
"""
Generate FreeFEM reference data for benchmark testing.

Runs FreeFEM++ on .edp scripts, reads output .dat files,
and packages results as .npz files for use by test_freefem_benchmark.py.
"""

import subprocess
import numpy as np
from pathlib import Path
from pyBlockGrid import polygon

FREEFEM_DIR = Path(__file__).parent
OUTPUT_DIR = FREEFEM_DIR.parent / 'freefem_reference_data'
DELTA = 0.05

GEOMETRIES = {
    'unit_square': {
        'vertices': np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
    },
    'l_shape': {
        'vertices': np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float),
    },
    'square_with_hole': {
        'vertices': np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
        'hole': np.array([[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]])[::-1],
    },
}

# Per-geometry BC case names: unit_square/square_with_hole use the right edge
# (hot_right), l_shape uses the left edge (hot_left).
BC_NAMES = {
    'unit_square': ['hot_bottom', 'hot_right', 'mixed'],
    'l_shape': ['hot_bottom', 'hot_left', 'mixed'],
    'square_with_hole': ['hot_bottom', 'hot_right', 'mixed'],
}


def run_freefem(script_name):
    """Run a FreeFEM++ script."""
    result = subprocess.run(
        ['FreeFem++', script_name],
        cwd=str(FREEFEM_DIR),
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"FreeFem++ failed on {script_name}:\n{result.stderr}"
        )
    print(f"  FreeFem++ completed: {script_name}")


def compute_grid_and_mask(vertices, hole_vertices=None):
    """Compute the Volkov-identical grid and mask for a geometry."""
    poly = polygon(vertices)
    if hole_vertices is not None:
        poly.add_hole(hole_vertices, [[0.0]] * 4, [True] * 4)

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


def read_dat_file(filepath, nx, ny):
    """Read a FreeFEM .dat output file into a (ny, nx) array."""
    data = np.loadtxt(filepath)
    assert data.shape == (ny * nx, 3), \
        f"Expected {ny*nx} rows, got {data.shape[0]} in {filepath}"
    return data[:, 2].reshape(ny, nx)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for geom_name, geom_info in GEOMETRIES.items():
        print(f"\nProcessing {geom_name}...")
        vertices = geom_info['vertices']
        hole = geom_info.get('hole')

        # Run FreeFEM script
        run_freefem(f"{geom_name}.edp")

        # Compute grid and mask (identical to Volkov)
        x_coords, y_coords, nx, ny, mask = compute_grid_and_mask(vertices, hole)

        # Read each BC output and save as .npz
        for bc_name in BC_NAMES[geom_name]:
            dat_file = FREEFEM_DIR / f"{geom_name}_{bc_name}.dat"
            solution = read_dat_file(dat_file, nx, ny)

            output_file = OUTPUT_DIR / f"freefem_{geom_name}_{bc_name}.npz"
            np.savez_compressed(
                output_file,
                solution=solution,
                mask=mask,
                x_coords=x_coords,
                y_coords=y_coords,
                delta=DELTA,
                geometry_name=geom_name,
                bc_name=bc_name,
                mesh_density=200,
                element_type='P2',
            )
            print(f"  Saved: {output_file.name}")

    print(f"\nAll reference data saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
