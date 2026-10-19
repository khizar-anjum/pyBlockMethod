#!/usr/bin/env python3
"""
Analytical (closed-form) reference solutions for benchmark geometries.

Reviewer 1 correctly notes the benchmark comparison is not rigorous. A hidden
part of that problem is that the *ground truth* used to grade every method was
itself a FreeFEM P2 solution at mesh density 200 (see
tests/freefem/generate_freefem_reference.py, which stamps element_type='P2').
Grading FEM against a fine copy of FEM is circular and flatters FEM.

Where a closed form exists we replace that reference with the exact solution.
For the unit square with piecewise-constant Dirichlet data (one hot edge, the
other three at zero) the Laplace solution is a classical Fourier series:

    hot_bottom (u=1 on y=0, u=0 elsewhere):
        u(x,y) = sum_{k odd} (4/(k pi)) sin(k pi x) sinh(k pi (1-y)) / sinh(k pi)

    hot_left (u=1 on x=0, u=0 elsewhere) is the same field rotated:
        u(x,y) = sum_{k odd} (4/(k pi)) sin(k pi y) sinh(k pi (1-x)) / sinh(k pi)

The sinh ratio is evaluated in a numerically stable exp form so it neither
overflows nor loses terms near the hot edge (where convergence is slowest).

Usage:
    python benchmarks/analytical_reference.py            # regenerate + compare to P2 ref
"""

import numpy as np
from pathlib import Path
from matplotlib.path import Path as MplPath

REPO_ROOT = Path(__file__).parent.parent

DELTA = 0.05
OUTPUT_DIR = REPO_ROOT / 'benchmarks' / 'analytical_reference_data'
P2_REF_DIR = REPO_ROOT / 'tests' / 'freefem_reference_data'

# Unit-square vertices, shared by all unit-square BC cases.
UNIT_SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)


def _sinh_ratio(k, a):
    """sinh(k*pi*a) / sinh(k*pi), stable for large k (a in [0,1])."""
    kpi = k * np.pi
    # sinh(kpi*a)/sinh(kpi) = exp(-kpi*(1-a)) * (1 - exp(-2*kpi*a)) / (1 - exp(-2*kpi))
    return np.exp(-kpi * (1.0 - a)) * (1.0 - np.exp(-2.0 * kpi * a)) / (1.0 - np.exp(-2.0 * kpi))


def hot_edge_solution(x, y, edge, n_terms=2000):
    """Exact Laplace solution on the unit square with u=1 on one geometric edge.

    edge in {'bottom','top','left','right'} names the actual edge carrying u=1;
    all other edges carry u=0. Built from the separable series
        sum_{k odd} (4/(k pi)) sin(k pi s) sinh(k pi a)/sinh(k pi),
    where (s, a) are chosen so the hot edge is at a=1 and the opposite at a=0.
    """
    coord = {
        'bottom': (x, 1.0 - y),   # u=1 at y=0
        'top':    (x, y),          # u=1 at y=1
        'left':   (y, 1.0 - x),    # u=1 at x=0
        'right':  (y, x),          # u=1 at x=1
    }
    if edge not in coord:
        raise ValueError(f"unsupported hot edge: {edge}")
    s, a = coord[edge]
    u = np.zeros_like(x, dtype=float)
    for k in range(1, 2 * n_terms, 2):  # odd k only
        u += (4.0 / (k * np.pi)) * np.sin(k * np.pi * s) * _sinh_ratio(k, a)
    return u


# Map benchmark BC-case names to the physical superposition of hot edges.
UNIT_SQUARE_BCS = {
    'hot_bottom': [('bottom', 1.0)],                 # on(1)=1
    'hot_right':  [('right', 1.0)],                  # on(2)=1
    'mixed':      [('bottom', 1.0), ('right', 0.5)], # on(1)=1, on(2)=0.5
}


def unit_square_solution(X, Y, bc_name):
    """Exact solution for a named unit-square BC case, by linear superposition."""
    u = np.zeros_like(X, dtype=float)
    for edge, weight in UNIT_SQUARE_BCS[bc_name]:
        u += weight * hot_edge_solution(X, Y, edge)
    return u


def standard_grid(vertices):
    """Reproduce the exact evaluation grid + mask used by the benchmarks.

    Mask uses matplotlib's point-in-polygon rather than polygon.is_inside: the
    C++ is_inside is currently unreliable for multi-point batches on the full
    evaluation grid (returns all-False deterministically) -- see the reviewer
    tracker (relates to R3.14). matplotlib agrees with the correct P2-reference
    mask (361 interior points for the unit square).
    """
    xmin, ymin = np.min(vertices, axis=0)
    xmax, ymax = np.max(vertices, axis=0)
    nx = int(np.ceil((xmax - xmin) / DELTA))
    ny = int(np.ceil((ymax - ymin) / DELTA))
    x = np.linspace(xmin + 0.5 * DELTA, xmax + 0.5 * DELTA, nx)
    y = np.linspace(ymin + 0.5 * DELTA, ymax + 0.5 * DELTA, ny)
    X, Y = np.meshgrid(x, y)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    mask = MplPath(vertices).contains_points(pts).reshape(ny, nx)
    return X, Y, x, y, mask


def generate_unit_square():
    OUTPUT_DIR.mkdir(exist_ok=True)
    X, Y, x, y, mask = standard_grid(UNIT_SQUARE)

    for bc_name in UNIT_SQUARE_BCS:
        # Evaluate only on interior points: outside points have y>1 (or x>1),
        # which would overflow the sinh ratio and are masked out anyway.
        sol = np.zeros_like(X)
        sol[mask] = unit_square_solution(X[mask], Y[mask], bc_name)
        out = OUTPUT_DIR / f'analytical_unit_square_{bc_name}.npz'
        np.savez_compressed(out, solution=sol, mask=mask, x_coords=x, y_coords=y,
                            delta=DELTA, geometry_name='unit_square', bc_name=bc_name,
                            element_type='analytical')
        print(f"  saved {out.name}")

        # Quantify how far the old P2 reference was from exact truth.
        p2_file = P2_REF_DIR / f'freefem_unit_square_{bc_name}.npz'
        if p2_file.exists():
            p2 = np.load(p2_file)['solution']
            m = mask
            rmse = np.sqrt(np.mean((p2[m] - sol[m]) ** 2))
            maxerr = np.max(np.abs(p2[m] - sol[m]))
            print(f"    OLD P2 reference error vs exact: RMSE={rmse:.3e}, max={maxerr:.3e}")


if __name__ == '__main__':
    print("Generating analytical references...")
    generate_unit_square()
