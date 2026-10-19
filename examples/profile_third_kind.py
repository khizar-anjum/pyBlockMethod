#!/usr/bin/env python3
"""Profile block covering steps to measure third-kind block creation cost."""

import time
import numpy as np
# Force Python backend to access internal methods
from pyBlockGrid.core.polygon import polygon
from pyBlockGrid.solvers.volkov import volkovSolver
from pyBlockGrid.utils.geometry import generate_random_polygon


def make_bcs(poly):
    nv = len(poly.vertices)
    boundary_conditions = [[1.0]] + [[0.0]] * (nv - 1)
    is_dirichlet = [True] * nv
    return boundary_conditions, is_dirichlet


def profile_one(poly, label, delta, n, radial_h, overlap_h, max_iter):
    bcs, is_dir = make_bcs(poly)

    solver = volkovSolver(
        poly=poly,
        boundary_conditions=bcs,
        is_dirichlet=is_dir,
        delta=delta,
        n=n,
        radial_heuristic=radial_h,
        overlap_heuristic=overlap_h,
        max_iter=max_iter,
    )

    # --- Step 1a: First kind blocks ---
    t0 = time.perf_counter()
    vertex_blocks = solver.block_covering._create_first_kind_blocks(solver.poly)
    t_first = time.perf_counter() - t0

    # --- Step 1b: Second kind blocks ---
    t0 = time.perf_counter()
    edge_blocks = solver.block_covering._create_second_kind_blocks(
        solver.poly, vertex_blocks, len(vertex_blocks)
    )
    t_second = time.perf_counter() - t0

    solver.blocks = vertex_blocks + edge_blocks
    solver.N = len(vertex_blocks)
    solver.L = solver.N + len(edge_blocks)
    solver.M = solver.L

    # --- Step 1c: Find uncovered points ---
    t0 = time.perf_counter()
    uncovered = solver.find_uncovered_points()
    t_uncovered = time.perf_counter() - t0

    n_uncovered = int((~uncovered.mask).sum())

    # --- Step 1d: Third kind blocks ---
    t0 = time.perf_counter()
    solver.create_third_kind_blocks(uncovered)
    t_third = time.perf_counter() - t0

    n_third = solver.M - solver.L

    # --- Step 2: Initialize solution ---
    t0 = time.perf_counter()
    solver.initialize_solution()
    t_init = time.perf_counter() - t0

    # --- Step 3: Boundary estimation ---
    t0 = time.perf_counter()
    solver.boundary_estimator.estimate(solver.state, solver.max_iter)
    t_boundary = time.perf_counter() - t0

    # --- Step 4: Interior solve ---
    t0 = time.perf_counter()
    solver.interior_solver.solve(solver.state, solver.tol)
    t_interior = time.perf_counter() - t0

    t_covering = t_first + t_second + t_uncovered + t_third
    t_total = t_covering + t_init + t_boundary + t_interior

    return {
        "label": label,
        "delta": delta,
        "N": solver.N,
        "L": solver.L,
        "M": solver.M,
        "n_uncovered": n_uncovered,
        "n_third_kind": n_third,
        "t_first": t_first,
        "t_second": t_second,
        "t_uncovered": t_uncovered,
        "t_third": t_third,
        "t_covering": t_covering,
        "t_init": t_init,
        "t_boundary": t_boundary,
        "t_interior": t_interior,
        "t_total": t_total,
    }


def main():
    np.random.seed(42)

    # Generate random polygons with many vertices — these create large domains
    # with complex shapes that need third-kind blocks
    geometries = []
    for nv in [8, 12, 16, 20]:
        verts = generate_random_polygon(nv, min_radius=1.0, max_radius=3.0)
        poly = polygon(verts)
        geometries.append((poly, f"rand_{nv}v"))

    configs = [
        # delta, n, radial_h, overlap_h, max_iter
        (0.05, 20, 0.85, 0.1, 10),
        (0.02, 20, 0.85, 0.1, 10),
    ]

    print(f"{'Label':<14} {'delta':>6} {'N':>4} {'L':>4} {'M':>4} "
          f"{'3rd#':>5} {'Uncov':>6} "
          f"{'1stK(ms)':>9} {'2ndK(ms)':>9} {'FindU(ms)':>10} {'3rdK(ms)':>9} "
          f"{'Cover(ms)':>10} {'Init(ms)':>9} {'Bndry(ms)':>10} {'Intri(ms)':>10} "
          f"{'Total(ms)':>10} {'3rd/Cov%':>9} {'3rd/Tot%':>9} {'Cov/Tot%':>9}")
    print("-" * 205)

    for poly, label in geometries:
        for delta, n, rho, omega, max_iter in configs:
            try:
                r = profile_one(poly, label, delta, n, rho, omega, max_iter)
            except Exception as e:
                print(f"{label:<14} {delta:>6.3f}  ERROR: {e}")
                continue
            pct_third_cov = 100 * r["t_third"] / r["t_covering"] if r["t_covering"] > 0 else 0
            pct_third_tot = 100 * r["t_third"] / r["t_total"] if r["t_total"] > 0 else 0
            pct_cov_tot = 100 * r["t_covering"] / r["t_total"] if r["t_total"] > 0 else 0

            print(
                f"{r['label']:<14} {r['delta']:>6.3f} "
                f"{r['N']:>4} {r['L']:>4} {r['M']:>4} "
                f"{r['n_third_kind']:>5} {r['n_uncovered']:>6} "
                f"{r['t_first']*1000:>9.2f} {r['t_second']*1000:>9.2f} "
                f"{r['t_uncovered']*1000:>10.2f} {r['t_third']*1000:>9.2f} "
                f"{r['t_covering']*1000:>10.2f} {r['t_init']*1000:>9.2f} "
                f"{r['t_boundary']*1000:>10.2f} {r['t_interior']*1000:>10.2f} "
                f"{r['t_total']*1000:>10.2f} "
                f"{pct_third_cov:>8.1f}% {pct_third_tot:>8.1f}% {pct_cov_tot:>8.1f}%"
            )
        print()


if __name__ == "__main__":
    main()
