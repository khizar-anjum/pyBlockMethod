#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate reference solution data for accuracy testing.

This script generates golden baseline solutions using the current implementation
for a comprehensive set of test cases covering different geometries, parameters,
and boundary conditions.
"""

import numpy as np
import os
from pathlib import Path
import itertools
from pyBlockGrid import polygon, volkovSolver

def get_test_geometries():
    """Define the test geometries."""
    geometries = {
        'unit_square': np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
        'l_shape': np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]])
    }
    return geometries

def get_test_parameters():
    """Define parameter combinations to test."""
    params = {
        'n': [10, 20],
        'radial_heuristic': [0.75, 0.85, 0.95],
        'overlap_heuristic': [0.1, 0.2, 0.4],
        'delta': 0.05,
        'max_iter': 10
    }
    return params

def get_boundary_conditions():
    """Define boundary condition test cases."""
    # For unit square (4 edges)
    square_bcs = {
        'hot_bottom': ([[1.0], [0.0], [0.0], [0.0]], [True, True, True, True]),
        'hot_left': ([[0.0], [1.0], [0.0], [0.0]], [True, True, True, True]),
        'mixed': ([[1.0], [0.5], [0.0], [0.0]], [True, True, True, True])
    }

    # For L-shape (6 edges)
    l_shape_bcs = {
        'hot_bottom': ([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [True, True, True, True, True, True]),
        'hot_left': ([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]], [True, True, True, True, True, True]),
        'mixed': ([[1.0], [0.0], [0.0], [0.5], [0.0], [0.0]], [True, True, True, True, True, True])
    }

    return {
        'unit_square': square_bcs,
        'l_shape': l_shape_bcs
    }

def generate_filename(geometry_name, n, radial_h, overlap_h, bc_name):
    """Generate standardized filename for reference data."""
    return f"{geometry_name}_n{n}_rad{radial_h:.2f}_ovl{overlap_h:.1f}_{bc_name}.npz"

def generate_reference_solution(geometry_name, vertices, n, radial_h, overlap_h,
                              boundary_conditions, is_dirichlet, bc_name):
    """Generate a single reference solution."""
    print(f"Generating: {geometry_name}, n={n}, rad={radial_h:.2f}, ovl={overlap_h:.1f}, bc={bc_name}")

    # Create polygon
    poly = polygon(vertices)

    # Create solver
    solver = volkovSolver(
        poly=poly,
        boundary_conditions=boundary_conditions,
        is_dirichlet=is_dirichlet,
        delta=0.05,
        n=n,
        max_iter=10,
        radial_heuristic=radial_h,
        overlap_heuristic=overlap_h
    )

    # Solve
    solution = solver.solve(verbose=False)

    # Get block covering info
    N, L, M, blocks = solver.find_block_covering()

    # Package the data
    reference_data = {
        'solution': solution.data,  # Full array including masked values
        'mask': solution.mask,      # Mask array
        'block_counts': {'N': N, 'L': L, 'M': M},
        'parameters': {
            'geometry_name': geometry_name,
            'vertices': vertices,
            'n': n,
            'radial_heuristic': radial_h,
            'overlap_heuristic': overlap_h,
            'delta': 0.05,
            'max_iter': 10,
            'boundary_conditions': boundary_conditions,
            'is_dirichlet': is_dirichlet,
            'bc_name': bc_name
        },
        'solution_stats': {
            'min': float(solution.min()),
            'max': float(solution.max()),
            'mean': float(solution.mean()),
            'std': float(solution.std()),
            'valid_points': int(solution.compressed().size)
        }
    }

    return reference_data

def main():
    """Generate all reference data."""
    # Create output directory
    output_dir = Path(__file__).parent / 'reference_data'
    output_dir.mkdir(exist_ok=True)

    # Get test definitions
    geometries = get_test_geometries()
    params = get_test_parameters()
    boundary_conditions = get_boundary_conditions()

    # Generate all combinations
    total_cases = 0
    generated_cases = 0

    for geometry_name, vertices in geometries.items():
        for n in params['n']:
            for radial_h in params['radial_heuristic']:
                for overlap_h in params['overlap_heuristic']:
                    for bc_name, (bc_values, is_dirichlet) in boundary_conditions[geometry_name].items():
                        total_cases += 1

                        # Generate filename
                        filename = generate_filename(geometry_name, n, radial_h, overlap_h, bc_name)
                        filepath = output_dir / filename

                        # Skip if already exists
                        if filepath.exists():
                            print(f"Skipping existing: {filename}")
                            continue

                        try:
                            # Generate reference solution
                            reference_data = generate_reference_solution(
                                geometry_name, vertices, n, radial_h, overlap_h,
                                bc_values, is_dirichlet, bc_name
                            )

                            # Save to file
                            np.savez_compressed(filepath, **reference_data)
                            generated_cases += 1

                        except Exception as e:
                            print(f"ERROR generating {filename}: {e}")
                            continue

    print(f"\nGeneration complete!")
    print(f"Total cases: {total_cases}")
    print(f"Generated: {generated_cases}")
    print(f"Output directory: {output_dir}")

    # Calculate storage
    total_size = sum(f.stat().st_size for f in output_dir.glob('*.npz'))
    print(f"Total storage: {total_size:,} bytes ({total_size/1024:.1f} KB)")

if __name__ == "__main__":
    main()