#!/usr/bin/env python3
"""
Simple example: Square domain with square hole using Volkov solver.

This example demonstrates the hole functionality with the Volkov block
method, showing the block covering for a domain with a hole.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pyBlockGrid import polygon, volkovSolver


def main():
    print("\n" + "=" * 60)
    print(" SQUARE WITH SQUARE HOLE - VOLKOV SOLVER EXAMPLE")
    print("=" * 60 + "\n")

    # Create main square: 2x2 centered at origin
    main_vertices = np.array(
        [
            [-1.0, -1.0],  # Bottom-left
            [1.0, -1.0],  # Bottom-right
            [1.0, 1.0],  # Top-right
            [-1.0, 1.0],  # Top-left
        ]
    )

    poly = polygon(main_vertices)
    print(f"Main square created:")
    print(f"  Area: {poly.area():.4f}")
    print(f"  Vertices: {len(main_vertices)}")

    # Add square hole: (c-shaped with width wi) (clockwise vertices)
    wi = 0.1
    hole_vertices = np.array(
        [
            [-0.4, -0.4],  # Bottom-left
            [-0.4, -0.4 + wi],  # Inside Bottom-left
            [0.4 - wi, -0.4 + wi],  # Inside Bottom-right
            [0.4 - wi, 0.4 - wi],  # Inside Top-right
            [-0.4, 0.4 - wi],  # Inside Top-left
            [-0.4, 0.4],  # Top-left
            [0.4, 0.4],  # Top-right
            [0.4, -0.4],  # Bottom-right
        ]
    )  # the array is already clockwise

    # Boundary conditions for hole (cold on all sides)
    hole_bc = [[0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
    hole_is_dirichlet = [True, True, True, True, True, True, True, True]

    # hole_vertices = np.array(
    #     [
    #         [-0.4, -0.4],  # Bottom-left
    #         [0.4, -0.4],  # Inside Bottom-left
    #         [0.4, 0.4],  # Top-right
    #         [-0.4, 0.4],  # Bottom-right
    #     ]
    # )[::-1]  # the array is already clockwise

    # # Boundary conditions for hole (cold on all sides)
    # hole_bc = [[0.0], [0.0], [0.0], [1.0]]
    # hole_is_dirichlet = [True, True, True, True]

    # Add hole to polygon
    hole = poly.add_hole(hole_vertices, hole_bc, hole_is_dirichlet)
    print(f"\nSquare hole added:")
    print(f"  Hole area: {hole.area():.4f}")
    print(f"  Effective domain area: {poly.area_with_holes():.4f}")

    # Set boundary conditions for main polygon:
    # Temperature = 1 on bottom edge, 0 elsewhere (Dirichlet conditions)
    boundary_conditions = [[0.0], [0.0], [0.0], [0.0]]
    is_dirichlet = [True] * 4  # Dirichlet conditions on all edges

    print("\nBoundary conditions:")
    print("  Main polygon:")
    print(f"    Bottom edge: {boundary_conditions[0][0]} (Dirichlet)")
    print(f"    Other edges: 0.0 (Dirichlet)")
    print("  Hole:")
    print(f"    All edges: 0.0 (Dirichlet)")

    # Create solver with hole support
    solver = volkovSolver(
        poly=poly,
        boundary_conditions=boundary_conditions,
        is_dirichlet=is_dirichlet,
        delta=0.02,  # Grid spacing
        n=30,  # Number of angular divisions
        max_iter=10,  # Maximum iterations
        overlap_heuristic=0.3,  # Overlap factor
        radial_heuristic=0.84,  # Radial overlap factor
    )

    # Find block covering
    print("\n" + "=" * 50)
    print("Finding block covering...")
    print("=" * 50)
    solver.find_block_covering()

    # Get block statistics
    N, L, M = solver.N, solver.L, solver.M
    print(f"\nBlock covering statistics:")
    print(f"  Total blocks: {M}")
    print(f"  First kind blocks (vertices): {N}")
    print(f"    - Main polygon vertices: 4")
    print(f"    - Hole vertices: 4")
    print(f"  Second kind blocks (edges): {L - N}")
    print(f"  Third kind blocks (interior): {M - L}")

    # Analyze blocks by boundary
    main_blocks = [
        b for b in solver.blocks if getattr(b, "boundary_type", "main") == "main"
    ]
    hole_blocks = [
        b for b in solver.blocks if getattr(b, "boundary_type", "main") == "hole"
    ]

    print(f"\nBlocks by boundary:")
    print(f"  Main boundary blocks: {len(main_blocks)}")
    print(f"  Hole boundary blocks: {len(hole_blocks)}")

    # Solve the problem
    print("\n" + "=" * 50)
    print("Solving the Laplace equation...")
    print("=" * 50)

    try:
        solution = solver.solve(verbose=True)
        print(f"\nSolution computed successfully!")
        print(f"  Solution shape: {solution.shape}")
        print(f"  Valid points: {np.sum(~solution.mask)}")
        print(f"  Solution range: [{solution.min():.4f}, {solution.max():.4f}]")

        # Validate the solution
        print("\n" + "=" * 50)
        print("Validating solution...")
        print("=" * 50)

        # Check hole boundary conditions
        hole_validation = solver.validate_hole_solution()
        if "holes" in hole_validation:
            for hole_result in hole_validation["holes"]:
                hole_id = hole_result["hole_id"]
                max_error = hole_result["max_error"]
                mean_error = hole_result["mean_error"]
                print(
                    f"  Hole {hole_id + 1}: max_error={max_error:.6f}, mean_error={mean_error:.6f}"
                )

        # Check solution continuity
        continuity = solver.check_solution_continuity()
        if "continuity_score" in continuity:
            print(f"  Continuity score: {continuity['continuity_score']:.4f}")
            print(f"  Max gradient: {continuity['max_gradient']:.4f}")
            print(f"  Mean gradient: {continuity['mean_gradient']:.4f}")

        solution_computed = True

    except Exception as e:
        print(f"\nSolution computation failed: {e}")
        print("Continuing with block covering visualization only...")
        solution_computed = False

    # Create visualization
    print("\n" + "=" * 50)
    print("Creating visualization...")
    print("=" * 50)

    # Save folder
    plot_folder = "plots"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # Plot 1: Block covering with boundary conditions
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    solver.plot_block_covering(
        ax=ax1, show_boundary_conditions=True, show_quantized_boundaries=False
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    plt.tight_layout()
    filename1 = os.path.join(plot_folder, "square_with_hole_blocks.pdf")
    plt.savefig(filename1, dpi=150)
    print(f"Block covering saved to '{filename1}'")

    if solution_computed:
        # Plot 2: Solution heatmap
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
        solver.plot_solution(ax2)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        plt.tight_layout()
        filename2 = os.path.join(plot_folder, "square_with_hole_solution.pdf")
        plt.savefig(filename2, dpi=150)
        print(f"Solution heatmap saved to '{filename2}'")

        # Plot 3: Gradient field
        fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))
        solver.plot_gradient(ax3, decimation_factor=4, scale=35)
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        plt.tight_layout()
        filename3 = os.path.join(plot_folder, "square_with_hole_gradient.pdf")
        plt.savefig(filename3, dpi=150)
        print(f"Gradient field saved to '{filename3}'")

    plt.show()

    print("\n" + "=" * 60)
    if solution_computed:
        print(" HOLE-AWARE VOLKOV SOLUTION COMPLETE!")
        print(
            f" Solution validation: {'PASSED' if all(h['max_error'] < 0.1 for h in hole_validation.get('holes', [])) else 'CHECK REQUIRED'}"
        )
    else:
        print(" HOLE-AWARE BLOCK COVERING COMPLETE!")
        print(" (Solution computation encountered errors)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
