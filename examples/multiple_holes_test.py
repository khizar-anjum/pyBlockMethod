#!/usr/bin/env python3
"""
Advanced test case: Rectangle with multiple holes using Volkov solver.

This example demonstrates the hole functionality with multiple holes
of different shapes, showing the full solution computation and validation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pyBlockGrid import polygon, volkovSolver

def main():
    print("\n" + "="*70)
    print(" MULTIPLE HOLES TEST - VOLKOV SOLVER VALIDATION")
    print("="*70 + "\n")

    # Create main rectangle: 3x2 centered at origin
    main_vertices = np.array([
        [-1.5, -1.0],  # Bottom-left
        [ 1.5, -1.0],  # Bottom-right
        [ 1.5,  1.0],  # Top-right
        [-1.5,  1.0]   # Top-left
    ])

    poly = polygon(main_vertices)
    print(f"Main rectangle created:")
    print(f"  Area: {poly.area():.4f}")
    print(f"  Vertices: {len(main_vertices)}")

    # Add first hole: small square on the left
    hole1_vertices = np.array([
        [-1.0, -0.3],  # Bottom-left
        [-1.0,  0.3],  # Top-left
        [-0.4,  0.3],  # Top-right
        [-0.4, -0.3]   # Bottom-right
    ])[::-1]  # Reverse to make clockwise

    hole1_bc = [[0.0], [0.0], [0.0], [0.0]]  # Cold hole
    hole1_is_dirichlet = [True, True, True, True]

    hole1 = poly.add_hole(hole1_vertices, hole1_bc, hole1_is_dirichlet)
    print(f"\nHole 1 (square) added:")
    print(f"  Area: {hole1.area():.4f}")

    # Add second hole: triangular hole on the right
    hole2_vertices = np.array([
        [0.4, -0.4],   # Bottom
        [1.0,  0.0],   # Right
        [0.4,  0.4]    # Top
    ])[::-1]  # Reverse to make clockwise

    hole2_bc = [[0.5], [0.5], [0.5]]  # Warm hole
    hole2_is_dirichlet = [True, True, True]

    hole2 = poly.add_hole(hole2_vertices, hole2_bc, hole2_is_dirichlet)
    print(f"\nHole 2 (triangle) added:")
    print(f"  Area: {hole2.area():.4f}")

    print(f"\nTotal effective domain area: {poly.area_with_holes():.4f}")

    # Set boundary conditions for main polygon:
    # Left edge hot (T=1), right edge cold (T=0), top/bottom insulated
    boundary_conditions = [[0.0], [1.0], [0.0], [0.0]]  # left, bottom, right, top
    is_dirichlet = [False, True, True, False]  # Neumann on left/top, Dirichlet on bottom/right

    print(f"\nBoundary conditions:")
    print(f"  Main polygon:")
    print(f"    Left edge: Neumann (insulated)")
    print(f"    Bottom edge: T=1.0 (hot)")
    print(f"    Right edge: T=0.0 (cold)")
    print(f"    Top edge: Neumann (insulated)")
    print(f"  Hole 1 (square): T=0.0 (cold)")
    print(f"  Hole 2 (triangle): T=0.5 (warm)")

    # Create solver
    solver = volkovSolver(
        poly=poly,
        boundary_conditions=boundary_conditions,
        is_dirichlet=is_dirichlet,
        delta=0.08,    # Slightly coarser grid for complex geometry
        n=40,          # More angular divisions for better accuracy
        max_iter=50    # More iterations for convergence
    )

    # Find block covering
    print("\n" + "="*60)
    print("Finding block covering...")
    print("="*60)
    solver.find_block_covering()

    N, L, M = solver.N, solver.L, solver.M
    print(f"\nBlock covering statistics:")
    print(f"  Total blocks: {M}")
    print(f"  First kind blocks (vertices): {N}")
    print(f"    - Main polygon vertices: 4")
    print(f"    - Hole 1 vertices: 4")
    print(f"    - Hole 2 vertices: 3")
    print(f"  Second kind blocks (edges): {L - N}")
    print(f"  Third kind blocks (interior): {M - L}")

    # Analyze blocks by boundary
    main_blocks = [b for b in solver.blocks if getattr(b, 'boundary_type', 'main') == 'main']
    hole_blocks = [b for b in solver.blocks if getattr(b, 'boundary_type', 'main') == 'hole']
    hole1_blocks = [b for b in hole_blocks if getattr(b, 'boundary_id', 0) == 0]
    hole2_blocks = [b for b in hole_blocks if getattr(b, 'boundary_id', 0) == 1]

    print(f"\nBlocks by boundary:")
    print(f"  Main boundary blocks: {len(main_blocks)}")
    print(f"  Hole 1 blocks: {len(hole1_blocks)}")
    print(f"  Hole 2 blocks: {len(hole2_blocks)}")

    # Solve the problem
    print("\n" + "="*60)
    print("Solving the Laplace equation...")
    print("="*60)

    try:
        solution = solver.solve(verbose=True)
        print(f"\nSolution computed successfully!")
        print(f"  Solution shape: {solution.shape}")
        print(f"  Valid points: {np.sum(~solution.mask)}")
        print(f"  Solution range: [{solution.min():.4f}, {solution.max():.4f}]")

        # Validate the solution
        print("\n" + "="*60)
        print("Validating solution...")
        print("="*60)

        # Check hole boundary conditions
        hole_validation = solver.validate_hole_solution()
        if 'holes' in hole_validation:
            for hole_result in hole_validation['holes']:
                hole_id = hole_result['hole_id']
                max_error = hole_result['max_error']
                mean_error = hole_result['mean_error']
                print(f"  Hole {hole_id + 1}: max_error={max_error:.6f}, mean_error={mean_error:.6f}")

        # Check solution continuity
        continuity = solver.check_solution_continuity()
        if 'continuity_score' in continuity:
            print(f"  Continuity score: {continuity['continuity_score']:.4f}")
            print(f"  Max gradient: {continuity['max_gradient']:.4f}")
            print(f"  Mean gradient: {continuity['mean_gradient']:.4f}")

        solution_computed = True

    except Exception as e:
        print(f"\nSolution computation failed: {e}")
        print("Error details:")
        import traceback
        traceback.print_exc()
        solution_computed = False

    # Create comprehensive visualization
    print("\n" + "="*60)
    print("Creating visualization...")
    print("="*60)

    if solution_computed:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Block covering
    solver.plot_block_covering(
        ax=ax1,
        show_boundary_conditions=True,
        show_quantized_boundaries=False
    )
    ax1.set_title('Block Covering with Multiple Holes\n(Mixed Boundary Conditions)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # Plot 2: Domain visualization
    ax2.set_title('Valid Domain\n(Rectangle with 2 holes)')

    # Create a fine grid to show the domain
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-1.5, 1.5, 150)
    X, Y = np.meshgrid(x, y)

    # Test each point to see if it's in the valid domain
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if poly._point_in_polygon(X[i, j], Y[i, j]):
                Z[i, j] = 1.0

    # Show valid domain
    ax2.contourf(X, Y, Z, levels=[0, 0.5, 1],
                 colors=['white', 'lightblue'], alpha=0.7)
    ax2.contour(X, Y, Z, levels=[0.5], colors='black', linewidths=2)

    # Overlay the polygon boundaries
    poly.plot(ax=ax2, show_holes=True)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # Add annotations
    ax2.text(-0.7, 0, 'HOLE 1\n(T=0)', ha='center', va='center', fontsize=10, color='blue')
    ax2.text(0.6, 0, 'HOLE 2\n(T=0.5)', ha='center', va='center', fontsize=10, color='orange')

    if solution_computed:
        # Plot 3: Solution heatmap
        solver.plot_solution(ax3, vmin=0, vmax=1)
        ax3.set_title('Solution Heatmap\n(Temperature Distribution)')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')

        # Plot 4: Gradient field
        solver.plot_gradient(ax4, decimation_factor=2, scale=25)
        ax4.set_title('Solution Gradient Field\n(Heat Flow)')
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')

        plt.suptitle('Multiple Holes Test - Complete Volkov Solution',
                     fontsize=16, fontweight='bold')
    else:
        plt.suptitle('Multiple Holes Test - Block Covering Only',
                     fontsize=16, fontweight='bold')

    plt.tight_layout()

    # Save figure
    plot_folder = "plots"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    filename = os.path.join(plot_folder, 'multiple_holes_test.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to '{filename}'")

    plt.show()

    print("\n" + "="*70)
    if solution_computed:
        validation_passed = all(h['max_error'] < 0.2 for h in hole_validation.get('holes', []))
        print(" MULTIPLE HOLES TEST COMPLETE!")
        print(f" Solution validation: {'PASSED' if validation_passed else 'CHECK REQUIRED'}")
        print(f" Solver successfully handled {len(poly.holes)} holes with mixed BC types")
    else:
        print(" MULTIPLE HOLES TEST FAILED!")
        print(" (Solution computation encountered errors)")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()