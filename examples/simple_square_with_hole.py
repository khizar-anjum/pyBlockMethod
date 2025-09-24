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
    print("\n" + "="*60)
    print(" SQUARE WITH SQUARE HOLE - VOLKOV SOLVER EXAMPLE")
    print("="*60 + "\n")

    # Create main square: 2x2 centered at origin
    main_vertices = np.array([
        [-1.0, -1.0],  # Bottom-left
        [ 1.0, -1.0],  # Bottom-right
        [ 1.0,  1.0],  # Top-right
        [-1.0,  1.0]   # Top-left
    ])

    poly = polygon(main_vertices)
    print(f"Main square created:")
    print(f"  Area: {poly.area():.4f}")
    print(f"  Vertices: {len(main_vertices)}")

    # Add square hole: 0.8x0.8 centered at origin (clockwise vertices)
    hole_vertices = np.array([
        [-0.4, -0.4],  # Bottom-left
        [-0.4,  0.4],  # Top-left
        [ 0.4,  0.4],  # Top-right
        [ 0.4, -0.4]   # Bottom-right
    ])[::-1]  # Reverse to make clockwise

    # Boundary conditions for hole (cold on all sides)
    hole_bc = [[0.0], [0.0], [0.0], [0.0]]
    hole_is_dirichlet = [True, True, True, True]

    # Add hole to polygon
    hole = poly.add_hole(hole_vertices, hole_bc, hole_is_dirichlet)
    print(f"\nSquare hole added:")
    print(f"  Hole area: {hole.area():.4f}")
    print(f"  Effective domain area: {poly.area_with_holes():.4f}")

    # Set boundary conditions for main polygon:
    # Temperature = 1 on bottom edge, 0 elsewhere (Dirichlet conditions)
    boundary_conditions = [[1.0], [0.0], [0.0], [0.0]]
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
        delta=0.1,   # Grid spacing
        n=50,        # Number of angular divisions
        max_iter=10  # Maximum iterations
    )

    # Find block covering
    print("\n" + "="*50)
    print("Finding block covering...")
    print("="*50)
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
    main_blocks = [b for b in solver.blocks if getattr(b, 'boundary_type', 'main') == 'main']
    hole_blocks = [b for b in solver.blocks if getattr(b, 'boundary_type', 'main') == 'hole']

    print(f"\nBlocks by boundary:")
    print(f"  Main boundary blocks: {len(main_blocks)}")
    print(f"  Hole boundary blocks: {len(hole_blocks)}")

    # Create visualization
    print("\n" + "="*50)
    print("Creating visualization...")
    print("="*50)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Block covering with boundary conditions
    solver.plot_block_covering(
        ax=ax1,
        show_boundary_conditions=True,
        show_quantized_boundaries=False
    )
    ax1.set_title('Block Covering with Boundary Conditions\n(Hole-aware Volkov Method)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # Plot 2: Domain visualization
    ax2.set_title('Valid Domain\n(Inside main, outside hole)')

    # Create a grid to show the domain
    x = np.linspace(-1.5, 1.5, 100)
    y = np.linspace(-1.5, 1.5, 100)
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
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # Add annotations
    ax2.text(0, 0, 'HOLE\n(excluded)',
             ha='center', va='center', fontsize=12, color='red')
    ax2.annotate('Hot boundary\n(T=1.0)',
                 xy=(0, -1), xytext=(0, -1.3),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2),
                 fontsize=10, ha='center', color='red')

    plt.suptitle('Square with Square Hole - Volkov Block Method',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    plot_folder = "plots"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    filename = os.path.join(plot_folder, 'square_with_hole_blocks.png')
    plt.savefig(filename, dpi=150)
    print(f"\nVisualization saved to '{filename}'")

    plt.show()

    print("\n" + "="*60)
    print(" HOLE-AWARE BLOCK COVERING COMPLETE!")
    print(" (Solution computation not yet implemented)")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()