#!/usr/bin/env python3
"""
Simple example: Square domain with square hole.

This is a minimal example demonstrating the polygon hole functionality
from Phase 1 of the implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pyBlockGrid.core.polygon import polygon


def create_square_with_square_hole():
    """Create a square domain with a square hole inside."""

    # Main square: 2x2 centered at origin
    # Counterclockwise vertices
    main_vertices = np.array([
        [-1.0, -1.0],  # Bottom-left
        [ 1.0, -1.0],  # Bottom-right
        [ 1.0,  1.0],  # Top-right
        [-1.0,  1.0]   # Top-left
    ])

    # Create main polygon
    poly = polygon(main_vertices)
    print(f"Main square created:")
    print(f"  Area: {poly.area():.4f}")
    print(f"  Vertices: {len(main_vertices)}")

    # Square hole: 0.8x0.8 centered at origin
    # Clockwise vertices (reversed order)
    hole_vertices = np.array([
        [-0.4, -0.4],  # Bottom-left
        [-0.4,  0.4],  # Top-left
        [ 0.4,  0.4],  # Top-right
        [ 0.4, -0.4]   # Bottom-right
    ])[::-1]  # Reverse to make clockwise

    # Boundary conditions for hole (will be used in later phases)
    # For now, just placeholders
    hole_bc = [[0, 0], [0, 0], [0, 0], [0, 0]]  # Cold on all sides
    hole_is_dirichlet = [True, True, True, True]  # All Dirichlet

    # Add hole to polygon
    hole = poly.add_hole(hole_vertices, hole_bc, hole_is_dirichlet)
    print(f"\nSquare hole added:")
    print(f"  Hole area: {hole.area():.4f}")
    print(f"  Hole vertices: {hole.n_vertices}")

    # Calculate area with hole
    total_area = poly.area_with_holes()
    expected_area = 4.0 - 0.64  # 2x2 - 0.8x0.8
    print(f"\nArea calculations:")
    print(f"  Main area: {poly.area():.4f}")
    print(f"  Hole area: {hole.area():.4f}")
    print(f"  Total area (main - hole): {total_area:.4f}")
    print(f"  Expected: {expected_area:.4f}")

    return poly


def visualize_square_with_hole(poly):
    """Visualize the square with hole."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Simple polygon visualization
    ax1.set_title("Square with Square Hole\n(Phase 1: Geometry Only)")
    poly.plot(ax1, show_holes=True)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # Add some annotations
    ax1.annotate('Main boundary\n(counterclockwise)',
                xy=(1.0, 1.0), xytext=(1.3, 1.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                fontsize=10)
    ax1.annotate('Hole boundary\n(clockwise)',
                xy=(0.4, 0.4), xytext=(0.7, 0.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=1),
                fontsize=10)

    # Right plot: Show domain (points inside polygon but outside hole)
    ax2.set_title("Valid Domain\n(Inside main, outside hole)")

    # Create a grid of points
    x = np.linspace(-1.5, 1.5, 100)
    y = np.linspace(-1.5, 1.5, 100)
    X, Y = np.meshgrid(x, y)

    # Test each point
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if poly._point_in_polygon(X[i, j], Y[i, j]):
                Z[i, j] = 1.0

    # Show as filled contour
    ax2.contourf(X, Y, Z, levels=[0, 0.5, 1],
                 colors=['white', 'lightblue'], alpha=0.7)
    ax2.contour(X, Y, Z, levels=[0.5], colors='black', linewidths=2)

    # Overlay the polygon boundaries
    poly.plot(ax2, show_holes=True)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    # Add text in the hole
    ax2.text(0, 0, 'HOLE\n(excluded)',
            ha='center', va='center', fontsize=12, color='red')

    plt.suptitle('Phase 1 Complete: Polygon with Hole Support',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def test_points_in_domain(poly):
    """Test various points to verify domain logic."""
    print("\n" + "="*50)
    print("Testing point-in-domain logic:")
    print("="*50)

    test_points = [
        (0.0, 0.0),     # Center (in hole)
        (0.7, 0.7),     # Corner between main and hole
        (-0.7, -0.7),   # Opposite corner
        (0.0, 0.5),     # Top middle
        (0.0, -0.5),    # Bottom middle
        (1.5, 0.0),     # Outside main
        (0.2, 0.2),     # Inside hole
        (-0.2, -0.2),   # Inside hole
        (0.5, 0.0),     # Between hole and main boundary
    ]

    for x, y in test_points:
        in_domain = poly._point_in_polygon(x, y)
        in_main = poly._point_in_polygon_original(x, y)
        in_hole = any(hole.point_in_hole(x, y) for hole in poly.holes)

        status = "✓ In domain" if in_domain else "✗ Excluded"
        location = []
        if in_main:
            location.append("in main")
        if in_hole:
            location.append("in hole")
        if not in_main:
            location.append("outside")

        print(f"  ({x:5.2f}, {y:5.2f}): {status:15} ({', '.join(location)})")


def main():
    """Run the example."""
    print("\n" + "="*60)
    print(" SIMPLE SQUARE WITH SQUARE HOLE EXAMPLE")
    print(" Phase 1: Polygon Hole Geometry")
    print("="*60 + "\n")

    # Create the geometry
    poly = create_square_with_square_hole()

    # Test point logic
    test_points_in_domain(poly)

    # Get boundary information
    print("\n" + "="*50)
    print("Boundary Information:")
    print("="*50)
    boundaries = poly.get_all_boundaries()
    print(f"Main boundary:")
    print(f"  Vertices: {boundaries['main']['n_vertices']}")
    print(f"  Edges: {len(boundaries['main']['edges'])}")

    for i, hole_info in enumerate(boundaries['holes']):
        print(f"\nHole {i+1}:")
        print(f"  Vertices: {hole_info['n_vertices']}")
        print(f"  Boundary conditions: {len(hole_info['boundary_conditions'])} edges")
        print(f"  All Dirichlet: {all(hole_info['is_dirichlet'])}")

    # Visualize
    print("\n" + "="*50)
    print("Creating visualization...")
    print("="*50)

    fig = visualize_square_with_hole(poly)
    plot_folder = "plots"
    plt.savefig(os.path.join(plot_folder,
        'square_with_hole_phase1.png'), dpi=150)
    print("Visualization saved to 'square_with_hole_phase1.png'")

    plt.show()

    print("\n" + "="*60)
    print(" PHASE 1 IMPLEMENTATION COMPLETE!")
    print(" Ready for Phase 2: Solution State Updates")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
