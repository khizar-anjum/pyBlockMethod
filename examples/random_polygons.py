#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pyBlockGrid import polygon, volkovSolver
from pyBlockGrid.utils.geometry import generate_random_polygon

def analyze_polygon(vertices, boundary_value=1.0):
    """Analyze a single polygon with given vertices."""
    poly = polygon(vertices)
    
    # Set boundary conditions (boundary_value on first edge, 0 elsewhere)
    boundary_conditions = [[0.0] for _ in range(len(vertices))]
    boundary_conditions[0] = [boundary_value]
    
    # All Dirichlet conditions
    is_dirichlet = [True] * len(vertices)
    
    # Create solver
    solver = volkovSolver(
        poly=poly,
        boundary_conditions=boundary_conditions,
        is_dirichlet=is_dirichlet,
        delta=0.05,
        n=50,
        max_iter=10
    )
    
    # Solve
    solution = solver.solve(verbose=True)
    
    return solver, solution

def main():
    # Generate and analyze multiple random polygons
    n_vertices = [6, 8, 10]  # Different numbers of vertices
    n_cols = len(n_vertices)
    
    fig, axes = plt.subplots(3, n_cols, figsize=(5*n_cols, 15))
    
    for col, n in enumerate(n_vertices):
        # Generate random polygon
        vertices = generate_random_polygon(n, min_radius=1.0, max_radius=2.0)
        
        # Analyze polygon
        solver, solution = analyze_polygon(vertices)
        
        # Plot results
        solver.plot_block_covering(axes[0, col])
        axes[0, col].set_title(f'{n} Vertices: Block Covering')
        
        solver.plot_solution(axes[1, col])
        axes[1, col].set_title('Temperature Distribution')
        
        solver.plot_gradient(axes[2, col])
        axes[2, col].set_title('Heat Flow')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 