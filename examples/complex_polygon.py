#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pyBlockGrid import polygon, volkovSolver

def main():
    # Create an L-shaped polygon
    vertices = np.array([
        [0, 0],
        [2, 0],
        [2, 1],
        [1, 1],
        [1, 2],
        [0, 2]
    ])
    poly = polygon(vertices)
    
    # Set mixed boundary conditions:
    # - Temperature = 1 on bottom edge
    # - Temperature = 0 on top edge
    # - Insulated (Neumann) conditions on other edges
    boundary_conditions = [
        [1.0],  # Bottom edge
        [0.0],  # Right bottom edge
        [0.0],  # Right top edge
        [0.0],  # Inner horizontal edge
        [0.0],  # Inner vertical edge
        [0.0]   # Left edge
    ]
    
    # True for Dirichlet, False for Neumann
    is_dirichlet = [
        True,   # Bottom edge: Dirichlet
        True,  # Right bottom edge: Dirichlet
        True,  # Right top edge: Dirichlet
        True,  # Inner horizontal edge: Dirichlet
        True,  # Inner vertical edge: Dirichlet
        True    # Left edge: Dirichlet
    ]
    
    # Create solver with custom parameters
    solver = volkovSolver(
        poly=poly,
        boundary_conditions=boundary_conditions,
        is_dirichlet=is_dirichlet,
        delta=0.02,      # Finer grid
        n=100,           # More angular divisions
        max_iter=20,     # More iterations
        radial_heuristic=0.9  # Different block sizing
    )
    
    # Solve and visualize
    solution = solver.solve(verbose=True)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plot polygon shape
    poly.plot(axes[0, 0])
    axes[0, 0].set_title('Polygon Geometry')
    
    # Plot block covering
    solver.plot_block_covering(axes[0, 1])
    axes[0, 1].set_title('Block Covering')
    
    # Plot solution
    solver.plot_solution(axes[1, 0], solution)
    axes[1, 0].set_title('Temperature Distribution')
    
    # Plot gradient
    solver.plot_gradient(axes[1, 1], solution)
    axes[1, 1].set_title('Heat Flow')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 