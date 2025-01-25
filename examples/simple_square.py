#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pyBlockGrid import polygon, volkovSolver

def main():
    # Create a simple square polygon
    vertices = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])
    poly = polygon(vertices)
    
    # Set boundary conditions:
    # Temperature = 1 on bottom edge, 0 elsewhere (Dirichlet conditions)
    boundary_conditions = [[1.0], [0.0], [0.0], [0.0]]
    is_dirichlet = [True] * 4  # Dirichlet conditions on all edges
    
    # Create solver
    solver = volkovSolver(
        poly=poly,
        boundary_conditions=boundary_conditions,
        is_dirichlet=is_dirichlet,
        delta=0.05,  # Grid spacing
        n=50,        # Number of angular divisions
        max_iter=10  # Maximum iterations
    )
    
    # Solve and get solution
    solution = solver.solve(verbose=True)
    
    # Test assertions
    assert len(solution) > 0, "Solution should contain values"
    assert all(isinstance(val, float) for val in solution.values()), "All solution values should be floats"
    
    # Solve and plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot the block covering
    solver.find_block_covering()
    solver.plot_block_covering(ax=ax1)
    ax1.set_title('Block Covering')
    
    # Solve and plot solution
    solver.plot_solution(ax=ax2, solution=solution)
    ax2.set_title('Temperature Distribution')
    
    # Plot gradient (heat flow)
    solver.plot_gradient(ax=ax3, solution=solution)
    ax3.set_title('Temperature Gradient')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 