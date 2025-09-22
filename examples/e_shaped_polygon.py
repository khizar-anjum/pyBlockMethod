#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pyBlockGrid import polygon, volkovSolver

def main():
    # Create an E-shaped polygon
    # The polygon is oriented like an E opening to the right
    vertices = np.array([
        [0, 0],      # Bottom left outer
        [3, 0],      # Bottom right outer
        [3, 1.5],    # Bottom right inner
        [1, 1.5],    # Bottom left inner
        [1, 2],      # Middle left inner
        [3, 2],      # Middle right inner
        [3, 3],      # Top right inner
        [1, 3],      # Top left inner
        [1, 3.5],    # Top right outer
        [3, 3.5],    # Top right outer
        [3, 5],      # Top outer
        [0, 5],      # Top left outer
    ])
    poly = polygon(vertices)

    # Set boundary conditions:
    # Temperature = 1 on the left vertical edge (edge 11), 0 elsewhere (Dirichlet conditions)
    # This simulates heat source on the left side of the E
    boundary_conditions = []
    for i in range(len(vertices)):
        if i == 11:  # Left vertical edge
            boundary_conditions.append([1.0])
        else:
            boundary_conditions.append([0.0])

    is_dirichlet = [True] * len(vertices)  # Dirichlet conditions on all edges

    # Create solver with parameters tuned for E-shaped geometry
    solver = volkovSolver(
        poly=poly,
        boundary_conditions=boundary_conditions,
        is_dirichlet=is_dirichlet,
        delta=0.05,  # Grid spacing
        n=50,        # Number of angular divisions
        max_iter=10,  # Maximum iterations
        overlap_heuristic = 0.3,
        radial_heuristic = 0.9,
    )

    # Solve and get solution
    solution = solver.solve(verbose=True)

    # Get block statistics
    N, L, M, _ = solver.find_block_covering()
    print(f"N, L, M = {N}, {L}, {M}")

    # Create visualization with 1x3 layout like other examples
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the block covering
    solver.plot_block_covering(ax=ax1)
    ax1.set_title('Block Covering')

    # Plot solution
    solver.plot_solution(ax=ax2)
    ax2.set_title('Temperature Distribution')

    # Plot gradient (heat flow)
    solver.plot_gradient(ax=ax3)
    ax3.set_title('Temperature Gradient')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
