#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
from src.pyBlockGrid.core.polygon import polygon
from src.pyBlockGrid.solvers.volkov import volkovSolver
from src.pyBlockGrid.visualization.plotting import plot_3by3_solution_steps

def test_polygon(poly):
    print(f"Polygon area: {poly.area():.2f}")
    print(f"Is polygon convex? {poly.verify_convexity()}")
    print(f"Are vertices in counter-clockwise order? {poly.verify_vertex_order()}")
    print(f"Number of vertices: {len(poly.vertices)}")
    _, ax = plt.subplots()
    poly.plot(ax=ax)
    plt.show()

def test_block_covering(poly, boundary_conditions, is_dirichlet, delta, n, max_iter, radial_heuristics_iter,
                        overlap_heuristic_iter):
    # Plot block covering
    plt.figure(figsize=(5 * len(radial_heuristics_iter), 5))
    for i, radial_heuristic in enumerate(radial_heuristics_iter):
        ax = plt.subplot(1, len(radial_heuristics_iter), i+1)
        solver = volkovSolver(poly, boundary_conditions, is_dirichlet, delta=delta, n=n, max_iter=max_iter, 
                                     radial_heuristic=radial_heuristic, overlap_heuristic=overlap_heuristic_iter[i])
        print("Finding block covering...")
        N, L, M, uncovered_points = solver.find_block_covering()
        print(f'N, L, M: {N}, {L}, {M}')
        print("Plotting block covering...")
        solver.plot_block_covering(ax=ax, uncovered_points=uncovered_points, show_boundary_conditions=False, show_quantized_boundaries=False)
        ax.set_title(f'Block Covering: radial_heuristic = {radial_heuristic}')
    
    plt.tight_layout()
    plt.show()

def test_approximate_solution(poly, boundary_conditions, is_dirichlet, delta, n, max_iter, radial_heuristics_iter, N = 100):
    # Plot approximate solutions
    plt.figure(figsize=(5 * len(radial_heuristics_iter), 5))
    for i, radial_heuristic in enumerate(radial_heuristics_iter):
        ax = plt.subplot(1, len(radial_heuristics_iter), i+1)
        solver = volkovSolver(poly, boundary_conditions, is_dirichlet, delta=delta, n=n, max_iter=max_iter, 
                                     radial_heuristic=radial_heuristic)
        solution = solver.solve(plot=False, verbose=True)
        solver.plot_solution(ax=ax, solution=solution, N = N)
        ax.set_title(f'Solution: radial_heuristic = {radial_heuristic}')
    
    plt.tight_layout()
    plt.show()
    
def test_gradient(poly, boundary_conditions, is_dirichlet, delta, n, max_iter):
    # Create figure and axes for plotting
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    
    solver = volkovSolver(poly, boundary_conditions, is_dirichlet, delta=delta, n=n, max_iter=max_iter)
    solution = solver.solve(plot=False, verbose=True)
    solver.plot_gradient(ax=ax, solution=solution)
    plt.show()

def test_solution_steps(poly, boundary_conditions, is_dirichlet, delta, n, max_iter, radial_heuristics_iter,
                        output_folder = None):
    # Plot the three steps in the solution process
    plt.figure(figsize=(15, 5))
    
    for i, radial_heuristic in enumerate(radial_heuristics_iter):
        # Create solver instance
        solver = volkovSolver(poly, boundary_conditions, is_dirichlet, delta=delta, n=n, max_iter=max_iter,
                                     radial_heuristic=radial_heuristic)
        
        # Plot block covering
        ax1 = plt.subplot(1, 3, 1)
        solution = solver.solve(plot=False, verbose=True)
        solver.plot_block_covering(ax=ax1, show_boundary_conditions=False, 
                                   show_quantized_boundaries=False)
        # ax1.set_title('Step 1: Block Covering')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Plot solution
        ax2 = plt.subplot(1, 3, 2)
        solver.plot_solution(ax=ax2, solution=solution)
        # ax2.set_title('Step 2: Solution')
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Plot gradient
        ax3 = plt.subplot(1, 3, 3)
        solver.plot_gradient(ax=ax3, solution=solution)
        # ax3.set_title('Step 3: Gradient')
        ax3.set_xticks([])
        ax3.set_yticks([])

    # Save figure if output folder is provided
    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        plt.savefig(os.path.join(output_folder, 'solution_steps.pdf'), 
                    format='pdf', bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

def test_3by3_plotting():
    v1 = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]) # square with a square depression # [1, 0],
    v2 = np.array([[-0.58677897, 1.62158457], [-0.63274991, 0.8863274], [-1.74993861, 0.30467836],
                        [-1.25531598, 0.19937933], [-1.35251473, 0.09726827], [-1.84362218, 0.11634993],
                        [-1.84289172, -0.18032796], [0.19833559, -1.43784708], [0.2842006, -1.56029036],
                        [1.1250711, -1.1409574]])
    v3 = np.array([[ 0.9599168 ,  0.65081101], [ 0.26401952 , 1.81835665], [-0.16186728 , 1.15558642], 
                         [-0.54884342 , 1.78631676], [-0.4325186,   1.11517468], [-1.129355,   -0.94655537], 
                         [-0.01205465, -1.22329523], [ 1.1184654,  -1.40151058], [ 0.88881931, -1.10564833], 
                         [ 1.34336699, -0.19043799]])
    # Create arrays of polygons, boundary conditions and Dirichlet flags
    poly_array = [polygon(v1), polygon(v2), polygon(v3)]
    
    # Define boundary conditions - zero everywhere except first edge of each polygon
    boundary_conditions_array = []
    for poly in poly_array:
        bc = [[0.0] for _ in range(len(poly.vertices))]
        bc[0] = [1.0] # Set first edge to 1.0
        boundary_conditions_array.append(bc)
        
    # All Dirichlet boundary conditions
    is_dirichlet_array = [
        [True for _ in range(len(poly.vertices))] for poly in poly_array
    ]
    
    # Different radial heuristics for each polygon
    radial_heuristics_iter = [0.95, 0.95, 0.95]
    
    # Call plotting function
    plot_3by3_solution_steps(
        poly_array=poly_array,
        boundary_conditions_array=boundary_conditions_array, 
        is_dirichlet_array=is_dirichlet_array,
        delta=0.05,
        n=50,
        max_iter=10,
        radial_heuristics_iter=radial_heuristics_iter,
        output_folder="plots"
    )