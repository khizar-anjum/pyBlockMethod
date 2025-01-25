#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from ..core.polygon import polygon
from ..solvers.volkov import volkovSolver
from ..utils.geometry import generate_random_polygon
import os

def plot_three_polygons_block_covering(delta, n, max_iter, radial_heuristic=1.0, save_path_prefix="polygon"):
    """
    Plot three pre-defined polygons with their block coverings, without axes and titles.
    Save each as a separate PDF.
    
    Parameters:
        delta: Delta parameter for block covering
        n: n parameter for block covering
        max_iter: Maximum iterations
        radial_heuristic: Radial heuristic parameter
        save_path_prefix: Prefix for the PDF output filenames
    """
    # Define width for all polygons
    width = 3.0
    
    # Define rectangle vertices (3x1 aspect ratio)
    rect_vertices = np.array([
        [-width/2, -width/6],
        [width/2, -width/6], 
        [width/2, width/6],
        [-width/2, width/6]
    ])
    
    # Define obtuse triangle vertices (2x1 aspect ratio)
    tri_vertices = np.array([
        [-width/2, -width/4],
        [width/2, -width/4],
        [0, width/4]
    ])
    
    # Define regular hexagon vertices (1x1 aspect ratio)
    hex_radius = width/2  # Radius to match width
    hex_vertices = np.array([
        [hex_radius, 0],
        [hex_radius/2, hex_radius*0.866],
        [-hex_radius/2, hex_radius*0.866],
        [-hex_radius, 0],
        [-hex_radius/2, -hex_radius*0.866],
        [hex_radius/2, -hex_radius*0.866]
    ])

    # Create polygon objects
    polys = [
        polygon(rect_vertices),
        polygon(tri_vertices),
        polygon(hex_vertices)
    ]

    # Define boundary conditions and Dirichlet flags for each polygon
    boundary_conditions_list = [
        [0] * len(p.vertices) for p in polys
    ]
    is_dirichlet_list = [
        [True] * len(p.vertices) for p in polys  
    ]

    # Plot each polygon separately
    for i, name in enumerate(['rectangle', 'triangle', 'hexagon']):
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        
        solver = volkovSolver(polys[i], boundary_conditions_list[i], 
                                     is_dirichlet_list[i], delta=delta, n=n, 
                                     max_iter=max_iter, radial_heuristic=radial_heuristic)
        solver.find_block_covering()
        solver.plot_block_covering(ax=ax, show_boundary_conditions=False, 
                                 show_quantized_boundaries=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        
        save_path = f"{save_path_prefix}_{name}.pdf"
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

def plot_blocks(blocks):
    fig, ax = plt.subplots()
    for blk in blocks:
        # Draw circle for each block using center and radius
        circle = plt.Circle(blk.center, blk.length, fill=False, color='black')
        ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1.5)
    plt.show()

def plot_3by3_solution_steps(poly_array, boundary_conditions_array, is_dirichlet_array, delta, n, max_iter, radial_heuristics_iter, output_folder = None):
    # Create three separate figures
    fig1, (ax1a, ax1b, ax1c) = plt.subplots(1, 3, figsize=(15, 5))
    fig2, (ax2a, ax2b, ax2c) = plt.subplots(1, 3, figsize=(15, 5))
    fig3, (ax3a, ax3b, ax3c) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Lists to store solutions and their values for consistent colormap
    solutions = []
    solver_array = []
    solution_values = []
    
    # First get all solutions to determine common colormap range
    for i in range(len(poly_array)):
        solver_array.append(volkovSolver(poly_array[i], boundary_conditions_array[i], 
                                     is_dirichlet_array[i], delta=delta, n=n, 
                                     max_iter=max_iter, radial_heuristic=radial_heuristics_iter[i]))
        solution = solver_array[i].solve(plot=False, verbose=True)
        solutions.append(solution)
        solution_values.extend(list(solution.values()))
    
    # Get common value range for colormap
    vmin, vmax = min(solution_values), max(solution_values)
    
    # Plot polygons with block covering
    for i in range(len(poly_array)):
        if i == 0:
            solver_array[i].plot_block_covering(ax=ax1a, show_boundary_conditions=False,
                                     show_quantized_boundaries=False)
        elif i == 1:
            solver_array[i].plot_block_covering(ax=ax1b, show_boundary_conditions=False,
                                     show_quantized_boundaries=False)
        else:
            solver_array[i].plot_block_covering(ax=ax1c, show_boundary_conditions=False,
                                     show_quantized_boundaries=False)
    
    ax1a.set_aspect('equal')
    ax1b.set_aspect('equal')
    ax1c.set_aspect('equal')
    #fig1.suptitle('Block Covering')
    
    # Plot solution heatmaps
    for i in range(len(solutions)):
        if i == 0:
            solver_array[i].plot_solution(ax=ax2a, solution=solutions[i], vmin=vmin, vmax=vmax)
        elif i == 1:
            solver_array[i].plot_solution(ax=ax2b, solution=solutions[i], vmin=vmin, vmax=vmax)
        else:
            solver_array[i].plot_solution(ax=ax2c, solution=solutions[i], vmin=vmin, vmax=vmax)
    
    ax2a.set_aspect('equal')
    ax2b.set_aspect('equal')
    ax2c.set_aspect('equal')
    # fig2.suptitle('Solution')
    
    # Plot gradients
    for i in range(len(solutions)):
        if i == 0:
            solver_array[i].plot_gradient(ax=ax3a, solution=solutions[i])
        elif i == 1:
            solver_array[i].plot_gradient(ax=ax3b, solution=solutions[i])
        else:
            solver_array[i].plot_gradient(ax=ax3c, solution=solutions[i])
    
    ax3a.set_aspect('equal')
    ax3b.set_aspect('equal')
    ax3c.set_aspect('equal')
    # fig3.suptitle('Gradient')
    
    # Save plots if output folder is specified
    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        fig1.savefig(os.path.join(output_folder, 'block_covering.pdf'),
                    bbox_inches='tight', pad_inches=0)
        fig2.savefig(os.path.join(output_folder, 'solution.pdf'),
                    bbox_inches='tight', pad_inches=0)
        fig3.savefig(os.path.join(output_folder, 'gradient.pdf'),
                    bbox_inches='tight', pad_inches=0)

def plot_heuristic_analysis(output_folder = None):
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    axes = [ax1, ax2, ax3]
    
    # Radial and overlap heuristic values to test
    radial_heuristics = [0.75, 0.8, 0.85, 0.90, 0.95, 0.99]
    overlap_heuristics = [0.1, 0.2, 0.3, 0.4, 0.45]
    markers = ['o', 's', 'D', 'v', 'X']
    
    print("Starting percentage increase analysis...")
    
    # Generate results for different numbers of vertices
    delta = 0.05
    num_vertices = [10, 20, 30]
    for vertex_idx, num_vertex in enumerate(num_vertices):
        print(f"\nAnalyzing with {num_vertex} vertices")
        
        # Store results across all polygons
        all_results = {oh: [] for oh in overlap_heuristics}
        
        # Generate multiple random polygons for averaging
        n_polygons = 5
        areas = []
        for poly_idx in range(n_polygons):
            print(f"  Analyzing polygon {poly_idx + 1}/{n_polygons}...")
            vertices = generate_random_polygon(num_vertex, min_radius=1, max_radius=2)
            poly = polygon(vertices)
            areas.append(poly.area())
            boundary_conditions = [[0.0] for _ in range(len(vertices))]
            is_dirichlet = [True for _ in range(len(vertices))]
            
            # Initialize results for this polygon
            results = {oh: [] for oh in overlap_heuristics}
            
            print("    Calculating baseline M values...")
            # Get baseline M values
            baselines = {}
            for oh in overlap_heuristics:
                solver = volkovSolver(poly, boundary_conditions, is_dirichlet,
                                             delta=delta, n=50, max_iter=10, 
                                             radial_heuristic=0.75, overlap_heuristic=oh)
                _, _, baseline_M = solver.find_block_covering()[:3]
                baselines[oh] = baseline_M
            
            # Calculate percentage increase for each radial heuristic
            print("    Testing radial heuristics...")
            for rh_idx, rh in enumerate(radial_heuristics):
                for oh in overlap_heuristics:
                    solver = volkovSolver(poly, boundary_conditions, is_dirichlet,
                                                 delta=delta, n=50, max_iter=10, 
                                                 radial_heuristic=rh, overlap_heuristic=oh)
                    _, _, M = solver.find_block_covering()[:3]
                    pct_increase = ((M - baselines[oh]) / baselines[oh]) * 100
                    results[oh].append(pct_increase)
            
            # Store results for this polygon
            for oh in overlap_heuristics:
                all_results[oh].append(results[oh])
        
        mean_area = sum(areas)/len(areas)
        area_variance = np.var(areas)
        print(f"  Average polygon area: {mean_area:.2f}, variance: {area_variance:.2f}")
        print("  Calculating statistics and plotting...")
        # Calculate mean and standard deviation
        for oh_idx, oh in enumerate(overlap_heuristics):
            results_array = np.array(all_results[oh])
            means = np.mean(results_array, axis=0)
            stds = np.std(results_array, axis=0)
            
            # Plot with error bars
            axes[vertex_idx].errorbar(radial_heuristics, means, yerr=stds,
                                   label=f'$\\omega$={oh}', marker=markers[oh_idx],
                                   capsize=3)
            
        axes[vertex_idx].set_xlabel('Radial Heuristic ($\\rho$)')
        axes[vertex_idx].set_ylabel('Percent change in number of blocks')
        axes[vertex_idx].set_title(f'N = {num_vertex}\nArea = {mean_area:.2f} ± {area_variance:.2f}')
        axes[vertex_idx].grid(True)
        axes[vertex_idx].legend()
    
    print("\nFinalizing plot...")
    plt.tight_layout()
    # Set y-axis limits for all subplots
    for ax in axes:
        ax.set_ylim(-90, 10)
    # Save as PDF
    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(os.path.join(output_folder, 'heuristic_analysis.pdf'), bbox_inches='tight')
    
    print("Analysis complete. Displaying plot...")
    plt.show()