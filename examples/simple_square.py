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
    assert all(isinstance(val, (float, np.floating)) for val in solution.compressed()), "All solution values should be floats"
    
    # Solve and plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot the block covering
    solver.find_block_covering()
    solver.plot_block_covering(ax=ax1)
    ax1.set_title('Block Covering')
    
    # Solve and plot solution
    solver.plot_solution(ax=ax2)
    ax2.set_title('Temperature Distribution')
    
    # Plot gradient (heat flow)
    solver.plot_gradient(ax=ax3)
    ax3.set_title('Temperature Gradient')
    
    plt.tight_layout()
    plt.show()

def generate_qualitative_comparison_rh(output_folder):
    """
    Generate fig:qualtitative_comparison_rh from the paper.

    Top row: block coverings for rho = 0.75, 0.83, 0.91, 0.99
    Bottom row: approximate solutions with MSE and ||Δu||_2 annotations
    Uses unit square with u(x,1)=sin(πx), u=0 on other edges.
    """
    import os
    os.makedirs(output_folder, exist_ok=True)

    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    # Taylor expansion of sin(πx) around x=0.5 (first 3 terms)
    # sin(πx) ≈ 0.02 + 2.905x + 1.153x² - 8.117x³ + 4.059x⁴
    # Edge 2 goes from (1,1) to (0,1), arc length r maps to x = 1-r,
    # but sin(π(1-r)) = sin(πr), so same polynomial coefficients apply.
    sin_pi_x_coeffs = [0.02, 2.905, 1.153, -8.117, 4.059]

    # u=0 on bottom, right, left; sin(πx) on top (edge 2)
    boundary_conditions = [[0.0], [0.0], sin_pi_x_coeffs, [0.0]]
    is_dirichlet = [True] * 4

    # Analytical solution: u(x,y) = sin(πx) * sinh(πy) / sinh(π)
    def analytical_solution(X, Y):
        return np.sin(np.pi * X) * np.sinh(np.pi * Y) / np.sinh(np.pi)

    rho_values = [0.75, 0.83, 0.91, 0.99]
    n = 15
    max_iter = 5
    delta = 0.01

    print("Generating qualitative comparison for radial heuristic...")
    for rho in rho_values:
        print(f"\n  rho = {rho}")
        poly = polygon(vertices)
        solver = volkovSolver(
            poly=poly,
            boundary_conditions=boundary_conditions,
            is_dirichlet=is_dirichlet,
            delta=delta,
            n=n,
            max_iter=max_iter,
            radial_heuristic=rho,
        )
        solution = solver.solve(verbose=True)
        M = solver.M

        # --- Block covering figure ---
        fig, ax = plt.subplots(figsize=(5, 5))
        solver.plot_block_covering(ax=ax, show_boundary_conditions=False,
                                   show_quantized_boundaries=False)
        # Re-draw polygon edges on top so they aren't hidden by block circles
        verts_closed = np.vstack([vertices, vertices[0]])
        ax.plot(verts_closed[:, 0], verts_closed[:, 1], 'k-', linewidth=2, zorder=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        fname = os.path.join(output_folder, f'block_covering_radial_heuristic_{rho}.pdf')
        fig.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"    Saved {fname}  (M = {M})")

        # --- Approximate solution figure ---
        X, Y = solver._X, solver._Y
        mask = ~solution.mask

        # MSE vs analytical
        u_analytical = analytical_solution(X, Y)
        mse = np.mean((solution[mask] - u_analytical[mask]) ** 2)

        # Consistency error: 5-point stencil Laplacian norm
        # Only include points where all 4 stencil neighbors are also valid
        sol_data = solution.filled(0.0)
        inner = sol_data[1:-1, 1:-1]
        lap = (sol_data[1:-1, 2:] + sol_data[1:-1, :-2]
               + sol_data[2:, 1:-1] + sol_data[:-2, 1:-1]
               - 4 * inner) / (delta ** 2)
        valid = (mask[1:-1, 1:-1] & mask[1:-1, 2:] & mask[1:-1, :-2]
                 & mask[2:, 1:-1] & mask[:-2, 1:-1])
        n_valid = np.sum(valid)
        laplacian_norm = np.sqrt(np.sum(lap[valid] ** 2) / n_valid)

        fig, ax = plt.subplots(figsize=(5, 5))
        solver.plot_solution(ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        fname = os.path.join(output_folder, f'approximate_solution_radial_heuristic_{rho}.pdf')
        fig.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"    Saved {fname}")
        print(f"    M = {M}, MSE = {mse:.2e}, ||Δu||_2 ≈ {laplacian_norm:.3f}")

    print("\nDone. Update LaTeX subcaptions with the values above.")


def generate_mse_vs_n_fitted(output_folder):
    """
    Generate fig:mse_vs_n_fitted from the paper.

    MSE vs n on log-scale for 6 values of rho, with linear fits
    showing exponential error decay |u - u_exact| <= c*exp(-d*n).
    """
    import os
    os.makedirs(output_folder, exist_ok=True)

    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    sin_pi_x_coeffs = [0.02, 2.905, 1.153, -8.117, 4.059]
    boundary_conditions = [[0.0], [0.0], sin_pi_x_coeffs, [0.0]]
    is_dirichlet = [True] * 4

    def analytical_solution(X, Y):
        return np.sin(np.pi * X) * np.sinh(np.pi * Y) / np.sinh(np.pi)

    rho_values = [0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    n_values = list(range(5, 105, 5))
    max_iter = 5
    delta = 0.01

    fig, ax = plt.subplots(figsize=(6, 4.2))
    markers = ['o', 's', 'D', '^', 'v', 'X']

    print("Generating MSE vs n fitted plot...")
    for rho_idx, rho in enumerate(rho_values):
        mse_list = []
        print(f"\n  rho = {rho}")
        for n_val in n_values:
            poly = polygon(vertices)
            solver = volkovSolver(
                poly=poly,
                boundary_conditions=boundary_conditions,
                is_dirichlet=is_dirichlet,
                delta=delta, n=n_val, max_iter=max_iter,
                radial_heuristic=rho,
            )
            solution = solver.solve(verbose=False)
            X, Y = solver._X, solver._Y
            mask = ~solution.mask
            u_analytical = analytical_solution(X, Y)
            mse = np.mean((solution[mask] - u_analytical[mask]) ** 2)
            mse_list.append(mse)
            print(f"    n={n_val:3d}  MSE={mse:.2e}")

        mse_arr = np.array(mse_list)
        n_arr = np.array(n_values)

        # Plot data points only (no connecting lines)
        ax.semilogy(n_arr, mse_arr, marker=markers[rho_idx], label=f'$\\rho$={rho}',
                     markersize=5, linestyle='none')

        # Linear fit on log-scale (only on points above the noise floor)
        above_floor = mse_arr > 1e-6
        if np.sum(above_floor) >= 2:
            coeffs = np.polyfit(n_arr[above_floor], np.log10(mse_arr[above_floor]), 1)
            fit_line = 10 ** np.polyval(coeffs, n_arr)
            ax.semilogy(n_arr, fit_line, '--', color=ax.get_lines()[-1].get_color(),
                         alpha=0.5, linewidth=0.8)

    ax.set_xlabel('$n$', fontsize=16)
    ax.set_ylabel('Mean Squared Error', fontsize=16)
    ax.legend(fontsize=13)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(bottom=1e-7)

    fname = os.path.join(output_folder, 'MSE_vs_n_fitted_6_rho_5_max_iter.pdf')
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    print(f"\nSaved {fname}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "qualitative_rh":
        output = sys.argv[2] if len(sys.argv) > 2 else "plots"
        generate_qualitative_comparison_rh(output)
    elif len(sys.argv) > 1 and sys.argv[1] == "mse_vs_n":
        output = sys.argv[2] if len(sys.argv) > 2 else "plots"
        generate_mse_vs_n_fitted(output)
    else:
        main()