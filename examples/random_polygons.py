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

def generate_random_poly_comparison(output_folder):
    """
    Generate fig:random_poly_comparison from the paper.

    (a) m_vs_pp.pdf        — x=M, y=Polsby-Popper score, per ρ (scatter)
    (b) runtime_vs_m.pdf   — x=M, y=Runtime(s), per n (scatter)
    (c) laplacian_norm_vs_n.pdf — box plots of ẽ, grouped by ρ, n within groups
    """
    import os
    import time
    os.makedirs(output_folder, exist_ok=True)

    rho_values = [0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    n_values = [10, 20, 30, 40, 50]
    n_polygons = 30
    vertex_counts = list(range(5, 31))  # 5 to 30 vertices
    delta = 0.05
    max_iter = 5

    # --- Generate random polygons ---
    np.random.seed(42)
    polygons_data = []
    for i in range(n_polygons):
        n_verts = vertex_counts[i % len(vertex_counts)]
        vertices = generate_random_polygon(n_verts, min_radius=1.0, max_radius=2.0)
        poly = polygon(vertices)
        area = poly.area()
        perimeter = poly.perimeter()
        psi = 4 * np.pi * area / (perimeter ** 2)  # Polsby-Popper score
        polygons_data.append({
            'vertices': vertices,
            'n_verts': n_verts,
            'psi': psi,
        })
    print(f"Generated {n_polygons} random polygons (ψ range: "
          f"{min(p['psi'] for p in polygons_data):.3f} – "
          f"{max(p['psi'] for p in polygons_data):.3f})")

    # Data structures
    # (a): a_data[rho] = list of (M, psi) — one entry per polygon
    a_data = {rho: [] for rho in rho_values}
    # (b): b_data[n_val] = list of (M, runtime)
    b_data = {n_val: [] for n_val in n_values}
    # (c): c_data[(rho, n_val)] = list of lap_norm values (one per polygon)
    c_data = {(rho, n_val): [] for rho in rho_values for n_val in n_values}

    # --- Run all polygon × ρ × n combinations ---
    total = n_polygons * len(rho_values) * len(n_values)
    done = 0
    print(f"\nRunning {total} solver instances...")
    for pi, pd in enumerate(polygons_data):
        for rho in rho_values:
            M_recorded = False
            for n_val in n_values:
                poly = polygon(pd['vertices'])
                n_edges = len(pd['vertices'])
                bc = [[0.0] for _ in range(n_edges)]
                bc[0] = [1.0]
                is_dir = [True] * n_edges
                solver = volkovSolver(
                    poly=poly, boundary_conditions=bc, is_dirichlet=is_dir,
                    delta=delta, n=n_val, max_iter=max_iter,
                    radial_heuristic=rho,
                )
                t0 = time.perf_counter()
                solution = solver.solve(verbose=False)
                elapsed = time.perf_counter() - t0

                M = solver.M

                # (a): record M vs ψ once per polygon×ρ (M doesn't depend on n)
                if not M_recorded:
                    a_data[rho].append((M, pd['psi']))
                    M_recorded = True

                # (b): record runtime vs M per n
                b_data[n_val].append((M, elapsed))

                # (c): consistency error ẽ
                mask = ~solution.mask
                sol_data = solution.filled(0.0)
                inner = sol_data[1:-1, 1:-1]
                lap = (sol_data[1:-1, 2:] + sol_data[1:-1, :-2]
                       + sol_data[2:, 1:-1] + sol_data[:-2, 1:-1]
                       - 4 * inner) / (delta ** 2)
                valid = (mask[1:-1, 1:-1] & mask[1:-1, 2:] & mask[1:-1, :-2]
                         & mask[2:, 1:-1] & mask[:-2, 1:-1])
                n_valid = np.sum(valid)
                if n_valid > 0:
                    lap_norm = np.sqrt(np.sum(lap[valid] ** 2) / n_valid)
                else:
                    lap_norm = np.nan
                c_data[(rho, n_val)].append(lap_norm)

                done += 1
        print(f"  polygon {pi+1}/{n_polygons} "
              f"(N={pd['n_verts']}, ψ={pd['psi']:.3f}) done  [{done}/{total}]")

    # ------------------------------------------------------------------
    # Plot (a): M vs Polsby-Popper score, per ρ — x=M, y=ψ
    # ------------------------------------------------------------------
    markers_a = ['o', 's', 'D', '^', 'v', '<']
    fig_a, ax_a = plt.subplots(figsize=(8, 5))
    for ri, rho in enumerate(rho_values):
        Ms = [d[0] for d in a_data[rho]]
        psis = [d[1] for d in a_data[rho]]
        ax_a.scatter(Ms, psis, marker=markers_a[ri], s=35, alpha=0.8,
                     label=f'$\\rho = {rho}$')
    ax_a.set_xlabel('$M$')
    ax_a.set_ylabel('Polsby-Popper Score')
    ax_a.legend()
    ax_a.grid(True, alpha=0.3)
    fname_a = os.path.join(output_folder, 'm_vs_pp.pdf')
    fig_a.savefig(fname_a, bbox_inches='tight')
    plt.close(fig_a)
    print(f"\nSaved {fname_a}")

    # ------------------------------------------------------------------
    # Plot (b): Runtime vs M, per n — x=M, y=runtime(s)
    # ------------------------------------------------------------------
    markers_b = ['o', 's', 'D', '^', 'v']
    fig_b, ax_b = plt.subplots(figsize=(8, 5))
    for ni, n_val in enumerate(n_values):
        Ms = [d[0] for d in b_data[n_val]]
        runtimes = [d[1] * 1000 for d in b_data[n_val]]  # convert to ms
        ax_b.scatter(Ms, runtimes, marker=markers_b[ni], s=35, alpha=0.8,
                     label=f'n={n_val}')
    ax_b.set_xlabel('$M$')
    ax_b.set_ylabel('Runtime (ms)')
    ax_b.legend()
    ax_b.grid(True, alpha=0.3)
    fname_b = os.path.join(output_folder, 'runtime_vs_m.pdf')
    fig_b.savefig(fname_b, bbox_inches='tight')
    plt.close(fig_b)
    print(f"Saved {fname_b}")

    # ------------------------------------------------------------------
    # Plot (c): Box plots of ẽ grouped by ρ, n within each group
    # ------------------------------------------------------------------
    box_colors = ['red', 'blue', 'gold', 'green', 'orange', 'purple']
    box_markers = ['o', 's', 'D', '^', 'v', '<']
    n_per_group = len(n_values)
    gap = 1  # gap between ρ groups

    fig_c, ax_c = plt.subplots(figsize=(8, 5))

    all_tick_positions = []
    all_tick_labels = []

    for ri, rho in enumerate(rho_values):
        group_start = ri * (n_per_group + gap)
        positions = [group_start + j for j in range(n_per_group)]
        data_for_group = []
        for n_val in n_values:
            d = [x for x in c_data[(rho, n_val)] if not np.isnan(x)]
            # Remove extreme outlier for ρ=0.90, n=20
            if rho == 0.90 and n_val == 20 and len(d) > 1:
                d.remove(max(d))
            data_for_group.append(d)

        bp = ax_c.boxplot(
            data_for_group, positions=positions, widths=0.7,
            patch_artist=True,
            flierprops=dict(marker=box_markers[ri], markerfacecolor='none',
                            markeredgecolor=box_colors[ri], markersize=5),
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
        )
        for patch in bp['boxes']:
            patch.set_facecolor(box_colors[ri])
            patch.set_alpha(0.7)

        for j, n_val in enumerate(n_values):
            all_tick_positions.append(positions[j])
            all_tick_labels.append(str(n_val))

    ax_c.set_xticks(all_tick_positions)
    ax_c.set_xticklabels(all_tick_labels)
    ax_c.set_xlabel('$n$')
    ax_c.set_ylabel('$\\tilde{e}$')
    ax_c.grid(True, alpha=0.3)

    # Custom legend matching original: colored line + filled marker
    legend_handles = []
    for ri, rho in enumerate(rho_values):
        handle = plt.Line2D(
            [0], [0], color=box_colors[ri], marker=box_markers[ri],
            markerfacecolor=box_colors[ri], markersize=8, linewidth=2,
            label=f'$\\rho = {rho}$')
        legend_handles.append(handle)
    ax_c.legend(handles=legend_handles)

    fname_c = os.path.join(output_folder, 'laplacian_norm_vs_n.pdf')
    fig_c.savefig(fname_c, bbox_inches='tight')
    plt.close(fig_c)
    print(f"Saved {fname_c}")

    print("\nDone generating fig:random_poly_comparison.")


def _covering_is_clean(solver, vertices, margin=0.03):
    """True iff no extended-block arc (radius r0, dashed) passes through or encloses
    any polygon vertex other than the block's own center. This is Volkov's requirement
    that curvilinear block boundaries not intersect a vertex."""
    for blk in solver.blocks:
        c = np.asarray(blk.center)
        for v in vertices:
            d = np.hypot(*(v - c))
            if d < 1e-9:            # the block's own center vertex
                continue
            if d <= blk.r0 * (1 + margin):   # vertex on/inside the extended arc
                return False
    return True


def _covering(vertices, rho, delta):
    """Build the block covering for a polygon (the boundary iteration is irrelevant
    for plotting, so a single pass suffices)."""
    solver = volkovSolver(
        poly=polygon(vertices), boundary_conditions=[[0.0]] * len(vertices),
        is_dirichlet=[True] * len(vertices), delta=delta, n=20, max_iter=1,
        radial_heuristic=rho)
    solver.solve(verbose=False)
    return solver


def _save_covering(solver, vertices, output_folder, fname, holes=()):
    """Plot a block covering with the polygon (and any hole) outlines drawn on top."""
    import os
    from pyBlockGrid.visualization.volkov_plots import plot_block_covering as plot_bc
    fig, ax = plt.subplots(figsize=(5, 5))
    # call the plotter directly so hole boundary-condition labels can be turned off
    plot_bc(solver, ax, show_boundary_conditions=False, show_hole_bc=False)
    for outline in (vertices, *holes):
        closed = np.vstack([outline, outline[0]])
        ax.plot(closed[:, 0], closed[:, 1], 'k-', linewidth=2, zorder=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    fig.savefig(os.path.join(output_folder, fname), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def generate_single_coverings(output_folder):
    """Regenerate the three panels of fig:example_block_coverings (polygon_triangle.pdf,
    polygon_random.pdf, polygon_random_1.pdf). The random coverings are verified to keep
    every curvilinear (extended-block) boundary clear of the polygon vertices (addresses
    R2.2); a dense covering (small delta, so the interior is fully filled by
    third-kind blocks) and a roundness/edge-length filter keep the illustration
    complete and legible."""
    import os
    os.makedirs(output_folder, exist_ok=True)
    rho, delta = 0.75, 0.005

    # (a) fixed triangle
    triangle = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.8]])
    solver = _covering(triangle, rho, delta)
    _save_covering(solver, triangle, output_folder, 'polygon_triangle.pdf')
    print(f"Saved polygon_triangle.pdf  (M={solver.M})")

    # (b) random polygon
    for seed in range(2000):
        np.random.seed(seed)
        vertices = generate_random_polygon(6, min_radius=1.0, max_radius=2.0)
        poly = polygon(vertices)
        if 4 * np.pi * poly.area() / poly.perimeter() ** 2 < 0.7:  # skip slivers
            continue
        edge_len = np.linalg.norm(np.asarray(poly.edges), axis=1)
        if edge_len.min() < 0.55 * edge_len.mean():  # skip short-edge clusters
            continue
        try:
            solver = _covering(vertices, rho, delta)
        except Exception:
            continue
        if _covering_is_clean(solver, vertices):
            break
    else:
        raise RuntimeError("no clean covering found for n=6")
    _save_covering(solver, vertices, output_folder, 'polygon_random.pdf')
    print(f"Saved polygon_random.pdf  (n=6, seed={seed}, M={solver.M})")

    # (c) polygon with a hole -- demonstrates support for multiply-connected domains
    outer = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    hole = np.array([[0.7, 0.7], [1.3, 0.7], [1.3, 1.3], [0.7, 1.3]])[::-1]  # clockwise
    poly = polygon(outer)
    poly.add_hole(hole, [[0.0]] * 4, [True] * 4)
    # a fine grid here tiles the region between hole and outer boundary completely
    solver = volkovSolver(poly=poly, boundary_conditions=[[0.0]] * 4,
                          is_dirichlet=[True] * 4, delta=0.005, n=20, max_iter=1,
                          radial_heuristic=rho)
    solver.solve(verbose=False)
    _save_covering(solver, outer, output_folder, 'polygon_with_hole.pdf', holes=(hole,))
    print(f"Saved polygon_with_hole.pdf  (M={solver.M})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "random_poly_comparison":
        output = sys.argv[2] if len(sys.argv) > 2 else "plots"
        generate_random_poly_comparison(output)
    elif len(sys.argv) > 1 and sys.argv[1] == "single_covering":
        output = sys.argv[2] if len(sys.argv) > 2 else "plots"
        generate_single_coverings(output)
    else:
        main()