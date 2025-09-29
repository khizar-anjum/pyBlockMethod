#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volkov-specific visualization functions.

This module provides specialized plotting functions for visualizing the
Volkov block grid method results, including block coverings, solution
heatmaps, and gradient fields.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_block_covering(
    solver,
    ax,
    uncovered_points=None,
    show_boundary_conditions=True,
    show_quantized_boundaries=False,
    show_holes=True,
    show_hole_bc=True,
):
    """
    Visualize the block covering of the domain with holes support.

    Shows blocks with colors based on block type:
    - First kind blocks (vertex sectors): Blue
    - Second kind blocks (edge half-disks): Red
    - Third kind blocks (interior disks): Green

    Blocks from both main and hole boundaries use the same color scheme.

    Parameters:
        solver: Volkov solver instance with computed blocks
        ax: Matplotlib axis to plot on
        uncovered_points: Optional array of uncovered points to highlight
        show_boundary_conditions (bool): Whether to show main boundary condition labels
        show_quantized_boundaries (bool): Whether to show discretized boundary points
        show_holes (bool): Whether to show hole boundaries
        show_hole_bc (bool): Whether to show hole boundary condition labels
    """
    # Plot the polygon boundary (main + holes)
    solver.poly.plot(ax=ax, show_holes=show_holes)

    # Plot each block according to its type and boundary
    for i, blk in enumerate(solver.blocks):
        # Determine colors based on boundary type
        boundary_type = getattr(blk, "boundary_type", "main")

        if blk.block_kind == 1:
            # First kind blocks (sectors from vertices) - Blue
            _plot_first_kind_block(
                ax, blk, solver.poly, solver.state, i, show_quantized_boundaries
            )
        elif blk.block_kind == 2:
            # Second kind blocks (half-disks from edges) - Red
            _plot_second_kind_block(
                ax, blk, solver.poly, solver.state, i, show_quantized_boundaries
            )
        elif blk.block_kind == 3:
            # Third kind blocks (interior disks) - Green
            _plot_third_kind_block(ax, blk, solver.state, i, show_quantized_boundaries)

    # Show boundary conditions for main polygon
    if show_boundary_conditions:
        _plot_boundary_conditions(ax, solver.poly, solver.boundary_conditions)

    # Show boundary conditions for holes
    if show_hole_bc and solver.poly.holes:
        _plot_hole_boundary_conditions(ax, solver.poly)

    if uncovered_points is not None:
        _plot_uncovered_points(ax, uncovered_points, solver.delta)

    ax.set_aspect("equal")
    ax.grid(True)


def _plot_first_kind_block(
    ax, blk, poly, state, block_index, show_quantized_boundaries
):
    """Plot a first kind block (vertex sector)."""
    # Handle both main polygon and hole boundaries
    boundary_type = getattr(blk, "boundary_type", "main")

    if boundary_type == "main":
        edge_angle = np.arctan2(
            poly.edges[blk.edge_j_index][1], poly.edges[blk.edge_j_index][0]
        )
    else:  # hole boundary
        # For holes, we need to get the edge angle from the hole
        hole_id = getattr(blk, "boundary_id", 0)
        vertex_id = getattr(blk, "vertex_id", 0)
        if hole_id < len(poly.holes):
            hole = poly.holes[hole_id]
            edge_angle = np.arctan2(hole.edges[vertex_id][1], hole.edges[vertex_id][0])
        else:
            edge_angle = 0  # fallback

    # Create sector
    theta = np.linspace(edge_angle, edge_angle + blk.angle, 100)

    # Plot inner radius r (solid line)
    r = blk.length
    x = blk.center[0] + r * np.cos(theta)
    y = blk.center[1] + r * np.sin(theta)
    ax.plot(x, y, "b-", alpha=0.5)

    # Plot outer radius r0 (dashed line)
    r0 = blk.r0
    x0 = blk.center[0] + r0 * np.cos(theta)
    y0 = blk.center[1] + r0 * np.sin(theta)
    ax.plot(x0, y0, "b--", alpha=0.5)

    # Plot block center and label
    ax.plot(blk.center[0], blk.center[1], "b.")
    ax.text(
        blk.center[0],
        blk.center[1],
        f"$P_{{{blk.id_ + 1}}}$",
        horizontalalignment="right",
        verticalalignment="bottom",
    )

    # Plot quantized boundaries with solution values if requested
    if show_quantized_boundaries and state is not None:
        _plot_quantized_boundaries(ax, state, blk.id_, "g")


def _plot_second_kind_block(
    ax, blk, poly, state, block_index, show_quantized_boundaries
):
    """Plot a second kind block (edge half-disk)."""
    # Handle both main polygon and hole boundaries
    boundary_type = getattr(blk, "boundary_type", "main")

    if boundary_type == "main":
        edge_angle = np.arctan2(
            poly.edges[blk.edge_i_index][1], poly.edges[blk.edge_i_index][0]
        )
    else:  # hole boundary
        # For holes, we need to get the edge angle from the hole
        hole_id = getattr(blk, "boundary_id", 0)
        edge_i_index = blk.edge_i_index
        if hole_id < len(poly.holes):
            hole = poly.holes[hole_id]
            edge_angle = np.arctan2(
                hole.edges[edge_i_index][1], hole.edges[edge_i_index][0]
            )
        else:
            edge_angle = 0  # fallback

    # Create half-disk
    theta = np.linspace(edge_angle, edge_angle + np.pi, 100)

    # Plot inner radius r (solid line)
    r = blk.length
    x = blk.center[0] + r * np.cos(theta)
    y = blk.center[1] + r * np.sin(theta)
    ax.plot(x, y, "r-", alpha=0.5)

    # Plot outer radius r0 (dashed line)
    r0 = blk.r0
    x0 = blk.center[0] + r0 * np.cos(theta)
    y0 = blk.center[1] + r0 * np.sin(theta)
    ax.plot(x0, y0, "r--", alpha=0.5)

    # Plot block center and label
    ax.plot(blk.center[0], blk.center[1], "r.")
    ax.text(
        blk.center[0],
        blk.center[1],
        f"$P_{{{blk.id_ + 1}}}$",
        horizontalalignment="right",
        verticalalignment="bottom",
    )

    # Plot quantized boundaries if requested
    if show_quantized_boundaries and state is not None:
        _plot_quantized_boundaries(ax, state, blk.id_, "g")


def _plot_third_kind_block(ax, blk, state, block_index, show_quantized_boundaries):
    """Plot a third kind block (interior disk)."""
    # Create full disk
    theta = np.linspace(0, 2 * np.pi, 100)

    # Plot inner radius r (solid line)
    r = blk.length
    x = blk.center[0] + r * np.cos(theta)
    y = blk.center[1] + r * np.sin(theta)
    ax.plot(x, y, "g-", alpha=0.5)

    # Plot outer radius r0 (dashed line)
    r0 = blk.r0
    x0 = blk.center[0] + r0 * np.cos(theta)
    y0 = blk.center[1] + r0 * np.sin(theta)
    ax.plot(x0, y0, "g--", alpha=0.5)

    # Plot block center and label
    ax.plot(blk.center[0], blk.center[1], "g.")
    ax.text(
        blk.center[0],
        blk.center[1],
        f"$P_{{{blk.id_ + 1}}}$",
        horizontalalignment="right",
        verticalalignment="bottom",
    )

    # Plot quantized boundaries if requested
    if show_quantized_boundaries and state is not None:
        _plot_quantized_boundaries(ax, state, blk.id_, "g")


def _plot_quantized_boundaries(ax, state, block_id, color):
    """Plot quantized boundary points for a block."""
    if hasattr(state, "quantized_boundary_points") and hasattr(
        state, "boundary_estimates"
    ):
        # Get unmasked points for this block
        if block_id < state.quantized_boundary_points.shape[0]:
            points = state.quantized_boundary_points[block_id][
                ~state.quantized_boundary_points[block_id].mask.any(axis=1)
            ]
            if len(points) > 0:
                ax.plot(points[:, 0], points[:, 1], f"{color}.")
                for j, point in enumerate(points):
                    if (
                        block_id < state.boundary_estimates.shape[0]
                        and j < state.boundary_estimates.shape[1]
                    ):
                        ax.text(
                            point[0],
                            point[1],
                            f"{state.boundary_estimates[block_id, j]:.2f}",
                            fontsize=8,
                            horizontalalignment="right",
                        )


def _plot_boundary_conditions(ax, poly, boundary_conditions):
    """Plot boundary condition labels on edges."""
    for i in range(len(poly.vertices)):
        # Get midpoint of edge for text placement
        midpoint = (poly.vertices[i] + poly.vertices[(i + 1) % len(poly.vertices)]) / 2

        # Calculate edge direction angle
        edge = poly.edges[i]
        angle = np.arctan2(edge[1], edge[0]) * 180 / np.pi

        # Add boundary condition value as text, rotated to match edge direction
        ax.text(
            midpoint[0],
            midpoint[1],
            f"$\\phi_{{{i + 1}}} = {boundary_conditions[i]}$",
            horizontalalignment="left",
            verticalalignment="top",
            rotation=angle,
        )


def _plot_hole_boundary_conditions(ax, poly):
    """Plot boundary condition labels on hole edges."""
    for hole_id, hole in enumerate(poly.holes):
        for i in range(len(hole.vertices)):
            # Get midpoint of hole edge for text placement
            midpoint = (
                hole.vertices[i] + hole.vertices[(i + 1) % len(hole.vertices)]
            ) / 2
            # Calculate edge direction angle
            edge = hole.edges[i]
            angle = np.arctan2(edge[1], edge[0]) * 180 / np.pi
            # Add hole boundary condition value as text
            bc_text = (
                f"$\\phi_{{H{hole_id + 1},{i + 1}}} = {hole.boundary_conditions[i]}$"
            )
            ax.text(
                midpoint[0],
                midpoint[1],
                bc_text,
                horizontalalignment="center",
                verticalalignment="center",
                rotation=angle,
                fontsize=7,
            )


def _plot_uncovered_points(ax, uncovered_points, delta):
    """Plot uncovered points as red dots."""
    y_coords, x_coords = np.where(~uncovered_points.mask)
    if len(x_coords) > 0:
        ax.scatter(x_coords * delta, y_coords * delta, c="red", s=1, alpha=0.5)


def plot_solution_heatmap(solver, ax, vmin=None, vmax=None):
    """
    Plot the solution as a heatmap with proper masking.

    Parameters:
        solver: Volkov solver instance with computed solution
        ax: Matplotlib axis to plot on
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Plot heatmap with masked data
    im = ax.pcolormesh(
        solver.state.X,
        solver.state.Y,
        solver.solution,
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    # Use axes divider to create colorbar with exact same height as plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)

    ax.set_aspect("equal")


def plot_gradient_field(solver, ax, decimation_factor=2, scale=20):
    """
    Plot the gradient field of the solution.

    Parameters:
        solver: Volkov solver instance with computed solution
        ax: Matplotlib axis to plot on
        decimation_factor (int): Factor to reduce vector density
        scale (float): Scale factor for arrow size
    """
    # Calculate gradients using np.gradient
    dy, dx = np.gradient(solver.solution)

    # Normalize the gradient field
    magnitude = np.sqrt(dx**2 + dy**2)
    # Avoid division by zero
    magnitude = np.where(magnitude == 0, 1, magnitude)
    dx_norm = dx / magnitude
    dy_norm = dy / magnitude

    # Plot normalized vector field using quiver
    ax.quiver(
        solver.state.cartesian_grid[:, :, 0][::decimation_factor, ::decimation_factor],
        solver.state.cartesian_grid[:, :, 1][::decimation_factor, ::decimation_factor],
        dx_norm[::decimation_factor, ::decimation_factor],
        dy_norm[::decimation_factor, ::decimation_factor],
        scale=scale,
    )
    ax.set_aspect("equal")
