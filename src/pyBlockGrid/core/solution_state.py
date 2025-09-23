#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution state and data structures for the Volkov solver.

This module defines the data structures that hold all the solution-related
information during the Volkov block grid method computation. These structures
organize the numerous arrays and parameters needed for the algorithm.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np
from .polygon import polygon
from .block import block


@dataclass
class SolutionState:
    """
    Container for all solution-related data and grid information.

    This class organizes all the arrays and parameters needed during the
    Volkov method computation, making it easier to pass data between
    different components of the algorithm.

    Now supports polygons with holes by handling multiple boundaries.
    """

    # Grid information
    nx: int
    ny: int
    delta: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    X: np.ndarray
    Y: np.ndarray

    # Solution arrays
    solution: np.ma.MaskedArray
    inside_block_ids: np.ma.MaskedArray
    cartesian_grid: np.ma.MaskedArray

    # Block parameters
    alpha_mu: np.ndarray
    r0_mu: np.ndarray
    r_mu: np.ndarray
    center_mu: np.ndarray
    ref_theta: np.ndarray
    block_type: np.ndarray

    # Discretization parameters
    n_mu: np.ndarray
    beta_mu: np.ndarray
    theta_mu: np.ma.MaskedArray

    # Boundary data
    phi_ij: np.ma.MaskedArray
    block_nu: np.ma.MaskedArray
    boundary_estimates: np.ma.MaskedArray
    quantized_boundary_points: np.ma.MaskedArray

    # Intermediate calculations
    tau_block_ids: np.ma.MaskedArray
    tau_block_centers: np.ma.MaskedArray
    tau_ref_thetas: np.ma.MaskedArray
    tau_block_polar_coordinates: np.ma.MaskedArray
    tau_carrier_function_values: np.ma.MaskedArray
    mu_carrier_function_values: np.ma.MaskedArray
    poisson_kernel_values: np.ma.MaskedArray

    # NEW: Support for holes
    polygon: polygon  # Store the polygon with holes
    boundary_segments: List  # List of (boundary_type, boundary_id, edge_id) tuples
    boundary_conditions_flat: List  # Flattened boundary conditions for all boundaries
    is_dirichlet_flat: List  # Flattened Dirichlet flags for all boundaries
    block_boundary_info: np.ndarray  # Maps blocks to their boundary segments

    @classmethod
    def from_polygon_and_blocks(cls, poly: polygon, blocks: List[block], delta: float,
                               n: int, boundary_conditions: List[List[float]],
                               is_dirichlet: List[bool], tolerance: float = 1e-10):
        """
        Factory method to create solution state from polygon and blocks.

        This method initializes all the necessary arrays and parameters for the
        Volkov method computation based on the given polygon, blocks, and parameters.
        Now supports polygons with holes.

        Parameters:
            poly (polygon): The polygon domain (may contain holes)
            blocks (List[block]): List of blocks covering the domain
            delta (float): Grid spacing
            n (int): Angular discretization parameter
            boundary_conditions (List[List[float]]): Boundary conditions for main polygon edges
            is_dirichlet (List[bool]): Boundary condition types for main polygon edges
            tolerance (float): Numerical tolerance

        Returns:
            SolutionState: Initialized solution state object with hole support
        """
        # Calculate grid dimensions and bounds
        x_min, y_min = np.min(poly.vertices, axis=0)
        x_max, y_max = np.max(poly.vertices, axis=0)

        ny = int(np.ceil((y_max - y_min) / delta))
        nx = int(np.ceil((x_max - x_min) / delta))

        # Create grid points
        x = np.linspace(x_min + 0.5 * delta, x_max + 0.5 * delta, nx)
        y = np.linspace(y_min + 0.5 * delta, y_max + 0.5 * delta, ny)
        X, Y = np.meshgrid(x, y)

        # Create points array and mask (accounting for holes)
        points = np.stack([X.ravel(), Y.ravel()], axis=1)

        # Use _point_in_polygon for each point to properly handle holes
        mask = np.zeros(len(points), dtype=bool)
        for i, point in enumerate(points):
            mask[i] = poly._point_in_polygon(point[0], point[1])
        mask = mask.reshape(ny, nx)

        # Flatten boundary conditions from main polygon and holes
        boundary_segments = []
        boundary_conditions_flat = []
        is_dirichlet_flat = []

        # Main polygon boundaries
        for edge_id in range(len(poly.vertices)):
            boundary_segments.append(('main', 0, edge_id))
            boundary_conditions_flat.append(boundary_conditions[edge_id])
            is_dirichlet_flat.append(is_dirichlet[edge_id])

        # Hole boundaries
        for hole_id, hole in enumerate(poly.holes):
            for edge_id in range(hole.n_vertices):
                boundary_segments.append(('hole', hole_id, edge_id))
                boundary_conditions_flat.append(hole.boundary_conditions[edge_id])
                is_dirichlet_flat.append(hole.is_dirichlet[edge_id])

        # Initialize solution arrays
        solution = np.ma.masked_array(np.zeros((ny, nx)), mask=~mask)
        inside_block_ids = np.ma.masked_array(np.zeros((ny, nx), dtype=int), mask=~mask)
        cartesian_grid = np.ma.array(
            np.stack([X, Y], axis=2),
            mask=np.repeat(~mask[:, :, np.newaxis], 2, axis=2)
        )

        # Initialize block parameters
        num_blocks = len(blocks)
        alpha_mu = np.zeros(num_blocks)
        r0_mu = np.zeros(num_blocks)
        r_mu = np.zeros(num_blocks)
        center_mu = np.zeros((num_blocks, 2))
        ref_theta = np.zeros(num_blocks)
        block_type = np.zeros(num_blocks, dtype=int)

        # Initialize boundary condition arrays
        num_bounds = [len(x) for x in boundary_conditions]
        max_bounds = max(num_bounds) if num_bounds else 1
        phi_ij = np.ma.masked_array(np.zeros((num_blocks, max_bounds, 2)), mask=True)
        block_nu = np.ma.masked_array(np.zeros((num_blocks, 2), dtype=bool), mask=True)

        # Fill block parameter arrays
        for blk in blocks:
            alpha_mu[blk.id_] = blk.angle / np.pi
            r0_mu[blk.id_] = blk.r0
            center_mu[blk.id_] = blk.center
            r_mu[blk.id_] = blk.length
            block_type[blk.id_] = blk.block_kind

            if blk.block_kind == 1:
                # First kind block
                ref_theta[blk.id_] = np.arctan2(
                    poly.edges[blk.edge_j_index][1],
                    poly.edges[blk.edge_j_index][0]
                )
                phi_ij[blk.id_, :num_bounds[blk.edge_i_index], 0] = boundary_conditions[blk.edge_i_index]
                phi_ij.mask[blk.id_, num_bounds[blk.edge_i_index]:, 0] = True
                phi_ij[blk.id_, :num_bounds[blk.edge_j_index], 1] = boundary_conditions[blk.edge_j_index]
                phi_ij.mask[blk.id_, num_bounds[blk.edge_j_index]:, 1] = True
                block_nu[blk.id_, 0] = is_dirichlet[blk.edge_i_index]
                block_nu[blk.id_, 1] = is_dirichlet[blk.edge_j_index]
                block_nu.mask[blk.id_, :] = False
            elif blk.block_kind == 2:
                # Second kind block
                ref_theta[blk.id_] = np.arctan2(
                    poly.edges[blk.edge_i_index][1],
                    poly.edges[blk.edge_i_index][0]
                )
                phi_ij[blk.id_, :num_bounds[blk.edge_i_index], 0] = boundary_conditions[blk.edge_i_index]
                phi_ij.mask[blk.id_, num_bounds[blk.edge_i_index]:, 0] = True
                phi_ij.mask[blk.id_, :, 1] = True
                block_nu[blk.id_, 0] = is_dirichlet[blk.edge_i_index]
                block_nu.mask[blk.id_, 0] = False
                block_nu.mask[blk.id_, 1] = True
            else:
                # Third kind block - no boundary conditions
                block_nu.mask[blk.id_, :] = True
                phi_ij.mask[blk.id_, :, :] = True

        # Calculate discretization parameters
        n_mu = np.maximum(4, np.floor(n * alpha_mu))
        beta_mu = np.pi * np.divide(alpha_mu, n_mu)

        # Create theta_mu array with proper masking
        max_n = int(np.max(n_mu))
        theta_mu = np.ma.masked_array(
            np.matmul(beta_mu.reshape(-1, 1),
                     (np.arange(1, max_n + 1) - 0.5).reshape(1, -1)),
            mask=np.array([k >= n_mu[i] for i, k in
                          np.ndindex(num_blocks, max_n)]).reshape(num_blocks, -1)
        )

        # Initialize quantized boundary points
        quantized_boundary_points = np.repeat(theta_mu[:, :, np.newaxis], 2, axis=2)
        quantized_boundary_points.data[:, :, 0] = np.repeat(r0_mu.reshape(-1, 1), max_n, axis=1)

        # Convert to Cartesian coordinates
        from ..utils.coordinate_transforms import CoordinateTransforms
        coord_transform = CoordinateTransforms()
        quantized_boundary_points = coord_transform.polar_to_cartesian(
            quantized_boundary_points,
            center_mu[:, np.newaxis, :],
            ref_theta[:, np.newaxis]
        )

        # Initialize boundary estimates
        boundary_estimates = np.ma.zeros_like(theta_mu)

        # Initialize placeholder arrays for tau block calculations
        # These will be filled during the solution process
        tau_block_ids = np.ma.masked_array(np.zeros_like(theta_mu, dtype=int), mask=theta_mu.mask)
        tau_block_centers = np.ma.masked_array(np.zeros((*theta_mu.shape, 2)), mask=np.repeat(theta_mu.mask[:, :, np.newaxis], 2, axis=2))
        tau_ref_thetas = np.ma.masked_array(np.zeros_like(theta_mu), mask=theta_mu.mask)
        tau_block_polar_coordinates = np.ma.masked_array(np.zeros((*theta_mu.shape, 2)), mask=np.repeat(theta_mu.mask[:, :, np.newaxis], 2, axis=2))
        tau_carrier_function_values = np.ma.zeros_like(theta_mu)
        mu_carrier_function_values = np.ma.zeros_like(theta_mu)
        poisson_kernel_values = np.ma.zeros_like(theta_mu)

        # Initialize block boundary info (will be updated when blocks handle holes)
        block_boundary_info = np.zeros((num_blocks, 3), dtype=int)  # boundary_type, boundary_id, edge_id

        return cls(
            nx=nx, ny=ny, delta=delta, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
            X=X, Y=Y, solution=solution, inside_block_ids=inside_block_ids,
            cartesian_grid=cartesian_grid, alpha_mu=alpha_mu, r0_mu=r0_mu, r_mu=r_mu,
            center_mu=center_mu, ref_theta=ref_theta, block_type=block_type,
            n_mu=n_mu, beta_mu=beta_mu, theta_mu=theta_mu, phi_ij=phi_ij,
            block_nu=block_nu, boundary_estimates=boundary_estimates,
            quantized_boundary_points=quantized_boundary_points,
            tau_block_ids=tau_block_ids, tau_block_centers=tau_block_centers,
            tau_ref_thetas=tau_ref_thetas, tau_block_polar_coordinates=tau_block_polar_coordinates,
            tau_carrier_function_values=tau_carrier_function_values,
            mu_carrier_function_values=mu_carrier_function_values,
            poisson_kernel_values=poisson_kernel_values,
            polygon=poly, boundary_segments=boundary_segments,
            boundary_conditions_flat=boundary_conditions_flat,
            is_dirichlet_flat=is_dirichlet_flat,
            block_boundary_info=block_boundary_info
        )