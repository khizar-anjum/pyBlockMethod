#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interior point solution algorithm for the Volkov block grid method.

This module implements the interior solution algorithm as described in
E.A. Volkov's book "Block Method for Solving the Laplace Equation and for
Constructing Conformal Mappings (1994)" ISBN: 0849394066, specifically
equation (5.1).

The interior solver computes the solution at all interior grid points using
the carrier functions, Poisson kernels, and boundary estimates.
"""

import numpy as np


class InteriorSolver:
    """
    Computes solution at interior grid points using equation (5.1).

    This class implements the final step of the Volkov method, computing
    the solution at all interior points by combining carrier functions
    and Poisson kernels with the boundary estimates.
    """

    def __init__(self, carrier_functions, poisson_kernels):
        """
        Initialize the interior solver with mathematical components.

        Parameters:
            carrier_functions: CarrierFunctions instance for carrier calculations
            poisson_kernels: PoissonKernels instance for kernel calculations
        """
        self.carrier_calc = carrier_functions
        self.poisson_calc = poisson_kernels

    def solve(self, state, tolerance=1e-10):
        """
        Compute interior solution using equation (5.1) from Volkov's book.

        This method combines carrier functions (Q) and Poisson kernels (R)
        with boundary estimates to compute the final solution at all interior
        grid points.

        Parameters:
            state: Solution state object containing all necessary arrays and parameters
            tolerance (float): Tolerance for checking points at the pole

        Returns:
            np.ma.MaskedArray: Solution array with values at interior points
        """
        # First, compute Q and R arrays for all interior points
        inner_sol_Q, inner_sol_R = self._find_inner_points_Q_and_R(state, tolerance)

        # Apply equation (5.1) from Volkov's book to compute final solution
        N, M = state.solution.shape
        for i in range(N):
            for j in range(M):
                if not state.solution.mask[i, j]:
                    block_id = state.inside_block_ids[i, j]
                    state.solution[i, j] = inner_sol_Q[i, j] + state.beta_mu[block_id] * \
                        np.sum((state.boundary_estimates[block_id] -
                               state.mu_carrier_function_values[block_id]) * \
                               inner_sol_R[i, j])

        return state.solution

    def _find_inner_points_Q_and_R(self, state, tolerance):
        """
        Find the inner points Q (carrier function) and R (Poisson kernel).

        This implements the computations needed for equation (5.1) in the book.
        For each interior grid point, we calculate:
        - Q: The carrier function value at that point
        - R: The Poisson kernel values relating that point to boundary points

        Parameters:
            state: Solution state object
            tolerance (float): Tolerance for checking points at the pole

        Returns:
            tuple: (Q, R) where Q is carrier function array and R is Poisson kernel array
        """
        Q = np.ma.zeros_like(state.solution)
        R = state.theta_mu[state.inside_block_ids]  # Initialize with proper shape
        N, M = Q.shape
        k = np.arange(state.phi_ij.shape[1])

        # Get polar coordinates for all interior points
        points_polar = self._get_inner_points_in_polar_coordinates(state)

        # Pre-compute arrays for vectorized operations
        boundary_identifiers = np.squeeze(state.block_nu.dot(np.array([[2], [1]])))[state.inside_block_ids]
        block_kinds = state.block_type[state.inside_block_ids]
        nu_ij = state.block_nu[state.inside_block_ids]
        phi_ij = state.phi_ij[state.inside_block_ids]
        r0_ = state.r0_mu[state.inside_block_ids]
        eta_ij = state.theta_mu[state.inside_block_ids]
        alpha_ij = state.alpha_mu[state.inside_block_ids]

        for i in range(N):
            for j in range(M):
                # Skip masked points or points at the pole
                if state.inside_block_ids.mask[i, j] or points_polar[i, j, 0] < tolerance:
                    continue

                # Calculate carrier function value (Q)
                Q[i, j] = self.carrier_calc.calculate(
                    block_kinds[i, j],
                    boundary_identifiers[i, j],
                    points_polar[i, j, 0],
                    points_polar[i, j, 1],
                    k,
                    phi_ij[i, j, :, 0],
                    phi_ij[i, j, :, 1],
                    alpha_ij[i, j]
                )

                # Calculate Poisson kernel value (R)
                R[i, j] = self.poisson_calc.calculate(
                    block_kinds[i, j],
                    nu_ij[i, j, 0],
                    nu_ij[i, j, 1],
                    points_polar[i, j, 0],
                    r0_[i, j],
                    points_polar[i, j, 1],
                    eta_ij[i, j],
                    alpha_ij[i, j]
                )

        return Q, R

    def _get_inner_points_in_polar_coordinates(self, state):
        """
        Convert interior grid points to polar coordinates relative to their containing blocks.

        Parameters:
            state: Solution state object

        Returns:
            np.ma.array: Array of polar coordinates [r, theta] for each grid point
        """
        # Get points relative to block centers
        points = state.cartesian_grid - state.center_mu[state.inside_block_ids]
        ref_theta = state.ref_theta[state.inside_block_ids]

        # Convert to polar coordinates with proper angle normalization
        return np.ma.array(
            np.stack([
                np.linalg.norm(points, axis=2),
                np.mod(np.arctan2(points[:, :, 1], points[:, :, 0]) - ref_theta, 2 * np.pi)
            ], axis=2),
            mask=points.mask
        )