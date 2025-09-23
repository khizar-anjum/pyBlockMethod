#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boundary solution estimation algorithm for the Volkov block grid method.

This module implements the iterative boundary estimation algorithm as described in
E.A. Volkov's book "Block Method for Solving the Laplace Equation and for
Constructing Conformal Mappings (1994)" ISBN: 0849394066, specifically
equations (4.14) and (4.15).

The boundary estimator iteratively refines the solution values on the curved
boundaries of blocks until convergence.
"""

import numpy as np


class BoundaryEstimator:
    """
    Iteratively estimates solution on curved boundaries of blocks.

    This class implements the core algorithm for estimating the solution on
    the boundaries of the extended blocks (T_mu) using carrier functions
    and Poisson kernels.
    """

    def __init__(self, carrier_functions, poisson_kernels):
        """
        Initialize the boundary estimator with mathematical components.

        Parameters:
            carrier_functions: CarrierFunctions instance for carrier calculations
            poisson_kernels: PoissonKernels instance for kernel calculations
        """
        self.carrier_calc = carrier_functions
        self.poisson_calc = poisson_kernels

    def estimate(self, state, max_iter=100):
        """
        Iteratively estimate boundary solution using equations (4.14) and (4.15).

        This method performs the iterative refinement of boundary values on the
        curved boundaries of blocks. The iteration continues until max_iter is reached.

        Parameters:
            state: Solution state object containing all necessary arrays and parameters
            max_iter (int): Maximum number of iterations

        The method updates state.boundary_estimates in place.
        """
        iter_count = 0
        N, M = state.boundary_estimates.shape

        while True:
            for i in range(N):
                for j in range(M):
                    # Skip masked points or points without tau blocks
                    if state.boundary_estimates.mask[i, j] or state.tau_block_ids.mask[i, j]:
                        continue

                    tau_block_id = state.tau_block_ids[i, j]
                    beta_tau = state.beta_mu[tau_block_id]
                    poisson_kernel_vector = state.poisson_kernel_values[i, j]

                    # Apply equation (4.14) from Volkov's book
                    # This estimates the solution at point P_mu that lies on the boundary
                    # of extended block T_mu and inside basic block T_tau
                    state.boundary_estimates[i, j] = state.tau_carrier_function_values[i, j] + \
                        beta_tau * np.sum((state.boundary_estimates[tau_block_id] -
                                          state.mu_carrier_function_values[tau_block_id]) * \
                                         poisson_kernel_vector / max(1.0, beta_tau * np.sum(poisson_kernel_vector)))

            iter_count += 1
            if iter_count > max_iter:
                break

    def verify_unique_solution(self, poisson_kernel_values, beta_mu, tau_block_ids, theta_mu, M):
        """
        Verify the existence of a unique solution for the given Poisson kernel values.

        This implements Lemma 4.2 and equation (4.12) from Volkov's book to ensure
        that the iterative process will converge to a unique solution.

        Parameters:
            poisson_kernel_values: Array of Poisson kernel values
            beta_mu: Array of beta parameters for blocks
            tau_block_ids: Array mapping boundary points to tau blocks
            theta_mu: Angular discretization array
            M: Total number of blocks

        Returns:
            bool: True if a unique solution exists, False otherwise
        """
        epsilon = np.ma.ones_like(theta_mu)
        beta_tau = beta_mu[tau_block_ids]

        for _ in range(M + 1):
            epsilon = beta_tau * np.sum(poisson_kernel_values * epsilon[:, :, np.newaxis], axis=2)

        if np.max(np.abs(epsilon)) < 1.0:
            return True
        else:
            return False