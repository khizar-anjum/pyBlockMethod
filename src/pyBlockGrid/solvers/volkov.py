#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pyBlockGrid.core.block import block
from pyBlockGrid.core.polygon import polygon
from .base import PDESolver

class volkovSolver(PDESolver):
    def __init__(self, poly : polygon, boundary_conditions : list[list[float]] = None, is_dirichlet : list[bool] = None,
                 n : int = 10, delta : float = 0.01, max_iter : int = 100, radial_heuristic : float = 0.8,
                 overlap_heuristic : float = 0.1, tolerance : float = 1e-10, verify_solution : bool = False):
        """
        Initialize the volkovSolver class. Solves using block grid method by E. Volkov, as outlined in the
        book "Block Method for Solving the Laplace Equation and for Constructing Conformal Mappings (1994)" ISBN: 0849394066
        The basic boundary-value problem is given by:
            -\\Delta u = 0 in the domain \\Omega
            \\nu_j * u + (1-\\nu_j) * \\partial_j u = \\phi_j on the boundary \\Gamma_j, j = 1, ..., N
        where \\nu_j is a boolean parameter that specifies the type of boundary condition on the jth edge. The value of \\nu_j is 
        independently chosen for each edge. When \\nu_1 = \\nu_2 = ... = \\nu_N = 1, we recover the standard Dirichlet problem, and when
        \\nu_1 = \\nu_2 = ... = \\nu_N = 0, we recover the standard Neumann problem. \\phi_j is the polynomial that defines the boundary
        condition on the jth edge. See page 7 in the book for more details.

        Parameters:
        poly (polygon): The polygon to solve the Laplace equation on. The polygon also has the boundary conditions \\phi_j for the jth edge.
        boundary_conditions (list[list[float]]): The boundary conditions for the polygon. List of List of floats, where the inner
            list contains the coefficients of the polynomials (\\phi_j) that define the boundary conditions. The list has a length equal
            to the number of edges of the polygon. 
        is_dirichlet (list[bool]): The parameter \\nu_j for the Laplace equation. List of booleans, where the jth boolean indicates the type of 
            boundary condition on the jth edge. If not provided, Dirichlet boundary conditions are assumed on all edges.
        n (int): The natural number parameter n in the block grid method (see page 35)
        delta (float): The grid spacing for the block grid
        max_iter (int): The maximum number of iterations to solve the Laplace equation
        radial_heuristic (float): The heuristic for tapering off the length of the blocks of first and second kind from the maximum
            length, can be between 0 and 1.
        tolerance (float): The tolerance for floating point errors, should be very small, 1e-10 is recommended.
        overlap_heuristic (float): The heuristic for the overlap of the blocks, can be between 0 and 0.5.

        Returns:
        None
        """
        self.poly = poly
        self.max_iter = max_iter
        self.boundary_conditions = boundary_conditions
        assert len(self.boundary_conditions) == len(self.poly.edges), "The number of boundary conditions must be equal to \
            the number of edges of the polygon"
        self.is_dirichlet = is_dirichlet if is_dirichlet else [True for _ in range(len(self.poly.edges))]
        assert len(self.is_dirichlet) == len(self.poly.edges), "The number of Dirichlet boundary conditions must be equal to \
            the number of edges of the polygon"
        self.delta = delta
        self.n = n
        self.radial_heuristic = radial_heuristic
        assert 0 < self.radial_heuristic < 1, "The radial heuristic must be between 0 and 1"
        assert self.radial_heuristic > 0.707, "The radial heuristic must be greater than 0.707 (sqrt(2)/2)"
        self.overlap_heuristic = overlap_heuristic
        assert 0 < self.overlap_heuristic < 0.5, "The overlap heuristic must be between 0 and 0.5"
        self.tol = tolerance
        assert self.tol > 0, "The tolerance must be greater than 0"
        self.verify_solution = verify_solution
        self.blocks = []
        self.N = 0
        self.L = 0
        self.M = 0
        self.x_min, self.y_min = np.min(self.poly.vertices, axis=0) 
        self.x_max, self.y_max = np.max(self.poly.vertices, axis=0) 
        self.min_xy = np.array([self.x_min, self.y_min])[np.newaxis, :]
        # Create solution array and mask points outside polygon
        # Calculate grid dimensions
        self.ny = int(np.ceil((self.y_max - self.y_min) / self.delta))
        self.nx = int(np.ceil((self.x_max - self.x_min) / self.delta))
        
        # Create grid points more efficiently using meshgrid
        x = np.linspace(self.x_min + 0.5 * self.delta, self.x_max + 0.5 * self.delta, self.nx)
        y = np.linspace(self.y_min + 0.5 * self.delta, self.y_max + 0.5 * self.delta, self.ny)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Create points array without using column_stack
        points = np.stack([self.X.ravel(), self.Y.ravel()], axis=1)
        
        # Create mask using vectorized operations
        mask = self.poly.is_inside(points).reshape(self.ny, self.nx)
        
        # Create masked array directly
        self.solution = np.ma.masked_array(np.zeros((self.ny, self.nx)), mask=~mask)
        self.inside_block_ids_ = np.ma.masked_array(np.zeros((self.ny, self.nx), dtype=int), mask=~mask)
        self.cartesian_grid = np.ma.array(np.stack([self.X, self.Y], axis = 2), 
                                          mask=np.repeat(~mask[:, :, np.newaxis], 2, axis=2))

    def solve(self, verbose = False, plot = False, ax = None):
        if verbose:
            print("Starting solution process...")
            print("Step 1: Finding block covering...")
        self.find_block_covering()
        
        if verbose:
            print("Step 2: Initializing solution data structures...")
        self.initialize_solution()
        
        if verbose:
            print("Step 3: Estimating solution over curved boundaries...")
        self.estimate_solution_over_curved_boundaries()
        
        if verbose:
            print("Step 4: Estimating solution over inner points...")
        self.estimate_solution_over_inner_points()
        
        if verbose:
            print("Solution process complete.")

        if verbose:
            print(f"N, L, M = {self.N}, {self.L}, {self.M}")
        if plot:
            self.plot_solution(ax)
            
        return self.solution

    def plot_solution(self, ax, vmin = None, vmax = None):
        # Plot the solution on the given axis
        # Plot heatmap with masked data
        im = ax.pcolormesh(self.X, self.Y, self.solution, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        # plt.colorbar(im, ax=ax, orientation='horizontal')
        plt.colorbar(im, ax=ax)

    def estimate_solution_over_inner_points(self):
        # Estimate the solution over the inner points
        # This is done by using the equation (5.1) in the book
        self.inner_sol_Q, self.inner_sol_R = self.find_inner_points_Q_and_R()

        N, M = self.solution.shape
        for i in range(N):
            for j in range(M):
                if not self.solution.mask[i, j]:
                    block_id = self.inside_block_ids_[i, j]
                    self.solution[i, j] = self.inner_sol_Q[i, j] + self.beta_mu[block_id] * \
                        np.sum((self.boundary_estimates[block_id] - self.mu_carrier_function_values[block_id]) * \
                               self.inner_sol_R[i, j])
        
    def find_inner_points_Q_and_R(self):
        # Find the inner points Q (carrier function) and R (poisson kernel)
        # This is done according to equation (5.1) in the book
        Q = np.ma.zeros_like(self.solution)
        R = self.theta_mu[self.inside_block_ids_] # init array to store the poisson kernel values with proper shape
        N, M = Q.shape
        k = np.arange(self.phi_ij.shape[1])
        points_polar = self.get_inner_points_in_polar_coordinates()
        boundary_identifiers = np.squeeze(self.block_nu.dot(np.array([[2], [1]])))[self.inside_block_ids_]
        block_kinds = self.block_type[self.inside_block_ids_]
        nu_ij = self.block_nu[self.inside_block_ids_]
        phi_ij = self.phi_ij[self.inside_block_ids_]
        r0_ = self.r0_mu[self.inside_block_ids_]
        eta_ij = self.theta_mu[self.inside_block_ids_]
        alpha_ij = self.alpha_mu[self.inside_block_ids_]
        for i in range(N):
            for j in range(M):
                if not (self.inside_block_ids_.mask[i, j] or points_polar[i, j, 0] < self.tol): # also check if point is at the pole
                    Q[i, j] = self.calculate_carrier_function_value(block_kinds[i, j], boundary_identifiers[i, j], \
                                                                    points_polar[i, j, 0], points_polar[i, j, 1], k, phi_ij[i, j, :, 0], \
                                                                    phi_ij[i, j, :, 1], alpha_ij[i, j])
                    R[i, j] = self.calculate_poisson_kernel_value(block_kinds[i, j], nu_ij[i, j, 0], nu_ij[i, j, 1], \
                                                                    points_polar[i, j, 0], r0_[i, j], points_polar[i, j, 1], \
                                                                    eta_ij[i, j], alpha_ij[i, j])
        return Q, R
    
    def get_inner_points_in_polar_coordinates(self):
        # Get the inner points in polar coordinates
        points = self.cartesian_grid - self.center_mu[self.inside_block_ids_]
        ref_theta = self.ref_theta[self.inside_block_ids_]
        return np.ma.array(np.stack([np.linalg.norm(points, axis=2), 
                                        np.mod(np.arctan2(points[:, :, 1], points[:, :, 0]) - \
                                            ref_theta, 2 * np.pi)], axis=2),
                            mask=points.mask)

    def initialize_solution(self):
        """
        Initialize the solution data structures for the block grid method.

        Returns:
            None
        """
        # make parameters!
        self.alpha_mu = np.zeros(len(self.blocks))
        self.r0_mu = np.zeros(len(self.blocks))
        self.r_mu = np.zeros(len(self.blocks))
        self.center_mu = np.zeros((len(self.blocks), 2))
        self.ref_theta = np.zeros(len(self.blocks))
        self.block_type = np.zeros(len(self.blocks), dtype=int)

        num_bounds = [len(x) for x in self.boundary_conditions]
        max_bounds = max(num_bounds)
        self.phi_ij = np.ma.masked_array(np.zeros((len(self.blocks), max_bounds, 2)), mask=False)
        self.block_nu = np.ma.masked_array(np.zeros((len(self.blocks), 2), dtype=bool), mask=False)
        for blk in self.blocks:
            self.alpha_mu[blk.id_] = blk.angle / np.pi
            self.r0_mu[blk.id_] = blk.r0
            self.center_mu[blk.id_] = blk.center
            self.r_mu[blk.id_] = blk.length
            self.block_type[blk.id_] = blk.block_kind
            if blk.block_kind == 1:
                self.ref_theta[blk.id_] = np.arctan2(self.poly.edges[blk.edge_j_index][1], self.poly.edges[blk.edge_j_index][0])
                self.phi_ij[blk.id_, :num_bounds[blk.edge_i_index], 0] = self.boundary_conditions[blk.edge_i_index]
                self.phi_ij.mask[blk.id_, num_bounds[blk.edge_i_index]:, 0] = True
                self.phi_ij[blk.id_, :num_bounds[blk.edge_j_index], 1] = self.boundary_conditions[blk.edge_j_index]
                self.phi_ij.mask[blk.id_, num_bounds[blk.edge_j_index]:, 1] = True
                self.block_nu[blk.id_, 0] = self.is_dirichlet[blk.edge_i_index]
                self.block_nu[blk.id_, 1] = self.is_dirichlet[blk.edge_j_index]
            elif blk.block_kind == 2:
                self.ref_theta[blk.id_] = np.arctan2(self.poly.edges[blk.edge_i_index][1], self.poly.edges[blk.edge_i_index][0])
                self.phi_ij[blk.id_, :num_bounds[blk.edge_i_index], 0] = self.boundary_conditions[blk.edge_i_index]
                self.phi_ij.mask[blk.id_, num_bounds[blk.edge_i_index]:, 0] = True
                self.phi_ij.mask[blk.id_, :, 1] = True
                self.block_nu[blk.id_, 0] = self.is_dirichlet[blk.edge_i_index]
                self.block_nu.mask[blk.id_, 1] = True
            else:
                self.block_nu.mask[blk.id_, :] = True
                self.phi_ij.mask[blk.id_, :, :] = True
            
        self.n_mu = np.maximum(4, np.floor(self.n * self.alpha_mu))
        self.beta_mu = np.pi * np.divide(self.alpha_mu, self.n_mu)
        # Create array with max size and mask values beyond n_mu for each block
        max_n = int(np.max(self.n_mu))
        self.theta_mu = np.ma.masked_array(
            np.matmul(self.beta_mu.reshape(-1, 1),
                     (np.arange(1, max_n + 1) - 0.5).reshape(1, -1)),
            mask=np.array([k >= self.n_mu[i] for i, k in 
                          np.ndindex(len(self.blocks), max_n)]).reshape(len(self.blocks), -1)
        )
        # Calculate parameters for each block

        # Initialize array for quantized boundary points with same shape as theta_mu
        self.quantized_boundary_points = np.repeat(self.theta_mu[:, :, np.newaxis], 2, axis = 2)
        self.quantized_boundary_points.data[:, :, 0] = np.repeat(self.r0_mu.reshape(-1, 1), max_n, axis=1)
        
        self.boundary_estimates = np.ma.zeros_like(self.theta_mu)
        self.quantized_boundary_points = self.convert_to_cartesian_coordinates(self.quantized_boundary_points)

        # Find the tau block ids for the P_mu points
        self.tau_block_ids = self.find_tau_block_ids_for_P_mu()

        # Create array of tau block centers with same shape as theta_mu
        self.tau_block_centers = np.ma.array(
            self.center_mu[self.tau_block_ids],
            mask=np.repeat(self.tau_block_ids.mask[:, :, np.newaxis], 2, axis=2)
        )

        # Create array of ref theta with same shape as theta_mu
        self.tau_ref_thetas = np.ma.array(
            self.ref_theta[self.tau_block_ids],
            mask=self.tau_block_ids.mask
        )

        # Find the tau block polar coordinates for the P_mu points
        self.tau_block_polar_coordinates = self.find_tau_block_polar_coordinates()

        self.poisson_kernel_values = self.find_poisson_kernel_values()
        
        if self.verify_solution:
            assert self.verify_unique_solution(self.poisson_kernel_values), \
                "No unique solution exists for the natural number parameter n = " + str(self.n)

        self.tau_carrier_function_values, self.mu_carrier_function_values = self.find_carrier_function_values_on_blocks()
        
    def convert_to_cartesian_coordinates(self, points : np.ma.MaskedArray):
        # Convert the points from polar coordinates to Cartesian coordinates
        return np.ma.array(np.stack([self.center_mu[:, 0][:, np.newaxis] + points[:, :, 0] * \
                                        np.cos(points[:, :, 1] + self.ref_theta[:, np.newaxis]),
                                     self.center_mu[:, 1][:, np.newaxis] + points[:, :, 0] * \
                                        np.sin(points[:, :, 1] + self.ref_theta[:, np.newaxis])], 
                                   axis=-1),
                          mask=points.mask)
    
    def convert_to_polar_coordinates(self, points : np.ma.MaskedArray):
        # Convert the points from Cartesian coordinates to polar coordinates
        points = points - self.center_mu[:, np.newaxis, :]
        return np.ma.array(np.stack([np.linalg.norm(points, axis=2), 
                                     np.mod(np.arctan2(points[:, :, 1], points[:, :, 0]) - \
                                        self.ref_theta[:, np.newaxis], 2 * np.pi)], axis=2),
                          mask=points.mask)
    
    def find_tau_block_polar_coordinates(self):
        # Find the tau block polar coordinates for the P_mu points
        points = self.quantized_boundary_points - self.tau_block_centers
        return np.ma.array(np.stack([np.linalg.norm(points, axis=2), 
                                     np.mod(np.arctan2(points[:, :, 1], points[:, :, 0]) - \
                                        self.tau_ref_thetas, 2 * np.pi)], axis=2),
                          mask=points.mask)

    def find_tau_block_ids_for_P_mu(self):
        # Find the block ids that contain the point_mu that lies on the boundary of the extended block T_mu, but
        # also lies inside basic block T_tau. See equation (4.4) for the definition of T_mu and T_tau.
        # id_mu is the id of the block T_mu
        points = np.repeat(self.quantized_boundary_points[np.newaxis, :], len(self.blocks), axis = 0) - \
            self.center_mu[:, np.newaxis, np.newaxis]
        dists = np.linalg.norm(points, axis = 3) <= self.r_mu[:, np.newaxis, np.newaxis]
        # Find the smallest index that is true in the first axis
        # For each point in quantized_boundary_points, find the first block that contains it
        tau_block_ids = np.ma.array(np.argmax(dists, axis=0), mask=~np.any(dists, axis=0))
        return tau_block_ids

    def find_carrier_function_values_on_blocks(self):
        # Find the carrier function values on the blocks
        tau_carrier_function_values = np.ma.zeros_like(self.theta_mu) # carrier function values on tau blocks for r_taumu, theta_taumu
        mu_carrier_function_values = np.ma.zeros_like(self.theta_mu) # carrier function values on mu blocks for r_mu, theta_mu

        mu_block_boundary_identifiers = np.squeeze(self.block_nu.dot(np.array([[2], [1]])))
        tau_block_boundary_identifiers = mu_block_boundary_identifiers[self.tau_block_ids]
        tau_block_boundary_identifiers.mask = tau_block_boundary_identifiers.mask | self.tau_block_ids.mask

        tau_block_kind_identifiers = np.ma.masked_array(self.block_type[self.tau_block_ids], mask = self.tau_block_ids.mask)

        N, M = tau_carrier_function_values.shape

        k = np.arange(self.phi_ij.shape[1])

        for i in range(N):
            for j in range(M):
                # the check for self.tau_block_ids.mask is to deal with improper covering (i.e., not tau block for P_mu)
                if not (tau_carrier_function_values.mask[i, j] or self.tau_block_ids.mask[i, j]):
                    tau_id_ = self.tau_block_ids[i, j]
                    tau_alpha_j = self.alpha_mu[tau_id_]
                    r_ = self.tau_block_polar_coordinates[i, j, 0]
                    theta_ = self.tau_block_polar_coordinates[i, j, 1]
                    a_ = self.phi_ij[tau_id_, :, 0]
                    b_ = self.phi_ij[tau_id_, :, 1]
                    tau_carrier_function_values[i, j] = self.calculate_carrier_function_value(tau_block_kind_identifiers[i, j], \
                                                                 tau_block_boundary_identifiers[i, j], r_, theta_, k, a_, b_, tau_alpha_j)
        for i in range(N):
            for j in range(M):
                if not mu_carrier_function_values.mask[i, j]:
                    mu_id_ = i
                    alpha_j = self.alpha_mu[mu_id_]
                    r_ = self.r0_mu[i]
                    theta_ = self.theta_mu[i, j]
                    a_ = self.phi_ij[mu_id_, :, 0]
                    b_ = self.phi_ij[mu_id_, :, 1]
                    mu_carrier_function_values[i, j] = self.calculate_carrier_function_value(self.block_type[mu_id_], \
                                                                 mu_block_boundary_identifiers[mu_id_], r_, theta_, k, a_, b_, alpha_j)
        return tau_carrier_function_values, mu_carrier_function_values

    def calculate_carrier_function_value(self, block_kind, block_boundary_identifier, r_, theta_, k, a_, b_, alpha_j):
        if block_kind == 1:
            # first kind block
            if block_boundary_identifier == 0:
                # both edges are Neumann (v_{j-1} = 0, v_{j} = 0)
                cond_ = (np.abs(np.sin((k + 1) * alpha_j * np.pi)) < self.tol).astype(int)
                
                sigma_jk = (k + 1) * ((1 - cond_) * np.sin((k + 1) * alpha_j * np.pi) + \
                    cond_ * (- alpha_j * np.pi * np.cos((k + 1) * alpha_j * np.pi)))
                eta_jk = (1 - cond_) * (np.cos((k + 1) * theta_)) + \
                    cond_ * (theta_ * np.sin((k + 1) * theta_) - np.log(r_) * np.cos((k + 1) * theta_))
                
                return np.sum((a_ / sigma_jk) * np.power(r_, k + 1) * eta_jk) + \
                    np.sum((b_ / (k + 1)) * np.power(r_, k + 1) * \
                            (np.sin((k + 1) * theta_) + ((k + 1) * np.cos((k + 1) * alpha_j * np.pi) / sigma_jk) * eta_jk))
                    
            elif block_boundary_identifier == 1:
                # edge j-1 is Neumann, edge j is Dirichlet (v_{j-1} = 0, v_{j} = 1)
                cond_ = (np.abs(np.cos(k * alpha_j * np.pi)) < self.tol).astype(int)

                omega_jk = k * ((1 - cond_) * (- np.cos(k * alpha_j * np.pi)) + \
                    cond_ * (alpha_j * np.pi * np.sin(k * alpha_j * np.pi)))
                psi_jk = (1 - cond_) * np.sin(k * theta_) + \
                    cond_ * (theta_ * np.cos(k * theta_) + np.log(r_) * np.sin(k * theta_))
                psi_jkp1 = (1 - cond_) * np.sin((k + 1) * theta_) + \
                    cond_ * (theta_ * np.cos((k + 1) * theta_) + np.log(r_) * np.sin((k + 1) * theta_))
                kp1 = k[1:]
                
                return np.sum((a_ / omega_jk) * np.power(r_, k + 1) * psi_jkp1) + \
                    b_[0] + np.sum(b_[1:] * np.power(r_, kp1) * \
                                    (np.cos(kp1 * theta_) - (kp1 * np.sin(kp1 * alpha_j * np.pi) / omega_jk[1:]) * psi_jk[1:]))
            elif block_boundary_identifier == 2:
                # edge j-1 is Dirichlet, edge j is Neumann (v_{j-1} = 1, v_{j} = 0)
                cond_ = (np.abs(np.cos((k + 1) * alpha_j * np.pi)) < self.tol).astype(int)
                kappa_jk = (1 - cond_) * (np.cos((k + 1) * theta_)) + \
                    cond_ * (theta_ * np.sin((k + 1) * theta_) - np.log(r_) * np.cos((k + 1) * theta_))
                kappa_1k = (1 - cond_) * (np.cos((k + 1) * alpha_j * np.pi)) + \
                    cond_ * (alpha_j * np.pi * np.sin((k + 1) * alpha_j * np.pi) - np.log(1) * np.cos((k + 1) * alpha_j * np.pi))
                kappa_jkn1 = (1 - cond_) * (np.cos(k * theta_)) + \
                    cond_ * (theta_ * np.sin(k * theta_) - np.log(r_) * np.cos(k * theta_))
                kappa_1kn1 = (1 - cond_) * (np.cos(k * alpha_j * np.pi)) + \
                    cond_ * (alpha_j * np.pi * np.sin(k * alpha_j * np.pi) - np.log(1) * np.cos(k * alpha_j * np.pi))
                
                return np.sum((a_ / kappa_1kn1) * np.power(r_, k) * kappa_jkn1) + \
                    np.sum((b_ / (k + 1)) * np.power(r_, k + 1) * \
                            (np.sin((k + 1) * theta_) - (np.sin((k + 1) * alpha_j * np.pi) / kappa_1k) * kappa_jk))
            else:
                # both edges are Dirichlet (v_{j-1} = 1, v_{j} = 1)
                cond_ = (np.abs(np.sin(k * alpha_j * np.pi)) < self.tol).astype(int)

                zeta_jk = (1 - cond_) * np.sin(k * theta_) + \
                    cond_ * (theta_ * np.cos(k * theta_) + np.log(r_) * np.sin(k * theta_))
                zeta_1k = (1 - cond_) * np.sin(k * alpha_j * np.pi) + \
                    cond_ * (alpha_j * np.pi * np.cos(k * alpha_j * np.pi) + np.log(1) * np.sin(k * alpha_j * np.pi))

                return np.sum((a_ / zeta_1k) * np.power(r_, k) * zeta_jk) + \
                    np.sum((b_ * np.power(r_, k) * (np.cos(k * theta_) - (np.cos(k * alpha_j * np.pi) / zeta_1k) * zeta_jk)))
        elif block_kind == 2:
            # second kind block
            nu_jq = block_boundary_identifier // 2

            return np.sum(a_ * np.power(r_, k) * \
                            (nu_jq * (np.cos(k * theta_)) + (1 - nu_jq) * \
                            r_ * (np.sin((k + 1) * theta_) / (k + 1))))
        else:
            # third kind block
            return 0.0
   
    def estimate_solution_over_curved_boundaries(self):
        # Estimate the solution over the block curved boundaries using the boundary conditions and the carrier function
        # This is done by iterating over the blocks and estimating the solution over the curved boundaries
        # The iteration is stopped based on the max_iter parameter 
        # see equations (4.14) and (4.15) in the book
        iter = 0
        N, M = self.boundary_estimates.shape
        while True: 
            for i in range(N):
                for j in range(M):
                    if self.boundary_estimates.mask[i, j] or self.tau_block_ids.mask[i, j]:
                        continue

                    tau_block_id = self.tau_block_ids[i, j]
                    beta_tau = self.beta_mu[tau_block_id]
                    poisson_kernel_vector = self.poisson_kernel_values[i, j]

                    self.boundary_estimates[i, j] = self.tau_carrier_function_values[i, j] + \
                        beta_tau * np.sum((self.boundary_estimates[tau_block_id] - self.mu_carrier_function_values[tau_block_id]) * \
                                    poisson_kernel_vector / max(1.0, beta_tau * np.sum(poisson_kernel_vector)))
            iter += 1
            if iter > self.max_iter:
                break       

    def find_poisson_kernel_values(self):
        # Find the poisson kernel vector for the given r_ and theta_, with the given theta_tau array
        eta_tau = self.theta_mu[self.tau_block_ids]
        poisson_kernel = np.ma.zeros_like(eta_tau)
        block_type_tau = self.block_type[self.tau_block_ids]
        nu_ij = self.block_nu[self.tau_block_ids]
        alpha_j = self.alpha_mu[self.tau_block_ids]
        r0_tau = self.r0_mu[self.tau_block_ids]
        r_ = self.tau_block_polar_coordinates[:, :, 0]
        theta_ = self.tau_block_polar_coordinates[:, :, 1]

        N, M = self.tau_block_ids.shape
        for i in range(N):
            for j in range(M):
                if self.tau_block_ids.mask[i, j]:
                    continue

                poisson_kernel[i, j] = self.calculate_poisson_kernel_value(block_type_tau[i, j], nu_ij[i, j, 0], nu_ij[i, j, 1], \
                                                                           r_[i, j], r0_tau[i, j], theta_[i, j], eta_tau[i, j], alpha_j[i, j])
        return poisson_kernel

    def verify_unique_solution(self, poisson_kernel_values):
        # verify the existance of a unique solution for the given poisson kernel values (see lemma 4.2 and (4.12) in the book)
        epsilon = np.ma.ones_like(self.theta_mu)
        beta_tau = self.beta_mu[self.tau_block_ids]
        for _ in range(self.M + 1):
            epsilon = beta_tau * np.sum(poisson_kernel_values * epsilon[:,:,np.newaxis], axis = 2)
        
        if np.max(np.abs(epsilon)) < 1.0:
            return True
        else:
            return False

    def calculate_poisson_kernel_value(self, block_kind, nu_i, nu_j, r_, r0_, theta_, eta_, alpha_j):
        main_kernel = lambda r1, theta1, eta1: (1 - r1**2) / (2 * np.pi * (1 - 2*r1*np.cos(theta1 - eta1) + r1**2))
        if block_kind == 1:
            # block of first kind
            
            lambda_j = 1/((2 - nu_i * nu_j - (1 - nu_i) * (1 - nu_j)) * alpha_j)
            
            if nu_i == nu_j: # both are equal (3.17), (3,12)
                return lambda_j * (main_kernel((r_ / r0_) ** lambda_j, lambda_j * theta_, lambda_j * eta_) + \
                                        (-1)**nu_j * main_kernel((r_ / r0_) ** lambda_j, lambda_j * theta_, -lambda_j * eta_))
            else: # they are different (3.17), (3.13)
                return lambda_j * (main_kernel((r_ / r0_) ** lambda_j, lambda_j * theta_, lambda_j * eta_) + \
                                        (-1)**nu_j * main_kernel((r_ / r0_) ** lambda_j, lambda_j * theta_, -lambda_j * eta_) - \
                                        (-1)**nu_j * \
                                        (main_kernel((r_ / r0_) ** lambda_j, lambda_j * theta_, lambda_j * (np.pi - eta_)) + \
                                        (-1)**nu_j * main_kernel((r_ / r0_) ** lambda_j, lambda_j * theta_, -lambda_j * (np.pi - eta_)))
                                    )
        elif block_kind == 2:
            # block of second kind (3.16)
            return main_kernel(r_ / r0_, theta_, eta_) + (-1)**nu_i * (main_kernel(r_ / r0_, theta_, -eta_))
        else:
            # block of third kind (3.15)
            return main_kernel(r_ / r0_, theta_, eta_)
           
    def plot_block_covering(self, ax, uncovered_points = None, show_boundary_conditions = True,
                            show_quantized_boundaries = False):
        # Plot polygon
        self.poly.plot(ax=ax)
        
        # Plot first kind blocks (sectors from vertices)
        for i, block in enumerate(self.blocks):
            if block.block_kind == 1:
                edge_angle = np.arctan2(self.poly.edges[block.edge_j_index][1], self.poly.edges[block.edge_j_index][0])
                # Create sector
                theta = np.linspace(edge_angle, edge_angle + block.angle, 100)
                # Plot inner radius r
                r = block.length
                x = block.center[0] + r * np.cos(theta)
                y = block.center[1] + r * np.sin(theta)
                ax.plot(x, y, 'b-', alpha=0.5)
                # Plot outer radius r0 
                r0 = block.r0
                x0 = block.center[0] + r0 * np.cos(theta)
                y0 = block.center[1] + r0 * np.sin(theta)
                ax.plot(x0, y0, 'b--', alpha=0.5)
                # Plot block center and id
                ax.plot(block.center[0], block.center[1], 'b.')
                ax.text(block.center[0], block.center[1], f'$P_{{ {block.id_ + 1} }}$', 
                        horizontalalignment='right', verticalalignment='bottom')
                # Plot quantized boundaries with solution values
                if show_quantized_boundaries:
                    # Get unmasked points for this block
                    points = self.quantized_boundary_points[block.id_][~self.quantized_boundary_points[block.id_].mask.any(axis=1)]
                    if len(points) > 0:
                        ax.plot(points[:, 0], points[:, 1], 'g.')
                        for j, point in enumerate(points):
                                ax.text(point[0], point[1], f'{self.boundary_estimates[i, j]:.2f}',
                                      fontsize=8, horizontalalignment='right')
        
            # Plot second kind blocks (half-disks from edges)
            elif block.block_kind == 2:
                edge_angle = np.arctan2(self.poly.edges[block.edge_i_index][1], self.poly.edges[block.edge_i_index][0])
                # Create half-disk
                theta = np.linspace(edge_angle, edge_angle + np.pi, 100)
                # Plot inner radius r
                r = block.length
                x = block.center[0] + r * np.cos(theta)
                y = block.center[1] + r * np.sin(theta)
                ax.plot(x, y, 'r-', alpha=0.5)
                # Plot outer radius r0
                r0 = block.r0
                x0 = block.center[0] + r0 * np.cos(theta)
                y0 = block.center[1] + r0 * np.sin(theta)
                ax.plot(x0, y0, 'r--', alpha=0.5)
                # Plot block center and id
                ax.plot(block.center[0], block.center[1], 'r.')
                ax.text(block.center[0], block.center[1], f'$P_{{ {block.id_ + 1} }}$',
                        horizontalalignment='right', verticalalignment='bottom')
                # Plot quantized boundaries
                if show_quantized_boundaries:
                    # Get unmasked points for this block
                    points = self.quantized_boundary_points[block.id_][~self.quantized_boundary_points[block.id_].mask.any(axis=1)]
                    if len(points) > 0:
                        ax.plot(points[:, 0], points[:, 1], 'g.')
                        for j, point in enumerate(points):
                                ax.text(point[0], point[1], f'{self.boundary_estimates[i, j]:.2f}',
                                      fontsize=8, horizontalalignment='right')
            
            # Plot third kind blocks (interior disks)
            elif block.block_kind == 3:
                # Create full disk
                theta = np.linspace(0, 2*np.pi, 100)
                # Plot inner radius r
                r = block.length
                x = block.center[0] + r * np.cos(theta)
                y = block.center[1] + r * np.sin(theta)
                ax.plot(x, y, 'g-', alpha=0.5)
                # Plot outer radius r0
                r0 = block.r0
                x0 = block.center[0] + r0 * np.cos(theta)
                y0 = block.center[1] + r0 * np.sin(theta)
                ax.plot(x0, y0, 'g--', alpha=0.5)
                # Plot block center and id
                ax.plot(block.center[0], block.center[1], 'g.')
                ax.text(block.center[0], block.center[1], f'$P_{{ {block.id_ + 1} }}$',
                        horizontalalignment='right', verticalalignment='bottom')
                # Plot quantized boundaries
                if show_quantized_boundaries:
                    # Get unmasked points for this block
                    points = self.quantized_boundary_points[block.id_][~self.quantized_boundary_points[block.id_].mask.any(axis=1)]
                    if len(points) > 0:
                        ax.plot(points[:, 0], points[:, 1], 'g.')
                        for j, point in enumerate(points):
                                ax.text(point[0], point[1], f'{self.boundary_estimates[i, j]:.2f}',
                                      fontsize=8, horizontalalignment='right')

        if show_boundary_conditions:
            # Plot boundary conditions
            for i in range(len(self.poly.vertices)):
                # Get midpoint of edge for text placement
                midpoint = (self.poly.vertices[i] + self.poly.vertices[(i+1) % len(self.poly.vertices)]) / 2
                # Calculate edge direction angle
                edge = self.poly.edges[i]
                angle = np.arctan2(edge[1], edge[0]) * 180/np.pi
                # Add boundary condition value as text, rotated to match edge direction
                ax.text(midpoint[0], midpoint[1], f'$\\phi_{{ {i+1} }} = {self.boundary_conditions[i]}$',
                        horizontalalignment='left', verticalalignment='top',
                        rotation=angle)
            
        # Plot uncovered points
        # Get coordinates of unmasked points
        if uncovered_points is not None:
            y_coords, x_coords = np.where(~uncovered_points.mask)
            if len(x_coords) > 0:
                ax.scatter(x_coords * self.delta, y_coords * self.delta, c='red', s=1, alpha=0.5)
            
        ax.set_aspect('equal')
        ax.grid(True)

    def distances_from_edges(self, v: np.ndarray, w: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Return distances between line segments vw and point p
        
        Parameters:
            v: Start points of line segments (Nx2 numpy array)
            w: End points of line segments (Nx2 numpy array) 
            p: Point to check distance from (2D numpy array)
            
        Returns:
            numpy array: Distances from point p to each line segment vw
        """
        # Calculate length squared of line segments
        l2 = np.sum((w - v) ** 2, axis=1)  # |w-v|^2
        
        # Handle segments that are actually points
        point_mask = l2 == 0.0
        distances = np.zeros(len(v))
        distances[point_mask] = np.linalg.norm(p - v[point_mask], axis=1)
        
        # For actual segments, find projection of p onto each line
        segment_mask = ~point_mask
        if np.any(segment_mask):
            # Calculate t = [(p-v) . (w-v)] / |w-v|^2 for each segment
            t = np.sum((p - v[segment_mask]) * (w[segment_mask] - v[segment_mask]), axis=1) / l2[segment_mask]
            t = np.clip(t, 0, 1)
            
            # Calculate projections
            projections = v[segment_mask] + t[:, np.newaxis] * (w[segment_mask] - v[segment_mask])
            
            # Calculate distances
            distances[segment_mask] = np.linalg.norm(p - projections, axis=1)
            
        return distances

    def find_block_covering(self):
        """
        This function will find the block covering of the polygon and return the number of blocks of each kind.
        The blocks of first kind are the ones extending from the vertex (their number denoted by N)
        The blocks of second kind are the ones extending from the edge (their number denoted by L)
        The blocks of third kind are the ones inside the polygon (their number denoted by M)
        """

        self.blocks = []
        self.N = 0
        self.L = 0
        self.M = 0
        block_id_counter = len(self.poly.vertices)

        # construct the blocks of first and second kind
        second_kind_blocks = []
        num_vertices = len(self.poly.vertices)
        for i, vertex in enumerate(self.poly.vertices):
            # Get angle from polygon class
            angle = self.poly.angles[i]

            w = np.roll(self.poly.vertices, -i-1, axis = 0)[:num_vertices-2]
            v = np.roll(self.poly.vertices, -i-2, axis = 0)[:num_vertices-2]
            r0 = self.radial_heuristic * min(self.distances_from_edges(v, w, vertex))
            length = self.radial_heuristic * r0

            # create a block of first kind at the vertex
            # for first kind blocks, the id_ is the index of the vertex
            self.blocks.append(block(vertex, angle, length, r0, block_kind=1, id_=i, edge_i_index=i-1, edge_j_index=i)) # first kind block
            self.N += 1
            
            # Check if there exists a gap between the blocks of first kind on previous vertex and current vertex
            if i == 0: # first vertex, we don't need to check for gaps
                continue
            prev_vertex = self.poly.vertices[i-1]
            this_vertex = self.poly.vertices[i]

            dist_covered = self.blocks[-1].length + self.blocks[-2].length

            distance = np.linalg.norm(this_vertex - prev_vertex)
            if distance > dist_covered:
                d = distance - dist_covered
                unit_edge_vector = (this_vertex - prev_vertex) / distance
                r0_half_disk = min(self.blocks[-1].r0, self.blocks[-2].r0)
                radius_half_disk = self.radial_heuristic * r0_half_disk
                o_rad = self.overlap_heuristic * radius_half_disk
                # make the edge_index_i positive
                edge_i_index = i-1 if i-1 >= 0 else len(self.poly.vertices) - 1
                max_dist = d + 0.5 * radius_half_disk
                if self.blocks[-2].r0 < self.blocks[-1].r0:
                    start_vertex = prev_vertex
                else: 
                    start_vertex = prev_vertex + (self.blocks[-2].length - self.blocks[-1].length) * unit_edge_vector

                edge_starts = np.roll(self.poly.vertices, -i, axis=0)[:num_vertices-1]
                edge_ends = np.roll(self.poly.vertices, -(i+1), axis=0)[:num_vertices-1]
                if d < 2 * radius_half_disk:
                    new_vertex = start_vertex + (radius_half_disk + 0.5 * d) * unit_edge_vector
                    r0_half_disk = self.radial_heuristic * min(self.distances_from_edges(edge_starts, edge_ends, new_vertex))
                    radius_half_disk = self.radial_heuristic * r0_half_disk
                    second_kind_blocks.append(block(new_vertex, np.pi, radius_half_disk, r0_half_disk, block_kind=2, 
                                                id_=block_id_counter, edge_i_index=edge_i_index, edge_j_index=None)) # second kind block
                    self.L += 1
                    block_id_counter += 1
                else:
                    # Calculate number of second kind blocks needed
                    num_second_kind_blocks = int(np.round((d + radius_half_disk) / (2 *(radius_half_disk - o_rad))))
                    # Add new vertices evenly spaced between prev_vertex and next_vertex
                    nextj = 0
                    for j in range(num_second_kind_blocks):
                        if j < nextj:
                            continue
                        # start adding blocks from the prev_vertex
                        next_loc = min(2 * (j + 1) * (radius_half_disk - o_rad), max_dist)
                        new_vertex = start_vertex + next_loc * unit_edge_vector 
                        max_allowed_radius = self.radial_heuristic * min(self.distances_from_edges(edge_starts, edge_ends, new_vertex))
                        max_radius_ratio = np.floor(self.radial_heuristic * max_allowed_radius / (2*(radius_half_disk - o_rad)))
                        if max_radius_ratio > 1:
                            second_kind_blocks.append(block(new_vertex, np.pi, max_allowed_radius * self.radial_heuristic, \
                                                    max_allowed_radius, block_kind=2, \
                                                    id_=block_id_counter, edge_i_index=edge_i_index, edge_j_index=None)) # second kind block
                            self.L += 1
                            block_id_counter += 1
                            nextj = nextj + max_radius_ratio
                        elif max_allowed_radius > r0_half_disk:
                            second_kind_blocks.append(block(new_vertex, np.pi, radius_half_disk, r0_half_disk, block_kind=2, 
                                                        id_=block_id_counter, edge_i_index=edge_i_index, edge_j_index=None)) # second kind block
                            self.L += 1
                            block_id_counter += 1
                            nextj = nextj + 1
                        else:
                            num_blocks, new_blocks = self.break_second_kind_block(new_vertex, r0_half_disk, max_allowed_radius, edge_starts, 
                                                                        edge_ends, unit_edge_vector, edge_i_index, block_id_counter)
                            second_kind_blocks.extend(new_blocks)
                            self.L += num_blocks
                            block_id_counter += num_blocks
                            nextj = nextj + 1
        
        # Check if there exists a gap between the blocks of first kind on last vertex and first vertex
        prev_vertex = self.poly.vertices[-1]
        this_vertex = self.poly.vertices[0]

        dist_covered = self.blocks[-1].length + self.blocks[0].length
        distance = np.linalg.norm(this_vertex - prev_vertex)
        if distance > dist_covered:
            d = distance - dist_covered
            unit_edge_vector = (this_vertex - prev_vertex) / distance
            r0_half_disk = min(self.blocks[0].r0, self.blocks[-1].r0)
            radius_half_disk = self.radial_heuristic * r0_half_disk
            o_rad = self.overlap_heuristic * radius_half_disk
            # make the edge_index_i positive
            edge_i_index = -1
            max_dist = d + 0.5 * radius_half_disk
            if self.blocks[-1].r0 < self.blocks[0].r0:
                start_vertex = prev_vertex
            else: 
                start_vertex = prev_vertex + (self.blocks[-1].length - self.blocks[0].length) * unit_edge_vector

            edge_starts = self.poly.vertices[:num_vertices-1]
            edge_ends = self.poly.vertices[1:]
            if d < 2 * radius_half_disk:
                new_vertex = start_vertex + (radius_half_disk + 0.5 * d) * unit_edge_vector
                r0_half_disk = self.radial_heuristic * min(self.distances_from_edges(edge_starts, edge_ends, new_vertex))
                radius_half_disk = self.radial_heuristic * r0_half_disk
                second_kind_blocks.append(block(new_vertex, np.pi, radius_half_disk, r0_half_disk, block_kind=2, 
                                            id_=block_id_counter, edge_i_index=edge_i_index, edge_j_index=None)) # second kind block
                self.L += 1
                block_id_counter += 1
            else:
                # Calculate number of second kind blocks needed
                num_second_kind_blocks = int(np.round((d + radius_half_disk) / (2 *(radius_half_disk - o_rad))))
                # Add new vertices evenly spaced between prev_vertex and next_vertex
                nextj = 0
                for j in range(num_second_kind_blocks):
                    if j < nextj:
                        continue
                    # start adding blocks from the prev_vertex
                    next_loc = min(2 * (j + 1) * (radius_half_disk - o_rad), max_dist)
                    new_vertex = start_vertex + next_loc * unit_edge_vector 
                    max_allowed_radius = self.radial_heuristic * min(self.distances_from_edges(edge_starts, edge_ends, new_vertex))
                    max_radius_ratio = np.floor(self.radial_heuristic * max_allowed_radius / (2*(radius_half_disk - o_rad)))
                    if max_radius_ratio > 1:
                        second_kind_blocks.append(block(new_vertex, np.pi, max_allowed_radius * self.radial_heuristic, \
                                                max_allowed_radius, block_kind=2, \
                                                id_=block_id_counter, edge_i_index=edge_i_index, edge_j_index=None)) # second kind block
                        self.L += 1
                        block_id_counter += 1
                        nextj = nextj + max_radius_ratio
                    elif max_allowed_radius > r0_half_disk:
                        second_kind_blocks.append(block(new_vertex, np.pi, radius_half_disk, r0_half_disk, block_kind=2, 
                                                    id_=block_id_counter, edge_i_index=edge_i_index, edge_j_index=None)) # second kind block
                        self.L += 1
                        block_id_counter += 1
                        nextj = nextj + 1
                    else:
                        num_blocks, new_blocks = self.break_second_kind_block(new_vertex, r0_half_disk, max_allowed_radius, edge_starts, 
                                                                    edge_ends, unit_edge_vector, edge_i_index, block_id_counter)
                        second_kind_blocks.extend(new_blocks)
                        self.L += num_blocks
                        block_id_counter += num_blocks
                        nextj = nextj + 1
        # Insert second kind blocks into self.blocks at their id_ positions
        for blk in second_kind_blocks:
            self.blocks.insert(blk.id_, blk)
            
        self.L = self.L + self.N
        uncovered_points = self.find_uncovered_points()
        self.create_third_kind_blocks(uncovered_points) 

        return self.N, self.L, self.M, uncovered_points

    def break_second_kind_block(self, old_vertex : np.ndarray, r0_old_block : float, 
                                max_allowed_radius : float, edge_starts : np.ndarray, 
                                edge_ends : np.ndarray, unit_edge_vector : np.ndarray,
                                edge_i_index : int, block_id_counter : int):
        # This function will break a second kind block into smaller blocks to cover the gap without leaking
        # The old_vertex is the center of the old block
        # The r0_old_block is the radius of the old block
        # The max_allowed_radius is the maximum radius of the old block
        # The edge_starts and edge_ends are the vertices of other edges to check leaking
        # The unit_edge_vector is the unit vector of the edge
        # This function will return a list of new blocks

        ratio = int(2 ** np.floor(np.log2(r0_old_block / max_allowed_radius))) + 1 # a starting value
        done = False
        radius_old_block = self.radial_heuristic * r0_old_block
        
        vertices = []
        while not done:
            done = True
            n_min = ratio + 1
            new_radius = radius_old_block / ratio
            overlap = 1 / (n_min - 1) # minimum overlap possible
            for i in range(n_min):
                new_vertex = old_vertex + (- radius_old_block + new_radius + 2 * i * (1 - overlap) * new_radius) * unit_edge_vector
                vertices.append(new_vertex)
                if min(self.distances_from_edges(edge_starts, edge_ends, new_vertex)) < new_radius:
                    done = False
                    break
            if not done:
                ratio = ratio * 2
                vertices = []
        new_blocks = []
        new_r0 = new_radius / self.radial_heuristic
        for i in range(n_min):
            new_blocks.append(block(vertices[i], np.pi, new_radius, new_r0, block_kind=2, 
                                  id_=block_id_counter, edge_i_index=edge_i_index, edge_j_index=None))
            block_id_counter += 1
        return n_min, new_blocks

    def find_uncovered_points(self):
        # Find points in self.solution that are not covered by any block
        uncovered_points = self.solution.copy()
        # Create mask for points covered by blocks
        covered_mask = np.zeros_like(uncovered_points, dtype=bool)
        remaining_points = self.cartesian_grid.reshape((-1, 2))
        X, Y = np.where(~covered_mask)
            
        # Check each block for points inside it
        # Keep track of remaining points to check for next blocks
        for blk in self.blocks:
            inside_mask = blk.is_inside(remaining_points)
            inside_points = remaining_points[inside_mask]
            self.inside_block_ids_[X[inside_mask], Y[inside_mask]] = blk.id_
            if len(inside_points) > 0:
                covered_mask[X[inside_mask], Y[inside_mask]] = True
                # Remove points that are inside this block from remaining points
                remaining_points = remaining_points[~inside_mask]
                X = X[~inside_mask]
                Y = Y[~inside_mask]
                if len(remaining_points) == 0:
                    break
                    
        # Update mask to include both points outside polygon and points inside blocks
        uncovered_points.mask = uncovered_points.mask | covered_mask
        self.inside_block_ids_.mask = self.solution.mask
        
        return uncovered_points

    def create_third_kind_blocks(self, uncovered_points : np.ma.MaskedArray):
        # This function will add blocks of third kind to the list of blocks
        # Convert uncovered points to numpy array for easier manipulation
        self.M = self.L
        
        # Skip if no uncovered points
        if uncovered_points.mask.all():
            return
            
        # Find connected components/contours by clustering points that are within delta distance
        visited = uncovered_points.mask
        
        # Create arrays for start and end points of edges
        edge_starts = self.poly.vertices
        edge_ends = np.roll(self.poly.vertices, -1, axis=0)
        
        # Get coordinates of unmasked points
        y_coords, x_coords = np.where(~uncovered_points.mask)
        points = np.column_stack((x_coords * self.delta + self.x_min, y_coords * self.delta + self.y_min))
        
        # Randomize order of processing points
        indices = np.random.permutation(len(points))
        for i in indices:
            if visited[y_coords[i], x_coords[i]]:
                continue
                
            # Get current point as center
            current_center = points[i] + 0.5 * self.delta
            visited[y_coords[i], x_coords[i]] = True
            
            # Calculate minimum distance from edges to determine radius
            radius = np.min(self.distances_from_edges(edge_starts, edge_ends, current_center))
            
            # Create third kind block with calculated center and radius
            r0 = self.radial_heuristic * radius
            length = self.radial_heuristic * r0
            new_block = block(current_center, 2 * np.pi, length, r0, block_kind=3,
                          id_=self.M)
            
            # Find all points connected to this one
            # Check distances to all unvisited points
            distances = np.linalg.norm(points - current_center, axis=1)
            neighbors = np.where((distances <= length) & (~visited[y_coords, x_coords]))[0]

            # Update visited array for neighbors
            visited[y_coords[neighbors], x_coords[neighbors]] = True
            # Add neighbors to inner points of the block
            self.inside_block_ids_[y_coords[i], x_coords[i]] = self.M
            self.inside_block_ids_[y_coords[neighbors], x_coords[neighbors]] = self.M

            self.blocks.append(new_block)
            self.M += 1
    
    def plot_gradient(self, ax, decimation_factor = 2, scale = 20):
        # Plot the solution on the given axis
        # Convert solution points and values to grid format for heatmap
        
        # Calculate gradients using np.gradient
        dy, dx = np.gradient(self.solution)
        
        # Normalize the gradient field
        magnitude = np.sqrt(dx**2 + dy**2)
        # Avoid division by zero
        magnitude = np.where(magnitude == 0, 1, magnitude)
        dx_norm = dx / magnitude
        dy_norm = dy / magnitude
        
        # Plot normalized vector field using quiver on provided axes
        ax.quiver(self.cartesian_grid[:, :, 0][::decimation_factor, ::decimation_factor], self.cartesian_grid[:, :, 1][::decimation_factor, ::decimation_factor],
                 dx_norm[::decimation_factor, ::decimation_factor], dy_norm[::decimation_factor, ::decimation_factor],
                 scale=scale)  # Reduce scale to make arrows bigger