#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from pyBlockGrid.core.block import block
from pyBlockGrid.core.polygon import polygon
import random
import warnings
from .base import PDESolver

class volkovSolver(PDESolver):
    def __init__(self, poly : polygon, boundary_conditions : list[list[float]] = None, is_dirichlet : list[bool] = None,
                 n : int = 10, delta : float = 0.01, max_iter : int = 100, radial_heuristic : float = 0.8,
                 overlap_heuristic : float = 0.1):
        """
        Initialize the VolkovSolver class. Solves using block grid method by E. Volkov, as outlined in the
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
        self.blocks = []
        self.N = 0
        self.L = 0
        self.M = 0
        self.solution = {} # final solution u(x, y)

    def solve(self, verbose = False, plot = False, ax = None, show_boundary_conditions = True,
              show_quantized_boundaries = False):
        if verbose:
            print("Starting solution process...")
            print("Step 1: Finding block covering...")
        self.find_block_covering()
        
        if verbose:
            print("Step 2: Initializing solution data structures...")
        self._initialize_solution()
        
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
            self.plot_block_covering(ax = ax, show_boundary_conditions = show_boundary_conditions, 
                                    show_quantized_boundaries = show_quantized_boundaries)
            
        return self.solution

    def plot_solution(self, ax, solution, N = 100, vmin = None, vmax = None):
        # Plot the solution on the given axis
        # Convert solution points and values to grid format for heatmap
        x_coords = [point[0] for point in solution.keys()]
        y_coords = [point[1] for point in solution.keys()]
        values = list(solution.values())
        
        # Create regular grid
        xi = np.linspace(min(x_coords), max(x_coords), N)
        yi = np.linspace(min(y_coords), max(y_coords), N)
        xi, yi = np.meshgrid(xi, yi)
        
        # Interpolate scattered data to regular grid using scipy's griddata
        zi = griddata((x_coords, y_coords), values, (xi, yi), method='linear')
        
        # Create mask for points outside polygon
        mask = np.zeros_like(zi, dtype=bool)
        for i in range(len(xi)):
            for j in range(len(yi)):
                point = np.array([xi[i,j], yi[i,j]])
                # Check if point is inside polygon using ray casting algorithm
                inside = self.poly.is_inside(point)
                mask[i,j] = inside
        
        # Mask points outside polygon
        zi_masked = np.ma.masked_array(zi, ~mask)
        
        # Plot heatmap with masked data
        im = ax.pcolormesh(xi, yi, zi_masked, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        # plt.colorbar(im, ax=ax, orientation='horizontal')
        plt.colorbar(im, ax=ax)

    def estimate_solution_over_inner_points(self):
        # Estimate the solution over the inner points
        # This is done by using the equation (5.1) in the book
        self.solution = {} # reset the solution dictionary
        for blk in self.blocks:
            # get carrier function and poisson kernel for the current block
            for point in blk.inner_points:
                r, theta = self.to_block_polar_coordinates(blk, point)
                sum_ = 0
                for k in range(self.n_mu[blk.id_]):
                    poisson_kernel_value = self.get_poisson_kernel_value_in_block_coordinate_system(blk, r, theta, self.theta_mu[blk.id_][k])
                    sum_ += self.beta_mu[blk.id_] * (self.boundary_estimates[tuple(self.quantized_boundary_points[blk.id_][k])] - \
                                                        self.get_carrier_function_value_in_block_coordinate_system(blk, blk.r0, self.theta_mu[blk.id_][k])) * poisson_kernel_value
                self.solution[tuple(point)] = self.get_carrier_function_value_in_block_coordinate_system(blk, r, theta) + sum_
        
    def _initialize_solution(self):
        """
        Initialize the solution data structures for the block grid method.
        This method sets up the following instance variables:
        - alpha_mu: Dictionary mapping block IDs to angle parameters (block angle / pi)
        - n_mu: Dictionary mapping block IDs to number of angular divisions
        - beta_mu: Dictionary mapping block IDs to angular step sizes
        - theta_mu: Dictionary mapping block IDs to lists of angular coordinates
        - quantized_boundaries: Dictionary mapping block IDs to lists of boundary points
        - quantized_inner_points: Dictionary mapping block IDs to lists of interior points
        - solution: Dictionary mapping point coordinates to solution values (initialized to 0)

        The method calculates block parameters and quantizes each block into a grid of points
        based on the delta spacing parameter. Points are stored in global coordinates.

        Returns:
            None
        """
        self.alpha_mu = {}
        self.n_mu = {}
        self.beta_mu = {}
        self.theta_mu = {}
        self.quantized_boundary_points = {}
        self.boundary_estimates = {}
        
        # Calculate parameters for each block
        for blk in self.blocks:
            alpha, n, beta, theta = self.initialize_parameters_for_block_grid_method(blk)
            boundary_points = self.quantize_block(blk, theta)
            self.alpha_mu[blk.id_] = alpha
            self.n_mu[blk.id_] = n
            self.beta_mu[blk.id_] = beta
            self.theta_mu[blk.id_] = theta
            self.quantized_boundary_points[blk.id_] = boundary_points
        # Initialize solution values for all points
        for blk_id in self.quantized_boundary_points.keys():
            # Initialize boundary points
            for point in self.quantized_boundary_points[blk_id]:
                self.boundary_estimates[tuple(point)] = 0.0

    def initialize_parameters_for_block_grid_method(self, block: block):
        # see page 35 in the book, equations (4.1), (4.2), and (4.3)
        alpha_mu = block.angle / np.pi
        n_mu = np.max([4, int(np.abs(self.n * alpha_mu))])
        beta_mu = block.angle / n_mu
        theta_mu = [(k - 0.5) * beta_mu for k in range(1, n_mu + 1)]
        return alpha_mu, n_mu, beta_mu, theta_mu
    
    def quantize_block(self, block : block, theta_mu : list[float]):
        # Generate inner points by varying r and theta
        q_boundary_points = []
        for theta in theta_mu:
            q_boundary_points.append(self.from_block_polar_coordinates(block, block.r0, theta)) # convert to global coordinates
        return q_boundary_points

    def to_block_polar_coordinates(self, block: block, point : np.ndarray):
        # Get the polar coordinates of the point relative to the block's center
        v = point - block.center
        r = np.linalg.norm(v)
        # Calculate angle relative to reference vector
        theta = np.arctan2(v[1], v[0])
        
        # Choose reference vector based on block kind
        if block.block_kind == 1:  # First kind - use edge_j as reference
            theta = theta - np.arctan2(self.poly.edges[block.edge_j_index][1], self.poly.edges[block.edge_j_index][0])
        elif block.block_kind == 2:  # Second kind - use edge_i as reference 
            theta = theta - np.arctan2(self.poly.edges[block.edge_i_index][1], self.poly.edges[block.edge_i_index][0])   

        # Normalize theta to [0, 2*pi]
        while theta < 0:
            theta += 2*np.pi
        while theta >= 2*np.pi:
            theta -= 2*np.pi
            
        return r, theta
    
    def from_block_polar_coordinates(self, block: block, r : float, theta : float):
        # Normalize theta to [-pi, pi]
        while theta > np.pi:
            theta -= 2*np.pi
        while theta <= -np.pi:
            theta += 2*np.pi
        # Adjust theta based on block kind and reference vector
        if block.block_kind == 1:  # First kind - use edge_j as reference
            theta = theta + np.arctan2(self.poly.edges[block.edge_j_index][1], self.poly.edges[block.edge_j_index][0])
        elif block.block_kind == 2:  # Second kind - use edge_i as reference
            theta = theta + np.arctan2(self.poly.edges[block.edge_i_index][1], self.poly.edges[block.edge_i_index][0])
        # Convert polar coordinates to Cartesian coordinates
        x = block.center[0] + r * np.cos(theta)
        y = block.center[1] + r * np.sin(theta)
        return np.array([x, y])

    def find_tau_block_for_P_mu(self, point_mu : np.ndarray, id_mu : int):
        # Find the block that contains the point_mu that lies on the boundary of the extended block T_mu, but
        # also lies inside basic block T_tau. See equation (4.4) for the definition of T_mu and T_tau.
        # id_mu is the id of the block T_mu
        # Find all blocks that contain point_mu, excluding block id_mu
        containing_block_ids = []
        for blk in self.blocks:
            if blk.id_ != id_mu:
                # Calculate distance from point to block center
                distance = np.linalg.norm(point_mu - blk.center)
                # Check if point lies within block radius
                if distance <= blk.length:
                    containing_block_ids.append(blk.id_)
        
        # Return random id if any blocks found, otherwise None
        return random.choice(containing_block_ids) if containing_block_ids else None

    def get_boundary_conditions_in_block_coordinate_system(self, block : block):
        # Transform the boundary conditions to the block's polar coordinate system with origin at the block's center 
        # and angle 0 along the block's edge_j direction, this is only relevant to the first and second kind blocks
        # TODO: Implement higher order boundary conditions
        if block.block_kind == 1:
            if len(self.boundary_conditions[block.edge_i_index]) > 1 or len(self.boundary_conditions[block.edge_j_index]) > 1:
                raise NotImplementedError("The boundary conditions must be a polynomial of order 0")
            phi_i = self.boundary_conditions[block.edge_i_index]
            phi_j = self.boundary_conditions[block.edge_j_index]
            return phi_i, phi_j
        if block.block_kind == 2:
            if len(self.boundary_conditions[block.edge_i_index]) > 1:
                raise NotImplementedError("The boundary conditions must be a polynomial of order 0")
            phi_i = self.boundary_conditions[block.edge_i_index]
            return phi_i, None
        return None, None
    
    def get_carrier_function_value_in_block_coordinate_system(self, block : block, r : float, theta : float):
        # Get the carrier function in the block's polar coordinate system with origin at the block's center 
        # and angle 0 along the block's edge_j direction
        # TODO: Implement higher order boundary conditions, and allow other types of boundary conditions
        # for the carrier functions
        phi_i, phi_j = self.get_boundary_conditions_in_block_coordinate_system(block)
        if block.block_kind == 1:
            if self.is_dirichlet[block.edge_i_index] is True and self.is_dirichlet[block.edge_j_index] is True:
                # carrier function given by equation (3.2) in the book, below is incomplete function definition
                return phi_j[0] + ((phi_i[0] - phi_j[0]) / block.angle) * theta
            elif self.is_dirichlet[block.edge_i_index] is False and self.is_dirichlet[block.edge_j_index] is True:
                # carrier function given by equation (3.6) in the book
                raise NotImplementedError("Not implemented, see equation (3.6) in the book")
            elif self.is_dirichlet[block.edge_i_index] is True and self.is_dirichlet[block.edge_j_index] is False:
                # carrier function given by equation (3.9) in the book
                raise NotImplementedError("Not implemented, see equation (3.9) in the book")
            else:
                # carrier function given by equation (3.4) in the book
                raise NotImplementedError("Not implemented, see equation (3.4) in the book")
        elif block.block_kind == 2:
            # see equation (3.10), with k = 0
            return phi_i[0] * (self.is_dirichlet[block.edge_i_index] + \
                             (1 - self.is_dirichlet[block.edge_i_index]) * r * np.sin(theta))
        else:
            # see equation (3.11)
            return 0
        
    def get_poisson_kernel_value_in_block_coordinate_system(self, block : block, r : float, theta : float, eta : float):
        # Get the Poisson kernel in the block's polar coordinate system with origin at the block's center 
        # and angle 0 along the block's edge_j direction, see page 24 in the book. Also see equations (3.12), (3.13),
        # (3.14), (3.15), (3.16), (3.17), and (3.18) in the book
        main_kernel = lambda r1, theta1, eta1: (1 - r1**2) / (2 * np.pi * (1 - 2*r1*np.cos(theta1 - eta1) + r1**2))

        if block.block_kind == 1:
            # block of first kind
            nu_i = int(self.is_dirichlet[block.edge_i_index])
            nu_j = int(self.is_dirichlet[block.edge_j_index])
            alpha_j = block.angle / np.pi
            lambda_j = 1/((2 - nu_i * nu_j - (1 - nu_i) * (1 - nu_j)) * alpha_j)
            
            if nu_i == nu_j: # both are equal (3.17), (3,12)
                return lambda_j * (main_kernel((r / block.r0) ** lambda_j, lambda_j * theta, lambda_j * eta) + \
                                        (-1)**nu_j * main_kernel((r / block.r0) ** lambda_j, lambda_j * theta, -lambda_j * eta))
            else: # they are different (3.17), (3.13)
                return lambda_j * (main_kernel((r / block.r0) ** lambda_j, lambda_j * theta, lambda_j * eta) + \
                                        (-1)**nu_j * main_kernel((r / block.r0) ** lambda_j, lambda_j * theta, -lambda_j * eta) - \
                                        (-1)**nu_j * \
                                        (main_kernel((r / block.r0) ** lambda_j, lambda_j * theta, lambda_j * (np.pi - eta)) + \
                                        (-1)**nu_j * main_kernel((r / block.r0) ** lambda_j, lambda_j * theta, -lambda_j * (np.pi - eta)))
                                    )
        elif block.block_kind == 2:
            # block of second kind (3.16)
            m = int(self.is_dirichlet[block.edge_i_index])
            return main_kernel(r / block.r0, theta, eta) + (-1)**m * (main_kernel(r / block.r0, theta, -eta))
        else:
            # block of third kind (3.15)
            return main_kernel(r / block.r0, theta, eta)
        
    def estimate_solution_over_curved_boundaries(self):
        # Estimate the solution over the block curved boundaries using the boundary conditions and the carrier function
        # This is done by iterating over the blocks and estimating the solution over the curved boundaries
        # The iteration is stopped based on the max_iter parameter 
        # see equations (4.14) and (4.15) in the book
        i = 0
        while True: 
            for blk in self.blocks:
                # find the tau block for the current block
                for point in self.quantized_boundary_points[blk.id_]:
                    tau_block_id = self.find_tau_block_for_P_mu(point, blk.id_)
                    if tau_block_id is None:
                        warnings.warn(f"No tau block found for point {point}", RuntimeWarning)
                        continue
                    r_taumu, theta_taumu = self.to_block_polar_coordinates(self.blocks[tau_block_id], point)
                    sum_ = 0
                    sum_poisson_kernel = 0
                    for m in range(self.n_mu[tau_block_id]):
                        poisson_kernel_value = self.get_poisson_kernel_value_in_block_coordinate_system(self.blocks[tau_block_id], r_taumu, theta_taumu, self.theta_mu[tau_block_id][m])
                        sum_ += self.beta_mu[tau_block_id] * (self.boundary_estimates[tuple(self.quantized_boundary_points[tau_block_id][m])] - \
                                                                self.get_carrier_function_value_in_block_coordinate_system(self.blocks[tau_block_id], self.blocks[tau_block_id].length, self.theta_mu[tau_block_id][m])) * \
                                                                poisson_kernel_value
                        sum_poisson_kernel += poisson_kernel_value
                    sum_ = sum_ / max(1, self.beta_mu[tau_block_id] * sum_poisson_kernel)
                    if np.isnan(sum_):
                        print(f"Warning: NaN value encountered at point {point}")
                    self.boundary_estimates[tuple(point)] = self.get_carrier_function_value_in_block_coordinate_system(self.blocks[tau_block_id], r_taumu, theta_taumu) + sum_

            i += 1
            if i > self.max_iter:
                break       

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
                    for point in self.quantized_boundary_points[block.id_]:
                        ax.plot(point[0], point[1], 'b.')
                        if tuple(point) in self.boundary_estimates:
                            ax.text(point[0], point[1], f'{self.boundary_estimates[tuple(point)]:.2f}', 
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
                    for point in self.quantized_boundary_points[block.id_]:
                        ax.plot(point[0], point[1], 'r.')
                        if tuple(point) in self.boundary_estimates:
                            ax.text(point[0], point[1], f'{self.boundary_estimates[tuple(point)]:.2f}',
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
                    for point in self.quantized_boundary_points[block.id_]:
                        ax.plot(point[0], point[1], 'g.')
                        if tuple(point) in self.boundary_estimates:
                            ax.text(point[0], point[1], f'{self.boundary_estimates[tuple(point)]:.2f}',
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
                ax.text(midpoint[0], midpoint[1], f'$\phi_{{ {i+1} }} = {self.boundary_conditions[i]}$',
                        horizontalalignment='left', verticalalignment='top',
                        rotation=angle)
            
        # Plot uncovered points
        if uncovered_points:
            uncovered_points = np.array(uncovered_points)
            ax.scatter(uncovered_points[:,0], uncovered_points[:,1], c='red', s=1, alpha=0.5)
            
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

            w = np.array([self.poly.vertices[m % num_vertices] for m in range(i + 1, i + num_vertices - 1)])
            v = np.array([self.poly.vertices[m % num_vertices] for m in range(i + 2, i + num_vertices)])
            # NOTE: If distance from all the other edges is not checked, it is known that if the polygon is not convex, 
            # the blocks of first kind might leak outside the polygon
            r0 = self.radial_heuristic * min(self.distances_from_edges(v, w, vertex))
            length = self.radial_heuristic * r0

            # create a block of first kind at the vertex
            # for first kind blocks, the id_ is the index of the vertex
            self.blocks.append(block(vertex, angle, length, r0, block_kind=1, id_=i, edge_i_index=i-1, edge_j_index=i)) # first kind block
            self.N += 1
            
            # Check if there exists a gap between the blocks of first kind on previous vertex and current vertex
            # TODO: if a second kind block is leaking, it could be decomposed into smaller blocks to cover the same gap
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

                edge_starts = np.array([self.poly.vertices[m % num_vertices] for m in range(i, i + num_vertices - 1)])
                edge_ends = np.array([self.poly.vertices[m % num_vertices] for m in range(i + 1, i + num_vertices)])
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
                    for j in range(num_second_kind_blocks):
                        # start adding blocks from the prev_vertex
                        next_loc = min(2 * (j + 1) * (radius_half_disk - o_rad), max_dist)
                        new_vertex = start_vertex + next_loc * unit_edge_vector 
                        max_allowed_radius = self.radial_heuristic * min(self.distances_from_edges(edge_starts, edge_ends, new_vertex))
                        if max_allowed_radius > r0_half_disk:
                            second_kind_blocks.append(block(new_vertex, np.pi, radius_half_disk, r0_half_disk, block_kind=2, 
                                                    id_=block_id_counter, edge_i_index=edge_i_index, edge_j_index=None)) # second kind block
                            self.L += 1
                            block_id_counter += 1        
                        else:
                            num_blocks, new_blocks = self.break_second_kind_block(new_vertex, r0_half_disk, max_allowed_radius, edge_starts, 
                                                                        edge_ends, unit_edge_vector, edge_i_index, block_id_counter)
                            second_kind_blocks.extend(new_blocks)
                            self.L += num_blocks
                            block_id_counter += num_blocks
        
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

            edge_starts = np.array([self.poly.vertices[m % num_vertices] for m in range(0, num_vertices - 1)])
            edge_ends = np.array([self.poly.vertices[m % num_vertices] for m in range(1, num_vertices)])
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
                for j in range(num_second_kind_blocks):
                    # start adding blocks from the prev_vertex
                    next_loc = min(2 * (j + 1) * (radius_half_disk - o_rad), max_dist)
                    new_vertex = start_vertex + next_loc * unit_edge_vector 
                    max_allowed_radius = self.radial_heuristic * min(self.distances_from_edges(edge_starts, edge_ends, new_vertex))
                    if max_allowed_radius > r0_half_disk:
                        second_kind_blocks.append(block(new_vertex, np.pi, radius_half_disk, r0_half_disk, block_kind=2, 
                                                id_=block_id_counter, edge_i_index=edge_i_index, edge_j_index=None)) # second kind block
                        self.L += 1
                        block_id_counter += 1        
                    else:
                        num_blocks, new_blocks = self.break_second_kind_block(new_vertex, r0_half_disk, max_allowed_radius, edge_starts, 
                                                                    edge_ends, unit_edge_vector, edge_i_index, block_id_counter)
                        second_kind_blocks.extend(new_blocks)
                        self.L += num_blocks
                        block_id_counter += num_blocks

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
                new_vertex = old_vertex + (- radius_old_block + new_radius + 2 * i * overlap * new_radius) * unit_edge_vector
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
        # Create a fine grid of points
        x_min, y_min = np.min(self.poly.vertices, axis=0) 
        x_max, y_max = np.max(self.poly.vertices, axis=0) 
        x = np.linspace(x_min, x_max, int((x_max - x_min) / self.delta))
        y = np.linspace(y_min, y_max, int((y_max - y_min) / self.delta))
        X, Y = np.meshgrid(x, y)
        points = np.column_stack((X.ravel(), Y.ravel()))
        
        # Find points that are in polygon but not in any block
        uncovered_points = []
        for point in points:
            if self.poly.is_inside(point):
                covered = False
                # Check first kind blocks
                for blk in self.blocks:
                    if blk.is_inside(point):
                        covered = True
                        blk.inner_points.append(point)
                        break
                if not covered:
                    uncovered_points.append(point)

        return uncovered_points

    def create_third_kind_blocks(self, uncovered_points : list[np.ndarray]):
        # This function will add blocks of third kind to the list of blocks
        # Convert uncovered points to numpy array for easier manipulation
        uncovered_points = np.array(uncovered_points)
        self.M = self.L
        
        # Skip if no uncovered points
        if len(uncovered_points) == 0:
            return
            
        # Find connected components/contours by clustering points that are within delta distance
        visited = np.zeros(len(uncovered_points), dtype=bool)
        
        # Create arrays for start and end points of edges
        edge_starts = self.poly.vertices
        edge_ends = np.vstack([self.poly.vertices[1:], self.poly.vertices[0]])
        
        # Randomize order of processing uncovered points
        indices = np.random.permutation(len(uncovered_points))
        for i in indices:
            if visited[i]:
                continue
                
            # Get current point as center
            current_center = uncovered_points[i]
            visited[i] = True
            
            # Calculate minimum distance from edges to determine radius
            radius = np.min(self.distances_from_edges(edge_starts, edge_ends, current_center))
            
            # Create third kind block with calculated center and radius
            r0 = self.radial_heuristic * radius 
            length = self.radial_heuristic * r0
            new_block = block(current_center, 2 * np.pi, length, r0, block_kind=3, 
                          id_=self.M)
            
            # Find all points connected to this one
            # Check distances to all unvisited points
            distances = np.linalg.norm(uncovered_points - current_center, axis=1)
            neighbors = np.where((distances <= length) & (~visited))[0]

            # Update visited array for neighbors
            visited[neighbors] = True
            # Add neighbors to inner points of the block
            for neighbor_idx in neighbors:
                new_block.inner_points.append(uncovered_points[neighbor_idx])

            self.blocks.append(new_block)
            self.M += 1
    
    def plot_gradient(self, ax, solution, N = 50):
        # Plot the solution on the given axis
        # Convert solution points and values to grid format for heatmap
        # Plot the solution on the given axis
        # Convert solution points and values to grid format for heatmap
        x_coords = [point[0] for point in solution.keys()]
        y_coords = [point[1] for point in solution.keys()]
        values = list(solution.values())
        
        # Create regular grid
        xi = np.linspace(min(x_coords), max(x_coords), N)
        yi = np.linspace(min(y_coords), max(y_coords), N)
        xi, yi = np.meshgrid(xi, yi)
        
        # Interpolate scattered data to regular grid using scipy's griddata
        zi = griddata((x_coords, y_coords), values, (xi, yi), method='linear')
        
        # Create mask for points outside polygon
        mask = np.zeros_like(zi, dtype=bool)
        for i in range(len(xi)):
            for j in range(len(yi)):
                point = np.array([xi[i,j], yi[i,j]])
                # Check if point is inside polygon using ray casting algorithm
                inside = self.poly.is_inside(point)
                mask[i,j] = inside
        
        # Mask points outside polygon
        zi_masked = np.ma.masked_array(zi, ~mask)
        
        # Calculate gradients using np.gradient
        dy, dx = np.gradient(zi_masked)
        
        # Normalize the gradient field
        magnitude = np.sqrt(dx**2 + dy**2)
        # Avoid division by zero
        magnitude = np.where(magnitude == 0, 1, magnitude)
        dx_norm = dx / magnitude
        dy_norm = dy / magnitude
        
        # Plot normalized vector field using quiver on provided axes
        ax.quiver(xi[::2, ::2], yi[::2, ::2],
                 dx_norm[::2, ::2], dy_norm[::2, ::2],
                 scale=40)  # Reduce scale to make arrows bigger