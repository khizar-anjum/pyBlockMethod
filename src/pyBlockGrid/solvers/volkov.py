#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volkov solver - Clean orchestrator for the block grid method.

This module implements E.A. Volkov's block grid method for solving the Laplace
equation on polygonal domains as described in the book "Block Method for Solving
the Laplace Equation and for Constructing Conformal Mappings (1994)" ISBN: 0849394066.

The refactored implementation delegates to specialized mathematical and core
components while maintaining the same public API as the original solver.
"""

import numpy as np
import matplotlib.pyplot as plt
from ..core.block import block
from ..core.polygon import polygon
from ..core.block_covering import BlockCoveringStrategy
from ..core.solution_state import SolutionState
from ..mathematical.carrier_functions import CarrierFunctions
from ..mathematical.poisson_kernels import PoissonKernels
from ..mathematical.boundary_estimator import BoundaryEstimator
from ..mathematical.interior_solver import InteriorSolver
from ..utils.coordinate_transforms import CoordinateTransforms
from .base import PDESolver


class volkovSolver(PDESolver):
    """
    Implements E.A. Volkov's block grid method for solving Laplace equation on polygonal domains.

    This is a refactored version that maintains full backward compatibility while
    providing a clean, modular architecture.
    """

    def __init__(self, poly: polygon, boundary_conditions: list[list[float]] = None,
                 is_dirichlet: list[bool] = None, n: int = 10, delta: float = 0.01,
                 max_iter: int = 100, radial_heuristic: float = 0.8,
                 overlap_heuristic: float = 0.1, tolerance: float = 1e-10,
                 verify_solution: bool = False):
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
        verify_solution (bool): Whether to verify the uniqueness of the solution using Lemma 4.2

        Returns:
        None
        """
        # Store and validate input parameters
        self.poly = poly
        self.max_iter = max_iter
        self.boundary_conditions = boundary_conditions
        assert len(self.boundary_conditions) == len(self.poly.edges), \
            "The number of boundary conditions must be equal to the number of edges of the polygon"

        self.is_dirichlet = is_dirichlet if is_dirichlet else [True for _ in range(len(self.poly.edges))]
        assert len(self.is_dirichlet) == len(self.poly.edges), \
            "The number of Dirichlet boundary conditions must be equal to the number of edges of the polygon"

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

        # Initialize component classes
        self.block_covering = BlockCoveringStrategy(self.radial_heuristic, self.overlap_heuristic)
        self.carrier_calc = CarrierFunctions(self.tol)
        self.poisson_calc = PoissonKernels()
        self.coord_transform = CoordinateTransforms()
        self.boundary_estimator = BoundaryEstimator(self.carrier_calc, self.poisson_calc)
        self.interior_solver = InteriorSolver(self.carrier_calc, self.poisson_calc)

        # These will be populated during solve()
        self.blocks = []
        self.N = 0  # Number of first kind blocks
        self.L = 0  # Number of first + second kind blocks
        self.M = 0  # Total number of blocks
        self.state = None
        self.solution = None

        # Initialize grid arrays immediately (like original implementation)
        self.x_min, self.y_min = np.min(self.poly.vertices, axis=0)
        self.x_max, self.y_max = np.max(self.poly.vertices, axis=0)
        self.min_xy = np.array([self.x_min, self.y_min])[np.newaxis, :]

        # Create solution array and mask points outside polygon (from original __init__)
        # Calculate grid dimensions
        self.ny = int(np.ceil((self.y_max - self.y_min) / self.delta))
        self.nx = int(np.ceil((self.x_max - self.x_min) / self.delta))

        # Create grid points more efficiently using meshgrid
        x = np.linspace(self.x_min + 0.5 * self.delta, self.x_max + 0.5 * self.delta, self.nx)
        y = np.linspace(self.y_min + 0.5 * self.delta, self.y_max + 0.5 * self.delta, self.ny)
        self._X, self._Y = np.meshgrid(x, y)

        # Create points array without using column_stack
        points = np.stack([self._X.ravel(), self._Y.ravel()], axis=1)

        # Create mask using vectorized operations (now handles holes)
        mask = self.poly.is_inside(points).reshape(self.ny, self.nx)

        # Create masked array directly
        self.solution = np.ma.masked_array(np.zeros((self.ny, self.nx)), mask=~mask)
        self.inside_block_ids_ = np.ma.masked_array(np.zeros((self.ny, self.nx), dtype=int), mask=~mask)
        self.cartesian_grid = np.ma.array(np.stack([self._X, self._Y], axis=2),
                                         mask=np.repeat(~mask[:, :, np.newaxis], 2, axis=2))

    def solve(self, verbose=False, plot=False, ax=None):
        """
        Solve the Laplace equation using Volkov's block grid method.

        This method orchestrates the complete solution process through four main steps:
        1. Create block covering of the domain
        2. Initialize solution data structures
        3. Estimate solution on curved boundaries
        4. Compute solution at interior points

        Parameters:
            verbose (bool): Whether to print progress information
            plot (bool): Whether to plot the solution
            ax: Matplotlib axis for plotting (if plot=True)

        Returns:
            np.ma.MaskedArray: Solution on the grid
        """
        if verbose:
            print("Starting solution process...")

        # Step 1: Find block covering (exactly like original)
        if verbose:
            print("Step 1: Finding block covering...")
        self.find_block_covering()

        # Step 2: Initialize solution data structures
        if verbose:
            print("Step 2: Initializing solution data structures...")
        self.initialize_solution()

        # Step 3: Estimate solution over curved boundaries
        if verbose:
            print("Step 3: Estimating solution over curved boundaries...")
        self.boundary_estimator.estimate(self.state, self.max_iter)

        # Step 4: Estimate solution over inner points
        if verbose:
            print("Step 4: Estimating solution over inner points...")
        self.solution = self.interior_solver.solve(self.state, self.tol)

        if verbose:
            print("Solution process complete.")
            print(f"N, L, M = {self.N}, {self.L}, {self.M}")

        if plot and ax is not None:
            self.plot_solution(ax)

        return self.solution

    def _complete_solution_initialization(self):
        """
        Complete the solution initialization with components that require all blocks.

        This handles the tau block calculations and other initialization steps that
        need the complete set of blocks.
        """
        # Find tau block IDs for P_mu points
        self.state.tau_block_ids = self._find_tau_block_ids_for_P_mu()

        # Create tau block centers array
        self.state.tau_block_centers = np.ma.array(
            self.state.center_mu[self.state.tau_block_ids],
            mask=np.repeat(self.state.tau_block_ids.mask[:, :, np.newaxis], 2, axis=2)
        )

        # Create tau ref thetas array
        self.state.tau_ref_thetas = np.ma.array(
            self.state.ref_theta[self.state.tau_block_ids],
            mask=self.state.tau_block_ids.mask
        )

        # Find tau block polar coordinates
        self.state.tau_block_polar_coordinates = self._find_tau_block_polar_coordinates()

        # Calculate Poisson kernel values
        self.state.poisson_kernel_values = self._find_poisson_kernel_values()

        # Verify solution uniqueness if requested
        if self.verify_solution:
            assert self.boundary_estimator.verify_unique_solution(
                self.state.poisson_kernel_values, self.state.beta_mu,
                self.state.tau_block_ids, self.state.theta_mu, self.M
            ), f"No unique solution exists for the natural number parameter n = {self.n}"

        # Calculate carrier function values
        self._find_carrier_function_values_on_blocks()

    def _find_tau_block_ids_for_P_mu(self):
        """Find block IDs that contain P_mu points lying on extended block boundaries."""
        points = np.repeat(self.state.quantized_boundary_points[np.newaxis, :], len(self.blocks), axis=0) - \
            self.state.center_mu[:, np.newaxis, np.newaxis]
        dists = np.linalg.norm(points, axis=3) <= self.state.r_mu[:, np.newaxis, np.newaxis]

        # Find the smallest index that is true in the first axis
        tau_block_ids = np.ma.array(np.argmax(dists, axis=0), mask=~np.any(dists, axis=0))
        return tau_block_ids

    def _find_tau_block_polar_coordinates(self):
        """Find polar coordinates of P_mu points relative to their tau blocks."""
        points = self.state.quantized_boundary_points - self.state.tau_block_centers
        return np.ma.array(
            np.stack([
                np.linalg.norm(points, axis=2),
                np.mod(np.arctan2(points[:, :, 1], points[:, :, 0]) - self.state.tau_ref_thetas, 2 * np.pi)
            ], axis=2),
            mask=points.mask
        )

    def _find_poisson_kernel_values(self):
        """Calculate Poisson kernel values for all boundary points."""
        eta_tau = self.state.theta_mu[self.state.tau_block_ids]
        poisson_kernel = np.ma.zeros_like(eta_tau)
        block_type_tau = self.state.block_type[self.state.tau_block_ids]
        nu_ij = self.state.block_nu[self.state.tau_block_ids]
        alpha_j = self.state.alpha_mu[self.state.tau_block_ids]
        r0_tau = self.state.r0_mu[self.state.tau_block_ids]
        r_ = self.state.tau_block_polar_coordinates[:, :, 0]
        theta_ = self.state.tau_block_polar_coordinates[:, :, 1]

        N, M = self.state.tau_block_ids.shape
        for i in range(N):
            for j in range(M):
                if self.state.tau_block_ids.mask[i, j]:
                    continue

                poisson_kernel[i, j] = self.poisson_calc.calculate(
                    block_type_tau[i, j], nu_ij[i, j, 0], nu_ij[i, j, 1],
                    r_[i, j], r0_tau[i, j], theta_[i, j], eta_tau[i, j], alpha_j[i, j]
                )

        return poisson_kernel

    def _find_carrier_function_values_on_blocks(self):
        """Calculate carrier function values on tau and mu blocks."""
        tau_carrier_function_values = np.ma.zeros_like(self.state.theta_mu)
        mu_carrier_function_values = np.ma.zeros_like(self.state.theta_mu)

        mu_block_boundary_identifiers = np.squeeze(self.state.block_nu.dot(np.array([[2], [1]])))
        tau_block_boundary_identifiers = mu_block_boundary_identifiers[self.state.tau_block_ids]
        tau_block_boundary_identifiers.mask = tau_block_boundary_identifiers.mask | self.state.tau_block_ids.mask

        tau_block_kind_identifiers = np.ma.masked_array(
            self.state.block_type[self.state.tau_block_ids],
            mask=self.state.tau_block_ids.mask
        )

        N, M = tau_carrier_function_values.shape
        k = np.arange(self.state.phi_ij.shape[1])

        # Calculate tau carrier function values
        for i in range(N):
            for j in range(M):
                if not (tau_carrier_function_values.mask[i, j] or self.state.tau_block_ids.mask[i, j]):
                    tau_id_ = self.state.tau_block_ids[i, j]
                    tau_alpha_j = self.state.alpha_mu[tau_id_]
                    r_ = self.state.tau_block_polar_coordinates[i, j, 0]
                    theta_ = self.state.tau_block_polar_coordinates[i, j, 1]
                    a_ = self.state.phi_ij[tau_id_, :, 0]
                    b_ = self.state.phi_ij[tau_id_, :, 1]

                    tau_carrier_function_values[i, j] = self.carrier_calc.calculate(
                        tau_block_kind_identifiers[i, j],
                        tau_block_boundary_identifiers[i, j],
                        r_, theta_, k, a_, b_, tau_alpha_j
                    )

        # Calculate mu carrier function values
        for i in range(N):
            for j in range(M):
                if not mu_carrier_function_values.mask[i, j]:
                    mu_id_ = i
                    alpha_j = self.state.alpha_mu[mu_id_]
                    r_ = self.state.r0_mu[i]
                    theta_ = self.state.theta_mu[i, j]
                    a_ = self.state.phi_ij[mu_id_, :, 0]
                    b_ = self.state.phi_ij[mu_id_, :, 1]

                    mu_carrier_function_values[i, j] = self.carrier_calc.calculate(
                        self.state.block_type[mu_id_],
                        mu_block_boundary_identifiers[mu_id_],
                        r_, theta_, k, a_, b_, alpha_j
                    )

        self.state.tau_carrier_function_values = tau_carrier_function_values
        self.state.mu_carrier_function_values = mu_carrier_function_values


    # Visualization methods - delegate to visualization module
    def plot_solution(self, ax, vmin=None, vmax=None):
        """Plot the solution on the given axis."""
        from ..visualization.volkov_plots import plot_solution_heatmap
        plot_solution_heatmap(self, ax, vmin, vmax)

    def plot_block_covering(self, ax, uncovered_points=None, show_boundary_conditions=True,
                           show_quantized_boundaries=False):
        """Plot the block covering visualization."""
        from ..visualization.volkov_plots import plot_block_covering
        plot_block_covering(self, ax, uncovered_points, show_boundary_conditions, show_quantized_boundaries)

    def plot_gradient(self, ax, decimation_factor=2, scale=20):
        """Plot the gradient field of the solution."""
        from ..visualization.volkov_plots import plot_gradient_field
        plot_gradient_field(self, ax, decimation_factor, scale)

    def initialize_solution(self):
        """
        Initialize solution data structures.

        This creates the solution state from the blocks that were created during find_block_covering().
        """
        # Create the solution state with the blocks we already have
        self.state = SolutionState.from_polygon_and_blocks(
            self.poly, self.blocks, self.delta, self.n,
            self.boundary_conditions, self.is_dirichlet, self.tol
        )

        # Copy over the grid arrays we already created
        self.state.X = self._X
        self.state.Y = self._Y
        self.state.solution = self.solution
        self.state.inside_block_ids = self.inside_block_ids_
        self.state.cartesian_grid = self.cartesian_grid

        self._complete_solution_initialization()

    def find_block_covering(self):
        """
        Find block covering of the polygon (following original implementation exactly).

        This function will find the block covering of the polygon and return the number of blocks of each kind.
        The blocks of first kind are the ones extending from the vertex (their number denoted by N)
        The blocks of second kind are the ones extending from the edge (their number denoted by L)
        The blocks of third kind are the ones inside the polygon (their number denoted by M)
        """
        # Create first and second kind blocks using the block covering strategy
        vertex_blocks = self.block_covering._create_first_kind_blocks(self.poly)
        edge_blocks = self.block_covering._create_second_kind_blocks(self.poly, vertex_blocks, len(vertex_blocks))

        self.blocks = vertex_blocks + edge_blocks
        self.N = len(vertex_blocks)
        self.L = self.N + len(edge_blocks)
        self.M = self.L  # Will be updated after third kind blocks

        # Find uncovered points using the already-created grid (like original)
        uncovered_points = self.find_uncovered_points()

        # Create third kind blocks
        self.create_third_kind_blocks(uncovered_points)

        return self.N, self.L, self.M, uncovered_points

    def find_uncovered_points(self):
        """
        Find points in self.solution that are not covered by any block.

        This is the original implementation that operates on the existing grid.
        """
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

    def create_third_kind_blocks(self, uncovered_points):
        """
        Create blocks of third kind (interior disks) from original implementation.
        Now supports polygons with holes.
        """
        # Skip if no uncovered points
        if uncovered_points.mask.all():
            return

        # Find connected components/contours by clustering points that are within delta distance
        visited = uncovered_points.mask

        # Get all constraining edges (main polygon + holes)
        edge_starts, edge_ends, _ = self.block_covering._get_all_edges_unified(self.poly)

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

            # Validate that the center is in valid domain (inside main polygon, outside holes)
            if not self.poly._point_in_polygon(current_center[0], current_center[1]):
                continue

            # Calculate minimum distance from ALL edges (main + holes) to determine radius
            radius = np.min(self.block_covering._distances_from_edges(edge_starts, edge_ends, current_center))

            # Create third kind block with calculated center and radius
            r0 = self.radial_heuristic * radius
            length = self.radial_heuristic * r0

            # Additional validation: ensure the block doesn't extend into holes
            block_is_valid = True
            for hole in self.poly.holes:
                # Check if block intersects with hole
                for edge_id in range(hole.n_vertices):
                    edge_distance = hole.distance_to_edge(current_center[0], current_center[1], edge_id)
                    if edge_distance < length:
                        # More precise check needed
                        block_is_valid = self._validate_third_kind_block(current_center, length, hole)
                        if not block_is_valid:
                            break
                if not block_is_valid:
                    break

            if not block_is_valid:
                continue

            new_block = block(current_center, 2 * np.pi, length, r0, block_kind=3, id_=self.M,
                            boundary_type='main', boundary_id=0)

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

    def _validate_third_kind_block(self, center, radius, hole):
        """
        Validate that a third kind block doesn't improperly intersect with a hole.

        Parameters:
            center: Block center coordinates
            radius: Block radius
            hole: PolygonHole object to check against

        Returns:
            bool: True if block placement is valid
        """
        # Check if block center is inside the hole
        if hole.point_in_hole(center[0], center[1]):
            return False

        # Check minimum distance to hole vertices
        for vertex in hole.vertices:
            if np.linalg.norm(center - vertex) < radius * 0.9:  # Small safety margin
                return False

        return True

    def validate_hole_solution(self, solution=None):
        """
        Validate that the solution satisfies boundary conditions on holes.

        Parameters:
            solution (np.ma.MaskedArray): Solution to validate (uses self.solution if None)

        Returns:
            dict: Validation results with boundary condition errors for each hole
        """
        if solution is None:
            solution = self.solution

        if not self.poly.holes:
            return {"message": "No holes to validate"}

        validation_results = {"holes": []}

        for hole_id, hole in enumerate(self.poly.holes):
            hole_errors = []

            # Check boundary conditions on each hole edge
            for edge_id in range(hole.n_vertices):
                edge_start = hole.vertices[edge_id]
                edge_end = hole.vertices[(edge_id + 1) % hole.n_vertices]

                # Sample points along the hole edge
                n_samples = 10
                edge_points = []
                for i in range(n_samples):
                    t = i / (n_samples - 1)
                    point = edge_start + t * (edge_end - edge_start)
                    edge_points.append(point)

                # Get boundary conditions for this edge
                bc = hole.boundary_conditions[edge_id]
                is_dirichlet = hole.is_dirichlet[edge_id]

                # Check boundary condition satisfaction
                for point in edge_points:
                    # Convert to grid coordinates
                    grid_x = int((point[0] - self.x_min) / self.delta)
                    grid_y = int((point[1] - self.y_min) / self.delta)

                    if 0 <= grid_x < self.nx and 0 <= grid_y < self.ny:
                        if not solution.mask[grid_y, grid_x]:
                            solution_value = solution[grid_y, grid_x]
                            expected_value = sum(bc[k] * point[0]**k for k in range(len(bc)))

                            if is_dirichlet:
                                error = abs(solution_value - expected_value)
                                hole_errors.append({
                                    "type": "Dirichlet",
                                    "edge": edge_id,
                                    "point": point.tolist(),
                                    "solution_value": float(solution_value),
                                    "expected_value": float(expected_value),
                                    "error": float(error)
                                })

            validation_results["holes"].append({
                "hole_id": hole_id,
                "errors": hole_errors,
                "max_error": max([e["error"] for e in hole_errors]) if hole_errors else 0.0,
                "mean_error": sum([e["error"] for e in hole_errors]) / len(hole_errors) if hole_errors else 0.0
            })

        return validation_results

    def check_solution_continuity(self, solution=None):
        """
        Check solution continuity across the domain, especially near hole boundaries.

        Parameters:
            solution (np.ma.MaskedArray): Solution to check (uses self.solution if None)

        Returns:
            dict: Continuity analysis results
        """
        if solution is None:
            solution = self.solution

        # Compute gradients
        dy, dx = np.gradient(solution.filled(0))
        gradient_magnitude = np.sqrt(dx**2 + dy**2)

        # Find maximum gradient locations (potential discontinuities)
        valid_mask = ~solution.mask
        valid_gradients = gradient_magnitude[valid_mask]

        if len(valid_gradients) == 0:
            return {"message": "No valid solution points"}

        max_gradient = np.max(valid_gradients)
        mean_gradient = np.mean(valid_gradients)

        # Find points with unusually high gradients
        high_gradient_threshold = mean_gradient + 3 * np.std(valid_gradients)
        high_gradient_locations = np.where((gradient_magnitude > high_gradient_threshold) & valid_mask)

        return {
            "max_gradient": float(max_gradient),
            "mean_gradient": float(mean_gradient),
            "std_gradient": float(np.std(valid_gradients)),
            "high_gradient_points": len(high_gradient_locations[0]),
            "continuity_score": float(mean_gradient / max_gradient) if max_gradient > 0 else 1.0
        }

    # Backward compatibility properties
    @property
    def X(self):
        """Backward compatibility property for grid X coordinates."""
        return self.state.X if self.state else None

    @property
    def Y(self):
        """Backward compatibility property for grid Y coordinates."""
        return self.state.Y if self.state else None