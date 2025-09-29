#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Block covering strategies for domain discretization in the Volkov method.

This module implements the block covering algorithm as described in
E.A. Volkov's book "Block Method for Solving the Laplace Equation and for
Constructing Conformal Mappings (1994)" ISBN: 0849394066.

The block covering creates three types of blocks:
1. First kind: Sectors extending from vertices
2. Second kind: Half-disks extending from edges
3. Third kind: Full disks inside the polygon
"""

from typing import List, Tuple
import numpy as np
from .block import block
from .polygon import polygon


class BlockCoveringStrategy:
    """
    Orchestrates creation of all three kinds of blocks to cover the polygonal domain.

    This class implements the complete block covering algorithm that creates
    a covering of the polygon using sectors, half-disks, and interior disks.
    """

    def __init__(self, radial_heuristic=0.8, overlap_heuristic=0.1):
        """
        Initialize the block covering strategy.

        Parameters:
            radial_heuristic (float): Heuristic for tapering block length (0.707-1.0)
            overlap_heuristic (float): Heuristic for block overlap (0-0.5)
        """
        self.radial_heuristic = radial_heuristic
        self.overlap_heuristic = overlap_heuristic

    def create_covering(self, poly: polygon) -> Tuple[List[block], int, int, int]:
        """
        Create complete block covering of the polygon domain.

        This function finds the block covering of the polygon and returns the blocks
        along with counts of each kind.

        Parameters:
            poly (polygon): The polygon to cover

        Returns:
            tuple: (blocks, N, L, M) where
                - blocks: List of all blocks
                - N: Number of first kind blocks (vertices)
                - L: N + number of second kind blocks (edges)
                - M: Total number of blocks (including third kind)
        """
        # Initialize counters
        N = 0  # First kind blocks
        L = 0  # First + second kind blocks
        M = 0  # Total blocks
        all_blocks = []
        block_id_counter = len(poly.vertices)

        # Step 1: Create first kind blocks (vertex sectors)
        vertex_blocks = self._create_first_kind_blocks(poly)
        all_blocks.extend(vertex_blocks)
        N = len(vertex_blocks)

        # Step 2: Create second kind blocks (edge half-disks)
        edge_blocks = self._create_second_kind_blocks(
            poly, vertex_blocks, block_id_counter
        )
        all_blocks.extend(edge_blocks)
        L = N + len(edge_blocks)

        # Step 3: Find uncovered points
        uncovered_points = self._find_uncovered_points(poly, all_blocks)

        # Step 4: Create third kind blocks (interior disks)
        interior_blocks = self._create_third_kind_blocks(poly, uncovered_points, L)
        all_blocks.extend(interior_blocks)
        M = L + len(interior_blocks)

        return all_blocks, N, L, M

    def _create_first_kind_blocks(self, poly: polygon) -> List[block]:
        """
        Create blocks of first kind (sectors from vertices).

        For each vertex of the main polygon and holes, creates a circular sector block
        extending from the vertex with angle equal to the interior angle.

        Parameters:
            poly (polygon): The polygon domain with possible holes

        Returns:
            List[block]: List of first kind blocks
        """
        vertex_blocks = []
        block_id = 0

        # Get unified edge arrays for distance constraints
        edge_starts, edge_ends, _ = self._get_all_edges_unified(poly)

        # Process each vertex using unified vertex iterator
        for (
            vertex,
            angle,
            boundary_type,
            boundary_id,
            vertex_id,
        ) in self._get_all_vertices_with_boundaries(poly):
            # Calculate distances to non-adjacent edges using original logic
            # Create non-adjacent edge arrays by excluding the two adjacent edges
            if boundary_type == "main":
                # For main polygon vertices, exclude adjacent main polygon edges
                num_vertices = len(poly.vertices)
                adjacent_mask = np.ones(len(edge_starts), dtype=bool)
                adjacent_mask[vertex_id] = False  # Current vertex's outgoing edge
                adjacent_mask[(vertex_id - 1) % num_vertices] = (
                    False  # Previous vertex's outgoing edge
                )
            else:
                # For hole vertices, exclude adjacent hole edges
                # Find the start index of this hole's edges in the unified array
                hole_start_idx = len(poly.vertices)  # Skip main polygon edges
                for h_idx in range(boundary_id):
                    hole_start_idx += len(poly.holes[h_idx].vertices)

                num_hole_vertices = len(poly.holes[boundary_id].vertices)
                adjacent_mask = np.ones(len(edge_starts), dtype=bool)
                adjacent_mask[hole_start_idx + vertex_id] = (
                    False  # Current vertex's outgoing edge
                )
                adjacent_mask[hole_start_idx + (vertex_id - 1) % num_hole_vertices] = (
                    False  # Previous edge
                )

            # Apply non-adjacent edge filtering
            non_adjacent_starts = edge_starts[adjacent_mask]
            non_adjacent_ends = edge_ends[adjacent_mask]

            # Extract adjacent edges for normal calculation
            adjacent_starts = edge_starts[~adjacent_mask]
            adjacent_ends = edge_ends[~adjacent_mask]

            # Calculate average normal at vertex
            avg_normal = self._get_vertex_average_normal(
                adjacent_starts, adjacent_ends, vertex
            )

            # Calculate radius constraint using filtered distances
            r0 = self.radial_heuristic * min(
                min(
                    self._distances_from_edges_filtered(
                        non_adjacent_starts,
                        non_adjacent_ends,
                        vertex,
                        avg_normal,
                        angle,
                    )
                ),
                np.linalg.norm(adjacent_starts - vertex),
                np.linalg.norm(adjacent_ends - vertex),
            )
            length = self.radial_heuristic * r0

            # Create first kind block at vertex
            if boundary_type == "main":
                vertex_blocks.append(
                    block(
                        vertex,
                        angle,
                        length,
                        r0,
                        block_kind=1,
                        id_=block_id,
                        edge_i_index=(vertex_id - 1) % num_vertices,
                        edge_j_index=vertex_id,
                        boundary_type=boundary_type,
                        boundary_id=boundary_id,
                        vertex_id=vertex_id,
                    )
                )
            else:
                vertex_blocks.append(
                    block(
                        vertex,
                        angle,
                        length,
                        r0,
                        block_kind=1,
                        id_=block_id,
                        edge_i_index=(vertex_id - 1) % num_hole_vertices,
                        edge_j_index=vertex_id,
                        boundary_type=boundary_type,
                        boundary_id=boundary_id,
                        vertex_id=vertex_id,
                    )
                )

            block_id += 1

        return vertex_blocks

    def _create_second_kind_blocks(
        self, poly: polygon, vertex_blocks: List[block], block_id_counter: int
    ) -> List[block]:
        """
        Create blocks of second kind (half-disks from edges).

        Fills gaps between first kind blocks with half-disk blocks along edges.

        Parameters:
            poly (polygon): The polygon domain with possible holes
            vertex_blocks (List[block]): Previously created vertex blocks
            block_id_counter (int): Starting ID for new blocks

        Returns:
            List[block]: List of second kind blocks
        """
        edge_blocks = []
        vertex_block_idx = 0

        # Process main polygon edges (original logic)
        num_main_vertices = len(poly.vertices)
        for i in range(num_main_vertices):
            prev_idx = i - 1 if i > 0 else num_main_vertices - 1
            curr_idx = i

            prev_vertex = poly.vertices[prev_idx]
            curr_vertex = poly.vertices[curr_idx]

            # Calculate coverage by adjacent vertex blocks
            prev_block = vertex_blocks[vertex_block_idx + prev_idx]
            curr_block = vertex_blocks[vertex_block_idx + curr_idx]
            dist_covered = prev_block.length + curr_block.length
            edge_distance = np.linalg.norm(curr_vertex - prev_vertex)

            if edge_distance > dist_covered:
                # Gap exists, need second kind blocks
                gap_blocks = self._fill_edge_gap(
                    poly,
                    prev_vertex,
                    curr_vertex,
                    prev_block,
                    curr_block,
                    prev_idx,
                    block_id_counter,
                    "main",
                    0,
                )
                edge_blocks.extend(gap_blocks)
                block_id_counter += len(gap_blocks)

        vertex_block_idx += num_main_vertices

        # Process hole edges (same logic applied to each hole)
        for hole_id, hole in enumerate(poly.holes):
            num_hole_vertices = len(hole.vertices)
            for i in range(num_hole_vertices):
                prev_idx = i - 1 if i > 0 else num_hole_vertices - 1
                curr_idx = i

                prev_vertex = hole.vertices[prev_idx]
                curr_vertex = hole.vertices[curr_idx]

                # Calculate coverage by adjacent vertex blocks
                prev_block = vertex_blocks[vertex_block_idx + prev_idx]
                curr_block = vertex_blocks[vertex_block_idx + curr_idx]
                dist_covered = prev_block.length + curr_block.length
                edge_distance = np.linalg.norm(curr_vertex - prev_vertex)

                if edge_distance > dist_covered:
                    # Gap exists, need second kind blocks
                    gap_blocks = self._fill_edge_gap(
                        poly,
                        prev_vertex,
                        curr_vertex,
                        prev_block,
                        curr_block,
                        prev_idx,
                        block_id_counter,
                        "hole",
                        hole_id,
                    )
                    edge_blocks.extend(gap_blocks)
                    block_id_counter += len(gap_blocks)

            vertex_block_idx += num_hole_vertices

        return edge_blocks

    def _fill_gap_recursive(
        self,
        start_vertex: np.ndarray,
        end_vertex: np.ndarray,
        start_radius: float,
        end_radius: float,
        edge_starts_filtered: np.ndarray,
        edge_ends_filtered: np.ndarray,
        edge_normal: np.ndarray,
        edge_index: int,
        block_id_counter: int,
        unit_edge_vector: np.ndarray,
        boundary_type: str,
        boundary_id: int,
    ) -> Tuple[List[block], int]:
        """
        Recursively fill gap between two points with optimally-sized blocks.

        Parameters:
            start_vertex: Start point of gap
            end_vertex: End point of gap
            start_radius: Radius of block at start
            end_radius: Radius of block at end
            edge_starts_filtered: Edge start points (current edge excluded)
            edge_ends_filtered: Edge end points (current edge excluded)
            edge_normal: Inward normal for visibility filtering
            edge_index: Index of current edge (local to boundary)
            block_id_counter: Current block ID counter
            unit_edge_vector: Unit vector along edge direction
            boundary_type: "main" or "hole"
            boundary_id: 0 for main polygon, hole index for holes

        Returns:
            Tuple of (blocks_list, next_block_id)
        """
        gap_distance = np.linalg.norm(end_vertex - start_vertex)

        # Base case: gap too small for meaningful block
        min_threshold = 1e-6  # Minimum meaningful gap size
        if gap_distance < min_threshold:
            return [], block_id_counter

        # Calculate midpoint and try to place a block there
        midpoint = start_vertex + 0.5 * gap_distance * unit_edge_vector

        # Calculate maximum allowed radius at midpoint
        max_allowed_radius = self.radial_heuristic * min(
            self._distances_from_edges_filtered(
                edge_starts_filtered, edge_ends_filtered, midpoint, edge_normal, np.pi
            )
        )
        actual_radius = self.radial_heuristic * max_allowed_radius

        # Check if single block can cover the relaxed gap
        if 2 * actual_radius >= gap_distance:
            # Single block covers the gap
            new_block = block(
                midpoint,
                np.pi,
                actual_radius,
                max_allowed_radius,
                block_kind=2,
                id_=block_id_counter,
                edge_i_index=edge_index,
                edge_j_index=None,
                boundary_type=boundary_type,
                boundary_id=boundary_id,
            )
            return [new_block], block_id_counter + 1
        else:
            # Need to split: place midpoint block and recursively solve sub-gaps
            midpoint_block = block(
                midpoint,
                np.pi,
                actual_radius,
                max_allowed_radius,
                block_kind=2,
                id_=block_id_counter,
                edge_i_index=edge_index,
                edge_j_index=None,
                boundary_type=boundary_type,
                boundary_id=boundary_id,
            )

            blocks = [midpoint_block]
            next_id = block_id_counter + 1

            # Recursively fill left gap (start to midpoint)
            left_blocks, next_id = self._fill_gap_recursive(
                start_vertex,
                midpoint
                - (1 - self.overlap_heuristic) * actual_radius * unit_edge_vector,
                start_radius,
                actual_radius,
                edge_starts_filtered,
                edge_ends_filtered,
                edge_normal,
                edge_index,
                next_id,
                unit_edge_vector,
                boundary_type,
                boundary_id,
            )
            blocks.extend(left_blocks)

            # Recursively fill right gap (midpoint to end)
            right_blocks, next_id = self._fill_gap_recursive(
                midpoint
                + (1 - self.overlap_heuristic) * actual_radius * unit_edge_vector,
                end_vertex,
                actual_radius,
                end_radius,
                edge_starts_filtered,
                edge_ends_filtered,
                edge_normal,
                edge_index,
                next_id,
                unit_edge_vector,
                boundary_type,
                boundary_id,
            )
            blocks.extend(right_blocks)

            return blocks, next_id

    def _fill_edge_gap(
        self,
        poly: polygon,
        prev_vertex: np.ndarray,
        curr_vertex: np.ndarray,
        prev_block: block,
        curr_block: block,
        edge_index: int,
        block_id_counter: int,
        boundary_type: str,
        boundary_id: int,
    ) -> List[block]:
        """
        Fill gap between two vertex blocks with second kind blocks.

        Parameters:
            poly (polygon): The polygon
            prev_vertex (np.ndarray): Previous vertex
            curr_vertex (np.ndarray): Current vertex
            prev_block (block): Previous vertex block
            curr_block (block): Current vertex block
            edge_index (int): Index of the edge (local to boundary)
            block_id_counter (int): Starting ID for new blocks
            boundary_type (str): "main" or "hole"
            boundary_id (int): 0 for main polygon, hole index for holes

        Returns:
            List[block]: List of blocks filling the gap
        """
        # Calculate basic gap parameters
        distance = np.linalg.norm(curr_vertex - prev_vertex)
        dist_covered = prev_block.length + curr_block.length
        d = distance - dist_covered

        # If no gap exists, return empty list
        if d <= 0:
            return []

        unit_edge_vector = (curr_vertex - prev_vertex) / distance

        # Determine start and end positions accounting for existing block coverage
        start_vertex = (
            prev_vertex
            + (1 - self.overlap_heuristic) * prev_block.length * unit_edge_vector
        )
        end_vertex = (
            curr_vertex
            - (1 - self.overlap_heuristic) * curr_block.length * unit_edge_vector
        )

        # Create edge arrays for constraint checking (exclude current edge)
        edge_starts, edge_ends, _ = self._get_all_edges_unified(poly)

        # Calculate global edge index for filtering
        if boundary_type == "main":
            global_edge_index = edge_index
        else:  # hole
            # Calculate offset: main polygon edges + previous holes
            global_edge_index = len(poly.vertices)
            for h_idx in range(boundary_id):
                global_edge_index += len(poly.holes[h_idx].vertices)
            global_edge_index += edge_index

        edge_starts_filtered = np.delete(edge_starts, global_edge_index, axis=0)
        edge_ends_filtered = np.delete(edge_ends, global_edge_index, axis=0)

        # Calculate edge normal for filtering
        edge_normal = self._get_edge_inward_normal(unit_edge_vector)

        # Use recursive method to fill the gap
        blocks, _ = self._fill_gap_recursive(
            start_vertex,
            end_vertex,
            prev_block.length,
            curr_block.length,
            edge_starts_filtered,
            edge_ends_filtered,
            edge_normal,
            edge_index,
            block_id_counter,
            unit_edge_vector,
            boundary_type,
            boundary_id,
        )

        return blocks

    def _create_third_kind_blocks(
        self, poly: polygon, uncovered_points: np.ma.MaskedArray, starting_id: int
    ) -> List[block]:
        """
        Create blocks of third kind (interior disks).

        Places circular blocks to cover remaining uncovered interior points.

        Parameters:
            poly (polygon): The polygon domain with possible holes
            uncovered_points (np.ma.MaskedArray): Points not covered by other blocks
            starting_id (int): Starting ID for new blocks

        Returns:
            List[block]: List of third kind blocks
        """
        interior_blocks = []

        # Skip if no uncovered points
        if uncovered_points.mask.all():
            return interior_blocks

        # Get unified edge arrays for distance calculations
        edge_starts, edge_ends, _ = self._get_all_edges_unified(poly)

        # Find uncovered point coordinates
        y_coords, x_coords = np.where(~uncovered_points.mask)
        delta = getattr(uncovered_points, "delta", 0.01)  # Get delta from context
        x_min = getattr(uncovered_points, "x_min", 0.0)
        y_min = getattr(uncovered_points, "y_min", 0.0)

        points = np.column_stack((x_coords * delta + x_min, y_coords * delta + y_min))

        # Process points in random order to avoid bias
        visited = uncovered_points.mask.copy()
        indices = np.random.permutation(len(points))
        block_id = starting_id

        for i in indices:
            if visited[y_coords[i], x_coords[i]]:
                continue

            # Place block at this point
            current_center = points[i] + 0.5 * delta
            visited[y_coords[i], x_coords[i]] = True

            # Simple hole intersection check: skip if center is inside any hole
            if any(hole.is_inside(current_center) for hole in poly.holes):
                continue

            # Calculate radius based on distance to edges using original algorithm
            radius = np.min(
                self._distances_from_edges(edge_starts, edge_ends, current_center)
            )
            r0 = self.radial_heuristic * radius
            length = self.radial_heuristic * r0

            # Create third kind block
            new_block = block(
                current_center, 2 * np.pi, length, r0, block_kind=3, id_=block_id
            )

            # Mark nearby points as covered
            distances = np.linalg.norm(points - current_center, axis=1)
            neighbors = np.where(
                (distances <= length) & (~visited[y_coords, x_coords])
            )[0]
            visited[y_coords[neighbors], x_coords[neighbors]] = True

            interior_blocks.append(new_block)
            block_id += 1

        return interior_blocks

    def _find_uncovered_points(
        self,
        poly: polygon,
        blocks: List[block],
        solution_grid: np.ma.MaskedArray = None,
        cartesian_grid: np.ma.MaskedArray = None,
        inside_block_ids: np.ma.MaskedArray = None,
    ) -> np.ma.MaskedArray:
        """
        Find points that are not covered by any existing block.

        Parameters:
            poly (polygon): The polygon
            blocks (List[block]): Existing blocks
            solution_grid (np.ma.MaskedArray): Solution grid for shape reference
            cartesian_grid (np.ma.MaskedArray): Cartesian coordinates of grid points
            inside_block_ids (np.ma.MaskedArray): Array to track block assignments

        Returns:
            np.ma.MaskedArray: Mask indicating uncovered points
        """
        if solution_grid is None or cartesian_grid is None:
            # Fallback for when grid info is not available
            return np.ma.array([[]], mask=[[True]])

        # Find points in solution_grid that are not covered by any block
        uncovered_points = solution_grid.copy()
        # Create mask for points covered by blocks
        covered_mask = np.zeros_like(uncovered_points, dtype=bool)
        remaining_points = cartesian_grid.reshape((-1, 2))
        X, Y = np.where(~covered_mask)

        # Check each block for points inside it
        # Keep track of remaining points to check for next blocks
        for blk in blocks:
            inside_mask = blk.is_inside(remaining_points)
            inside_points = remaining_points[inside_mask]
            if inside_block_ids is not None:
                inside_block_ids[X[inside_mask], Y[inside_mask]] = blk.id_
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

        return uncovered_points

    def _distances_from_edges(
        self, v: np.ndarray, w: np.ndarray, p: np.ndarray
    ) -> np.ndarray:
        """
        Calculate distances from point p to line segments vw.

        Parameters:
            v (np.ndarray): Start points of line segments (Nx2)
            w (np.ndarray): End points of line segments (Nx2)
            p (np.ndarray): Point to check distance from (2D)

        Returns:
            np.ndarray: Distances from point p to each line segment
        """
        # Calculate length squared of line segments
        l2 = np.sum((w - v) ** 2, axis=1)

        # Handle segments that are actually points
        point_mask = l2 == 0.0
        distances = np.zeros(len(v))
        distances[point_mask] = np.linalg.norm(p - v[point_mask], axis=1)

        # For actual segments, find projection
        segment_mask = ~point_mask
        if np.any(segment_mask):
            t = (
                np.sum(
                    (p - v[segment_mask]) * (w[segment_mask] - v[segment_mask]), axis=1
                )
                / l2[segment_mask]
            )
            t = np.clip(t, 0, 1)

            projections = v[segment_mask] + t[:, np.newaxis] * (
                w[segment_mask] - v[segment_mask]
            )
            distances[segment_mask] = np.linalg.norm(p - projections, axis=1)

        return distances

    def _distances_from_edges_filtered(
        self,
        v: np.ndarray,
        w: np.ndarray,
        p: np.ndarray,
        inward_normal: np.ndarray,
        field_of_view_angle: float = np.pi,
    ) -> np.ndarray:
        """
        Calculate distances only to edges that could constrain a block.

        Parameters:
            v (np.ndarray): Start points of edges
            w (np.ndarray): End points of edges
            p (np.ndarray): Point to check
            inward_normal (np.ndarray): Inward normal vector for filtering
            field_of_view_angle (float): Full field-of-view angle in radians (default: π for 180°)

        Returns:
            np.ndarray: Filtered distances
        """
        # relaxing the field of view angle by around 1 degree on each side!
        cos_half_fov = np.cos((field_of_view_angle + 4e-2) / 2.0)

        # Filter edges based on geometric constraints
        v_dots = np.dot(v - p, inward_normal)
        w_dots = np.dot(w - p, inward_normal)

        v_norms = np.linalg.norm(v - p, axis=1)
        w_norms = np.linalg.norm(w - p, axis=1)

        keep_mask = (v_dots / v_norms >= cos_half_fov) | (
            w_dots / w_norms >= cos_half_fov
        )

        if np.any(keep_mask):
            return self._distances_from_edges(v[keep_mask], w[keep_mask], p)

        return np.array([np.inf])

    def _get_edge_inward_normal(self, edge_vector: np.ndarray) -> np.ndarray:
        """Get inward normal (90° CCW rotation) of edge vector."""
        return np.array([-edge_vector[1], edge_vector[0]]) / np.linalg.norm(edge_vector)

    def _get_vertex_average_normal(
        self, adjacent_starts: np.ndarray, adjacent_ends: np.ndarray, vertex: np.ndarray
    ) -> np.ndarray:
        """
        Calculate average inward normal at a vertex from its two adjacent edges.

        Parameters:
            adjacent_starts: Start points of the 2 adjacent edges (2x2 array)
            adjacent_ends: End points of the 2 adjacent edges (2x2 array)
            vertex: The vertex position (for determining edge direction)

        Returns:
            np.ndarray: Normalized average of the two edge normals
        """
        # Calculate the two edge vectors
        edge_vector1 = adjacent_ends[0] - adjacent_starts[0]
        edge_vector2 = adjacent_ends[1] - adjacent_starts[1]

        # Get normals for each edge (works for both main polygon and holes!)
        normal1 = self._get_edge_inward_normal(edge_vector1)
        normal2 = self._get_edge_inward_normal(edge_vector2)

        # Average and normalize
        avg_normal = (normal1 + normal2) / 2
        return avg_normal / np.linalg.norm(avg_normal)

    def _get_all_edges_unified(
        self, poly: polygon
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Get all edges (main polygon + holes) as unified arrays without phantom edges.

        Parameters:
            poly (polygon): The polygon domain with possible holes

        Returns:
            Tuple: (edge_starts, edge_ends, boundary_types) where each boundary
                  forms proper closed loops without phantom edges between boundaries
        """
        all_starts = []
        all_ends = []
        boundary_types = []

        # Main polygon edges (closed loop)
        main_starts = poly.vertices
        main_ends = np.roll(poly.vertices, -1, axis=0)
        all_starts.append(main_starts)
        all_ends.append(main_ends)
        boundary_types.extend(["main"] * len(poly.vertices))

        # Hole edges (each forms closed loop)
        for hole in poly.holes:
            hole_starts = hole.vertices
            hole_ends = np.roll(hole.vertices, -1, axis=0)
            all_starts.append(hole_starts)
            all_ends.append(hole_ends)
            boundary_types.extend(["hole"] * len(hole.vertices))

        # Concatenate all edge arrays (no phantom edges between boundaries)
        edge_starts = np.concatenate(all_starts, axis=0)
        edge_ends = np.concatenate(all_ends, axis=0)

        return edge_starts, edge_ends, boundary_types

    def _get_all_vertices_with_boundaries(self, poly: polygon):
        """
        Unified vertex iterator yielding vertex info with boundary context.

        Parameters:
            poly (polygon): The polygon domain with possible holes

        Yields:
            Tuple: (vertex, angle, boundary_type, boundary_id, vertex_id)
        """
        # Main polygon vertices
        for i, vertex in enumerate(poly.vertices):
            angle = poly.angles[i]
            yield (vertex, angle, "main", 0, i)

        # Hole vertices
        for hole_id, hole in enumerate(poly.holes):
            for vertex_id, vertex in enumerate(hole.vertices):
                angle = hole.interior_angles[vertex_id]
                yield (vertex, angle, "hole", hole_id, vertex_id)
