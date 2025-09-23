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
        edge_blocks = self._create_second_kind_blocks(poly, vertex_blocks, block_id_counter)
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

        For each vertex of the polygon, creates a circular sector block
        extending from the vertex with angle equal to the interior angle.

        Parameters:
            poly (polygon): The polygon

        Returns:
            List[block]: List of first kind blocks
        """
        vertex_blocks = []
        num_vertices = len(poly.vertices)

        for i, vertex in enumerate(poly.vertices):
            # Get interior angle from polygon
            angle = poly.angles[i]

            # Calculate distances to non-adjacent edges for radius constraint
            w = np.roll(poly.vertices, -i-1, axis=0)[:num_vertices-2]
            v = np.roll(poly.vertices, -i-2, axis=0)[:num_vertices-2]
            r0 = self.radial_heuristic * min(self._distances_from_edges(v, w, vertex))
            length = self.radial_heuristic * r0

            # Create first kind block at vertex
            vertex_blocks.append(block(
                vertex, angle, length, r0, block_kind=1, id_=i,
                edge_i_index=i-1, edge_j_index=i
            ))

        return vertex_blocks

    def _create_second_kind_blocks(self, poly: polygon, vertex_blocks: List[block],
                                  block_id_counter: int) -> List[block]:
        """
        Create blocks of second kind (half-disks from edges).

        Fills gaps between first kind blocks with half-disk blocks along edges.

        Parameters:
            poly (polygon): The polygon
            vertex_blocks (List[block]): Previously created vertex blocks
            block_id_counter (int): Starting ID for new blocks

        Returns:
            List[block]: List of second kind blocks
        """
        edge_blocks = []
        num_vertices = len(poly.vertices)

        # Check gaps between adjacent vertex blocks
        for i in range(num_vertices):
            prev_idx = i - 1 if i > 0 else num_vertices - 1
            curr_idx = i

            prev_vertex = poly.vertices[prev_idx]
            curr_vertex = poly.vertices[curr_idx]

            # Calculate coverage by adjacent vertex blocks
            dist_covered = vertex_blocks[prev_idx].length + vertex_blocks[curr_idx].length
            edge_distance = np.linalg.norm(curr_vertex - prev_vertex)

            if edge_distance > dist_covered:
                # Gap exists, need second kind blocks
                gap_blocks = self._fill_edge_gap(
                    poly, prev_vertex, curr_vertex, vertex_blocks[prev_idx],
                    vertex_blocks[curr_idx], prev_idx, block_id_counter
                )
                edge_blocks.extend(gap_blocks)
                block_id_counter += len(gap_blocks)

        return edge_blocks

    def _fill_edge_gap(self, poly: polygon, prev_vertex: np.ndarray, curr_vertex: np.ndarray,
                      prev_block: block, curr_block: block, edge_index: int,
                      block_id_counter: int) -> List[block]:
        """
        Fill gap between two vertex blocks with second kind blocks.

        Parameters:
            poly (polygon): The polygon
            prev_vertex (np.ndarray): Previous vertex
            curr_vertex (np.ndarray): Current vertex
            prev_block (block): Previous vertex block
            curr_block (block): Current vertex block
            edge_index (int): Index of the edge
            block_id_counter (int): Starting ID for new blocks

        Returns:
            List[block]: List of blocks filling the gap
        """
        gap_blocks = []
        distance = np.linalg.norm(curr_vertex - prev_vertex)
        dist_covered = prev_block.length + curr_block.length
        d = distance - dist_covered

        unit_edge_vector = (curr_vertex - prev_vertex) / distance
        r0_half_disk = min(prev_block.r0, curr_block.r0)
        radius_half_disk = self.radial_heuristic * r0_half_disk
        o_rad = self.overlap_heuristic * radius_half_disk

        # Determine starting position
        if prev_block.r0 < curr_block.r0:
            start_vertex = prev_vertex
        else:
            start_vertex = prev_vertex + (prev_block.length - curr_block.length) * unit_edge_vector

        # Create edge arrays for constraint checking
        num_vertices = len(poly.vertices)
        if edge_index >= 0:
            edge_starts = np.roll(poly.vertices, -edge_index-1, axis=0)[:num_vertices-1]
            edge_ends = np.roll(poly.vertices, -edge_index-2, axis=0)[:num_vertices-1]
        else:  # Handle wrap-around case
            edge_starts = poly.vertices[:num_vertices-1]
            edge_ends = poly.vertices[1:]

        max_dist = d + 0.5 * radius_half_disk

        # Try single block first
        tentative_vertex = start_vertex + (radius_half_disk + 0.5 * d) * unit_edge_vector
        actual_r0 = self.radial_heuristic * min(
            self._distances_from_edges_filtered(edge_starts, edge_ends, tentative_vertex, edge_index, poly)
        )
        actual_radius = self.radial_heuristic * actual_r0

        if d < 2 * actual_radius:
            # Single block can cover the gap
            gap_blocks.append(block(
                tentative_vertex, np.pi, actual_radius, actual_r0, block_kind=2,
                id_=block_id_counter, edge_i_index=edge_index, edge_j_index=None
            ))
        else:
            # Multiple blocks needed
            num_second_kind_blocks = int(np.round((d + radius_half_disk) / (2 * (radius_half_disk - o_rad))))
            nextj = 0

            for j in range(num_second_kind_blocks):
                if j < nextj:
                    continue

                next_loc = min(2 * (j + 1) * (radius_half_disk - o_rad), max_dist)
                new_vertex = start_vertex + next_loc * unit_edge_vector
                max_allowed_radius = self.radial_heuristic * min(
                    self._distances_from_edges_filtered(edge_starts, edge_ends, new_vertex, edge_index, poly)
                )
                max_radius_ratio = np.floor(
                    self.radial_heuristic * max_allowed_radius / (2 * (radius_half_disk - o_rad))
                )

                if max_radius_ratio > 1:
                    gap_blocks.append(block(
                        new_vertex, np.pi, max_allowed_radius * self.radial_heuristic,
                        max_allowed_radius, block_kind=2,
                        id_=block_id_counter + len(gap_blocks),
                        edge_i_index=edge_index, edge_j_index=None
                    ))
                    nextj = nextj + max_radius_ratio
                elif max_allowed_radius > r0_half_disk:
                    gap_blocks.append(block(
                        new_vertex, np.pi, radius_half_disk, r0_half_disk, block_kind=2,
                        id_=block_id_counter + len(gap_blocks),
                        edge_i_index=edge_index, edge_j_index=None
                    ))
                    nextj = nextj + 1
                else:
                    # Need to break the block into smaller pieces
                    broken_blocks = self._break_second_kind_block(
                        new_vertex, r0_half_disk, max_allowed_radius, edge_starts,
                        edge_ends, unit_edge_vector, edge_index,
                        block_id_counter + len(gap_blocks)
                    )
                    gap_blocks.extend(broken_blocks)
                    nextj = nextj + 1

        return gap_blocks

    def _break_second_kind_block(self, old_vertex: np.ndarray, r0_old_block: float,
                                max_allowed_radius: float, edge_starts: np.ndarray,
                                edge_ends: np.ndarray, unit_edge_vector: np.ndarray,
                                edge_i_index: int, block_id_counter: int) -> List[block]:
        """
        Break a second kind block into smaller blocks to prevent leaking.

        Parameters:
            old_vertex (np.ndarray): Center of the block to break
            r0_old_block (float): Radius of the block to break
            max_allowed_radius (float): Maximum allowed radius
            edge_starts (np.ndarray): Start points of constraining edges
            edge_ends (np.ndarray): End points of constraining edges
            unit_edge_vector (np.ndarray): Unit vector along the edge
            edge_i_index (int): Index of the edge
            block_id_counter (int): Starting ID for new blocks

        Returns:
            List[block]: List of smaller blocks
        """
        ratio = int(2 ** np.floor(np.log2(r0_old_block / max_allowed_radius))) + 1
        done = False
        radius_old_block = self.radial_heuristic * r0_old_block

        vertices = []
        while not done:
            done = True
            n_min = ratio + 1
            new_radius = radius_old_block / ratio
            overlap = 1 / (n_min - 1)  # minimum overlap possible

            for i in range(n_min):
                new_vertex = old_vertex + (
                    -radius_old_block + new_radius + 2 * i * (1 - overlap) * new_radius
                ) * unit_edge_vector
                vertices.append(new_vertex)

                if min(self._distances_from_edges_filtered(
                    edge_starts, edge_ends, new_vertex, edge_i_index, poly=None
                )) < new_radius:
                    done = False
                    break

            if not done:
                ratio = ratio * 2
                vertices = []

        new_blocks = []
        new_r0 = new_radius / self.radial_heuristic
        for i in range(n_min):
            new_blocks.append(block(
                vertices[i], np.pi, new_radius, new_r0, block_kind=2,
                id_=block_id_counter + i, edge_i_index=edge_i_index, edge_j_index=None
            ))

        return new_blocks

    def _create_third_kind_blocks(self, poly: polygon, uncovered_points: np.ma.MaskedArray,
                                 starting_id: int) -> List[block]:
        """
        Create blocks of third kind (interior disks).

        Places circular blocks to cover remaining uncovered interior points.

        Parameters:
            poly (polygon): The polygon
            uncovered_points (np.ma.MaskedArray): Points not covered by other blocks
            starting_id (int): Starting ID for new blocks

        Returns:
            List[block]: List of third kind blocks
        """
        interior_blocks = []

        # Skip if no uncovered points
        if uncovered_points.mask.all():
            return interior_blocks

        # Get edge arrays for distance calculations
        edge_starts = poly.vertices
        edge_ends = np.roll(poly.vertices, -1, axis=0)

        # Find uncovered point coordinates
        y_coords, x_coords = np.where(~uncovered_points.mask)
        delta = getattr(uncovered_points, 'delta', 0.01)  # Get delta from context
        x_min = getattr(uncovered_points, 'x_min', 0.0)
        y_min = getattr(uncovered_points, 'y_min', 0.0)

        points = np.column_stack((
            x_coords * delta + x_min,
            y_coords * delta + y_min
        ))

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

            # Calculate radius based on distance to edges
            radius = np.min(self._distances_from_edges(edge_starts, edge_ends, current_center))
            r0 = self.radial_heuristic * radius
            length = self.radial_heuristic * r0

            # Create third kind block
            new_block = block(
                current_center, 2 * np.pi, length, r0, block_kind=3, id_=block_id
            )

            # Mark nearby points as covered
            distances = np.linalg.norm(points - current_center, axis=1)
            neighbors = np.where((distances <= length) & (~visited[y_coords, x_coords]))[0]
            visited[y_coords[neighbors], x_coords[neighbors]] = True

            interior_blocks.append(new_block)
            block_id += 1

        return interior_blocks

    def _find_uncovered_points(self, poly: polygon, blocks: List[block],
                                 solution_grid: np.ma.MaskedArray = None,
                                 cartesian_grid: np.ma.MaskedArray = None,
                                 inside_block_ids: np.ma.MaskedArray = None) -> np.ma.MaskedArray:
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

    def _distances_from_edges(self, v: np.ndarray, w: np.ndarray, p: np.ndarray) -> np.ndarray:
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
            t = np.sum((p - v[segment_mask]) * (w[segment_mask] - v[segment_mask]), axis=1) / l2[segment_mask]
            t = np.clip(t, 0, 1)

            projections = v[segment_mask] + t[:, np.newaxis] * (w[segment_mask] - v[segment_mask])
            distances[segment_mask] = np.linalg.norm(p - projections, axis=1)

        return distances

    def _distances_from_edges_filtered(self, v: np.ndarray, w: np.ndarray, p: np.ndarray,
                                      current_edge_index: int, poly: polygon = None) -> np.ndarray:
        """
        Calculate distances only to edges that could constrain a half-disk.

        Parameters:
            v (np.ndarray): Start points of edges
            w (np.ndarray): End points of edges
            p (np.ndarray): Point to check
            current_edge_index (int): Index of current edge
            poly (polygon): Polygon for edge information

        Returns:
            np.ndarray: Filtered distances
        """
        if poly is not None:
            current_edge = poly.edges[current_edge_index]
            inward_normal = self._get_edge_inward_normal(current_edge)

            # Filter edges based on geometric constraints
            v_dots = np.dot(v - p, inward_normal)
            w_dots = np.dot(w - p, inward_normal)
            keep_mask = (v_dots > -1e-10) | (w_dots > -1e-10)

            if np.any(keep_mask):
                return self._distances_from_edges(v[keep_mask], w[keep_mask], p)

        return np.array([np.inf])

    def _get_edge_inward_normal(self, edge_vector: np.ndarray) -> np.ndarray:
        """Get inward normal (90° CCW rotation) of edge vector."""
        return np.array([-edge_vector[1], edge_vector[0]]) / np.linalg.norm(edge_vector)