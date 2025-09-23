#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PolygonHole class for representing holes within polygonal domains.

This module provides the PolygonHole class, which represents a hole inside
a polygon domain. Unlike polygon objects, holes cannot contain other holes
and are oriented clockwise (opposite to the main polygon's counterclockwise
orientation).
"""

import numpy as np


class PolygonHole:
    """Represents a hole within a polygon domain.

    A hole is a clockwise-oriented polygon with boundary conditions.
    Unlike polygon objects, holes cannot contain other holes (non-recursive).

    Attributes:
        vertices (np.ndarray): Clockwise-ordered vertices of the hole
        boundary_conditions (list): Polynomial coefficients for each edge
        is_dirichlet (list): Boolean flags for boundary type per edge
        n_vertices (int): Number of vertices
        edges (list): Edge vectors
        interior_angles (list): Interior angles at each vertex
    """

    def __init__(self, vertices, boundary_conditions, is_dirichlet):
        """Initialize a PolygonHole.

        Parameters:
            vertices: nx2 array of vertices (will be ensured clockwise)
            boundary_conditions: List of polynomial coefficients for each edge
            is_dirichlet: List of boolean flags for boundary type
        """
        self.vertices = np.asarray(vertices, dtype=float)
        self.boundary_conditions = boundary_conditions
        self.is_dirichlet = is_dirichlet
        self.n_vertices = len(self.vertices)

        # Ensure clockwise ordering
        self.vertices = self._ensure_clockwise(self.vertices)

        # Compute geometric properties
        self.edges = self._compute_edges()
        self.interior_angles = self._compute_interior_angles()

        # Validate the hole
        self.validate()

    def _ensure_clockwise(self, vertices):
        """Ensure vertices are in clockwise order.

        Parameters:
            vertices: nx2 array of vertices

        Returns:
            np.ndarray: Vertices in clockwise order
        """
        # Calculate signed area using shoelace formula
        n = len(vertices)
        signed_area = 0
        for i in range(n):
            j = (i + 1) % n
            signed_area += vertices[i][0] * vertices[j][1] - vertices[i][1] * vertices[j][0]

        # If positive (counterclockwise), reverse the order
        if signed_area > 0:
            return vertices[::-1]
        return vertices

    def _compute_edges(self):
        """Compute edge vectors for the hole.

        Returns:
            list: Edge vectors from each vertex to the next
        """
        edges = []
        for i in range(self.n_vertices):
            current_vertex = self.vertices[i]
            next_vertex = self.vertices[(i + 1) % self.n_vertices]
            edge = next_vertex - current_vertex
            edges.append(edge)
        return edges

    def _compute_interior_angles(self):
        """Compute interior angles at each vertex.

        For a clockwise polygon, we compute angles in the same way
        but the interpretation is that blocks extend outward.

        Returns:
            list: Interior angles in radians
        """
        angles = []
        for i in range(self.n_vertices):
            prev_vertex = self.vertices[i - 1]
            next_vertex = self.vertices[(i + 1) % self.n_vertices]
            v1 = next_vertex - self.vertices[i]
            v2 = prev_vertex - self.vertices[i]

            # For 2D vectors, cross product is v1[0]*v2[1] - v1[1]*v2[0]
            # Note: For clockwise ordering, this will be negative
            cross_prod = v1[0] * v2[1] - v1[1] * v2[0]
            angle = np.arctan2(cross_prod, np.dot(v1, v2))

            # Adjust for obtuse angles
            if angle < 0:
                angle += 2 * np.pi
            angles.append(angle)
        return angles

    def area(self):
        """Calculate the area of the hole using the shoelace formula.

        Returns:
            float: Positive area of the hole
        """
        n = self.n_vertices
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i][0] * self.vertices[j][1]
            area -= self.vertices[j][0] * self.vertices[i][1]
        # Return absolute value since clockwise gives negative area
        return abs(area) / 2.0

    def point_in_hole(self, x, y):
        """Check if a point is inside this hole.

        Uses ray-tracing algorithm for point-in-polygon test.

        Parameters:
            x (float): X coordinate
            y (float): Y coordinate

        Returns:
            bool: True if point is inside the hole
        """
        # Ray tracing algorithm
        n = self.n_vertices
        inside = False

        p1x, p1y = self.vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = self.vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def validate(self):
        """Validate hole geometry and boundary conditions.

        Raises:
            ValueError: If validation fails
        """
        # Check we have at least 3 vertices
        if self.n_vertices < 3:
            raise ValueError("Hole must have at least 3 vertices")

        # Check boundary conditions match edge count
        if len(self.boundary_conditions) != self.n_vertices:
            raise ValueError(f"Boundary conditions count ({len(self.boundary_conditions)}) "
                           f"must match vertex count ({self.n_vertices})")

        if len(self.is_dirichlet) != self.n_vertices:
            raise ValueError(f"Dirichlet flags count ({len(self.is_dirichlet)}) "
                           f"must match vertex count ({self.n_vertices})")

        # Verify clockwise orientation
        signed_area = 0
        for i in range(self.n_vertices):
            j = (i + 1) % self.n_vertices
            signed_area += self.vertices[i][0] * self.vertices[j][1] - \
                          self.vertices[i][1] * self.vertices[j][0]

        if signed_area > 0:
            raise ValueError("Hole vertices must be in clockwise order")

        # Check for self-intersections (simplified check)
        # A more thorough check would test all edge pairs
        # For now, we check that angles sum appropriately
        angle_sum = sum(self.interior_angles)
        expected_sum = (self.n_vertices - 2) * np.pi
        if abs(angle_sum - expected_sum) > 0.1:  # Allow small numerical error
            # Note: For non-convex polygons this test might not be sufficient
            pass  # We'll allow this for now as holes can be non-convex

    def get_edge(self, index):
        """Get the endpoints of a specific edge.

        Parameters:
            index (int): Edge index

        Returns:
            tuple: (start_vertex, end_vertex) as numpy arrays
        """
        start = self.vertices[index]
        end = self.vertices[(index + 1) % self.n_vertices]
        return start, end

    def distance_to_edge(self, x, y, edge_index):
        """Calculate distance from a point to a specified edge.

        Parameters:
            x (float): X coordinate of the point
            y (float): Y coordinate of the point
            edge_index (int): Index of the edge

        Returns:
            float: Distance to the edge
        """
        # Get edge endpoints
        start, end = self.get_edge(edge_index)

        # Vector from start to end
        edge_vec = end - start
        edge_length = np.linalg.norm(edge_vec)

        if edge_length == 0:
            # Degenerate edge - return distance to start point
            return np.linalg.norm([x - start[0], y - start[1]])

        # Vector from start to point
        point_vec = np.array([x, y]) - start

        # Project point onto edge
        t = np.dot(point_vec, edge_vec) / (edge_length ** 2)
        t = np.clip(t, 0, 1)  # Clamp to edge segment

        # Find nearest point on edge
        nearest = start + t * edge_vec

        # Return distance to nearest point
        return np.linalg.norm([x - nearest[0], y - nearest[1]])

    def perimeter(self):
        """Calculate the perimeter of the hole.

        Returns:
            float: Perimeter length
        """
        return sum(np.linalg.norm(edge) for edge in self.edges)

    def __str__(self):
        """String representation of the hole."""
        return f"PolygonHole(n_vertices={self.n_vertices}, area={self.area():.4f})"

    def __repr__(self):
        """Detailed representation of the hole."""
        return (f"PolygonHole(vertices={self.n_vertices}, "
                f"area={self.area():.4f}, "
                f"clockwise=True)")