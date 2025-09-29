#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .polygon_hole import PolygonHole


class polygon:
    def __init__(self, vertices: np.ndarray):
        """
        Initialize the polygon class.

        Parameters:
        vertices (np.ndarray): The vertices of the polygon.
        boundary_conditions (list[list[float]]): The boundary conditions for the polygon. List of List of floats, where the inner
            list contains the coefficients of the polynomials (\\phi_j) that define the boundary conditions. The list has a length equal
            to the number of edges of the polygon. See the LaplaceEquationSolver class for more details.
        is_dirichlet (list[bool]): The parameter \\nu_j for the Laplace equation. List of booleans, where the jth boolean indicates the type of
            boundary condition on the jth edge. If not provided, Dirichlet boundary conditions are assumed on all edges.

        Returns:
        None
        """
        self.vertices = vertices.astype(float)
        assert self.verify_vertex_order(), "Vertices are not ordered counterclockwise"
        self.angles = self.calculate_angles()
        self.edges = self.calculate_edges()
        self.inner_connections = self.calculate_inner_connections()
        self.holes = []  # List of PolygonHole objects

    def calculate_inner_connections(self):
        # Calculate all inner connections between vertices that aren't adjacent
        connections = []
        n = len(self.vertices)
        # Check all pairs of vertices that aren't adjacent
        for i in range(n):
            for j in range(i + 2, n):
                # Skip if vertices are adjacent (including first-last vertex connection)
                if not (i == 0 and j == n - 1):
                    connections.append(
                        (self.vertices[i], self.vertices[j])
                    )  # from self.vertices[i] to self.vertices[j]
        return connections

    def calculate_angles(self):
        angles = []
        for i in range(len(self.vertices)):
            prev_vertex = self.vertices[i - 1]
            next_vertex = self.vertices[(i + 1) % len(self.vertices)]
            v1 = next_vertex - self.vertices[i]
            v2 = prev_vertex - self.vertices[i]
            # For 2D vectors, cross product is v1[0]*v2[1] - v1[1]*v2[0]
            cross_prod = v1[0] * v2[1] - v1[1] * v2[0]
            angle = np.arctan2(cross_prod, np.dot(v1, v2))
            # Adjust for obtuse angles - arctan2 returns values in [-pi, pi]
            if angle < 0:
                angle += 2 * np.pi  # Convert negative angles to [0, 2pi] range
            angles.append(angle)
        return angles

    def calculate_edges(self):
        edges = []
        for i in range(len(self.vertices)):
            current_vertex = self.vertices[i]
            next_vertex = self.vertices[(i + 1) % len(self.vertices)]
            edge = next_vertex - current_vertex
            edges.append(edge)
        return edges

    def plot(self, ax, show_holes=True):
        # Plot polygon edges
        vertices_closed = np.vstack(
            (self.vertices, self.vertices[0])
        )  # Add first vertex to end
        ax.plot(
            vertices_closed[:, 0],
            vertices_closed[:, 1],
            "k-",
            linewidth=2,
            label="Main boundary",
        )

        # Plot holes if requested
        if show_holes and self.holes:
            for i, hole in enumerate(self.holes):
                hole_vertices_closed = np.vstack((hole.vertices, hole.vertices[0]))
                ax.plot(
                    hole_vertices_closed[:, 0],
                    hole_vertices_closed[:, 1],
                    "k--",
                    linewidth=1.5,
                    label=f"Hole {i + 1}",
                )

    def plot_connections(self, ax):
        for connection in self.inner_connections:
            ax.plot(
                [connection[0][0], connection[1][0]],
                [connection[0][1], connection[1][1]],
                "k--",
            )

    def area(self):
        # Shoelace formula
        n = len(self.vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i][0] * self.vertices[j][1]
            area -= self.vertices[j][0] * self.vertices[i][1]
        return abs(area) / 2.0

    def perimeter(self):
        # Calculate the perimeter of the polygon
        return np.sum(np.linalg.norm(self.edges, axis=1))

    def verify_convexity(self):
        # Check if the polygon is convex
        # A polygon is convex if all interior angles are less than 180 degrees
        # We can check this by ensuring all cross products between consecutive edges have the same sign
        n = len(self.vertices)
        if n < 3:
            return False

        # Get vectors between consecutive vertices
        edges = np.roll(self.vertices, -1, axis=0) - self.vertices

        # Calculate cross products between consecutive edges
        # For 2D vectors, cross product is v1[0]*v2[1] - v1[1]*v2[0]
        next_edges = np.roll(edges, -1, axis=0)
        cross_products = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]

        # Check if all cross products have the same sign
        return np.all(cross_products >= 0) or np.all(cross_products <= 0)

    def verify_vertex_order(self):
        # Check if vertices are ordered counterclockwise
        # Calculate signed area using cross products
        n = len(self.vertices)
        signed_area = 0
        for i in range(n):
            j = (i + 1) % n
            # For 2D vectors, cross product is v1[0]*v2[1] - v1[1]*v2[0]
            signed_area += (
                self.vertices[i][0] * self.vertices[j][1]
                - self.vertices[i][1] * self.vertices[j][0]
            )
        # Positive signed area means counterclockwise ordering
        return signed_area > 0

    def is_inside(self, points: np.ndarray, ignore_holes=False):
        # assume that points is a numpy array of shape (N, 2)
        # Fully vectorized ray tracing algorithm for point-in-polygon test
        # Convert single point to array if needed
        points = np.atleast_2d(points)
        single_point = len(points) == 1

        # Get edges by pairing consecutive vertices
        edges = np.vstack((self.vertices, self.vertices[0]))
        edge_starts = edges[:-1]
        edge_ends = edges[1:]

        # Expand dimensions to allow broadcasting
        # points shape: (n_points, 2)
        # edge_starts shape: (n_edges, 2) -> (n_edges, 1, 2)
        # edge_ends shape: (n_edges, 2) -> (n_edges, 1, 2)
        edge_starts = edge_starts[:, np.newaxis, :]
        edge_ends = edge_ends[:, np.newaxis, :]

        # Get vectors from edge start to point and edge start to end
        v1 = points - edge_starts  # Shape: (n_edges, n_points, 2)
        v2 = edge_ends - edge_starts  # Shape: (n_edges, 1, 2)

        # Check if points are above/below each edge
        above = points[:, 1] >= edge_starts[:, :, 1]  # Shape: (n_edges, n_points)
        below = points[:, 1] < edge_ends[:, :, 1]  # Shape: (n_edges, n_points)
        # Or vice versa
        above_flip = points[:, 1] >= edge_ends[:, :, 1]
        below_flip = points[:, 1] < edge_starts[:, :, 1]

        # Find edges that could intersect with ray
        possible = (above & below) | (
            above_flip & below_flip
        )  # Shape: (n_edges, n_points)

        # Calculate intersection x coordinate where possible
        # Avoid division by zero by masking
        with np.errstate(divide="ignore", invalid="ignore"):
            t = v1[:, :, 1] / v2[:, :, 1]  # Shape: (n_edges, n_points)
            intersect_x = (
                edge_starts[:, :, 0] + t * v2[:, :, 0]
            )  # Shape: (n_edges, n_points)

        # Count valid intersections to the right of each point
        valid_intersections = (intersect_x > points[:, 0]) & possible
        intersections = np.sum(valid_intersections, axis=0)  # Shape: (n_points,)

        # Point is inside main polygon if number of intersections is odd
        inside = intersections % 2 == 1

        # Handle holes if present and not ignored
        if self.holes and not ignore_holes:
            # For each hole, check which points are inside it
            for hole in self.holes:
                # Vectorized hole check
                hole_edges = np.vstack((hole.vertices, hole.vertices[0]))
                hole_starts = hole_edges[:-1]
                hole_ends = hole_edges[1:]

                # Expand dimensions for broadcasting
                hole_starts = hole_starts[:, np.newaxis, :]
                hole_ends = hole_ends[:, np.newaxis, :]

                # Vectors for hole edges
                v1_hole = points - hole_starts
                v2_hole = hole_ends - hole_starts

                # Check intersections with hole edges
                above_h = points[:, 1] >= hole_starts[:, :, 1]
                below_h = points[:, 1] < hole_ends[:, :, 1]
                above_flip_h = points[:, 1] >= hole_ends[:, :, 1]
                below_flip_h = points[:, 1] < hole_starts[:, :, 1]

                possible_h = (above_h & below_h) | (above_flip_h & below_flip_h)

                with np.errstate(divide="ignore", invalid="ignore"):
                    t_h = v1_hole[:, :, 1] / v2_hole[:, :, 1]
                    intersect_x_h = hole_starts[:, :, 0] + t_h * v2_hole[:, :, 0]

                valid_intersections_h = (intersect_x_h > points[:, 0]) & possible_h
                intersections_h = np.sum(valid_intersections_h, axis=0)

                # Points are inside hole if intersections are odd
                inside_hole = intersections_h % 2 == 1

                # Exclude points that are inside this hole
                inside = inside & ~inside_hole

        # Return single bool if input was single point
        if single_point:
            return inside[0]
        return inside

    def add_hole(self, hole_vertices, boundary_conditions, is_dirichlet):
        """Add a hole to the polygon with validation.

        Parameters:
            hole_vertices: nx2 array of vertices (will be ensured clockwise)
            boundary_conditions: List of polynomial coefficients for hole edges
            is_dirichlet: List of boolean flags for hole boundary types

        Returns:
            PolygonHole: The created hole object
        """
        hole = PolygonHole(hole_vertices, boundary_conditions, is_dirichlet)
        self._validate_hole_placement(hole)
        self.holes.append(hole)
        return hole

    def _validate_hole_placement(self, hole):
        """Validate that a hole is inside the polygon and doesn't overlap other holes.

        Parameters:
            hole: PolygonHole object to validate

        Raises:
            ValueError: If hole placement is invalid
        """
        # Check all hole vertices are inside main polygon
        for vertex in hole.vertices:
            if not self._point_in_polygon_original(vertex[0], vertex[1]):
                raise ValueError(f"Hole vertex {vertex} is outside main polygon")

        # Check hole doesn't overlap existing holes
        for existing_hole in self.holes:
            if self._holes_overlap(hole, existing_hole):
                raise ValueError("Holes cannot overlap")

    def _holes_overlap(self, hole1, hole2):
        """Check if two holes overlap.

        Parameters:
            hole1: First PolygonHole
            hole2: Second PolygonHole

        Returns:
            bool: True if holes overlap
        """
        # Check if any vertex of hole1 is inside hole2
        for vertex in hole1.vertices:
            if hole2.point_in_hole(vertex[0], vertex[1]):
                return True

        # Check if any vertex of hole2 is inside hole1
        for vertex in hole2.vertices:
            if hole1.point_in_hole(vertex[0], vertex[1]):
                return True

        # TODO: Also check for edge intersections
        return False

    def _point_in_polygon_original(self, x, y):
        """Check if point is in main polygon (ignoring holes).

        This is the original is_inside method for single points.
        """
        points = np.array([[x, y]])
        # Use the existing is_inside logic but ignore holes
        edges = np.vstack((self.vertices, self.vertices[0]))
        edge_starts = edges[:-1]
        edge_ends = edges[1:]

        edge_starts = edge_starts[:, np.newaxis, :]
        edge_ends = edge_ends[:, np.newaxis, :]

        v1 = points - edge_starts
        v2 = edge_ends - edge_starts

        above = points[:, 1] >= edge_starts[:, :, 1]
        below = points[:, 1] < edge_ends[:, :, 1]
        above_flip = points[:, 1] >= edge_ends[:, :, 1]
        below_flip = points[:, 1] < edge_starts[:, :, 1]

        possible = (above & below) | (above_flip & below_flip)

        with np.errstate(divide="ignore", invalid="ignore"):
            t = v1[:, :, 1] / v2[:, :, 1]
            intersect_x = edge_starts[:, :, 0] + t * v2[:, :, 0]

        valid_intersections = (intersect_x > points[:, 0]) & possible
        intersections = np.sum(valid_intersections, axis=0)

        inside = intersections % 2 == 1
        return inside[0]

    def _point_in_polygon(self, x, y, ignore_holes=False):
        """Check if point is in polygon, accounting for holes.

        Parameters:
            x: X coordinate
            y: Y coordinate
            ignore_holes: If True, only check main polygon

        Returns:
            bool: True if point is in polygon and not in any hole
        """
        # First check if point is in main polygon
        if not self._point_in_polygon_original(x, y):
            return False

        if ignore_holes:
            return True

        # Check point is not inside any hole
        for hole in self.holes:
            if hole.point_in_hole(x, y):
                return False

        return True

    def area_with_holes(self):
        """Calculate area of polygon excluding hole areas.

        Returns:
            float: Total area minus hole areas
        """
        total_area = self.area()  # Main polygon area
        for hole in self.holes:
            total_area -= hole.area()
        return total_area

    def get_all_boundaries(self):
        """Return structured information about all boundaries (main + holes).

        Returns:
            dict: Dictionary containing main and hole boundary information
        """
        boundaries = {
            "main": {
                "vertices": self.vertices,
                "edges": self.edges,
                "angles": self.angles,
                "n_vertices": len(self.vertices),
            },
            "holes": [],
        }

        for hole in self.holes:
            boundaries["holes"].append(
                {
                    "vertices": hole.vertices,
                    "edges": hole.edges,
                    "angles": hole.interior_angles,
                    "n_vertices": hole.n_vertices,
                    "boundary_conditions": hole.boundary_conditions,
                    "is_dirichlet": hole.is_dirichlet,
                }
            )

        return boundaries
