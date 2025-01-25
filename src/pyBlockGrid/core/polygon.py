#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class polygon:
    def __init__(self, vertices : np.ndarray):
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
        self.angles= self.calculate_angles()
        self.edges = self.calculate_edges()
        self.inner_connections = self.calculate_inner_connections()

    def calculate_inner_connections(self):
        # Calculate all inner connections between vertices that aren't adjacent
        connections = []
        n = len(self.vertices)
        # Check all pairs of vertices that aren't adjacent
        for i in range(n):
            for j in range(i+2, n):
                # Skip if vertices are adjacent (including first-last vertex connection)
                if not (i == 0 and j == n-1):
                    connections.append((self.vertices[i], self.vertices[j])) # from self.vertices[i] to self.vertices[j]
        return connections

    def calculate_angles(self):
        angles = []
        for i in range(len(self.vertices)):
            prev_vertex = self.vertices[i-1]
            next_vertex = self.vertices[(i+1) % len(self.vertices)]
            angle = np.arctan2(np.cross(next_vertex - self.vertices[i], prev_vertex - self.vertices[i]), np.dot(next_vertex - self.vertices[i], prev_vertex - self.vertices[i]))
            # Adjust for obtuse angles - arctan2 returns values in [-pi, pi]
            if angle < 0:
                angle += 2 * np.pi  # Convert negative angles to [0, 2pi] range
            angles.append(angle)
        return angles
    
    def calculate_edges(self):
        edges = []
        for i in range(len(self.vertices)):
            current_vertex = self.vertices[i]
            next_vertex = self.vertices[(i+1) % len(self.vertices)]
            edge = next_vertex - current_vertex
            edges.append(edge)
        return edges

    def plot(self, ax):
        # Plot polygon edges
        vertices_closed = np.vstack((self.vertices, self.vertices[0]))  # Add first vertex to end
        ax.plot(vertices_closed[:,0], vertices_closed[:,1], 'k-')

    def plot_connections(self, ax):
        for connection in self.inner_connections:
            ax.plot([connection[0][0], connection[1][0]], [connection[0][1], connection[1][1]], 'k--')

    def area(self):
        # Shoelace formula
        n = len(self.vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i][0] * self.vertices[j][1]
            area -= self.vertices[j][0] * self.vertices[i][1]
        return abs(area) / 2.0
    
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
        cross_products = np.cross(edges, np.roll(edges, -1, axis=0))
        
        # Check if all cross products have the same sign
        return np.all(cross_products >= 0) or np.all(cross_products <= 0)
    
    def verify_vertex_order(self):
        # Check if vertices are ordered counterclockwise
        # Calculate signed area using cross products
        n = len(self.vertices)
        signed_area = 0
        for i in range(n):
            j = (i + 1) % n
            signed_area += np.cross(self.vertices[i], self.vertices[j])
        # Positive signed area means counterclockwise ordering
        return signed_area > 0

    def is_inside(self, point):
        # Ray casting algorithm to determine if point is inside polygon
        x, y = point
        inside = False
        j = len(self.vertices) - 1
        
        for i in range(len(self.vertices)):
            if ((self.vertices[i][1] > y) != (self.vertices[j][1] > y) and
                (x < (self.vertices[j][0] - self.vertices[i][0]) * (y - self.vertices[i][1]) /
                     (self.vertices[j][1] - self.vertices[i][1]) + self.vertices[i][0])):
                inside = not inside
            j = i
            
        return inside
    