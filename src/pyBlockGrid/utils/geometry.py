#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def break_polygon_edges_and_bcs(vertices, boundary_conditions, is_dirichlet, radial_heuristic):
    # break polygon edges and boundary conditions
    # Break edges that are too long
    i = 0
    while i < len(vertices):
        prev_vertex = vertices[i-1]
        next_vertex = vertices[i]
        
        # Get edges connected to each vertex
        prev_edges = [vertices[i-2] - prev_vertex, next_vertex - prev_vertex]
        next_edges = [prev_vertex - next_vertex, vertices[(i+1) % len(vertices)] - next_vertex]
        
        # Calculate min edge length for each vertex
        prev_min_edge = min(np.linalg.norm(edge) for edge in prev_edges)
        next_min_edge = min(np.linalg.norm(edge) for edge in next_edges)
        
        # Calculate threshold based on radial_heuristic * radial_heuristic * sum of min edges
        threshold = radial_heuristic * radial_heuristic * (prev_min_edge + next_min_edge)
        
        # Check if distance between vertices exceeds threshold
        distance = np.linalg.norm(next_vertex - prev_vertex)
        if distance > threshold:
            radius_half_disk = radial_heuristic * radial_heuristic * min(prev_min_edge, next_min_edge) 
            # Calculate number of segments needed
            num_segments = int(np.ceil(distance / radius_half_disk))
            # Add new vertices evenly spaced between prev_vertex and next_vertex
            for j in range(1, num_segments):
                t = j / num_segments
                new_vertex = prev_vertex + t * (next_vertex - prev_vertex)
                vertices = np.insert(vertices, i + j - 1, new_vertex, axis=0)
                boundary_conditions.insert(i + j - 1, boundary_conditions[i-1]) # insert boundary condition in a list
                is_dirichlet.insert(i + j - 1, is_dirichlet[i-1]) # insert nu_param in a list
            i += num_segments - 1  # Skip past all newly inserted vertices
        i += 1
    return vertices, boundary_conditions, is_dirichlet

def generate_random_polygon(n: int, min_radius: float = 1.0, max_radius: float = 2.0) -> np.ndarray:
    """
    Generate a random simply connected polygon with n vertices.
    Uses the following algorithm:
    1. Generate n random angles between 0 and 2π
    2. Sort angles to ensure vertices are ordered counterclockwise
    3. Generate random radii between min_radius and max_radius
    4. Convert polar coordinates to Cartesian coordinates
    
    Parameters:
    n (int): Number of vertices desired
    min_radius (float): Minimum radius from origin to any vertex
    max_radius (float): Maximum radius from origin to any vertex
    
    Returns:
    np.ndarray: Array of vertex coordinates with shape (n,2)
    """
    # Input validation
    if n < 3:
        raise ValueError("Polygon must have at least 3 vertices")
    if min_radius <= 0 or max_radius <= min_radius:
        raise ValueError("Invalid radius bounds")
        
    # Generate n random angles between 0 and 2π
    angles = np.random.uniform(0, 2*np.pi, n)
    # Sort angles to ensure counterclockwise ordering
    angles.sort()
    
    # Generate random radii between min_radius and max_radius
    radii = np.random.uniform(min_radius, max_radius, n)
    
    # Convert polar coordinates to Cartesian coordinates
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    # Stack x and y coordinates into vertices array
    vertices = np.column_stack((x, y))
    
    return vertices
