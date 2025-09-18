#!/usr/bin/env python3
"""
Example demonstrating Laplace equation solution on an elliptical slit polygon.
This simulates potential distribution in a biological membrane-like structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyBlockGrid import polygon, volkovSolver


def generate_elliptical_slit(a_outer, b_outer, thickness, n_points,
                             start_angle=0, end_angle=90, center=(0, 0)):
    """
    Generate a thin slit polygon formed by two elliptical arcs.

    Parameters:
    -----------
    a_outer : float
        Semi-major axis of the outer ellipse
    b_outer : float
        Semi-minor axis of the outer ellipse
    thickness : float
        Thickness of the slit (distance between inner and outer arcs)
    n_points : int
        Number of points on each arc
    start_angle : float
        Starting angle in degrees (default: 0)
    end_angle : float
        Ending angle in degrees (default: 90)
    center : tuple
        Center coordinates (cx, cy) of the ellipses

    Returns:
    --------
    polygon : ndarray
        Array containing the complete polygon vertices (outer arc + inner arc reversed)
    """
    # Convert angles to radians
    theta_start = np.deg2rad(start_angle)
    theta_end = np.deg2rad(end_angle)

    # Generate angles for the arcs
    theta = np.linspace(theta_start, theta_end, n_points)

    # Outer arc
    x_outer = center[0] + a_outer * np.cos(theta)
    y_outer = center[1] + b_outer * np.sin(theta)
    outer_arc = np.column_stack((x_outer, y_outer))

    # Inner arc (reduced by thickness)
    # For uniform thickness, we need to offset perpendicular to the curve
    # Simplified approach: reduce radii proportionally
    a_inner = a_outer - thickness
    b_inner = b_outer - thickness

    x_inner = center[0] + a_inner * np.cos(theta)
    y_inner = center[1] + b_inner * np.sin(theta)
    inner_arc = np.column_stack((x_inner, y_inner))

    # Create closed polygon: outer arc + connecting line + inner arc (reversed) + connecting line
    polygon = np.vstack([
        outer_arc,                          # Outer arc from start to end
        inner_arc[::-1],                    # Inner arc from end to start (reversed)
        outer_arc[0:1]                      # Close the polygon
    ])

    return polygon

def identify_inner_boundary_edges(vertices, n_points):
    """
    Identify which edges belong to the inner boundary of the elliptical slit.

    For the elliptical slit geometry:
    - First n_points edges: outer arc
    - Next n_points-1 edges: inner arc (these are the inner boundary)
    - Last 2 edges: connecting edges at the ends

    Parameters:
    -----------
    vertices : ndarray
        Polygon vertices
    n_points : int
        Number of points used to generate each arc

    Returns:
    --------
    inner_edge_indices : list
        Indices of edges that form the inner boundary
    """
    # The inner arc edges start after the outer arc edges
    # and continue for n_points-1 edges
    inner_edge_indices = list(range(n_points, 2*n_points - 1))
    return inner_edge_indices

def calculate_inner_boundary_conditions(vertices, nucleus_center, inner_edge_indices):
    """
    Calculate boundary conditions for inner edges based on inverse square distance from nucleus.

    Parameters:
    -----------
    vertices : ndarray
        Polygon vertices
    nucleus_center : tuple
        (x, y) coordinates of the nucleus center
    inner_edge_indices : list
        Indices of edges that form the inner boundary

    Returns:
    --------
    boundary_values : dict
        Dictionary mapping edge index to boundary condition value
    """
    boundary_values = {}
    min_distance = float('inf')
    edge_distances = {}

    # Calculate distance from nucleus to midpoint of each inner edge
    for edge_idx in inner_edge_indices:
        # Get edge vertices
        v1 = vertices[edge_idx]
        v2 = vertices[(edge_idx + 1) % len(vertices)]

        # Calculate midpoint of edge
        midpoint = (v1 + v2) / 2

        # Calculate distance from nucleus to midpoint
        distance = np.sqrt((midpoint[0] - nucleus_center[0])**2 +
                          (midpoint[1] - nucleus_center[1])**2)
        edge_distances[edge_idx] = distance
        min_distance = min(min_distance, distance)

    # Calculate boundary values using inverse square law
    # Normalize so minimum distance gets value 1.0
    for edge_idx, distance in edge_distances.items():
        # Inverse square relationship: value ∝ (min_distance/distance)²
        boundary_value = (min_distance / distance) # ** 2
        boundary_values[edge_idx] = boundary_value

    return boundary_values

def main():
    # Configuration parameters
    n_points = 15  # Number of points per arc - can be changed to any value

    # Generate elliptical slit polygon
    vertices = generate_elliptical_slit(
        a_outer=50,
        b_outer=30,
        thickness=5,
        n_points=n_points,
        start_angle=0,
        end_angle=90
    )

    # Remove the duplicate closing vertex
    vertices = vertices[:-1]

    # Create polygon object
    poly = polygon(vertices)

    # Define nucleus center and radius (oocyte nucleus position and size)
    # Place it inside the slit, closer to the inner boundary
    nucleus_center = (10, 20)  # Adjust these coordinates as needed
    nucleus_radius = 3.0  # Radius of the nucleus

    # Automatically identify inner boundary edges based on n_points
    inner_edge_indices = identify_inner_boundary_edges(vertices, n_points)

    # Calculate boundary conditions based on inverse square distance
    inner_boundary_values = calculate_inner_boundary_conditions(
        vertices, nucleus_center, inner_edge_indices
    )

    # Set boundary conditions:
    # Outer arc = 0, inner arc edges based on distance from nucleus
    n_edges = len(vertices)
    boundary_conditions = [[0.0]] * n_edges

    # Apply calculated values to inner edges
    for edge_idx, value in inner_boundary_values.items():
        boundary_conditions[edge_idx] = [value]

    is_dirichlet = [True] * n_edges  # All Dirichlet conditions

    # Create solver
    solver = volkovSolver(
        poly=poly,
        boundary_conditions=boundary_conditions,
        is_dirichlet=is_dirichlet,
        delta=0.5,   # Grid spacing
        n=100,        # Number of angular divisions
        max_iter=10,  # Maximum iterations
        radial_heuristic=0.95 # radial heuristic value
    )

    # Solve the Laplace equation
    solution = solver.solve(verbose=True)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot solution
    im = ax.pcolormesh(solver.X, solver.Y, solver.solution, shading='auto', cmap='viridis')

    # Overlay polygon boundary
    poly_vertices = np.vstack([vertices, vertices[0]])  # Close the polygon
    ax.plot(poly_vertices[:, 0], poly_vertices[:, 1], 'k-', linewidth=2)

    # Show discretized edge points as blue dots
    ax.plot(vertices[:, 0], vertices[:, 1], 'bo', markersize=4)

    # Add vertical line at x=0 starting from inner boundary vertex
    # Inner boundary starts at index n_points, find vertex closest to x=0
    inner_vertices = vertices[n_points:2*n_points-1]  # Inner boundary vertices
    # Find the vertex with x-coordinate closest to 0
    x_coords = inner_vertices[:, 0]
    closest_idx = np.argmin(np.abs(x_coords))
    start_vertex = inner_vertices[closest_idx]

    # Find minimum y-coordinate in the entire polygon (lower boundary)
    min_y = np.min(vertices[:, 1])

    # Draw vertical line from inner boundary vertex down to minimum y
    ax.plot([start_vertex[0], start_vertex[0]], [start_vertex[1], min_y],
            'k-', linewidth=2)

    # Add follicle cells (FCs) as rectangles connected to outer boundary
    fc_width = 1.8   # Width of follicle cells (along boundary)
    fc_height = 3.0  # Height of follicle cells (extending outward)

    # Use parametric approach with proper arc length spacing
    a_outer = 50
    b_outer = 30

    # Create many points and filter by arc length distance
    n_samples = 1000  # High resolution sampling
    theta_samples = np.linspace(0, np.pi/2, n_samples)

    # Calculate cumulative arc length
    x_samples = a_outer * np.cos(theta_samples)
    y_samples = b_outer * np.sin(theta_samples)

    # Calculate distances between consecutive points
    dx = np.diff(x_samples)
    dy = np.diff(y_samples)
    arc_lengths = np.sqrt(dx**2 + dy**2)
    cumulative_arc = np.concatenate([[0], np.cumsum(arc_lengths)])

    # Select points at regular arc length intervals
    total_arc = cumulative_arc[-1]
    n_fcs = int(total_arc / fc_width)
    target_arc_positions = np.linspace(0, total_arc - fc_width, n_fcs)

    # Shift all positions by half width to the left (toward theta=0)
    target_arc_positions = target_arc_positions + fc_width / 2

    # Find theta values corresponding to target arc positions
    theta_dense = np.interp(target_arc_positions, cumulative_arc, theta_samples)

    for i, theta in enumerate(theta_dense):
        # Point on outer ellipse
        x_outer = a_outer * np.cos(theta)
        y_outer = b_outer * np.sin(theta)

        # Calculate outward normal direction at this point on ellipse
        # For ellipse x²/a² + y²/b² = 1, normal vector is (x/a², y/b²)
        normal_x = x_outer / (a_outer**2)
        normal_y = y_outer / (b_outer**2)
        normal_magnitude = np.sqrt(normal_x**2 + normal_y**2)
        normal_x /= normal_magnitude
        normal_y /= normal_magnitude

        # Calculate tangent direction (perpendicular to normal) for rectangle orientation
        tangent_x = -normal_y  # Perpendicular to normal
        tangent_y = normal_x

        # Create rectangle corners manually to ensure proper positioning
        # Rectangle extends outward from boundary by fc_height and along boundary by fc_width
        half_width = fc_width / 2

        # Four corners of the rectangle
        # Start at boundary point, then create rectangle extending outward
        corner1 = np.array([x_outer - tangent_x * half_width, y_outer - tangent_y * half_width])  # Bottom left
        corner2 = np.array([x_outer + tangent_x * half_width, y_outer + tangent_y * half_width])  # Bottom right
        corner3 = corner2 + np.array([normal_x * fc_height, normal_y * fc_height])  # Top right
        corner4 = corner1 + np.array([normal_x * fc_height, normal_y * fc_height])  # Top left

        # Create polygon patch
        from matplotlib.patches import Polygon
        rect_corners = np.array([corner1, corner2, corner3, corner4])
        rect = Polygon(rect_corners, fill=True, facecolor='white', edgecolor='black', linewidth=1)
        ax.add_patch(rect)

        # Add Y-shaped receptor attached to this follicle cell
        # Y-receptor extends inward from the center of the follicle cell's inner edge
        receptor_length = 1  # Length of each arm of the Y
        receptor_angle = 30    # Angle between Y arms in degrees

        # Starting point: center of follicle cell's inner edge (at boundary)
        start_x = x_outer
        start_y = y_outer

        # Main stem of Y extends inward by receptor_length
        stem_end_x = start_x - normal_x * receptor_length
        stem_end_y = start_y - normal_y * receptor_length

        # Draw main stem
        ax.plot([start_x, stem_end_x], [start_y, stem_end_y], 'k-', linewidth=2)

        # Calculate Y arms - rotate the inward normal by ±receptor_angle
        angle_rad = np.deg2rad(receptor_angle)

        # Left arm direction (rotate normal by +angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        left_dir_x = -normal_x * cos_a - (-normal_y) * sin_a  # Inward and rotated
        left_dir_y = -normal_y * cos_a + (-normal_x) * sin_a

        # Right arm direction (rotate normal by -angle)
        right_dir_x = -normal_x * cos_a - (normal_y) * sin_a  # Inward and rotated
        right_dir_y = -normal_y * cos_a + (normal_x) * sin_a

        # End points of Y arms
        left_end_x = stem_end_x + left_dir_x * receptor_length * 0.7
        left_end_y = stem_end_y + left_dir_y * receptor_length * 0.7

        right_end_x = stem_end_x + right_dir_x * receptor_length * 0.7
        right_end_y = stem_end_y + right_dir_y * receptor_length * 0.7

        # Draw Y arms
        ax.plot([stem_end_x, left_end_x], [stem_end_y, left_end_y], 'k-', linewidth=2)
        ax.plot([stem_end_x, right_end_x], [stem_end_y, right_end_y], 'k-', linewidth=2)

    # Add nucleus as a circle
    nucleus_circle = plt.Circle(nucleus_center, nucleus_radius,
                                fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(nucleus_circle)
    ax.plot(nucleus_center[0], nucleus_center[1], 'r+', markersize=10)  # Center marker

    # Add gurkin ligands as small circles scattered around nucleus
    np.random.seed(42)  # For reproducible results
    n_ligands = 25  # Number of gurkin ligands
    ligand_radius = 0.4  # Radius of each ligand circle
    scatter_radius = 8  # Radius around nucleus to scatter ligands

    for _ in range(n_ligands):
        # Random position around nucleus, but outside the nucleus
        angle = np.random.uniform(0, 2*np.pi)
        # Ensure distance is at least nucleus_radius + ligand_radius to avoid overlap
        min_distance = nucleus_radius + ligand_radius + 0.2  # Small buffer
        distance = np.random.uniform(min_distance, scatter_radius)

        ligand_x = nucleus_center[0] + distance * np.cos(angle)
        ligand_y = nucleus_center[1] + distance * np.sin(angle)

        # Create small circle for gurkin ligand
        ligand_circle = plt.Circle((ligand_x, ligand_y), ligand_radius,
                                  fill=True, facecolor='orange', edgecolor='darkorange',
                                  linewidth=1, alpha=0.8)
        ax.add_patch(ligand_circle)

    # Add text annotations with arrows
    # 1. Point to nucleus
    ax.annotate('Nucleus', xy=(nucleus_center[0], nucleus_center[1]),
                xytext=(nucleus_center[0], nucleus_center[1] - 8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                fontsize=16, ha='center', fontfamily='sans-serif')

    # 2. Point to follicular cells (FCs) - use one of the follicle cells
    fc_target_theta = theta_dense[len(theta_dense)//3]  # Pick a cell roughly 1/3 along
    fc_target_x = a_outer * np.cos(fc_target_theta)
    fc_target_y = b_outer * np.sin(fc_target_theta)
    normal_x_target = fc_target_x / (a_outer**2)
    normal_y_target = fc_target_y / (b_outer**2)
    normal_mag = np.sqrt(normal_x_target**2 + normal_y_target**2)
    normal_x_target /= normal_mag
    normal_y_target /= normal_mag
    fc_center_x = fc_target_x + normal_x_target * (fc_height / 2)
    fc_center_y = fc_target_y + normal_y_target * (fc_height / 2)

    ax.annotate('FCs', xy=(fc_center_x, fc_center_y),
                xytext=(fc_center_x + 12, fc_center_y + 5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                fontsize=16, ha='center', fontfamily='sans-serif')

    # 3. Point to EGFR (Y-shaped receptor) - use same follicle cell as above
    egfr_x = fc_target_x - normal_x_target * 1  # Inside the boundary a bit
    egfr_y = fc_target_y - normal_y_target * 1

    ax.annotate('EGFR', xy=(egfr_x, egfr_y),
                xytext=(egfr_x - 10, egfr_y - 6),
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                fontsize=16, ha='center', fontfamily='sans-serif')

    # Remove all labels, ticks, and legend for clean visualization
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_aspect('equal')

    # Use axes divider to create colorbar with exact same height as plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)

    # Save the plot with a higher resolution
    import os
    plotdir = 'plots'
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    plt.savefig(f'{plotdir}/biological_polygon.pdf', bbox_inches='tight', dpi=100)

    plt.show()

if __name__ == "__main__":
    main()
