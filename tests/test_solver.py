#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest
from src.pyBlockGrid.core.polygon import polygon
from src.pyBlockGrid.solvers.volkov import volkovSolver
from src.pyBlockGrid.visualization.plotting import plot_3by3_solution_steps

@pytest.fixture
def square_polygon():
    return polygon(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))

@pytest.fixture
def boundary_conditions():
    return [[1.0], [0.0], [0.0], [0.0]]

@pytest.fixture
def is_dirichlet():
    return [True] * 4

@pytest.fixture
def solver_params():
    return {
        'delta': 0.05,
        'n': 50,
        'max_iter': 10
    }

def test_polygon(square_polygon):
    """Test basic polygon functionality."""
    assert square_polygon.area() == 1.0
    assert square_polygon.verify_convexity()
    assert square_polygon.verify_vertex_order()
    assert len(square_polygon.vertices) == 4

def test_solver_basic(square_polygon, boundary_conditions, is_dirichlet, solver_params):
    solver = volkovSolver(
        poly=square_polygon,
        boundary_conditions=boundary_conditions,
        is_dirichlet=is_dirichlet,
        **solver_params
    )
    
    solution = solver.solve(verbose=False)
    assert len(solution) > 0
    assert all(isinstance(val, float) for val in solution.values())

def test_block_covering(square_polygon, boundary_conditions, is_dirichlet, solver_params):
    solver = volkovSolver(
        poly=square_polygon,
        boundary_conditions=boundary_conditions,
        is_dirichlet=is_dirichlet,
        **solver_params
    )
    
    N, L, M, _ = solver.find_block_covering()
    assert N > 0
    assert L >= N
    assert M >= L

def test_3by3_plotting():
    # Test data
    v1 = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]])
    v2 = np.array([[-0.58677897, 1.62158457], [-0.63274991, 0.8863274], [-1.74993861, 0.30467836],
                  [-1.25531598, 0.19937933], [-1.35251473, 0.09726827], [-1.84362218, 0.11634993],
                  [-1.84289172, -0.18032796], [0.19833559, -1.43784708], [0.2842006, -1.56029036],
                  [1.1250711, -1.1409574]])
    v3 = np.array([[ 0.9599168 ,  0.65081101], [ 0.26401952 , 1.81835665], [-0.16186728 , 1.15558642], 
                  [-0.54884342 , 1.78631676], [-0.4325186,   1.11517468], [-1.129355,   -0.94655537], 
                  [-0.01205465, -1.22329523], [ 1.1184654,  -1.40151058], [ 0.88881931, -1.10564833], 
                  [ 1.34336699, -0.19043799]])
    
    # Create test data
    poly_array = [polygon(v1), polygon(v2), polygon(v3)]
    boundary_conditions_array = [[[1.0] if i == 0 else [0.0] for i in range(len(p.vertices))] for p in poly_array]
    is_dirichlet_array = [[True] * len(p.vertices) for p in poly_array]
    
    # Test plotting function
    plot_3by3_solution_steps(
        poly_array=poly_array,
        boundary_conditions_array=boundary_conditions_array, 
        is_dirichlet_array=is_dirichlet_array,
        delta=0.05,
        n=50,
        max_iter=10,
        radial_heuristics_iter=[0.95, 0.95, 0.95],
        output_folder="plots"
    )