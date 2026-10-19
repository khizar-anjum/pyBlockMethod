#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functional tests for the pyBlockGrid package.

These tests verify that core components have correct functionality.
For numerical accuracy testing, see test_accuracy.py.
"""

import numpy as np
import pytest
from pyBlockGrid import polygon, volkovSolver


@pytest.fixture
def square_polygon():
    """Create a unit square polygon for testing."""
    return polygon(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))


@pytest.fixture
def l_shape_polygon():
    """Create an L-shaped polygon for testing."""
    return polygon(np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]))


@pytest.fixture
def square_with_hole_polygon():
    """Create a unit square with a square hole for testing."""
    # Main square: 1x1
    poly = polygon(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))

    # Add hole: 0.4x0.4 centered square (clockwise vertices)
    hole_vertices = np.array([[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]])[::-1]
    hole_bc = [[0.0], [0.0], [0.0], [0.0]]
    hole_is_dirichlet = [True, True, True, True]

    poly.add_hole(hole_vertices, hole_bc, hole_is_dirichlet)
    return poly


class TestPolygonFunctionality:
    """Test core polygon geometric functionality."""

    def test_square_polygon_properties(self, square_polygon):
        """Test unit square polygon geometric calculations."""
        assert square_polygon.area() == 1.0
        assert square_polygon.verify_convexity()
        assert square_polygon.verify_vertex_order()
        assert len(square_polygon.vertices) == 4

    def test_l_shape_polygon_properties(self, l_shape_polygon):
        """Test L-shaped polygon geometric calculations."""
        assert l_shape_polygon.area() == 3.0  # 2×1 + 1×2 = 3
        assert not l_shape_polygon.verify_convexity()  # L-shape is non-convex
        assert l_shape_polygon.verify_vertex_order()
        assert len(l_shape_polygon.vertices) == 6


class TestPolygonWithHole:
    """Test polygon with hole functionality."""

    def test_square_with_hole_properties(self, square_with_hole_polygon):
        """Test square with hole geometric calculations."""
        poly = square_with_hole_polygon

        # Main square area is 1.0, hole area is 0.4*0.4 = 0.16
        expected_effective_area = 1.0 - 0.16
        assert abs(poly.area_with_holes() - expected_effective_area) < 1e-10, \
            f"Effective area should be {expected_effective_area}, got {poly.area_with_holes()}"

        # Should have one hole
        assert len(poly.holes) == 1, "Should have exactly one hole"

        # Hole should have 4 vertices
        assert len(poly.holes[0].vertices) == 4, "Hole should have 4 vertices"

        # Main polygon still has area 1.0
        assert poly.area() == 1.0, "Main polygon area should still be 1.0"

    def test_square_with_hole_point_inside(self, square_with_hole_polygon):
        """Test point-in-polygon checking with holes."""
        poly = square_with_hole_polygon

        # Point in main polygon but outside hole should be inside
        assert poly.is_inside(np.array([[0.1, 0.1]])).all(), \
            "Point in main polygon outside hole should be inside"

        # Point inside hole should be outside
        assert not poly.is_inside(np.array([[0.5, 0.5]])).any(), \
            "Point inside hole should be outside"

        # Point outside main polygon should be outside
        assert not poly.is_inside(np.array([[1.5, 1.5]])).any(), \
            "Point outside main polygon should be outside"


class TestBlockCovering:
    """Test block covering algorithm functionality."""

    def test_block_covering_logic(self, square_polygon):
        """Test that block covering follows expected mathematical relationships."""
        solver = volkovSolver(
            poly=square_polygon,
            boundary_conditions=[[1.0], [0.0], [0.0], [0.0]],
            is_dirichlet=[True, True, True, True],
            delta=0.05,
            n=10,
            max_iter=5
        )

        N, L, M, uncovered_points = solver.find_block_covering()

        # Mathematical relationships from Volkov method
        assert N > 0, "Should have vertex blocks (first kind)"
        assert L >= N, "Should have edge blocks in addition to vertex blocks"
        assert M >= L, "Should have interior blocks in addition to boundary blocks"

        # Specific expectations for square (4 vertices)
        assert N == 4, "Square should have exactly 4 vertex blocks"

    def test_block_covering_with_hole(self, square_with_hole_polygon):
        """Test that block covering works correctly with holes."""
        poly = square_with_hole_polygon

        solver = volkovSolver(
            poly=poly,
            boundary_conditions=[[1.0], [0.0], [0.0], [0.0]],
            is_dirichlet=[True, True, True, True],
            delta=0.05,
            n=10,
            max_iter=5
        )

        N, L, M, uncovered_points = solver.find_block_covering()

        # Mathematical relationships still hold
        assert N > 0, "Should have vertex blocks (first kind)"
        assert L >= N, "Should have edge blocks in addition to vertex blocks"
        assert M >= L, "Should have interior blocks in addition to boundary blocks"

        # Should have blocks for both main polygon (4 vertices) and hole (4 vertices)
        assert N == 8, "Should have 8 vertex blocks (4 main + 4 hole)"


if __name__ == "__main__":
    # Run functional tests
    pytest.main([__file__, "-v"])