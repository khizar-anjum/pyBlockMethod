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


if __name__ == "__main__":
    # Run functional tests
    pytest.main([__file__, "-v"])