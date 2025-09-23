#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coordinate transformation utilities for the Volkov block grid method.

This module provides functions for converting between Cartesian and polar
coordinate systems relative to block centers and reference angles. These
transformations are essential for the mathematical calculations in the
Volkov method.
"""

import numpy as np


class CoordinateTransforms:
    """
    Handles coordinate transformations for numerical computations in the Volkov method.

    All transformations are performed relative to block centers with reference angles
    to ensure proper orientation of the coordinate systems.
    """

    @staticmethod
    def cartesian_to_polar(points, centers, ref_thetas):
        """
        Convert Cartesian coordinates to polar relative to given centers
        with reference angles.

        This function converts points from Cartesian coordinates to polar coordinates
        where the polar system is centered at the given centers and oriented according
        to the reference angles.

        Parameters:
            points (np.ma.MaskedArray): Points in Cartesian coordinates, shape (..., 2)
            centers (np.ndarray): Center points for coordinate systems, shape (..., 2)
            ref_thetas (np.ndarray): Reference angles for angular orientation, shape (...)

        Returns:
            np.ma.array: Polar coordinates [r, theta] where theta is normalized to [0, 2π)
        """
        # Translate points relative to centers
        points_translated = points - centers

        # Calculate radial distances
        r = np.linalg.norm(points_translated, axis=-1)

        # Calculate angles and normalize relative to reference angles
        theta_raw = np.arctan2(points_translated[..., 1], points_translated[..., 0])
        theta = np.mod(theta_raw - ref_thetas, 2 * np.pi)

        # Stack r and theta together
        polar_coords = np.stack([r, theta], axis=-1)

        # Preserve mask from input points
        return np.ma.array(polar_coords, mask=getattr(points, 'mask', False))

    @staticmethod
    def polar_to_cartesian(polar_points, centers, ref_thetas):
        """
        Convert polar coordinates to Cartesian relative to given centers
        with reference angles.

        This function converts points from polar coordinates to Cartesian coordinates
        where the polar system is centered at the given centers and oriented according
        to the reference angles.

        Parameters:
            polar_points (np.ma.MaskedArray): Polar coordinates [r, theta], shape (..., 2)
            centers (np.ndarray): Center points for coordinate systems, shape (..., 2)
            ref_thetas (np.ndarray): Reference angles for angular orientation, shape (...)

        Returns:
            np.ma.array: Cartesian coordinates [x, y]
        """
        r = polar_points[..., 0]
        theta = polar_points[..., 1]

        # Calculate absolute angles by adding reference angles
        absolute_theta = theta + ref_thetas

        # Convert to Cartesian coordinates
        x = centers[..., 0] + r * np.cos(absolute_theta)
        y = centers[..., 1] + r * np.sin(absolute_theta)

        # Stack x and y coordinates
        cartesian_coords = np.stack([x, y], axis=-1)

        # Preserve mask from input points
        return np.ma.array(cartesian_coords, mask=getattr(polar_points, 'mask', False))

    @staticmethod
    def batch_cartesian_to_polar(points_grid, centers_array, ref_thetas_array):
        """
        Vectorized conversion for grid of points to polar coordinates.

        This is an optimized version for converting large grids of points
        where each point may belong to a different coordinate system.

        Parameters:
            points_grid (np.ma.MaskedArray): Grid of points in Cartesian coordinates
            centers_array (np.ndarray): Array of centers corresponding to each point
            ref_thetas_array (np.ndarray): Array of reference angles corresponding to each point

        Returns:
            np.ma.array: Polar coordinates for each point
        """
        # Vectorized version of the transformation
        points_translated = points_grid - centers_array

        r = np.linalg.norm(points_translated, axis=-1)
        theta_raw = np.arctan2(points_translated[..., 1], points_translated[..., 0])
        theta = np.mod(theta_raw - ref_thetas_array, 2 * np.pi)

        polar_coords = np.stack([r, theta], axis=-1)
        return np.ma.array(polar_coords, mask=getattr(points_grid, 'mask', False))

    @staticmethod
    def batch_polar_to_cartesian(polar_grid, centers_array, ref_thetas_array):
        """
        Vectorized conversion for grid of polar coordinates to Cartesian.

        This is an optimized version for converting large grids of polar points
        where each point may belong to a different coordinate system.

        Parameters:
            polar_grid (np.ma.MaskedArray): Grid of polar coordinates [r, theta]
            centers_array (np.ndarray): Array of centers corresponding to each point
            ref_thetas_array (np.ndarray): Array of reference angles corresponding to each point

        Returns:
            np.ma.array: Cartesian coordinates for each point
        """
        r = polar_grid[..., 0]
        theta = polar_grid[..., 1]

        absolute_theta = theta + ref_thetas_array

        x = centers_array[..., 0] + r * np.cos(absolute_theta)
        y = centers_array[..., 1] + r * np.sin(absolute_theta)

        cartesian_coords = np.stack([x, y], axis=-1)
        return np.ma.array(cartesian_coords, mask=getattr(polar_grid, 'mask', False))