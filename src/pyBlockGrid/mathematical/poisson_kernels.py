#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poisson kernel calculations for the Volkov block grid method.

This module implements the Poisson kernel calculations as described in
E.A. Volkov's book "Block Method for Solving the Laplace Equation and for
Constructing Conformal Mappings (1994)" ISBN: 0849394066.

The Poisson kernels are fundamental solutions used to propagate boundary
conditions from the boundaries of blocks to their interiors.
"""

import numpy as np


class PoissonKernels:
    """
    Poisson kernel calculations for different block types.

    The Poisson kernels are essential for the block grid method, providing
    the mechanism to extend the boundary solution into the interior of blocks.
    """

    @staticmethod
    def main_kernel(r1, theta1, eta1):
        """
        Calculate the fundamental Poisson kernel.

        This is the basic Poisson kernel formula for the unit disk, which forms
        the foundation for all block-specific kernels.

        Parameters:
            r1 (float or np.ndarray): Radial coordinate (normalized)
            theta1 (float or np.ndarray): Angular coordinate of the point
            eta1 (float or np.ndarray): Angular coordinate on the boundary

        Returns:
            float or np.ndarray: Poisson kernel value(s)
        """
        return (1 - r1**2) / (2 * np.pi * (1 - 2*r1*np.cos(theta1 - eta1) + r1**2))

    def calculate(self, block_kind, nu_i, nu_j, r_, r0_, theta_, eta_, alpha_j):
        """
        Calculate Poisson kernel value for given block type and parameters.

        Parameters:
            block_kind (int): Type of block (1=first kind, 2=second kind, 3=third kind)
            nu_i (int): Boundary condition type for first edge (0=Neumann, 1=Dirichlet)
            nu_j (int): Boundary condition type for second edge (0=Neumann, 1=Dirichlet)
            r_ (float): Radial coordinate
            r0_ (float): Outer radius of the block
            theta_ (float): Angular coordinate
            eta_ (float or np.ndarray): Angular coordinate(s) on boundary
            alpha_j (float): Normalized angle of the block

        Returns:
            float or np.ndarray: Poisson kernel value(s)
        """
        if block_kind == 1:
            # block of first kind
            return self._first_kind(nu_i, nu_j, r_, r0_, theta_, eta_, alpha_j)
        elif block_kind == 2:
            # block of second kind
            return self._second_kind(nu_i, r_, r0_, theta_, eta_)
        else:
            # block of third kind
            return self._third_kind(r_, r0_, theta_, eta_)

    def _first_kind(self, nu_i, nu_j, r_, r0_, theta_, eta_, alpha_j):
        """
        Poisson kernel for first kind blocks (sectors at vertices).

        Implements equations (3.12), (3.13), and (3.17) from Volkov's book.
        The kernel depends on the boundary condition types on both edges.

        Parameters:
            nu_i (int): Boundary condition type for edge i (0=Neumann, 1=Dirichlet)
            nu_j (int): Boundary condition type for edge j (0=Neumann, 1=Dirichlet)
            r_ (float): Radial coordinate
            r0_ (float): Outer radius of the block
            theta_ (float): Angular coordinate
            eta_ (float or np.ndarray): Angular coordinate(s) on boundary
            alpha_j (float): Normalized angle of the block

        Returns:
            float or np.ndarray: Poisson kernel value(s)
        """
        lambda_j = 1/((2 - nu_i * nu_j - (1 - nu_i) * (1 - nu_j)) * alpha_j)

        if nu_i == nu_j:  # both are equal (3.17), (3.12)
            return lambda_j * (self.main_kernel((r_ / r0_) ** lambda_j, lambda_j * theta_, lambda_j * eta_) + \
                                    (-1)**nu_j * self.main_kernel((r_ / r0_) ** lambda_j, lambda_j * theta_, -lambda_j * eta_))
        else:  # they are different (3.17), (3.13)
            return lambda_j * (self.main_kernel((r_ / r0_) ** lambda_j, lambda_j * theta_, lambda_j * eta_) + \
                                    (-1)**nu_j * self.main_kernel((r_ / r0_) ** lambda_j, lambda_j * theta_, -lambda_j * eta_) - \
                                    (-1)**nu_j * \
                                    (self.main_kernel((r_ / r0_) ** lambda_j, lambda_j * theta_, lambda_j * (np.pi - eta_)) + \
                                    (-1)**nu_j * self.main_kernel((r_ / r0_) ** lambda_j, lambda_j * theta_, -lambda_j * (np.pi - eta_)))
                                )

    def _second_kind(self, nu_i, r_, r0_, theta_, eta_):
        """
        Poisson kernel for second kind blocks (half-disks from edges).

        Implements equation (3.16) from Volkov's book.

        Parameters:
            nu_i (int): Boundary condition type for the edge (0=Neumann, 1=Dirichlet)
            r_ (float): Radial coordinate
            r0_ (float): Outer radius of the block
            theta_ (float): Angular coordinate
            eta_ (float or np.ndarray): Angular coordinate(s) on boundary

        Returns:
            float or np.ndarray: Poisson kernel value(s)
        """
        return self.main_kernel(r_ / r0_, theta_, eta_) + (-1)**nu_i * (self.main_kernel(r_ / r0_, theta_, -eta_))

    def _third_kind(self, r_, r0_, theta_, eta_):
        """
        Poisson kernel for third kind blocks (interior disks).

        Implements equation (3.15) from Volkov's book.

        Parameters:
            r_ (float): Radial coordinate
            r0_ (float): Outer radius of the block
            theta_ (float): Angular coordinate
            eta_ (float or np.ndarray): Angular coordinate(s) on boundary

        Returns:
            float or np.ndarray: Poisson kernel value(s)
        """
        return self.main_kernel(r_ / r0_, theta_, eta_)