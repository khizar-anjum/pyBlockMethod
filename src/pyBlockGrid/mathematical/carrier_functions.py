#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Carrier function calculations for the Volkov block grid method.

This module implements the carrier function calculations as described in
E.A. Volkov's book "Block Method for Solving the Laplace Equation and for
Constructing Conformal Mappings (1994)" ISBN: 0849394066.

The carrier functions are used to handle boundary conditions on the edges
of the polygonal domain for different block types.
"""

import numpy as np


class CarrierFunctions:
    """
    Carrier function calculations for different block types and boundary conditions.

    The carrier functions are essential components of the Volkov method that
    encode the boundary conditions into the solution process.
    """

    def __init__(self, tolerance=1e-10):
        """
        Initialize the carrier function calculator.

        Parameters:
            tolerance (float): Tolerance for floating point comparisons
        """
        self.tol = tolerance

    def calculate(self, block_kind, block_boundary_identifier, r_, theta_, k, a_, b_, alpha_j):
        """
        Calculate carrier function value for given block and parameters.

        Parameters:
            block_kind (int): Type of block (1=first kind, 2=second kind, 3=third kind)
            block_boundary_identifier (int): Boundary condition identifier
            r_ (float): Radial coordinate
            theta_ (float): Angular coordinate
            k (np.ndarray): Array of indices for polynomial terms
            a_ (np.ndarray): Coefficients for first edge boundary condition
            b_ (np.ndarray): Coefficients for second edge boundary condition
            alpha_j (float): Normalized angle of the block

        Returns:
            float: Carrier function value at the given point
        """
        if block_kind == 1:
            # first kind block
            if block_boundary_identifier == 0:
                # both edges are Neumann (v_{j-1} = 0, v_{j} = 0)
                return self._first_kind_neumann_neumann(r_, theta_, k, a_, b_, alpha_j)
            elif block_boundary_identifier == 1:
                # edge j-1 is Neumann, edge j is Dirichlet (v_{j-1} = 0, v_{j} = 1)
                return self._first_kind_neumann_dirichlet(r_, theta_, k, a_, b_, alpha_j)
            elif block_boundary_identifier == 2:
                # edge j-1 is Dirichlet, edge j is Neumann (v_{j-1} = 1, v_{j} = 0)
                return self._first_kind_dirichlet_neumann(r_, theta_, k, a_, b_, alpha_j)
            else:
                # both edges are Dirichlet (v_{j-1} = 1, v_{j} = 1)
                return self._first_kind_dirichlet_dirichlet(r_, theta_, k, a_, b_, alpha_j)
        elif block_kind == 2:
            # second kind block
            return self._second_kind(block_boundary_identifier, r_, theta_, k, a_)
        else:
            # third kind block
            return 0.0

    def _first_kind_neumann_neumann(self, r_, theta_, k, a_, b_, alpha_j):
        """
        Carrier function for first kind block with Neumann-Neumann boundary conditions.
        Both edges have Neumann conditions (v_{j-1} = 0, v_{j} = 0).
        """
        cond_ = (np.abs(np.sin((k + 1) * alpha_j * np.pi)) < self.tol).astype(int)

        sigma_jk = (k + 1) * ((1 - cond_) * np.sin((k + 1) * alpha_j * np.pi) + \
            cond_ * (- alpha_j * np.pi * np.cos((k + 1) * alpha_j * np.pi)))
        eta_jk = (1 - cond_) * (np.cos((k + 1) * theta_)) + \
            cond_ * (theta_ * np.sin((k + 1) * theta_) - np.log(r_) * np.cos((k + 1) * theta_))

        return np.sum((a_ / sigma_jk) * np.power(r_, k + 1) * eta_jk) + \
            np.sum((b_ / (k + 1)) * np.power(r_, k + 1) * \
                    (np.sin((k + 1) * theta_) + ((k + 1) * np.cos((k + 1) * alpha_j * np.pi) / sigma_jk) * eta_jk))

    def _first_kind_neumann_dirichlet(self, r_, theta_, k, a_, b_, alpha_j):
        """
        Carrier function for first kind block with Neumann-Dirichlet boundary conditions.
        Edge j-1 has Neumann condition, edge j has Dirichlet condition (v_{j-1} = 0, v_{j} = 1).
        """
        cond_ = (np.abs(np.cos(k * alpha_j * np.pi)) < self.tol).astype(int)

        omega_jk = k * ((1 - cond_) * (- np.cos(k * alpha_j * np.pi)) + \
            cond_ * (alpha_j * np.pi * np.sin(k * alpha_j * np.pi)))
        psi_jk = (1 - cond_) * np.sin(k * theta_) + \
            cond_ * (theta_ * np.cos(k * theta_) + np.log(r_) * np.sin(k * theta_))
        psi_jkp1 = (1 - cond_) * np.sin((k + 1) * theta_) + \
            cond_ * (theta_ * np.cos((k + 1) * theta_) + np.log(r_) * np.sin((k + 1) * theta_))
        kp1 = k[1:]

        return np.sum((a_ / omega_jk) * np.power(r_, k + 1) * psi_jkp1) + \
            b_[0] + np.sum(b_[1:] * np.power(r_, kp1) * \
                            (np.cos(kp1 * theta_) - (kp1 * np.sin(kp1 * alpha_j * np.pi) / omega_jk[1:]) * psi_jk[1:]))

    def _first_kind_dirichlet_neumann(self, r_, theta_, k, a_, b_, alpha_j):
        """
        Carrier function for first kind block with Dirichlet-Neumann boundary conditions.
        Edge j-1 has Dirichlet condition, edge j has Neumann condition (v_{j-1} = 1, v_{j} = 0).
        """
        cond_ = (np.abs(np.cos((k + 1) * alpha_j * np.pi)) < self.tol).astype(int)
        kappa_jk = (1 - cond_) * (np.cos((k + 1) * theta_)) + \
            cond_ * (theta_ * np.sin((k + 1) * theta_) - np.log(r_) * np.cos((k + 1) * theta_))
        kappa_1k = (1 - cond_) * (np.cos((k + 1) * alpha_j * np.pi)) + \
            cond_ * (alpha_j * np.pi * np.sin((k + 1) * alpha_j * np.pi) - np.log(1) * np.cos((k + 1) * alpha_j * np.pi))
        kappa_jkn1 = (1 - cond_) * (np.cos(k * theta_)) + \
            cond_ * (theta_ * np.sin(k * theta_) - np.log(r_) * np.cos(k * theta_))
        kappa_1kn1 = (1 - cond_) * (np.cos(k * alpha_j * np.pi)) + \
            cond_ * (alpha_j * np.pi * np.sin(k * alpha_j * np.pi) - np.log(1) * np.cos(k * alpha_j * np.pi))

        return np.sum((a_ / kappa_1kn1) * np.power(r_, k) * kappa_jkn1) + \
            np.sum((b_ / (k + 1)) * np.power(r_, k + 1) * \
                    (np.sin((k + 1) * theta_) - (np.sin((k + 1) * alpha_j * np.pi) / kappa_1k) * kappa_jk))

    def _first_kind_dirichlet_dirichlet(self, r_, theta_, k, a_, b_, alpha_j):
        """
        Carrier function for first kind block with Dirichlet-Dirichlet boundary conditions.
        Both edges have Dirichlet conditions (v_{j-1} = 1, v_{j} = 1).
        """
        cond_ = (np.abs(np.sin(k * alpha_j * np.pi)) < self.tol).astype(int)

        zeta_jk = (1 - cond_) * np.sin(k * theta_) + \
            cond_ * (theta_ * np.cos(k * theta_) + np.log(r_) * np.sin(k * theta_))
        zeta_1k = (1 - cond_) * np.sin(k * alpha_j * np.pi) + \
            cond_ * (alpha_j * np.pi * np.cos(k * alpha_j * np.pi) + np.log(1) * np.sin(k * alpha_j * np.pi))

        return np.sum((a_ / zeta_1k) * np.power(r_, k) * zeta_jk) + \
            np.sum((b_ * np.power(r_, k) * (np.cos(k * theta_) - (np.cos(k * alpha_j * np.pi) / zeta_1k) * zeta_jk)))

    def _second_kind(self, block_boundary_identifier, r_, theta_, k, a_):
        """
        Carrier function for second kind block (half-disk from edge).
        """
        nu_jq = block_boundary_identifier // 2

        return np.sum(a_ * np.power(r_, k) * \
                        (nu_jq * (np.cos(k * theta_)) + (1 - nu_jq) * \
                        r_ * (np.sin((k + 1) * theta_) / (k + 1))))