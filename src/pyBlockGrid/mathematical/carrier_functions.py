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

    def _eta(self, r_, theta_, k, alpha_j):
        """Volkov's eta_{jk} (Eq. 3.5): the Neumann-edge basis for a
        Neumann-Neumann first-kind block. Absorbs a flux datum r^k on the edge at
        theta=0. (Distinct from vartheta, the Neumann basis for a *mixed* block.)"""
        kp1 = k + 1
        s = np.sin(kp1 * alpha_j * np.pi)
        c = np.cos(kp1 * alpha_j * np.pi)
        sing = np.abs(s) < self.tol
        rk = np.power(r_, kp1)
        reg = rk * np.cos(kp1 * theta_) / (kp1 * np.where(sing, 1.0, s))
        singval = rk * (np.log(r_) * np.cos(kp1 * theta_) - theta_ * np.sin(kp1 * theta_)) \
            / (kp1 * alpha_j * np.pi * np.where(sing, c, 1.0))
        return np.where(sing, singval, reg)

    def _first_kind_neumann_neumann(self, r_, theta_, k, a_, b_, alpha_j):
        """
        Carrier function for first kind block with Neumann-Neumann boundary conditions.
        Both edges Neumann. Volkov Eq. (3.4): both series via eta. Unlike chi
        (which sets its datum at theta=0), eta imposes its flux at theta=alpha*pi,
        so the incoming a-series uses eta(r,theta) and the outgoing b-series uses
        eta(r, alpha*pi - theta).
        """
        return np.sum(a_ * self._eta(r_, theta_, k, alpha_j)) + \
            np.sum(b_ * self._eta(r_, alpha_j * np.pi - theta_, k, alpha_j))

    def _vartheta(self, r_, theta_, k, alpha_j):
        """Volkov's vartheta_{jk} (Eq. 3.7): the Neumann-edge basis for a mixed
        first-kind block. Absorbs a flux datum r^k on the edge at theta=0."""
        kp1 = k + 1
        c = np.cos(kp1 * alpha_j * np.pi)
        s = np.sin(kp1 * alpha_j * np.pi)
        sing = np.abs(c) < self.tol
        rk = np.power(r_, kp1)
        reg = -rk * np.sin(kp1 * theta_) / (kp1 * np.where(sing, 1.0, c))
        singval = rk * (theta_ * np.cos(kp1 * theta_) + np.log(r_) * np.sin(kp1 * theta_)) \
            / (kp1 * alpha_j * np.pi * np.where(sing, s, 1.0))
        return np.where(sing, singval, reg)

    def _chi(self, r_, theta_, k, alpha_j):
        """Volkov's chi_{jk} (Eq. 3.8): the Dirichlet-edge basis for a mixed
        first-kind block. Reproduces a value datum r^k on the Dirichlet edge
        (theta=0) while carrying zero flux on the Neumann edge (theta=alpha*pi).
        Zero far-edge flux forces the cos(k(theta-alpha*pi)) phase; plain
        cos(k*theta) only works for the resonant k where sin(k*alpha*pi)=0."""
        c = np.cos(k * alpha_j * np.pi)
        sing = np.abs(c) < self.tol
        rk = np.power(r_, k)
        reg = rk * np.cos(k * (theta_ - alpha_j * np.pi)) / np.where(sing, 1.0, c)
        singval = rk * (np.cos(k * theta_) - (theta_ * np.cos(k * theta_)
                        + np.log(r_) * np.sin(k * theta_)) / (alpha_j * np.pi))
        return np.where(sing, singval, reg)

    def _first_kind_neumann_dirichlet(self, r_, theta_, k, a_, b_, alpha_j):
        """
        Carrier function for first kind block with Neumann-Dirichlet boundary conditions.
        Edge j-1 (incoming, at theta=alpha*pi) is Neumann; edge j (outgoing, at theta=0)
        is Dirichlet. Volkov Eq. (3.6): a-series via vartheta, b-series via chi.
        vartheta imposes its flux at theta=alpha*pi, so the incoming a-series uses
        vartheta(r,theta); chi sets its value at theta=0, so b uses chi(r,theta).
        """
        return np.sum(a_ * self._vartheta(r_, theta_, k, alpha_j)) + \
            np.sum(b_ * self._chi(r_, theta_, k, alpha_j))

    def _first_kind_dirichlet_neumann(self, r_, theta_, k, a_, b_, alpha_j):
        """
        Carrier function for first kind block with Dirichlet-Neumann boundary conditions.
        Edge j-1 (incoming, at theta=alpha*pi) is Dirichlet; edge j (outgoing, at theta=0)
        is Neumann. Volkov Eq. (3.9): a-series via chi, b-series via vartheta.
        chi sets its value at theta=0, so the incoming a uses chi(r, alpha*pi - theta);
        vartheta imposes its flux at theta=alpha*pi, so the outgoing b uses
        vartheta(r, alpha*pi - theta).
        """
        return np.sum(a_ * self._chi(r_, alpha_j * np.pi - theta_, k, alpha_j)) + \
            np.sum(b_ * self._vartheta(r_, alpha_j * np.pi - theta_, k, alpha_j))

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