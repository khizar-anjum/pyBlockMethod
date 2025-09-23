#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematical module for pyBlockGrid.

This module contains all mathematical algorithms and functions used in the
Volkov block grid method for solving the Laplace equation on polygonal domains.

Components:
- carrier_functions: Carrier function calculations for different block types
- poisson_kernels: Poisson kernel calculations for the block grid method
- boundary_estimator: Algorithm for estimating solution on curved boundaries
- interior_solver: Algorithm for computing solution at interior points
"""

from .carrier_functions import CarrierFunctions
from .poisson_kernels import PoissonKernels
from .boundary_estimator import BoundaryEstimator
from .interior_solver import InteriorSolver

__all__ = [
    'CarrierFunctions',
    'PoissonKernels',
    'BoundaryEstimator',
    'InteriorSolver'
]