"""
pyBlockGrid: A Laplace equation solver on polygons using Volkov method.

This package provides tools for solving the Laplace equation on arbitrary
polygonal domains using the Volkov method.
"""

try:
    from ._pyblockgrid_cpp import polygon, block, volkovSolver
except ImportError:
    from .core.polygon import polygon
    from .core.block import block
    from .solvers.volkov import volkovSolver

__version__ = "0.1.0"
__all__ = ["polygon", "block", "volkovSolver"] 