#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Accuracy tests for mixed Dirichlet / homogeneous-Neumann boundary conditions.

These cover the second-kind (Neumann) boundary handling: manufactured harmonic
solutions with insulated (zero-flux) edges, and Motz's problem -- the standard
mixed Dirichlet-Neumann benchmark with an r^(1/2) corner singularity.
"""

import numpy as np
import pytest
from pyBlockGrid import polygon, volkovSolver

SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)


def _solve(vertices, bc, is_dirichlet, n=40, delta=0.02):
    solver = volkovSolver(poly=polygon(vertices.copy()), boundary_conditions=bc,
                          is_dirichlet=is_dirichlet, delta=delta, n=n,
                          radial_heuristic=0.75, overlap_heuristic=0.1, max_iter=40)
    sol = solver.solve(verbose=False)
    return solver, sol


def _rmse(sol, solver, exact_fn):
    mask = ~sol.mask
    err = sol[mask] - exact_fn(solver._X, solver._Y)[mask]
    return float(np.sqrt(np.mean(err ** 2)))


class TestHomogeneousNeumann:
    """Manufactured harmonic solutions with insulated (zero-flux) edges.

    Each edge normal derivative of the exact solution is zero on the Neumann
    edges, so these isolate the homogeneous-Neumann coupling from the Dirichlet
    data. The block method should recover them to near machine precision.
    """

    def test_constant_insulated_top_bottom(self):
        # u = 1: left/right Dirichlet = 1, top/bottom insulated.
        solver, sol = _solve(SQUARE, [[0.0], [1.0], [0.0], [1.0]],
                             [False, True, False, True])
        assert _rmse(sol, solver, lambda X, Y: np.ones_like(X)) < 1e-5

    def test_linear_x_insulated_top_bottom(self):
        # u = x: left/right Dirichlet 0/1, top/bottom insulated (u independent of y).
        solver, sol = _solve(SQUARE, [[0.0], [1.0], [0.0], [0.0]],
                             [False, True, False, True])
        assert _rmse(sol, solver, lambda X, Y: X) < 1e-5

    def test_linear_y_insulated_left_right(self):
        # u = y: bottom/top Dirichlet 0/1, left/right insulated (u independent of x).
        solver, sol = _solve(SQUARE, [[0.0], [0.0], [1.0], [0.0]],
                             [True, False, True, False])
        assert _rmse(sol, solver, lambda X, Y: Y) < 1e-5


class TestMotzProblem:
    """Motz's problem: the canonical mixed Dirichlet-Neumann benchmark.

    On the rectangle [-1,1] x [0,1] the bottom edge switches from u=0 (Dirichlet,
    x<0) to du/dn=0 (Neumann, x>0) at the origin, with u=500 on the right edge and
    insulated top/left. The solution has an r^(1/2) cos(theta/2) singularity at the
    origin; its leading coefficient is the well-established a0 = 401.1625.
    """

    # split the bottom edge at the origin so the BC-type change sits at a vertex
    VERTS = np.array([[-1, 0], [0, 0], [1, 0], [1, 1], [-1, 1]], float)
    BC = [[0.0], [0.0], [500.0], [0.0], [0.0]]
    IS_DIRICHLET = [True, False, True, False, False]

    @staticmethod
    def _leading_coefficient(sol, solver, n_modes=8, r_in=0.1, r_out=0.7):
        """Least-squares fit of the corner expansion sum a_k r^(k+1/2) cos((k+1/2)theta)
        about the origin; return a_0."""
        X, Y = solver._X, solver._Y
        r = np.hypot(X, Y)
        theta = np.arctan2(Y, X)
        sel = (~sol.mask) & (r >= r_in) & (r <= r_out) & (theta >= -1e-6) & (theta <= np.pi + 1e-6)
        basis = np.stack([r[sel] ** (k + 0.5) * np.cos((k + 0.5) * theta[sel])
                          for k in range(n_modes)], axis=1)
        coeffs, *_ = np.linalg.lstsq(basis, sol.data[sel], rcond=None)
        return coeffs[0]

    def test_leading_singular_coefficient(self):
        solver, sol = _solve(self.VERTS, self.BC, self.IS_DIRICHLET, n=40)
        a0 = self._leading_coefficient(sol, solver)
        assert a0 == pytest.approx(401.1625, abs=1.0)

    def test_maximum_principle(self):
        # zero-flux Neumann + Dirichlet data in [0, 500] => solution stays in [0, 500].
        solver, sol = _solve(self.VERTS, self.BC, self.IS_DIRICHLET, n=40)
        vals = sol.compressed()
        assert vals.min() > -1.0 and vals.max() < 501.0


class TestNonzeroFluxNeumann:
    """Prescribed non-zero flux (inhomogeneous Neumann) data.

    The flux is the inner-normal derivative. Both Neumann-Neumann corners and
    mixed (Dirichlet-Neumann) corners with non-zero flux are recovered to near
    machine precision.
    """

    def test_neumann_neumann_nonzero_flux(self):
        # u = y: bottom Dirichlet 0, top prescribed inner-normal flux d/d(-y) = -1,
        # left/right insulated. The flux edge's corners are Neumann-Neumann.
        solver, sol = _solve(SQUARE, [[0.0], [0.0], [-1.0], [0.0]],
                             [True, False, False, False])
        assert _rmse(sol, solver, lambda X, Y: Y) < 1e-4

    def test_mixed_corner_nonzero_flux(self):
        # u = x^2 - y^2 with prescribed non-zero flux (inner-normal d/d(-x) = -2) on
        # the right edge, whose corners are Dirichlet-Neumann / Neumann-Dirichlet.
        # Exercises chi_{jk} for k>=1 with non-constant Dirichlet data on the
        # adjacent edges (the constant-datum case leaves chi at its trivial k=0 term).
        bc = [[0.0, 0.0, 1.0], [-2.0], [0.0, -2.0, 1.0], [-1.0, 2.0, -1.0]]
        solver, sol = _solve(SQUARE, bc, [True, False, True, True])
        assert _rmse(sol, solver, lambda X, Y: X ** 2 - Y ** 2) < 1e-4

    def test_mixed_corner_linear_nonzero_flux(self):
        # u = x with non-constant Dirichlet data on the bottom/top edges and a
        # prescribed inner-normal flux d/d(-x) = -1 on the right (mixed corners).
        solver, sol = _solve(SQUARE, [[0.0, 1.0], [-1.0], [1.0, -1.0], [0.0]],
                             [True, False, True, True])
        assert _rmse(sol, solver, lambda X, Y: X) < 1e-4
