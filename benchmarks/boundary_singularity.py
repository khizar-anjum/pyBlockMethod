#!/usr/bin/env python3
"""
Boundary-data limit study: near-edge external singularity (reviewer items R1.5b, R3.6).

Referee 1 noted the sharpest case for the Block Method: a harmonic solution whose
continuation has a singularity just *outside* the domain, close to an edge. The
Block Method imposes boundary data as a per-edge Taylor (monomial) polynomial, so
this case stresses the *boundary representation* rather than the interior solve.

We make that precise with a manufactured solution on the unit square,

    u(x, y) = log |z - z0|,   z0 = (0.5, -d),

which is harmonic inside the square (the singularity sits a distance d below the
bottom edge). We sweep d (how close the singularity is to the edge) and the Taylor
degree of the per-edge boundary polynomial, and grade against the exact u.

Two things emerge, addressing the two comments together:
  * R3.6: the expansion degree is a genuine precision knob -- for a fixed d the
    error decreases as the degree grows, tracking the boundary-fit residual.
  * R1.5b: as d -> 0 the required degree grows quickly and a practical floor sets
    in (the high-degree fit becomes ill-conditioned). This is the regime where an
    hp-FEM or boundary-integral method, which do not represent the datum by a
    global polynomial per edge, are the appropriate tool.

Produces results/boundary_singularity.pdf.

Usage:
  python benchmarks/boundary_singularity.py [--no-plot]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
RESULTS = REPO / "benchmarks" / "results"

SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
DISTANCES = [0.5, 0.2, 0.1, 0.05]
DEGREES = [2, 4, 6, 8, 12, 16]
N_ANG = 40           # angular resolution, high enough that the BC fit dominates
DELTA = 0.02


def exact(d):
    """Harmonic u = log|z - z0| with z0 = (0.5, -d) just below the bottom edge."""
    x0, y0 = 0.5, -d
    return lambda X, Y: 0.5 * np.log((X - x0) ** 2 + (Y - y0) ** 2)


def edge_taylor(v0, v1, u, degree):
    """Least-squares monomial fit (in arc length) of u along one edge; return the
    coefficient vector and the max fit residual on the edge."""
    length = np.hypot(*(v1 - v0))
    s = np.linspace(0, length, 600)
    pts = v0 + (s / length)[:, None] * (v1 - v0)
    vals = u(pts[:, 0], pts[:, 1])
    basis = np.vander(s, degree + 1, increasing=True)
    coeffs, *_ = np.linalg.lstsq(basis, vals, rcond=None)
    return list(coeffs), float(np.max(np.abs(basis @ coeffs - vals)))


def run():
    from pyBlockGrid import polygon, volkovSolver
    print(f"\n{'=' * 66}\nNear-edge external singularity  u = log|z - (0.5, -d)|\n{'=' * 66}")
    results = {}   # d -> list of (degree, rmse, fit_residual)
    for d in DISTANCES:
        u = exact(d)
        rows = []
        for deg in DEGREES:
            bcs, fit = [], 0.0
            for i in range(4):
                c, r = edge_taylor(SQUARE[i], SQUARE[(i + 1) % 4], u, deg)
                bcs.append(c); fit = max(fit, r)
            s = volkovSolver(poly=polygon(SQUARE.copy()), boundary_conditions=bcs,
                             is_dirichlet=[True] * 4, delta=DELTA, n=N_ANG,
                             radial_heuristic=0.75, overlap_heuristic=0.1, max_iter=20)
            sol = s.solve(verbose=False)
            m = ~sol.mask
            err = np.abs(sol[m] - u(s._X, s._Y)[m])
            rows.append((deg, float(np.sqrt(np.mean(err ** 2))), fit))
        results[d] = rows
        print(f"\n d = {d}:")
        for deg, rms, fit in rows:
            print(f"   degree {deg:2d}:  RMSE = {rms:.2e}   (boundary-fit residual {fit:.1e})")
    return results


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 16, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 13, "figure.dpi": 200})
    markers = ["o", "s", "D", "^", "v", "X"]   # match Fig. MSE_vs_n color/marker scheme
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for (d, rows), mk in zip(results.items(), markers):
        deg = [r[0] for r in rows]
        rms = [r[1] for r in rows]
        ax.semilogy(deg, rms, marker=mk, linestyle="-", label=f"$d = {d}$")
    ax.set_xlabel("Boundary-data Taylor degree")
    ax.set_ylabel("RMSE vs exact solution")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="singularity depth")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "boundary_singularity.pdf"
    fig.savefig(out)
    print(f"\nSaved figure: {out}")


def plot_solution_field(d=0.1, degree=12):
    """Render the manufactured harmonic field u = log|z - z0| solved by the Block
    Method, highlighting the external singularity z0 = (0.5, -d) sitting just
    below the bottom edge, which is what sharpens the boundary trace as d -> 0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pyBlockGrid import polygon, volkovSolver
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 16, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 13, "figure.dpi": 200})

    u = exact(d)
    bcs = [edge_taylor(SQUARE[i], SQUARE[(i + 1) % 4], u, degree)[0] for i in range(4)]
    s = volkovSolver(poly=polygon(SQUARE.copy()), boundary_conditions=bcs,
                     is_dirichlet=[True] * 4, delta=DELTA, n=N_ANG,
                     radial_heuristic=0.75, overlap_heuristic=0.1, max_iter=20)
    sol = s.solve(verbose=False)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    cf = ax.contourf(s._X, s._Y, sol, levels=40, cmap="viridis")
    ax.contour(s._X, s._Y, sol, levels=12, colors="k", linewidths=0.4, alpha=0.5)
    verts_closed = np.vstack([SQUARE, SQUARE[0]])
    ax.plot(verts_closed[:, 0], verts_closed[:, 1], "k-", linewidth=1.5)
    ax.plot(0.5, -d, "*", color="white", markersize=15,
            markeredgecolor="black", markeredgewidth=0.8)
    ax.annotate(f"singularity ($d={d}$)", xy=(0.5, -d),
                xytext=(0.5, -0.27), fontsize=15, color="black", ha="center", va="top",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.38, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    fig.colorbar(cf, ax=ax, label="$u = \\log|z - z_0|$")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "boundary_solution.pdf"
    fig.savefig(out)
    print(f"Saved figure: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    try:
        import pyBlockGrid  # noqa: F401
    except ImportError:
        sys.exit("ERROR: pyBlockGrid not importable (build the C++ backend).")
    results = run()
    if not args.no_plot:
        plot(results)
        plot_solution_field()


if __name__ == "__main__":
    main()
