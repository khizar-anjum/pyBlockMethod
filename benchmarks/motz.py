#!/usr/bin/env python3
"""
Motz's problem: mixed Dirichlet-Neumann benchmark (reviewer item R2.4).

Motz's problem is the standard test for a boundary point where the condition
switches from Dirichlet to Neumann, producing an r^(1/2) cos(theta/2) corner
singularity. On the rectangle [-1,1] x [0,1]:

    u = 0            on the bottom-left edge  (Dirichlet),
    du/dn = 0        on the bottom-right edge (Neumann),
    u = 500          on the right edge        (Dirichlet),
    du/dn = 0        on the top and left      (Neumann, insulated).

The switch sits at the origin. We split the bottom edge there so the BC-type
change lands on a 180-degree vertex; the first-kind block at that vertex has
lambda = 1/(2*alpha) = 1/2, which reproduces the r^(1/2) behavior analytically.

We extract the leading singular coefficient a0 (the generalized stress/flux
intensity factor) by least squares and compare to the well-established benchmark
value a0 = 401.1625. Produces results/motz_convergence.pdf.

Usage:
  python benchmarks/motz.py [--no-plot]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
RESULTS = REPO / "benchmarks" / "results"

VERTS = np.array([[-1, 0], [0, 0], [1, 0], [1, 1], [-1, 1]], float)
BC = [[0.0], [0.0], [500.0], [0.0], [0.0]]
IS_DIRICHLET = [True, False, True, False, False]
A0_REFERENCE = 401.1625        # standard benchmark value
N_VALUES = [10, 15, 20, 25, 30, 40, 50]


def singular_coefficients(sol, X, Y, n_modes=8, r_in=0.1, r_out=0.7):
    """Fit u ~ sum_k a_k r^(k+1/2) cos((k+1/2) theta) about the origin."""
    r = np.hypot(X, Y)
    theta = np.arctan2(Y, X)
    sel = (~sol.mask) & (r >= r_in) & (r <= r_out) & (theta >= -1e-6) & (theta <= np.pi + 1e-6)
    basis = np.stack([r[sel] ** (k + 0.5) * np.cos((k + 0.5) * theta[sel])
                      for k in range(n_modes)], axis=1)
    coeffs, *_ = np.linalg.lstsq(basis, sol.data[sel], rcond=None)
    return coeffs


def run():
    from pyBlockGrid import polygon, volkovSolver
    print(f"\n{'=' * 60}\nMotz's problem (mixed Dirichlet-Neumann), R2.4\n{'=' * 60}")
    print(f"reference leading coefficient a0 = {A0_REFERENCE}\n")
    rows = []
    for n in N_VALUES:
        s = volkovSolver(poly=polygon(VERTS.copy()), boundary_conditions=BC,
                         is_dirichlet=IS_DIRICHLET, delta=0.02, n=n,
                         radial_heuristic=0.75, overlap_heuristic=0.1, max_iter=50)
        sol = s.solve(verbose=False)
        c = singular_coefficients(sol, s._X, s._Y)
        rows.append((n, c[0], abs(c[0] - A0_REFERENCE)))
        print(f"  n={n:3d}: a0={c[0]:.4f}  |a0 - ref|={abs(c[0] - A0_REFERENCE):.2e}  "
              f"(a1={c[1]:.3f}, a2={c[2]:.3f})")
    return rows


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 16, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 13, "figure.dpi": 200})
    n = [r[0] for r in rows]
    err = [max(r[2], 1e-16) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.semilogy(n, err, "o-", color="#1565C0")
    ax.set_xlabel("Angular resolution $n$")
    ax.set_ylabel(r"$|a_0 - a_0^{\mathrm{ref}}|$")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "motz_convergence.pdf"
    fig.savefig(out)
    print(f"\nSaved figure: {out}")


def plot_solution_field(n=40):
    """Render the Motz potential field, highlighting the origin where the
    boundary condition switches Dirichlet->Neumann and the r^(1/2) singularity sits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pyBlockGrid import polygon, volkovSolver
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 16, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 13, "figure.dpi": 200})

    s = volkovSolver(poly=polygon(VERTS.copy()), boundary_conditions=BC,
                     is_dirichlet=IS_DIRICHLET, delta=0.01, n=n,
                     radial_heuristic=0.75, overlap_heuristic=0.1, max_iter=50)
    sol = s.solve(verbose=False)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    cf = ax.contourf(s._X, s._Y, sol, levels=40, cmap="viridis")
    ax.contour(s._X, s._Y, sol, levels=12, colors="k", linewidths=0.4, alpha=0.5)
    # domain outline and the Dirichlet->Neumann switch point
    import matplotlib.patheffects as pe
    outline = [pe.withStroke(linewidth=2.5, foreground="black")]
    verts_closed = np.vstack([VERTS, VERTS[0]])
    ax.plot(verts_closed[:, 0], verts_closed[:, 1], "k-", linewidth=1.5)
    ax.plot(0, 0, "*", color="white", markersize=15,
            markeredgecolor="black", markeredgewidth=0.8)
    ax.annotate("D$\\to$N switch\n($r^{1/2}$ singularity)", xy=(0, 0), xytext=(-0.48, 0.72),
                fontsize=11, color="white", ha="center", path_effects=outline,
                arrowprops=dict(arrowstyle="->", color="white", lw=1.6))
    # boundary-condition markers on each edge (white text, dark outline)
    lbl = dict(fontsize=11, color="white", ha="center", va="center", path_effects=outline)
    ax.text(-0.5, 0.07, "$u=0$", **lbl)                    # bottom-left: Dirichlet
    ax.text(0.52, 0.07, "$\\partial_n u=0$", **lbl)        # bottom-right: Neumann
    ax.text(0.9, 0.5, "$u=500$", rotation=90, **lbl)       # right: Dirichlet
    ax.text(0.0, 0.92, "insulated", **lbl)                 # top: Neumann
    ax.text(-0.92, 0.5, "insulated", rotation=90, **lbl)   # left: Neumann
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    fig.colorbar(cf, ax=ax, label="potential $u$")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "motz_solution.pdf"
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
    rows = run()
    if not args.no_plot:
        plot(rows)
        plot_solution_field()


if __name__ == "__main__":
    main()
