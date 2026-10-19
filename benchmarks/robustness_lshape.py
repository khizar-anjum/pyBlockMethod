#!/usr/bin/env python3
"""
Robustness study on the L-shaped re-entrant corner (reviewer item R1.5).

Referee 1 asked for evidence that the Block Method holds up on a demanding,
singular problem. The L-shape carries a 270-degree re-entrant corner at (1,1):
the solution there behaves like the corner expansion

    u(r, theta) ~= sum_j c_j * r^(2j/3) * sin(2 j theta / 3),

so its gradient is unbounded (~ r^(-1/3)) -- the classic hard case for elliptic
solvers. We grade the Block Method and a properly configured deal.II hp-FEM
(with a sparse-direct solve and with a multigrid-preconditioned CG solve)
against the committed method-independent reference (Lightning cross-validated vs
hp-FEM, agree to RMSE 4.4e-9), on three robustness lenses:

  (1) full-domain accuracy vs compute cost,
  (2) accuracy restricted to the corner neighbourhood (r <= R_CORNER), where the
      singularity dominates,
  (3) convergence of the leading flux-intensity factor c_1 -- the generalized
      stress-intensity factor -- extracted by least squares from each solution.

The Block Method's first-kind block expands in exactly the corner modes above,
so it resolves c_1 to ~1e-7 at a fraction of the degrees of freedom hp-FEM needs.

This script reuses the solver drivers from reproduce_comparison.py; it adds only
the corner-specific analysis. It produces results/robustness_lshape.pdf.

Usage:
  python benchmarks/robustness_lshape.py [--reps N] [--no-plot]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks"))

from reproduce_comparison import (  # noqa: E402
    GEOMETRIES, HP_BIN, RESULTS, load_reference, run_hp, run_volkov, rmse,
)

CORNER = np.array([1.0, 1.0])   # the re-entrant corner
OMEGA = 1.5 * np.pi             # interior opening angle (270 deg)
LAMBDA = np.pi / OMEGA          # leading singular exponent = 2/3
R_CORNER = 0.5                  # radius of the corner-neighbourhood lens
N_MODES = 4                     # singular modes used in the flux-intensity fit


def corner_grid():
    """Return flattened (r, theta) of the shared 40x40 solution grid about the
    corner. theta runs 0..3pi/2 across the open sector (0 on one re-entrant edge,
    3pi/2 on the other), matching the corner expansion above."""
    from pyBlockGrid import polygon, volkovSolver
    cfg = GEOMETRIES["l_shape"]
    s = volkovSolver(poly=polygon(cfg["vertices"].copy()), boundary_conditions=cfg["bc"],
                     is_dirichlet=cfg["is_dirichlet"], delta=0.05, n=5,
                     radial_heuristic=0.85, overlap_heuristic=0.1, max_iter=1)
    s.find_block_covering()
    X, Y = np.asarray(s._X).ravel(), np.asarray(s._Y).ravel()
    r = np.hypot(X - CORNER[0], Y - CORNER[1])
    theta = np.mod(np.arctan2(Y - CORNER[1], X - CORNER[0]) - np.pi / 2, 2 * np.pi)
    return r, theta


def flux_intensity(sol, mask, r, theta):
    """Least-squares fit of the corner expansion to a solution; return c_1."""
    sel = mask & (r >= 0.05) & (r <= R_CORNER)
    basis = np.stack([r[sel] ** (j * LAMBDA) * np.sin(j * LAMBDA * theta[sel])
                      for j in range(1, N_MODES + 1)], axis=1)
    coeffs, *_ = np.linalg.lstsq(basis, sol[sel], rcond=None)
    return coeffs[0]


def corner_rmse(sol, ref, mask, r):
    """RMSE restricted to the corner neighbourhood r <= R_CORNER."""
    sel = mask & (r <= R_CORNER)
    return float(np.sqrt(np.mean((sol[sel] - ref[sel]) ** 2)))


def run(reps):
    cfg = GEOMETRIES["l_shape"]
    ref, mask = load_reference(cfg)
    r, theta = corner_grid()
    c1_ref = flux_intensity(ref, mask, r, theta)
    print(f"\n{'=' * 70}\nL-shape re-entrant-corner robustness (R1.5)\n{'=' * 70}")
    print(f"reference flux-intensity factor c_1 = {c1_ref:.6f}  "
          f"(leading exponent {LAMBDA:.4f})\n")

    volkov, hp, hp_amg = [], [], []
    print("Block Method (Volkov, C++):")
    for n in cfg["volkov_n"]:
        sol, full, _ = run_volkov(cfg, n, reps)
        row = (full, rmse(sol, ref, mask), corner_rmse(sol, ref, mask, r),
               abs(flux_intensity(sol, mask, r, theta) - c1_ref), n)
        volkov.append(row)
        print(f"  n={n:<3d}  RMSE={row[1]:.2e}  corner-RMSE={row[2]:.2e}  "
              f"|c1-c1*|={row[3]:.2e}  full={full * 1e3:7.1f} ms")

    # Same hp discretization with two linear solvers: sparse direct (UMFPACK)
    # and matrix-free CG preconditioned by p-multigrid (the O(N) solver).
    for label, rows, extra in (("UMFPACK", hp, None),
                               ("matrix-free MG", hp_amg, ["--solver", "mf-pmg"])):
        print(f"hp-FEM (deal.II, C++, adaptive, {label}):")
        for c in cfg["hp_cycles"]:
            sol, full, _, dofs = run_hp("l_shape", cfg, c, reps, extra=extra)
            row = (full, rmse(sol, ref, mask), corner_rmse(sol, ref, mask, r),
                   abs(flux_intensity(sol, mask, r, theta) - c1_ref), dofs)
            rows.append(row)
            print(f"  DOFs={dofs:<6d} RMSE={row[1]:.2e}  corner-RMSE={row[2]:.2e}  "
                  f"|c1-c1*|={row[3]:.2e}  full={full * 1e3:7.1f} ms")

    return (np.array(volkov, float), np.array(hp, float),
            np.array(hp_amg, float), c1_ref)


def plot(volkov, hp, hp_amg, c1_ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 16, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 13, "figure.dpi": 200})
    V, H, A = "#1565C0", "#D84315", "#5D4037"
    RESULTS.mkdir(exist_ok=True)

    # (a) accuracy vs cost: full domain (solid) and corner neighbourhood (dashed).
    # Saved as its own panel (no in-plot title) to match the other figures.
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(volkov[:, 0], volkov[:, 1], "o-", color=V, label="Block Method (ours)")
    ax.plot(volkov[:, 0], volkov[:, 2], "o--", color=V, alpha=0.55,
            label="Block Method, corner region")
    ax.plot(hp[:, 0], hp[:, 1], "s-", color=H, label="hp-FEM, direct (deal.II)")
    ax.plot(hp[:, 0], hp[:, 2], "s--", color=H, alpha=0.55,
            label="hp-FEM direct, corner region")
    ax.plot(hp_amg[:, 0], hp_amg[:, 1], "P-", color=A, label="hp-FEM, matrix-free MG")
    ax.plot(hp_amg[:, 0], hp_amg[:, 2], "P--", color=A, alpha=0.55,
            label="hp-FEM matrix-free MG, corner")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Compute time (s)"); ax.set_ylabel("RMSE vs reference")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "robustness_accuracy.pdf")
    plt.close(fig)
    print("Saved figure: robustness_accuracy.pdf")

    # (b) convergence of the leading flux-intensity factor
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(volkov[:, 0], volkov[:, 3], "o-", color=V, label="Block Method (ours)")
    ax.plot(hp[:, 0], hp[:, 3], "s-", color=H, label="hp-FEM, direct (deal.II)")
    ax.plot(hp_amg[:, 0], hp_amg[:, 3], "P-", color=A, label="hp-FEM, matrix-free MG")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Compute time (s)")
    ax.set_ylabel(r"$|c_1 - c_1^\star|$  (flux-intensity error)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "robustness_c1.pdf")
    plt.close(fig)
    print(f"Saved figure: robustness_c1.pdf  (c_1* = {c1_ref:.4f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=4, help="timing repeats (drop 1 warmup)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if not HP_BIN.exists():
        sys.exit(f"ERROR: deal.II solver not built at {HP_BIN} (see REPRODUCE.md).")
    try:
        import pyBlockGrid  # noqa: F401
    except ImportError:
        sys.exit("ERROR: pyBlockGrid not importable (build the C++ backend).")

    volkov, hp, hp_amg, c1_ref = run(args.reps)
    if not args.no_plot:
        plot(volkov, hp, hp_amg, c1_ref)


if __name__ == "__main__":
    main()
