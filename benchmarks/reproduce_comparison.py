#!/usr/bin/env python3
"""
Reproduce the Block Method vs hp-FEM head-to-head comparison.

Runs the Block Method and a properly configured deal.II hp-FEM (both C++, same
machine) on each benchmark geometry, grades them against method-independent
references, and reports the wall-clock speedup at matched accuracy plus a Pareto
plot (time vs error). The Lightning and AAA-LS spectral solvers -- now also C++,
addressing reviewer comment R1.4 -- are included as additional Pareto curves when
their binaries are built.

References are method-independent and committed to the repo, so NO MATLAB is
required to reproduce the comparison. Each reference is certified live against a
*different* solver than the one that produced it, so it can be trusted without
re-running the original tool:
  - unit_square:      exact Fourier-series solution (analytical, no check needed).
  - l_shape:          a Lightning rational-approximation solution, certified
                      against deal.II hp-FEM.
  - square_with_hole: a converged deal.II hp-FEM solution (Lightning cannot
                      handle multiply-connected domains), certified against a
                      converged Block-Method solution. Grading the hp-FEM sweep
                      against a fine hp-FEM is conservative -- it flatters the
                      competitor, not us.

Timing is apples-to-apples and reported two ways (R3.5): the FULL pipeline (our
block-covering generation + solve vs the hp-FEM's mesh generation + adaptation +
assembly + solve) and solve-only (both methods' preprocessing excluded). Only
deal.II's process startup is excluded -- a one-time cost, not per-solve compute.
The Block Method wins under both accountings.

Prerequisites (see REPRODUCE.md, or just use the Docker image):
  - pyBlockGrid C++ backend built (src/pyBlockGrid/_pyblockgrid_cpp*.so)
  - deal.II solver built (benchmarks/dealii/build/hp_laplace)

Usage:
  python benchmarks/reproduce_comparison.py
      [--geometry {unit_square,l_shape,square_with_hole,all}] [--reps N] [--no-plot]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
REF_DIR = REPO / "benchmarks" / "analytical_reference_data"
HP_BIN = REPO / "benchmarks" / "dealii" / "build" / "hp_laplace"
LIGHTNING_BIN = REPO / "benchmarks" / "spectral" / "build" / "lightning_laplace"
AAALS_BIN = REPO / "benchmarks" / "spectral" / "build" / "aaals_laplace"
RESULTS = REPO / "benchmarks" / "results"

# --- geometry configurations ------------------------------------------------
GEOMETRIES = {
    "unit_square": {
        "vertices": np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float),
        "bc": [[1.0], [0.0], [0.0], [0.0]],
        "is_dirichlet": [True] * 4,
        "reference": "analytical_unit_square_hot_bottom.npz",
        "ref_kind": "analytical Fourier series (exact)",
        "volkov_n": [5, 10, 15, 20, 30, 40],
        "hp_min_deg": 2, "hp_max_deg": 5,
        "hp_cycles": [4, 6, 8, 10, 12, 14, 16],
        "lightning_ok": True,
        "spectral_tols": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    },
    "l_shape": {
        "vertices": np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float),
        "bc": [[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
        "is_dirichlet": [True] * 6,
        "reference": "reference_l_shape_hot_bottom.npz",
        "ref_kind": "Lightning solver, cross-validated vs hp-FEM",
        "certifier": "hp",
        "volkov_n": [10, 15, 20, 25, 30, 35, 40],
        "hp_min_deg": 2, "hp_max_deg": 6,
        "hp_cycles": [4, 6, 8, 10, 12, 14],
        "lightning_ok": True,
        "spectral_tols": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    },
    "square_with_hole": {
        "vertices": np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float),
        "bc": [[1.0], [0.0], [0.0], [0.0]],
        "is_dirichlet": [True] * 4,
        "hole": {
            "vertices": np.array([[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]])[::-1],
            "bc": [[0.0]] * 4,
            "is_dirichlet": [True] * 4,
        },
        "reference": "reference_square_with_hole_hot_bottom.npz",
        "ref_kind": "hp-FEM (converged), cross-validated vs Volkov",
        "certifier": "volkov",
        "volkov_n": [5, 10, 15, 20, 30, 40, 50],
        "hp_min_deg": 2, "hp_max_deg": 6,
        "hp_cycles": [3, 4, 6, 8, 10, 12, 14],
        "lightning_ok": False,  # Lightning is simply-connected only
        "spectral_tols": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    },
}


def load_reference(cfg):
    data = np.load(REF_DIR / cfg["reference"])
    return data["solution"].ravel(), data["mask"].ravel()


def rmse(sol, ref, mask):
    return float(np.sqrt(np.mean((sol[mask] - ref[mask]) ** 2)))


def run_volkov(cfg, n, reps):
    """Return (solution_flat, full time, block-covering generation time).

    'full' is the whole solve() -- which already INCLUDES building the block
    covering (R3.5: our cost is not solve-only). find_block_covering() is timed
    separately so the covering-vs-solve split is reportable; solve-only = full - gen.
    """
    from pyBlockGrid import polygon, volkovSolver

    def build():
        poly = polygon(cfg["vertices"].copy())
        if "hole" in cfg:
            h = cfg["hole"]
            poly.add_hole(h["vertices"].copy(), h["bc"], h["is_dirichlet"])
        return volkovSolver(poly=poly, boundary_conditions=cfg["bc"],
                            is_dirichlet=cfg["is_dirichlet"], delta=0.05, n=n,
                            radial_heuristic=0.85, overlap_heuristic=0.1, max_iter=30)

    times, gens, sol = [], [], None
    for _ in range(reps):
        solver = build()
        t0 = time.perf_counter()
        out = solver.solve(verbose=False)
        times.append(time.perf_counter() - t0)
        sol = out.data if hasattr(out, "data") else np.asarray(out)
        gen_solver = build()
        tg = time.perf_counter()
        gen_solver.find_block_covering()
        gens.append(time.perf_counter() - tg)
    pick = lambda a: min(a[1:] if reps > 1 else a)
    return sol.ravel(), pick(times), pick(gens)


def run_hp(geom, cfg, cycles, reps, extra=None):
    """Return (solution_flat, full pipeline time, solve-only time, n_dofs).

    full        = mesh + setup + assembly + solve + adaptation (R3.5 end-to-end).
    solve-only  = setup + assembly + solve (excludes mesh generation + adaptation).
    """
    out_dat, out_t = "/tmp/_repro_hp.dat", "/tmp/_repro_hp_t.dat"
    args = [str(HP_BIN), "--geometry", geom, "--bc", "hot_bottom",
            "--refine-mode", "adaptive", "--min-degree", str(cfg["hp_min_deg"]),
            "--max-degree", str(cfg["hp_max_deg"]), "--init-refine", "2",
            "--cycles", str(cycles), "--out", out_dat, "--timing", out_t]
    if extra:
        args += extra
    full, solveonly, n_dofs, sol = None, None, None, None
    # The binary initializes MPI (for the Trilinos AMG preconditioner); on some
    # machines hwloc's GL component blocks MPI startup while probing the X
    # server, so that component is switched off.
    env = dict(os.environ)
    env.setdefault("HWLOC_COMPONENTS", "-gl")
    # One thread: like the other solvers, and faster than deal.II's task
    # parallelism at these problem sizes.
    env.setdefault("DEAL_II_NUM_THREADS", "1")
    for _ in range(reps):
        subprocess.run(args, check=True, capture_output=True, env=env)
        meta = dict(l.split()[:2] for l in open(out_t) if len(l.split()) >= 2)
        t_full = float(meta["total"])
        if full is None or t_full < full:  # keep the fastest rep's decomposition
            full = t_full
            solveonly = float(meta["setup"]) + float(meta["assembly"]) + float(meta["solve"])
            n_dofs = int(meta["n_dofs"])
        sol = np.loadtxt(out_dat)[:, 2]
    return sol, full, solveonly, n_dofs


def run_spectral(binary, geom, tol, reps):
    """Run a C++ spectral solver (Lightning/AAA-LS). Return (solution_flat, best time).

    The full solve time (pole placement + least squares) is the like-for-like C++
    replacement for the MATLAB spectral timings the reviewer flagged (R1.4).
    """
    out_dat, out_t = "/tmp/_repro_spec.dat", "/tmp/_repro_spec_t.dat"
    args = [str(binary), "--geometry", geom, "--bc", "hot_bottom",
            "--tol", f"{tol:.0e}", "--out", out_dat, "--timing", out_t]
    best, sol = None, None
    for _ in range(reps):
        subprocess.run(args, check=True, capture_output=True)
        meta = dict(l.split()[:2] for l in open(out_t) if len(l.split()) >= 2)
        t = float(meta["total"])
        best = t if best is None else min(best, t)
        sol = np.loadtxt(out_dat)[:, 2]
    return sol, best


def loglog_interp(rmses, times, target):
    """Time to reach a target RMSE by log-log interpolation (None if out of range)."""
    r = np.asarray(rmses); t = np.asarray(times)
    o = np.argsort(r); r, t = r[o], t[o]
    if target < r.min() or target > r.max():
        return None
    return float(np.exp(np.interp(np.log(target), np.log(r), np.log(t))))


def verify_reference(geom, cfg, ref, mask):
    """Certify the committed reference against a fresh solve by a *different*
    method than the one that produced it (so it needs no external tool to trust).
    """
    cert = cfg.get("certifier")
    if cert == "hp":
        sol, *_ = run_hp(geom, cfg, cycles=16, reps=1, extra=["--max-degree", "6"])
        via = "fresh converged hp-FEM"
    elif cert == "volkov":
        sol, *_ = run_volkov(cfg, n=max(cfg["volkov_n"]), reps=1)
        via = "fresh converged Volkov"
    else:
        return  # analytical reference: exact, no cross-check needed
    agree = rmse(sol, ref, mask)
    print(f"  reference self-check: committed ref vs {via}  ->  "
          f"RMSE {agree:.2e}  ({'OK' if agree < 1e-6 else 'MISMATCH'})")


def compare_geometry(geom, reps):
    cfg = GEOMETRIES[geom]
    ref, mask = load_reference(cfg)
    print(f"\n{'=' * 68}\n{geom}  (reference: {cfg['ref_kind']})\n{'=' * 68}")
    verify_reference(geom, cfg, ref, mask)

    volkov = []  # (n, full, gen, rmse)
    for n in cfg["volkov_n"]:
        sol, full, gen = run_volkov(cfg, n, reps)
        volkov.append((n, full, gen, rmse(sol, ref, mask)))
    hp = []  # (cycles, full, solveonly, n_dofs, rmse)
    for c in cfg["hp_cycles"]:
        sol, full, solveonly, d = run_hp(geom, cfg, c, reps)
        hp.append((c, full, solveonly, d, rmse(sol, ref, mask)))
    # Spectral competitors, now also C++ (R1.4): Lightning (simply-connected) + AAA-LS.
    spectral = {}  # name -> list of (tol, time, rmse)
    specs = [("AAA-LS", AAALS_BIN)]
    if cfg.get("lightning_ok"):
        specs.insert(0, ("Lightning", LIGHTNING_BIN))
    for name, binary in specs:
        if not binary.exists():
            continue
        rows = []
        for tol in cfg["spectral_tols"]:
            sol, t = run_spectral(binary, geom, tol, reps)
            rows.append((tol, t, rmse(sol, ref, mask)))
        spectral[name] = rows

    print("\n  Block Method (Volkov, C++)  [full = block-covering generation + solve]:")
    for n, full, gen, e in volkov:
        print(f"    n={n:<3d}  RMSE={e:.2e}  full={full * 1e3:8.1f} ms  "
              f"(covering {gen * 1e3:5.1f} ms)")
    print("  hp-FEM (deal.II, C++)  [full = mesh + assembly + solve + adaptation]:")
    for c, full, solveonly, d, e in hp:
        print(f"    cycles={c:<3d} DOFs={d:<6d} RMSE={e:.2e}  full={full * 1e3:8.1f} ms  "
              f"(setup+asm+solve {solveonly * 1e3:6.1f} ms)")
    for name, rows in spectral.items():
        print(f"  {name} (C++, full solve):")
        for tol, t, e in rows:
            print(f"    tol={tol:.0e}  RMSE={e:.2e}  time={t * 1e3:8.1f} ms")

    v_r = [e for *_, e in volkov]
    v_full = [full for _, full, _, _ in volkov]
    v_solve = [full - gen for _, full, gen, _ in volkov]
    h_r = [e for *_, e in hp]
    h_full = [full for _, full, _, _, _ in hp]
    h_solve = [so for _, _, so, _, _ in hp]

    def spd(vt, ht, tgt):
        tv, th = loglog_interp(v_r, vt, tgt), loglog_interp(h_r, ht, tgt)
        return f"{th / tv:.1f}x" if (tv and th) else "--"

    # Two apples-to-apples accountings (R3.5): full pipeline (our covering+solve
    # vs their mesh+adaptation+assembly+solve) and solve-only (both preprocessing
    # excluded). The Block Method wins under both.
    print("\n  Block-Method speedup vs hp-FEM at matched accuracy:")
    print(f"    {'target RMSE':>12} {'full pipeline':>15} {'solve only':>12}")
    for tgt in [1e-4, 1e-5, 1e-6]:
        print(f"    {tgt:12.0e} {spd(v_full, h_full, tgt):>15} "
              f"{spd(v_solve, h_solve, tgt):>12}")

    data = {"volkov": (v_full, v_r), "hp": (h_full, h_r)}
    for name, rows in spectral.items():
        data[name] = ([t for _, t, _ in rows], [e for _, _, e in rows])
    return data


def make_plot(all_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(all_data)
    style = {"volkov": ("o-", "#1565C0", "Block Method (ours)"),
             "hp": ("s-", "#D84315", "hp-FEM (deal.II)"),
             "Lightning": ("D-", "#2E7D32", "Lightning (C++)"),
             "AAA-LS": ("^-", "#7B1FA2", "AAA-LS (C++)")}
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
    for ax, (geom, d) in zip(axes[0], all_data.items()):
        for key, (fmt, color, label) in style.items():
            if key in d and d[key][0]:
                ax.plot(d[key][0], d[key][1], fmt, color=color, label=label)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Compute time (s)"); ax.set_ylabel("RMSE vs reference")
        ax.set_title(geom); ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "reproduce_pareto.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"\nSaved Pareto plot: {out}")


def plot_test_geometries(output_folder):
    """Draw the three benchmark geometries (Fig.~test_geometries) with enlarged
    fonts (reviewer R2.5). Each edge is labelled with its Dirichlet value from
    GEOMETRIES; the hot edge (u=1) is drawn in red."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly
    os.makedirs(output_folder, exist_ok=True)
    FS = 18
    for geom, cfg in GEOMETRIES.items():
        V = cfg["vertices"]
        centroid = V.mean(axis=0)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.add_patch(MplPoly(V, closed=True, facecolor="#d6e6f5", edgecolor="none"))
        for i in range(len(V)):
            a, b = V[i], V[(i + 1) % len(V)]
            val = cfg["bc"][i][0]
            hot = val != 0.0
            ax.plot([a[0], b[0]], [a[1], b[1]], color="red" if hot else "black",
                    lw=4 if hot else 2, zorder=2)
            mid = 0.5 * (a + b)
            ax.annotate(f"$u={val:g}$", mid + 0.12 * (centroid - mid), fontsize=FS,
                        color="red" if hot else "black", ha="center", va="center")
        if "hole" in cfg:
            H = cfg["hole"]["vertices"]
            ax.add_patch(MplPoly(H, closed=True, facecolor="white",
                                 edgecolor="black", lw=2, zorder=3))
            ax.annotate(f"$u={cfg['hole']['bc'][0][0]:g}$", H.mean(axis=0),
                        fontsize=FS, ha="center", va="center", zorder=4)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=FS - 3)
        ax.margins(0.12)
        ax.grid(True, alpha=0.3)
        out = os.path.join(output_folder, f"geometry_{geom}.pdf")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", default="all",
                    choices=["unit_square", "l_shape", "square_with_hole", "all"])
    ap.add_argument("--reps", type=int, default=4, help="timing repeats (drop 1 warmup)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--plot-geometries", metavar="OUTDIR",
                    help="just draw the geometry sketches (Fig. test_geometries) and exit")
    args = ap.parse_args()

    if args.plot_geometries:
        plot_test_geometries(args.plot_geometries)
        return

    if not HP_BIN.exists():
        sys.exit(f"ERROR: deal.II solver not built at {HP_BIN}\n"
                 f"Build it (see REPRODUCE.md), or use the Docker image.")
    try:
        import pyBlockGrid  # noqa: F401
    except ImportError:
        sys.exit("ERROR: pyBlockGrid not importable. Build the C++ backend "
                 "(see REPRODUCE.md), or use the Docker image.")

    geoms = list(GEOMETRIES) if args.geometry == "all" else [args.geometry]
    all_data = {g: compare_geometry(g, args.reps) for g in geoms}
    if not args.no_plot:
        make_plot(all_data)


if __name__ == "__main__":
    main()
