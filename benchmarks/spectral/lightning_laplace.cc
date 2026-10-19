// Lightning Laplace solver (Gopal & Trefethen, SIAM J. Numer. Anal. 2019),
// reimplemented in C++/Eigen from benchmarks/matlab/laplace.m.
//
// Reviewer 1 (R1.4) objected that the paper timed the C++ Block Method against a
// MATLAB Lightning solver. This is a like-for-like C++ port so the comparison is
// fair. It handles simply-connected polygons (unit_square, l_shape) with
// piecewise-constant Dirichlet data; the discontinuous data triggers the same
// corner-distance error weighting ('rel') the MATLAB code uses automatically.
//
// Method: rational approximation with poles exponentially clustered near each
// corner plus an Arnoldi-orthogonalised polynomial term, fit by weighted least
// squares; the number of poles per corner is grown adaptively until the boundary
// error meets the tolerance.
//
// Usage (matches hp_laplace.cc conventions):
//   ./lightning_laplace --geometry l_shape --bc hot_bottom --tol 1e-8 \
//       --delta 0.05 --out sol.dat --timing timing.dat

#include "spectral_common.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace spectral;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::VectorXcd;
using Eigen::VectorXd;

int main(int argc, char** argv) {
  const std::string geom = arg_str(argc, argv, "--geometry", "unit_square");
  const std::string bc = arg_str(argc, argv, "--bc", "hot_bottom");
  const double tol = arg_dbl(argc, argv, "--tol", 1e-8);
  const double delta = arg_dbl(argc, argv, "--delta", 0.05);
  const std::string out_file = arg_str(argc, argv, "--out", "sol.dat");
  const std::string time_file = arg_str(argc, argv, "--timing", "timing.dat");

  const std::vector<cd> w = outer_vertices(geom);
  if (!hole_vertices(geom).empty()) {
    std::cerr << "lightning_laplace: geometry '" << geom
              << "' is multiply-connected; Lightning supports only "
                 "simply-connected domains.\n";
    return 2;
  }
  const int nw = w.size();
  const std::vector<double> gside = bc_per_side(geom, bc);

  // Scale/translation invariants and per-corner outward bisector (65-79).
  double xmin = w[0].real(), xmax = xmin, ymin = w[0].imag(), ymax = ymin;
  for (const cd& z : w) {
    xmin = std::min(xmin, z.real()); xmax = std::max(xmax, z.real());
    ymin = std::min(ymin, z.imag()); ymax = std::max(ymax, z.imag());
  }
  const cd wc((xmin + xmax) / 2, (ymin + ymax) / 2);
  const double scl = std::max(xmax - xmin, ymax - ymin);
  std::vector<cd> outward(nw);
  std::vector<double> dw(nw);  // side lengths
  for (int k = 0; k < nw; ++k) {
    const int kn = (k + 1) % nw, kp = (k + nw - 1) % nw;
    dw[k] = std::abs(w[kn] - w[k]);
    const cd fwd = w[kn] - w[k], bwd = w[kp] - w[k];
    const cd tmp = cd(0, 1) * bwd * std::sqrt(-fwd / bwd);
    outward[k] = tmp / std::abs(tmp);
  }

  // Discontinuous data => corner-distance weighting (478-488).
  bool rel = false;
  for (int k = 0; k < nw; ++k)
    if (std::abs(gside[k] - gside[(k + nw - 1) % nw]) > tol) rel = true;

  const double q = 0.5;  // adaptive mode
  std::vector<int> nkv(nw, 0);

  // Best-so-far state kept for the evaluation step.
  int best_n = 0; cd best_wc = wc; double best_err = 1e300;
  MatrixXcd best_H; VectorXcd best_cc; std::vector<cd> best_pol; std::vector<double> best_d;

  auto t0 = std::chrono::high_resolution_clock::now();
  const int maxstep = 30;
  for (int stepno = 1; stepno <= maxstep; ++stepno) {
    // --- poles + clustered sample distances per side (111-128) ---
    std::vector<cd> pol; std::vector<double> d;
    std::vector<std::vector<double>> tt(nw);
    for (int k = 0; k < nw; ++k) {
      const int nk = nkv[k];
      std::vector<double> dk;
      std::vector<cd> polk;
      for (int i = 1; i <= nk; ++i) {
        const double sk = std::sqrt((double)i) - std::sqrt((double)nk);
        const double dki = scl * std::exp(4.0 * sk);
        if (dki > 1e-15 * scl) { dk.push_back(dki); polk.push_back(w[k] + outward[k] * dki); }
      }
      // Drop any pole that falls inside the domain (111: "don't allow poles in Omega").
      for (int i = 0; i < (int)polk.size(); ++i)
        if (!inside_polygon(polk[i], w)) { pol.push_back(polk[i]); d.push_back(dk[i]); }
      // Sample distances clustered like the poles, plus a base linspace (123-127).
      std::vector<double> dvec;
      for (double x : dk) { dvec.push_back(x / 3); dvec.push_back(2 * x / 3); dvec.push_back(x); }
      for (double x : dvec) if (x < dw[k]) tt[k].push_back(x);
      const int npt = std::max(30, nk);
      for (int i = 0; i < npt; ++i) tt[k].push_back(dw[k] * i / (npt - 1.0));
      const int kp = (k + nw - 1) % nw;
      for (double x : dvec) if (x < dw[kp]) tt[kp].push_back(dw[kp] - x);
    }

    // --- boundary sample points Z and data G (129-137) ---
    std::vector<cd> Zv; std::vector<double> Gv;
    for (int k = 0; k < nw; ++k) {
      std::sort(tt[k].begin(), tt[k].end());
      const cd dir = (w[(k + 1) % nw] - w[k]) / dw[k];
      for (double t : tt[k]) { Zv.push_back(w[k] + t * dir); Gv.push_back(gside[k]); }
    }
    const int M = Zv.size();
    VectorXcd Z = Eigen::Map<VectorXcd>(Zv.data(), M);
    VectorXd G = Eigen::Map<VectorXd>(Gv.data(), M);

    // --- assemble the real least-squares matrix A (147-180) ---
    const int n = 4 * stepno, Np = pol.size();
    MatrixXcd H, Q = arnoldi_fit(Z, n, wc, H);
    MatrixXcd P = pole_basis(Z, pol, d);
    const int N = 2 * n + 1 + 2 * Np;
    MatrixXd A(M, N);
    A.leftCols(n + 1) = Q.real();
    A.middleCols(n + 1, n) = Q.rightCols(n).imag();
    if (Np > 0) { A.middleCols(2 * n + 1, Np) = P.real(); A.rightCols(Np) = P.imag(); }

    // --- weights (nearest-corner distance if data discontinuous) ---
    VectorXd wt = VectorXd::Ones(M);
    std::vector<int> Kj(M);
    for (int m = 0; m < M; ++m) {
      double best = 1e300; int kbest = 0;
      for (int k = 0; k < nw; ++k) { double dd = std::abs(Z[m] - w[k]); if (dd < best) { best = dd; kbest = k; } }
      Kj[m] = kbest;
      if (rel) wt[m] = std::abs(Z[m] - w[kbest]) / scl;
    }
    const VectorXd sw = wt.array().sqrt();

    // --- weighted least-squares solve c = (W A) \ (W G) (195) ---
    MatrixXd WA = sw.asDiagonal() * A;
    VectorXd WG = sw.array() * G.array();
    VectorXd c = WA.colPivHouseholderQr().solve(WG);

    // complex coefficients cc = [c0; poly; poles] (196-197)
    VectorXcd cc(1 + n + Np);
    cc[0] = c[0];
    for (int i = 0; i < n; ++i) cc[1 + i] = cd(c[1 + i], -c[n + 1 + i]);
    for (int p = 0; p < Np; ++p) cc[1 + n + p] = cd(c[2 * n + 1 + p], -c[2 * n + 1 + Np + p]);

    // --- errors, global and per corner (202-206) ---
    VectorXd resid = (A * c - G).cwiseAbs().array() * wt.array();
    const double err = resid.maxCoeff();
    std::vector<double> errk(nw, 0.0);
    for (int m = 0; m < M; ++m) errk[Kj[m]] = std::max(errk[Kj[m]], resid[m]);

    if (err < best_err) {
      best_err = err; best_n = n; best_wc = wc; best_H = H; best_cc = cc;
      best_pol = pol; best_d = d;
    }
    if (err < 0.5 * tol) break;
    if (N > 1200 || stepno == maxstep || Np >= 100 * nw) break;

    // --- adapt the pole count per corner (208-215) ---
    for (int k = 0; k < nw; ++k) {
      if (errk[k] > q * err && nkv[k] < 100)
        nkv[k] += (int)std::ceil(1.0 + std::sqrt((double)nkv[k]));
      else
        nkv[k] = std::max(nkv[k], (int)std::ceil(stepno / 2.0));
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  const double t_total = std::chrono::duration<double>(t1 - t0).count();

  // --- evaluate u = real(f) on the grid (351-370) ---
  auto u = [&](const cd& z) -> double {
    VectorXcd Z(1); Z[0] = z;
    MatrixXcd Q = arnoldi_eval(Z, best_H, best_wc);
    MatrixXcd P = pole_basis(Z, best_pol, best_d);
    cd f = 0;
    for (int j = 0; j <= best_n; ++j) f += Q(0, j) * best_cc[j];
    for (int p = 0; p < (int)best_pol.size(); ++p) f += P(0, p) * best_cc[1 + best_n + p];
    return f.real();
  };

  const Grid grid = make_grid(geom, delta);
  write_solution(out_file, grid, w, hole_vertices(geom), u);
  write_timing(time_file, t_total, "Lightning_Cxx",
               {{"maxerr", best_err}, {"npoles", (double)best_pol.size()}});

  std::cout << "geom=" << geom << " bc=" << bc << " tol=" << tol
            << " npoles=" << best_pol.size() << " maxerr=" << best_err
            << " | total=" << t_total << "s\n";
  return 0;
}
