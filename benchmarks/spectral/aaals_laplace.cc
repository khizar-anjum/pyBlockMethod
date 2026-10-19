// AAA-LS Laplace solver (Costa & Trefethen, 2023), reimplemented in C++/Eigen
// from benchmarks/matlab/aaals_benchmark.m (+ aaa.m, VAorthog.m, VAeval.m).
//
// Reviewer 1 (R1.4) objected that the paper timed the C++ Block Method against a
// MATLAB AAA-LS solver. This is a like-for-like C++ port. AAA is used only to
// place poles near corners (from the boundary data, or an artificial corner
// indicator on the multiply-connected domain); the harmonic solution is then a
// least-squares fit in a Vandermonde+Arnoldi polynomial basis plus those poles
// (plus a Laurent/log basis around the hole for square_with_hole).
//
// Usage (matches hp_laplace.cc conventions):
//   ./aaals_laplace --geometry square_with_hole --bc hot_bottom --tol 1e-6 \
//       --delta 0.05 --out sol.dat --timing timing.dat

#include "spectral_common.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>
#include <Eigen/Dense>

using namespace spectral;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::VectorXcd;
using Eigen::VectorXd;

// LAPACK complex generalized eigenvalue solver (for AAA poles).
extern "C" void zggev_(const char*, const char*, const int*, cd*, const int*,
                       cd*, const int*, cd*, cd*, cd*, const int*, cd*,
                       const int*, cd*, const int*, double*, int*);

constexpr int N_POLY = 20;       // polynomial degree of the LS basis
constexpr int N_PTS_SIDE = 300;  // boundary sample points per side
constexpr int N_HOLE_POLY = 10;  // Laurent terms around the hole

// Clustered points on the segment z1->z2, tanh-bunched toward the ends (218-223).
static std::vector<cd> cluster_points(cd z1, cd z2, int np) {
  std::vector<cd> Z(np);
  const double s0 = std::tanh(-14.0), s1 = std::tanh(14.0);
  for (int i = 0; i < np; ++i) {
    const double t = std::tanh(-14.0 + 28.0 * i / (np - 1.0));
    Z[i] = z1 + ((t - s0) / (s1 - s0)) * (z2 - z1);
  }
  return Z;
}

// Finite generalized eigenvalues of the arrowhead pencil (E, B) via LAPACK zggev.
static std::vector<cd> pencil_poles(const std::vector<cd>& zj,
                                    const VectorXcd& wj) {
  const int m = zj.size(), n1 = m + 1;
  MatrixXcd E = MatrixXcd::Zero(n1, n1), B = MatrixXcd::Zero(n1, n1);
  for (int i = 0; i < m; ++i) {
    E(0, i + 1) = wj[i];       // first row: [0  wj.']
    E(i + 1, 0) = cd(1, 0);    // first col below: ones
    E(i + 1, i + 1) = zj[i];   // diag(zj)
    B(i + 1, i + 1) = cd(1, 0);
  }
  VectorXcd alpha(n1), beta(n1);
  int info = 0, lwork = -1;
  std::vector<double> rwork(8 * n1);
  cd wq;
  zggev_("N", "N", &n1, E.data(), &n1, B.data(), &n1, alpha.data(), beta.data(),
         nullptr, &n1, nullptr, &n1, &wq, &lwork, rwork.data(), &info);
  lwork = (int)wq.real();
  std::vector<cd> work(std::max(1, lwork));
  zggev_("N", "N", &n1, E.data(), &n1, B.data(), &n1, alpha.data(), beta.data(),
         nullptr, &n1, nullptr, &n1, work.data(), &lwork, rwork.data(), &info);

  double maxb = 0;
  for (int i = 0; i < n1; ++i) maxb = std::max(maxb, std::abs(beta[i]));
  std::vector<cd> pol;
  for (int i = 0; i < n1; ++i)
    if (std::abs(beta[i]) > 1e-8 * maxb) pol.push_back(alpha[i] / beta[i]);  // drop infinite
  return pol;
}

// AAA rational approximation core: return the poles of the approximant to data F
// on sample points Z at tolerance tol (aaa.m core; cleanup/Lawson gated off).
static std::vector<cd> aaa_poles(const VectorXcd& Fin, const VectorXcd& Zin, double tol) {
  // Drop repeated sample points (adjacent sides share corner nodes); a duplicate
  // Z would make 1/(Z-Z_j) infinite and poison the barycentric form (aaa.m 118).
  std::vector<cd> Zu, Fu;
  for (int i = 0; i < Zin.size(); ++i) {
    bool dup = false;
    for (cd z : Zu) if (std::abs(z - Zin[i]) < 1e-13) { dup = true; break; }
    if (!dup) { Zu.push_back(Zin[i]); Fu.push_back(Fin[i]); }
  }
  const int M = Zu.size();
  if (M < 3) return {};
  VectorXcd Z = Eigen::Map<VectorXcd>(Zu.data(), M);
  VectorXcd F = Eigen::Map<VectorXcd>(Fu.data(), M);
  const double Finf = F.cwiseAbs().maxCoeff();
  const double abstol = tol * Finf;
  VectorXcd R = VectorXcd::Constant(M, F.mean());
  std::vector<int> J(M);
  for (int i = 0; i < M; ++i) J[i] = i;
  std::vector<cd> zj;
  std::vector<cd> fj;
  std::vector<VectorXcd> C, A;   // Cauchy and Loewner columns (each length M)
  std::vector<int> support;      // rows that are support points
  VectorXcd wj;

  // Return the BEST rational found, and stop once the error stagnates. Smooth
  // data converges to `abstol`; a sampled boundary discontinuity (a corner
  // between a hot and cold side) cannot be driven to abstol, so unbounded
  // refinement would just scatter spurious poles -- stopping at the error elbow
  // keeps the poles clean, matching the accuracy AAA-LS achieves in MATLAB.
  std::vector<cd> best_zj; VectorXcd best_wj; double best_me = 1e300; int stall = 0;
  const int mmax = std::min(100, M - 1);
  for (int m = 1; m <= mmax; ++m) {
    // greedy: pick active index of maximal residual (NaN residuals are skipped,
    // mirroring MATLAB max; if none remain the approximant is degenerate -> stop)
    int jj = -1; double best = -1;
    for (int idx : J) { double r = std::abs(F[idx] - R[idx]); if (r > best) { best = r; jj = idx; } }
    if (jj < 0) break;
    zj.push_back(Z[jj]); fj.push_back(F[jj]); support.push_back(jj);
    VectorXcd cnew = (Z.array() - Z[jj]).inverse();
    cnew[jj] = 0;  // avoid Inf; this row is overridden below
    C.push_back(cnew);
    A.push_back((F.array() - fj.back()) * cnew.array());
    J.erase(std::remove(J.begin(), J.end(), jj), J.end());

    // weights = right singular vector of smallest singular value of A(J,:)
    MatrixXcd Aact(J.size(), m);
    for (int c = 0; c < m; ++c)
      for (int r = 0; r < (int)J.size(); ++r) Aact(r, c) = A[c][J[r]];
    // Right singular vector of the smallest singular value. Thin V (= full V for
    // a tall matrix) suffices when there are at least as many active rows as
    // columns; otherwise use full V so column m-1 exists (MATLAB null-space
    // branch, aaa.m 181-184). BDCSVD is far faster than JacobiSVD at these sizes.
    const unsigned vopt =
        ((int)J.size() >= m) ? Eigen::ComputeThinV : Eigen::ComputeFullV;
    VectorXd coln = Aact.colwise().norm();
    const double cond = coln.maxCoeff() / std::max(coln.minCoeff(), 1e-300);
    if (cond > 1.0 / (3.0 * std::numeric_limits<double>::epsilon())) {
      MatrixXcd As = Aact;
      for (int c = 0; c < m; ++c) As.col(c) /= coln[c];
      Eigen::JacobiSVD<MatrixXcd> svd(As, vopt);
      wj = svd.matrixV().col(m - 1);
      for (int c = 0; c < m; ++c) wj[c] /= coln[c];
      wj /= wj.norm();
    } else {
      Eigen::JacobiSVD<MatrixXcd> svd(Aact, vopt);
      wj = svd.matrixV().col(m - 1);
    }

    // rational approximant R = N/D with interpolation at support points
    VectorXcd N = VectorXcd::Zero(M), D = VectorXcd::Zero(M);
    for (int c = 0; c < m; ++c) { N += C[c] * (wj[c] * fj[c]); D += C[c] * wj[c]; }
    R = N.array() / D.array();
    for (int s = 0; s < m; ++s) R[support[s]] = fj[s];

    const double me = (F - R).cwiseAbs().maxCoeff();
    if (me < best_me * (1.0 - 1e-3)) { best_me = me; best_zj = zj; best_wj = wj; stall = 0; }
    else if (++stall >= 10) break;   // error has stagnated
    if (me <= abstol) { best_zj = zj; best_wj = wj; break; }
  }

  // drop zero-weight support points from the best approximant
  std::vector<cd> zj2; VectorXcd wj2(best_wj.size());
  int cnt = 0;
  for (int k = 0; k < (int)best_zj.size(); ++k)
    if (std::abs(best_wj[k]) > 0) { zj2.push_back(best_zj[k]); wj2[cnt++] = best_wj[k]; }
  wj2.conservativeResize(cnt);
  if (zj2.size() < 2) return {};
  return pencil_poles(zj2, wj2);
}

// Least-squares solve of the real system [Re(B) -Im(B)] [cr;ci] = H, returning
// the complex coefficient vector c = cr + i ci (aaals_benchmark 260-265).
static VectorXcd ls_complex_coeffs(const MatrixXcd& B, const VectorXd& H) {
  const int K = B.cols();
  MatrixXd A(B.rows(), 2 * K);
  A.leftCols(K) = B.real();
  A.rightCols(K) = -B.imag();
  // Rank-truncated minimum-norm least squares: the AAA pole basis is highly
  // ill-conditioned, so an ordinary QR/backslash solve blows up in the interior.
  // Truncating near-null directions (as MATLAB's rank-revealing backslash does)
  // keeps the fit bounded and matches the reference solution.
  Eigen::CompleteOrthogonalDecomposition<MatrixXd> cod(A.rows(), A.cols());
  cod.setThreshold(1e-10);
  cod.compute(A);
  VectorXd c = cod.solve(H);
  VectorXcd cc(K);
  for (int k = 0; k < K; ++k) cc[k] = cd(c[k], c[K + k]);
  return cc;
}

// Per-pole scale d_p = min_m |Z_m - pol_p|.
static std::vector<double> pole_scale(const VectorXcd& Z, const std::vector<cd>& pol) {
  std::vector<double> d(pol.size());
  for (int p = 0; p < (int)pol.size(); ++p) {
    double mn = 1e300;
    for (int m = 0; m < Z.size(); ++m) mn = std::min(mn, std::abs(Z[m] - pol[p]));
    d[p] = mn;
  }
  return d;
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--selftest") {
    // Recover known poles of f(z) = 1/(z-2) + 0.5/(z+1.3i) on the unit circle.
    const int M = 200;
    VectorXcd Z(M), F(M);
    for (int i = 0; i < M; ++i) {
      Z[i] = std::exp(cd(0, 1) * (2 * M_PI * i / M));
      F[i] = 1.0 / (Z[i] - cd(2, 0)) + 0.5 / (Z[i] - cd(0, -1.3));
    }
    for (cd p : aaa_poles(F, Z, 1e-11)) std::cout << "pole " << p << "\n";
    return 0;
  }
  const std::string geom = arg_str(argc, argv, "--geometry", "unit_square");
  const std::string bc = arg_str(argc, argv, "--bc", "hot_bottom");
  const double tol = arg_dbl(argc, argv, "--tol", 1e-6);
  const double delta = arg_dbl(argc, argv, "--delta", 0.05);
  const std::string out_file = arg_str(argc, argv, "--out", "sol.dat");
  const std::string time_file = arg_str(argc, argv, "--timing", "timing.dat");

  const std::vector<cd> outer = outer_vertices(geom);
  const std::vector<cd> hole = hole_vertices(geom);
  const std::vector<double> gside = bc_per_side(geom, bc);
  const bool doubly = !hole.empty();

  // --- boundary samples Z and Dirichlet data H ---
  std::vector<cd> Zv; std::vector<double> Hv;
  for (int k = 0; k < (int)outer.size(); ++k) {
    auto seg = cluster_points(outer[k], outer[(k + 1) % outer.size()], N_PTS_SIDE);
    for (cd z : seg) { Zv.push_back(z); Hv.push_back(gside[k]); }
  }
  // Hole corners (CCW) for AAA + sampling on the multiply-connected domain.
  std::vector<cd> hcorn;
  if (doubly) {
    hcorn = {cd(0.3, 0.3), cd(0.7, 0.3), cd(0.7, 0.7), cd(0.3, 0.7)};
    for (int k = 0; k < (int)hcorn.size(); ++k) {
      auto seg = cluster_points(hcorn[k], hcorn[(k + 1) % hcorn.size()], N_PTS_SIDE);
      for (cd z : seg) { Zv.push_back(z); Hv.push_back(0.0); }
    }
  }
  const int M = Zv.size();
  VectorXcd Z = Eigen::Map<VectorXcd>(Zv.data(), M);
  VectorXd H = Eigen::Map<VectorXd>(Hv.data(), M);

  auto t0 = std::chrono::high_resolution_clock::now();

  // --- generate poles near corners via local AAA fits ---
  std::vector<cd> corners = outer;
  if (doubly) corners.insert(corners.end(), hcorn.begin(), hcorn.end());

  // AAA target: boundary data (simple) or a corner-singularity indicator (doubly).
  VectorXcd Faaa = H.cast<cd>();
  if (doubly) {
    Faaa = VectorXcd::Ones(M);
    for (cd c : corners)
      for (int m = 0; m < M; ++m) Faaa[m] *= std::sqrt(std::abs(Z[m] - c));
  }

  std::vector<cd> pol;
  for (int k = 0; k < (int)corners.size(); ++k) {
    std::vector<int> ii;  // sample indices whose nearest corner is k
    for (int m = 0; m < M; ++m) {
      int kbest = 0; double mn = 1e300;
      for (int c = 0; c < (int)corners.size(); ++c) {
        double dd = std::abs(Z[m] - corners[c]);
        if (dd < mn) { mn = dd; kbest = c; }
      }
      if (kbest == k) ii.push_back(m);
    }
    if ((int)ii.size() < 3) continue;
    VectorXcd Fi(ii.size()), Zi(ii.size());
    for (int r = 0; r < (int)ii.size(); ++r) { Fi[r] = Faaa[ii[r]]; Zi[r] = Z[ii[r]]; }
    for (cd p : aaa_poles(Fi, Zi, tol)) {
      const bool in_out = inside_polygon(p, outer);
      const bool in_hole = doubly && inside_polygon(p, hcorn);
      if (doubly ? (!in_out || in_hole) : !in_out) pol.push_back(p);
    }
  }
  // Fallback: exponentially clustered poles near corners (aaals 245-255 / 299-317).
  if (pol.empty()) {
    for (int k = 0; k < (int)outer.size(); ++k) {
      cd dir = std::exp(cd(0, 1) * (std::arg(outer[(k + 1) % outer.size()] - outer[k]) + M_PI));
      for (int j = 1; j <= 20; ++j) {
        cd p = outer[k] + 0.5 * std::exp(-double(j)) * dir;
        if (!inside_polygon(p, outer)) pol.push_back(p);
      }
    }
  }

  // --- least-squares fit in [poly | poles | (hole basis)] ---
  const int Np = pol.size();
  const std::vector<double> d = pole_scale(Z, pol);
  MatrixXcd Hes, P = arnoldi_fit(Z, N_POLY, cd(0, 0), Hes);  // VAorthog (uncentred)
  MatrixXcd Qp = pole_basis(Z, pol, d);
  const cd zc = doubly ? std::accumulate(hcorn.begin(), hcorn.end(), cd(0, 0)) / cd((double)hcorn.size(), 0) : cd(0, 0);

  VectorXcd cc; MatrixXd cr_ci;  // coefficients (doubly uses a bespoke packing)
  VectorXd cr, ci;
  if (!doubly) {
    MatrixXcd B(M, P.cols() + Np);
    B.leftCols(P.cols()) = P;
    B.rightCols(Np) = Qp;
    cc = ls_complex_coeffs(B, H);
  } else {
    // basis blocks: [P Qp Qh log]; log has no imaginary partner (zeros column).
    MatrixXcd Qh(M, N_HOLE_POLY);
    for (int j = 1; j <= N_HOLE_POLY; ++j) Qh.col(j - 1) = (Z.array() - zc).pow(-j);
    VectorXd logc(M);
    for (int m = 0; m < M; ++m) logc[m] = std::log(std::abs(Z[m] - zc));
    const int nr = P.cols() + Np + N_HOLE_POLY + 1;
    MatrixXd A(M, 2 * nr);
    A.block(0, 0, M, P.cols()) = P.real();
    A.block(0, P.cols(), M, Np) = Qp.real();
    A.block(0, P.cols() + Np, M, N_HOLE_POLY) = Qh.real();
    A.col(P.cols() + Np + N_HOLE_POLY) = logc;
    A.block(0, nr, M, P.cols()) = -P.imag();
    A.block(0, nr + P.cols(), M, Np) = -Qp.imag();
    A.block(0, nr + P.cols() + Np, M, N_HOLE_POLY) = -Qh.imag();
    A.col(2 * nr - 1).setZero();
    VectorXd c = A.completeOrthogonalDecomposition().solve(H);
    cr = c.head(nr); ci = c.tail(nr);
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  const double t_total = std::chrono::duration<double>(t1 - t0).count();

  // --- evaluate u on the grid ---
  auto u = [&](const cd& z) -> double {
    VectorXcd zz(1); zz[0] = z;
    MatrixXcd Pz = arnoldi_eval(zz, Hes, cd(0, 0));
    MatrixXcd Qz = pole_basis(zz, pol, d);
    if (!doubly) {
      cd f = 0;
      for (int k = 0; k < Pz.cols(); ++k) f += Pz(0, k) * cc[k];
      for (int p = 0; p < Np; ++p) f += Qz(0, p) * cc[Pz.cols() + p];
      return f.real();
    }
    double rp = 0, ip = 0; int col = 0;
    for (int k = 0; k < Pz.cols(); ++k) { rp += Pz(0, k).real() * cr[col]; ip += -Pz(0, k).imag() * ci[col]; ++col; }
    for (int p = 0; p < Np; ++p) { rp += Qz(0, p).real() * cr[col]; ip += -Qz(0, p).imag() * ci[col]; ++col; }
    for (int j = 1; j <= N_HOLE_POLY; ++j) { cd qh = std::pow(z - zc, -j); rp += qh.real() * cr[col]; ip += -qh.imag() * ci[col]; ++col; }
    rp += std::log(std::abs(z - zc)) * cr[col];  // log has no imag partner
    return rp + ip;
  };

  const Grid grid = make_grid(geom, delta);
  write_solution(out_file, grid, outer, hole, u);
  write_timing(time_file, t_total, "AAA-LS_Cxx", {{"npoles", (double)Np}});

  std::cout << "geom=" << geom << " bc=" << bc << " tol=" << tol
            << " npoles=" << Np << " | total=" << t_total << "s\n";
  return 0;
}
