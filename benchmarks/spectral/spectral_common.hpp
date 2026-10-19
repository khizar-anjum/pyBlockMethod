// Shared helpers for the C++ spectral Laplace benchmark solvers (Lightning and
// AAA-LS). These reimplement, in C++/Eigen, the two MATLAB baselines so the
// timing comparison against the Block Method is like-for-like C++ (reviewer
// comment R1.4), not C++-vs-MATLAB.
//
// The CLI flags, evaluation grid and output-file formats match the deal.II
// solver (benchmarks/dealii/hp_laplace.cc) so the Python drivers parse all
// three the same way: a `.dat` of "px py u" (y-outer, x-inner) and a timing
// file whose `total` line is the full solve time (R3.5).

#ifndef SPECTRAL_COMMON_HPP
#define SPECTRAL_COMMON_HPP

#include <complex>
#include <cmath>
#include <fstream>
#include <functional>
#include <string>
#include <vector>
#include <Eigen/Dense>

namespace spectral {

using cd = std::complex<double>;
using Eigen::MatrixXcd;
using Eigen::VectorXcd;

// --- Vandermonde+Arnoldi orthogonalisation (laplace.m 147-161 / VAorthog.m) --
// Build an orthogonal polynomial basis Q (M x deg+1) on points Z centred at wc
// (wc=0 reproduces VAorthog's uncentred basis); fill the Hessenberg matrix H.
inline MatrixXcd arnoldi_fit(const VectorXcd& Z, int deg, cd wc, MatrixXcd& H) {
  const int M = Z.size();
  MatrixXcd Q(M, deg + 1);
  Q.col(0).setOnes();
  H = MatrixXcd::Zero(deg + 1, deg);
  for (int k = 0; k < deg; ++k) {
    VectorXcd v = (Z.array() - wc) * Q.col(k).array();
    for (int j = 0; j <= k; ++j) {
      H(j, k) = Q.col(j).dot(v) / double(M);  // dot() conjugates the first arg
      v -= H(j, k) * Q.col(j);
    }
    H(k + 1, k) = v.norm() / std::sqrt(double(M));
    Q.col(k + 1) = v / H(k + 1, k);
  }
  return Q;
}

// Rebuild the polynomial basis at new points from a stored H (fzeval / VAeval).
inline MatrixXcd arnoldi_eval(const VectorXcd& Z, const MatrixXcd& H, cd wc) {
  const int M = Z.size(), deg = H.cols();
  MatrixXcd Q(M, deg + 1);
  Q.col(0).setOnes();
  for (int k = 0; k < deg; ++k) {
    VectorXcd v = (Z.array() - wc) * Q.col(k).array();
    for (int j = 0; j <= k; ++j) v -= H(j, k) * Q.col(j);
    Q.col(k + 1) = v / H(k + 1, k);
  }
  return Q;
}

// Rational pole basis columns d_p / (Z_m - pol_p).
inline MatrixXcd pole_basis(const VectorXcd& Z, const std::vector<cd>& pol,
                            const std::vector<double>& d) {
  MatrixXcd P(Z.size(), pol.size());
  for (int p = 0; p < (int)pol.size(); ++p)
    P.col(p) = d[p] * (Z.array() - pol[p]).inverse();
  return P;
}

// --- tiny CLI helpers (same convention as hp_laplace.cc) -------------------
inline std::string arg_str(int argc, char** argv, const std::string& key,
                           const std::string& def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (key == argv[i]) return argv[i + 1];
  return def;
}
inline double arg_dbl(int argc, char** argv, const std::string& key, double def) {
  const std::string v = arg_str(argc, argv, key, "");
  return v.empty() ? def : std::stod(v);
}

// --- geometry --------------------------------------------------------------
// Outer polygon vertices (CCW) as complex numbers, matching the benchmark
// polygons used everywhere else in the repo.
inline std::vector<cd> outer_vertices(const std::string& geom) {
  if (geom == "unit_square" || geom == "square_with_hole")
    return {cd(0, 0), cd(1, 0), cd(1, 1), cd(0, 1)};
  if (geom == "l_shape")
    return {cd(0, 0), cd(2, 0), cd(2, 1), cd(1, 1), cd(1, 2), cd(0, 2)};
  throw std::runtime_error("unknown geometry: " + geom);
}

// Hole vertices (CW, i.e. reversed) for the multiply-connected geometry, empty
// otherwise. [0.3,0.7]^2 hole to match the benchmark.
inline std::vector<cd> hole_vertices(const std::string& geom) {
  if (geom == "square_with_hole")
    return {cd(0.3, 0.3), cd(0.3, 0.7), cd(0.7, 0.7), cd(0.7, 0.3)};
  return {};
}

// Dirichlet value on each outer side (side k runs from vertex k to k+1). Only
// hot_bottom is graded in the paper; hot_right/mixed mirror hp_laplace.cc.
inline std::vector<double> bc_per_side(const std::string& geom,
                                       const std::string& bc) {
  std::vector<double> g(outer_vertices(geom).size(), 0.0);
  if (bc == "hot_bottom")      g[0] = 1.0;                     // bottom side
  else if (bc == "hot_right")  g[1] = 1.0;                     // right side
  else if (bc == "mixed")    { g[0] = 1.0; g[1] = 0.5; }
  else throw std::runtime_error("unknown bc: " + bc);
  return g;
}

// --- point-in-polygon (ray casting) ----------------------------------------
inline bool inside_polygon(const cd& z, const std::vector<cd>& v) {
  const double x = z.real(), y = z.imag();
  bool in = false;
  const size_t n = v.size();
  for (size_t i = 0, j = n - 1; i < n; j = i++) {
    const double xi = v[i].real(), yi = v[i].imag();
    const double xj = v[j].real(), yj = v[j].imag();
    if (((yi > y) != (yj > y)) &&
        (x < (xj - xi) * (y - yi) / (yj - yi) + xi))
      in = !in;
  }
  return in;
}

// A grid point counts as interior iff inside the outer polygon and, for the
// multiply-connected case, outside the hole.
inline bool in_domain(const cd& z, const std::vector<cd>& outer,
                      const std::vector<cd>& hole) {
  if (!inside_polygon(z, outer)) return false;
  if (!hole.empty() && inside_polygon(z, hole)) return false;
  return true;
}

// --- evaluation grid (identical to hp_laplace.cc) --------------------------
struct Grid {
  int nx, ny;
  std::vector<cd> pts;  // row-major, y outer / x inner
};

inline Grid make_grid(const std::string& geom, double delta) {
  const double xmin = 0.0, ymin = 0.0;
  const double xmax = (geom == "l_shape") ? 2.0 : 1.0;
  const double ymax = xmax;
  Grid g;
  g.nx = static_cast<int>(std::ceil((xmax - xmin) / delta));
  g.ny = static_cast<int>(std::ceil((ymax - ymin) / delta));
  g.pts.reserve(static_cast<size_t>(g.nx) * g.ny);
  for (int j = 0; j < g.ny; ++j)
    for (int i = 0; i < g.nx; ++i) {
      const double px = xmin + 0.5 * delta + i * (xmax - xmin) / (g.nx - 1.0);
      const double py = ymin + 0.5 * delta + j * (ymax - ymin) / (g.ny - 1.0);
      g.pts.emplace_back(px, py);
    }
  return g;
}

// Write "px py u" on the grid; points outside the domain are written as 0.
inline void write_solution(const std::string& out_file, const Grid& grid,
                           const std::vector<cd>& outer,
                           const std::vector<cd>& hole,
                           const std::function<double(const cd&)>& u) {
  std::ofstream f(out_file);
  f.precision(12);
  f << std::scientific;
  for (const cd& z : grid.pts) {
    const double val = in_domain(z, outer, hole) ? u(z) : 0.0;
    f << z.real() << " " << z.imag() << " " << val << "\n";
  }
}

// Write the timing file. `total` is the full solve time (R3.5); extra key/value
// lines (e.g. poles, iterations) are appended verbatim.
inline void write_timing(const std::string& time_file, double total,
                         const std::string& solver,
                         const std::vector<std::pair<std::string, double>>& extra) {
  std::ofstream tf(time_file);
  tf.precision(12);
  tf << std::scientific;
  tf << "total " << total << "\n";
  for (const auto& [k, v] : extra) tf << k << " " << v << "\n";
  tf << "solver " << solver << "\n";
}

}  // namespace spectral

#endif  // SPECTRAL_COMMON_HPP
