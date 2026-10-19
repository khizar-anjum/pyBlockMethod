// hp-FEM Laplace solver (deal.II) for the pyBlockMethod benchmark comparison.
//
// Reviewer 1 asks for a proper hp-FEM baseline (not low-order P2) with the
// linear solver disclosed. This program solves the Dirichlet Laplace problem
// on a benchmark geometry and evaluates the solution on the exact delta-grid
// used by the Volkov solver, so MSE is directly comparable.
//
// Two refinement strategies (--refine-mode), sharing one hp code path:
//   global   : uniform h-refinement at fixed degree (high-order FEM sweep).
//   adaptive : true hp-adaptivity -- Kelly error estimate drives h-refinement,
//              Fourier-coefficient smoothness promotes smooth cells to
//              p-refinement (deal.II step-27 strategy). This breaks the
//              corner-singularity plateau that caps uniform p-refinement.
//
// Two linear solvers (--solver):
//   umfpack : sparse-direct (the original baseline, disclosed per R1.3).
//   cg-amg  : conjugate gradients preconditioned by algebraic multigrid
//             (Trilinos ML), warm-started from the previous cycle's solution.
//             Requested in the second review round as an O(N) alternative to
//             the O(N^{3/2}) sparse-direct solve. AMG setup is timed as part
//             of the solve.
//   cg-pmg  : conjugate gradients preconditioned by a p-multigrid V-cycle:
//             the polynomial degree is coarsened level by level on the same
//             mesh down to linear elements (Chebyshev-Jacobi smoothing on each
//             level), and algebraic multigrid handles the linear-element
//             coarse problem. This is the hp-multigrid strategy of deal.II's
//             step-75 with assembled level matrices; it avoids applying AMG
//             directly to high-order matrices, where AMG is weakest.
//   mf-pmg  : the same p-multigrid-preconditioned CG, but matrix-free: the hp
//             operator is never assembled. It is applied cell by cell with
//             sum factorization (deal.II MatrixFree), and its diagonal (for
//             the smoother) and the Dirichlet right-hand side are computed the
//             same way. Only the linear-element coarse matrix is assembled.
// Timing is
// reported cumulatively across all cycles and decomposed into mesh-generation/
// setup/assembly/solve/refinement; the total is the FULL end-to-end pipeline
// (mesh generation + adaptation + assembly + solve), per R3.5. Total DOF count
// is printed (R1.3, R3.5).
//
// Usage:
//   ./hp_laplace --geometry unit_square --bc hot_bottom \
//                --refine-mode adaptive --min-degree 2 --max-degree 5 \
//                --init-refine 2 --cycles 6 --delta 0.05 \
//                --out sol.dat --timing timing.dat
//   ./hp_laplace --bc hot_bottom --refine-mode global \
//                --min-degree 3 --init-refine 4 --cycles 1

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/mpi.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_series.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/refinement.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/tools.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/smoothness_estimator.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/solution_transfer.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>

using namespace dealii;

// --- tiny CLI helper -------------------------------------------------------
static std::string arg_str(int argc, char** argv, const std::string& key,
                           const std::string& def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (key == argv[i]) return argv[i + 1];
  return def;
}
static int arg_int(int argc, char** argv, const std::string& key, int def) {
  const std::string v = arg_str(argc, argv, key, "");
  return v.empty() ? def : std::stoi(v);
}
static double arg_dbl(int argc, char** argv, const std::string& key, double def) {
  const std::string v = arg_str(argc, argv, key, "");
  return v.empty() ? def : std::stod(v);
}

// Dirichlet values per boundary id, per geometry.
//
// unit_square: hyper_cube(colorize=true) ids 0=left(x=0), 1=right(x=1),
//   2=bottom(y=0), 3=top(y=1).
// square_with_hole [0,1]^2 minus [0.3,0.7]^2: 1=bottom(y=0), 2=right(x=1),
//   3=top(y=1), 4=left(x=0), 5=hole (all four hole edges, u=0).
// l_shape [0,2]^2 minus [1,2]^2, edges numbered to match the benchmark polygon
//   [[0,0],[2,0],[2,1],[1,1],[1,2],[0,2]]: 1=bottom(y=0), 2=right(x=2),
//   3=notch-horizontal(y=1), 4=notch-vertical(x=1), 5=top(y=2), 6=left(x=0).
static std::map<types::boundary_id, double> bc_values(const std::string& geom,
                                                      const std::string& bc) {
  if (geom == "unit_square") {
    if (bc == "hot_bottom") return {{2, 1.0}, {0, 0.0}, {1, 0.0}, {3, 0.0}};
    if (bc == "hot_right")  return {{1, 1.0}, {0, 0.0}, {2, 0.0}, {3, 0.0}};
    if (bc == "mixed")      return {{2, 1.0}, {1, 0.5}, {0, 0.0}, {3, 0.0}};
  } else if (geom == "square_with_hole") {
    std::map<types::boundary_id, double> m{{1, 0.0}, {2, 0.0}, {3, 0.0},
                                           {4, 0.0}, {5, 0.0}};
    if (bc == "hot_bottom") { m[1] = 1.0; return m; }
    if (bc == "hot_right")  { m[2] = 1.0; return m; }
    if (bc == "mixed")      { m[1] = 1.0; m[2] = 0.5; return m; }
  } else if (geom == "l_shape") {
    std::map<types::boundary_id, double> m{{1, 0.0}, {2, 0.0}, {3, 0.0},
                                           {4, 0.0}, {5, 0.0}, {6, 0.0}};
    if (bc == "hot_bottom") { m[1] = 1.0; return m; }
    if (bc == "hot_left")   { m[6] = 1.0; return m; }
    if (bc == "mixed")      { m[1] = 1.0; m[4] = 0.5; return m; }
  }
  throw std::runtime_error("unknown geom/bc: " + geom + "/" + bc);
}

// Build the coarse mesh and assign boundary ids for a geometry.
static void build_mesh(const std::string& geom, Triangulation<2>& tria) {
  if (geom == "unit_square") {
    GridGenerator::hyper_cube(tria, 0.0, 1.0, /*colorize=*/true);
    return;
  }
  if (geom == "square_with_hole") {
    // 3x3 grid of cells on the lines {0,0.3,0.7,1}, with the centre cell
    // [0.3,0.7]^2 removed -- an 8-cell square annulus matching the benchmark.
    const double c[4] = {0.0, 0.3, 0.7, 1.0};
    std::vector<Point<2>> vertices;
    for (int j = 0; j < 4; ++j)
      for (int i = 0; i < 4; ++i) vertices.emplace_back(c[i], c[j]);
    std::vector<CellData<2>> cells;
    for (int cj = 0; cj < 3; ++cj)
      for (int ci = 0; ci < 3; ++ci) {
        if (ci == 1 && cj == 1) continue;  // the hole
        CellData<2> cell;
        cell.vertices = {static_cast<unsigned int>(cj * 4 + ci),
                         static_cast<unsigned int>(cj * 4 + ci + 1),
                         static_cast<unsigned int>((cj + 1) * 4 + ci),
                         static_cast<unsigned int>((cj + 1) * 4 + ci + 1)};
        cells.push_back(cell);
      }
    tria.create_triangulation(vertices, cells, SubCellData());
    const double eps = 1e-9;
    for (const auto& cell : tria.active_cell_iterators())
      for (const auto& face : cell->face_iterators())
        if (face->at_boundary()) {
          const Point<2> ctr = face->center();
          if (std::abs(ctr[1] - 0) < eps)      face->set_boundary_id(1);
          else if (std::abs(ctr[0] - 1) < eps) face->set_boundary_id(2);
          else if (std::abs(ctr[1] - 1) < eps) face->set_boundary_id(3);
          else if (std::abs(ctr[0] - 0) < eps) face->set_boundary_id(4);
          else                                 face->set_boundary_id(5);  // hole
        }
    return;
  }
  if (geom == "l_shape") {
    // 3 unit cells: [0,1]x[0,1], [1,2]x[0,1], [0,1]x[1,2]. deal.II 2D cell
    // vertex order is lexicographic: (0,0),(1,0),(0,1),(1,1).
    const std::vector<Point<2>> vertices = {
        {0, 0}, {1, 0}, {2, 0}, {0, 1}, {1, 1}, {2, 1}, {0, 2}, {1, 2}};
    std::vector<CellData<2>> cells(3);
    cells[0].vertices = {0, 1, 3, 4};
    cells[1].vertices = {1, 2, 4, 5};
    cells[2].vertices = {3, 4, 6, 7};
    tria.create_triangulation(vertices, cells, SubCellData());
    const double eps = 1e-9;
    for (const auto& cell : tria.active_cell_iterators())
      for (const auto& face : cell->face_iterators())
        if (face->at_boundary()) {
          const Point<2> c = face->center();
          if (std::abs(c[1] - 0) < eps)      face->set_boundary_id(1);
          else if (std::abs(c[0] - 2) < eps) face->set_boundary_id(2);
          else if (std::abs(c[1] - 1) < eps) face->set_boundary_id(3);
          else if (std::abs(c[0] - 1) < eps) face->set_boundary_id(4);
          else if (std::abs(c[1] - 2) < eps) face->set_boundary_id(5);
          else if (std::abs(c[0] - 0) < eps) face->set_boundary_id(6);
        }
    return;
  }
  throw std::runtime_error("unknown geometry: " + geom);
}

// --- fine-level operators ---------------------------------------------------
// The p-multigrid only needs the action of the hp operator, so the assembled
// matrix and the matrix-free operator sit behind one interface.
struct FineOperator : public Subscriptor {
  virtual void vmult(Vector<double>& dst, const Vector<double>& src) const = 0;
  virtual unsigned int m() const = 0;
  // Required by PreconditionChebyshev's template; never called, because the
  // inverse diagonal is always supplied.
  double el(const unsigned int, const unsigned int) const {
    throw std::runtime_error("FineOperator::el is not available");
  }
};

struct AssembledOperator : public FineOperator {
  explicit AssembledOperator(const SparseMatrix<double>& A) : A(A) {}
  void vmult(Vector<double>& dst, const Vector<double>& src) const override {
    A.vmult(dst, src);
  }
  unsigned int m() const override { return A.m(); }
  const SparseMatrix<double>& A;
};

// Matrix-free hp Laplace operator. Constrained DoFs (hanging nodes, Dirichlet)
// are handled by MatrixFree; their rows act as the identity so the operator is
// symmetric positive definite on the whole vector space.
class MatrixFreeLaplace : public FineOperator {
 public:
  using VectorType = LinearAlgebra::distributed::Vector<double>;
  using CellIntegrator = FEEvaluation<2, -1, 0, 1, double>;

  // constraints: hanging nodes + homogeneous Dirichlet values.
  void reinit(const DoFHandler<2>& dof, const AffineConstraints<double>& constraints,
              const hp::QCollection<2>& quads, const bool threaded) {
    MatrixFree<2, double>::AdditionalData data;
    data.mapping_update_flags = update_gradients;
    if (!threaded)
      data.tasks_parallel_scheme = MatrixFree<2, double>::AdditionalData::none;
    hp::MappingCollection<2> mappings;
    mappings.push_back(MappingQ1<2>());
    matrix_free.reinit(mappings, dof, constraints, quads, data);
    constrained.clear();
    for (unsigned int i = 0; i < dof.n_dofs(); ++i)
      if (constraints.is_constrained(i)) constrained.push_back(i);
    matrix_free.initialize_dof_vector(src_);
    matrix_free.initialize_dof_vector(dst_);
    n = dof.n_dofs();
  }

  unsigned int m() const override { return n; }

  void vmult(Vector<double>& dst, const Vector<double>& src) const override {
    for (unsigned int i = 0; i < n; ++i) src_.local_element(i) = src(i);
    matrix_free.cell_loop(&MatrixFreeLaplace::apply_range, this, dst_, src_, true);
    for (unsigned int i = 0; i < n; ++i) dst(i) = dst_.local_element(i);
    for (const unsigned int i : constrained) dst(i) = src(i);
  }

  // Right-hand side for inhomogeneous Dirichlet data: -A u_D, where u_D carries
  // the boundary values (and is read without applying the constraints).
  void dirichlet_rhs(Vector<double>& rhs, const Vector<double>& u_dirichlet) const {
    for (unsigned int i = 0; i < n; ++i) src_.local_element(i) = u_dirichlet(i);
    matrix_free.cell_loop(&MatrixFreeLaplace::apply_range_plain, this, dst_, src_, true);
    for (unsigned int i = 0; i < n; ++i) rhs(i) = -dst_.local_element(i);
    for (const unsigned int i : constrained) rhs(i) = 0.0;
  }

  void inverse_diagonal(Vector<double>& inv_diag) const {
    VectorType diagonal;
    matrix_free.initialize_dof_vector(diagonal);
    MatrixFreeTools::compute_diagonal(matrix_free, diagonal,
                                      &MatrixFreeLaplace::apply_local, this);
    inv_diag.reinit(n);
    for (unsigned int i = 0; i < n; ++i) {
      const double d = diagonal.local_element(i);
      inv_diag(i) = std::abs(d) > 1e-14 ? 1.0 / d : 1.0;
    }
    for (const unsigned int i : constrained) inv_diag(i) = 1.0;
  }

 private:
  void apply_local(CellIntegrator& phi) const {
    phi.evaluate(EvaluationFlags::gradients);
    for (unsigned int q = 0; q < phi.n_q_points; ++q)
      phi.submit_gradient(phi.get_gradient(q), q);
    phi.integrate(EvaluationFlags::gradients);
  }
  void apply_range(const MatrixFree<2, double>& data, VectorType& dst,
                   const VectorType& src,
                   const std::pair<unsigned int, unsigned int>& range) const {
    CellIntegrator phi(data, range);
    for (unsigned int cell = range.first; cell < range.second; ++cell) {
      phi.reinit(cell);
      phi.read_dof_values(src);
      apply_local(phi);
      phi.distribute_local_to_global(dst);
    }
  }
  void apply_range_plain(const MatrixFree<2, double>& data, VectorType& dst,
                         const VectorType& src,
                         const std::pair<unsigned int, unsigned int>& range) const {
    CellIntegrator phi(data, range);
    for (unsigned int cell = range.first; cell < range.second; ++cell) {
      phi.reinit(cell);
      phi.read_dof_values_plain(src);
      apply_local(phi);
      phi.distribute_local_to_global(dst);
    }
  }

  MatrixFree<2, double> matrix_free;
  std::vector<unsigned int> constrained;
  mutable VectorType src_, dst_;
  unsigned int n = 0;
};

// --- p-multigrid preconditioner (cg-pmg) -----------------------------------
// Levels share the triangulation and differ in polynomial degree: level L is
// the hp system itself; each coarser level halves every cell's degree until all
// cells are linear. Level matrices are assembled; the transfer is the exact
// embedding of the coarser hp space (its transpose is the restriction).
class PMultigrid {
 public:
  // Cumulative setup-time split: level dofs+constraints, level assembly,
  // transfers, smoothers, coarse AMG.
  static inline double t_setup_parts[5] = {0, 0, 0, 0, 0};
  using Smoother = PreconditionChebyshev<SparseMatrix<double>, Vector<double>,
                                         DiagonalMatrix<Vector<double>>>;
  using TopSmoother = PreconditionChebyshev<FineOperator, Vector<double>,
                                            DiagonalMatrix<Vector<double>>>;

  PMultigrid(const Triangulation<2>& tria, const DoFHandler<2>& fine_dof,
             const FineOperator& fine_op,
             const Vector<double>& fine_inverse_diagonal,
             const std::vector<types::boundary_id>& dirichlet_ids,
             const unsigned int max_degree, const unsigned int cheb_degree,
             const double smoothing_range, const bool direct_to_linear,
             const TrilinosWrappers::PreconditionAMG::AdditionalData& amg_data)
      : fine_op(fine_op) {
    Timer part;
    for (unsigned int d = 1; d <= max_degree; ++d) {
      fes.push_back(FE_Q<2>(d));
      quads.push_back(QGauss<2>(d + 1));
    }

    // Degree of every cell on every level, finest first.
    std::vector<std::vector<unsigned int>> degrees(1);
    for (const auto& cell : fine_dof.active_cell_iterators())
      degrees[0].push_back(cell->get_fe().degree);
    do {  // always at least one coarse (linear-element) level
      std::vector<unsigned int> coarser(degrees.back());
      for (auto& d : coarser) d = direct_to_linear ? 1u : std::max(1u, d / 2);
      degrees.push_back(coarser);
    } while (*std::max_element(degrees.back().begin(), degrees.back().end()) > 1);
    const unsigned int n_coarse = degrees.size() - 1;
    levels.resize(n_coarse);

    // Coarse levels: level 0 is all-linear, level n_coarse-1 is just below fine.
    for (unsigned int l = 0; l < n_coarse; ++l) {
      Level& lev = levels[l];
      const auto& deg = degrees[n_coarse - l];
      lev.dof = std::make_unique<DoFHandler<2>>(tria);
      unsigned int c = 0;
      for (const auto& cell : lev.dof->active_cell_iterators())
        cell->set_active_fe_index(deg[c++] - 1);
      part.restart();
      lev.dof->distribute_dofs(fes);
      make_constraints(*lev.dof, dirichlet_ids, lev.constraints);
      t_setup_parts[0] += part.wall_time();
      part.restart();

      DynamicSparsityPattern dsp(lev.dof->n_dofs());
      DoFTools::make_sparsity_pattern(*lev.dof, dsp, lev.constraints, false);
      lev.sparsity.copy_from(dsp);
      lev.matrix.reinit(lev.sparsity);
      hp::FEValues<2> hp_fe_values(fes, quads, update_gradients | update_JxW_values);
      for (const auto& cell : lev.dof->active_cell_iterators()) {
        hp_fe_values.reinit(cell);
        const FEValues<2>& fv = hp_fe_values.get_present_fe_values();
        const unsigned int n = cell->get_fe().n_dofs_per_cell();
        FullMatrix<double> cell_matrix(n, n);
        std::vector<types::global_dof_index> local_dofs(n);
        for (unsigned int q = 0; q < fv.n_quadrature_points; ++q)
          for (unsigned int i = 0; i < n; ++i)
            for (unsigned int j = 0; j < n; ++j)
              cell_matrix(i, j) +=
                  fv.shape_grad(i, q) * fv.shape_grad(j, q) * fv.JxW(q);
        cell->get_dof_indices(local_dofs);
        lev.constraints.distribute_local_to_global(cell_matrix, local_dofs,
                                                   lev.matrix);
      }
      t_setup_parts[1] += part.wall_time();
    }
    part.restart();

    // Transfers: levels[l].prolongation maps level l to the next finer level.
    make_constraints(fine_dof, dirichlet_ids, fine_constraints);
    for (unsigned int l = 0; l < n_coarse; ++l) {
      const bool to_fine = (l + 1 == n_coarse);
      build_prolongation(*levels[l].dof,
                         to_fine ? fine_dof : *levels[l + 1].dof,
                         to_fine ? fine_constraints : levels[l + 1].constraints,
                         levels[l]);
    }

    t_setup_parts[2] += part.wall_time();
    part.restart();
    // Smoothers on every level above the coarsest (and on the fine system).
    smoothers.resize(n_coarse);
    for (unsigned int l = 1; l < n_coarse; ++l) {
      const SparseMatrix<double>& A = levels[l].matrix;
      Smoother::AdditionalData data;
      data.degree = cheb_degree;
      data.smoothing_range = smoothing_range;
      data.eig_cg_n_iterations = 10;
      data.preconditioner = std::make_shared<DiagonalMatrix<Vector<double>>>();
      Vector<double>& inv_diag = data.preconditioner->get_vector();
      inv_diag.reinit(A.m());
      for (unsigned int i = 0; i < A.m(); ++i) inv_diag(i) = 1.0 / A.diag_element(i);
      smoothers[l].initialize(A, data);
    }
    {
      TopSmoother::AdditionalData data;
      data.degree = cheb_degree;
      data.smoothing_range = smoothing_range;
      data.eig_cg_n_iterations = 10;
      data.preconditioner = std::make_shared<DiagonalMatrix<Vector<double>>>();
      data.preconditioner->get_vector() = fine_inverse_diagonal;
      top_smoother.initialize(fine_op, data);
    }

    t_setup_parts[3] += part.wall_time();
    part.restart();
    // Coarsest problem (linear elements, or the fine system if it is linear).
    amg.initialize(levels[0].matrix, amg_data);
    t_setup_parts[4] += part.wall_time();

    x.resize(n_coarse + 1); b.resize(n_coarse + 1); r.resize(n_coarse + 1);
    for (unsigned int l = 0; l <= n_coarse; ++l) {
      x[l].reinit(size(l)); b[l].reinit(size(l)); r[l].reinit(size(l));
    }
  }

  unsigned int n_levels() const { return levels.size() + 1; }

  // One V-cycle applied to src.
  void vmult(Vector<double>& dst, const Vector<double>& src) const {
    const unsigned int top = levels.size();
    b[top] = src;
    v_cycle(top);
    dst = x[top];
  }

 private:
  struct Level {
    std::unique_ptr<DoFHandler<2>> dof;
    AffineConstraints<double> constraints;
    SparsityPattern sparsity, prolongation_sparsity;
    SparseMatrix<double> matrix, prolongation;
  };

  unsigned int size(const unsigned int l) const {
    return l == levels.size() ? fine_op.m() : levels[l].matrix.m();
  }

  static void make_constraints(const DoFHandler<2>& dof,
                               const std::vector<types::boundary_id>& ids,
                               AffineConstraints<double>& constraints) {
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof, constraints);
    for (const auto id : ids)
      VectorTools::interpolate_boundary_values(
          dof, id, Functions::ZeroFunction<2>(), constraints);
    constraints.close();
  }

  // Embedding of the coarse hp space into the finer one on the same mesh. Each
  // unconstrained fine DoF takes its row from one adjacent cell: the coarse
  // function is conforming once its constraints are distributed, so any cell's
  // local interpolation gives the same value.
  void build_prolongation(const DoFHandler<2>& coarse, const DoFHandler<2>& fine,
                          const AffineConstraints<double>& fine_constr,
                          Level& lev) {
    const unsigned int invalid = numbers::invalid_unsigned_int;
    std::vector<unsigned int> owner(fine.n_dofs(), invalid);
    std::map<std::pair<unsigned int, unsigned int>, FullMatrix<double>> local;
    DynamicSparsityPattern dsp(fine.n_dofs(), coarse.n_dofs());
    std::vector<types::global_dof_index> fd, cd;
    for (unsigned int pass = 0; pass < 2; ++pass) {
      if (pass == 1) {
        lev.prolongation_sparsity.copy_from(dsp);
        lev.prolongation.reinit(lev.prolongation_sparsity);
      }
      auto cf = fine.begin_active();
      auto cc = coarse.begin_active();
      for (unsigned int c = 0; cf != fine.end(); ++cf, ++cc, ++c) {
        const auto& fe_f = cf->get_fe();
        const auto& fe_c = cc->get_fe();
        const auto key = std::make_pair(fe_c.degree, fe_f.degree);
        if (local.find(key) == local.end()) {
          FullMatrix<double> m(fe_f.n_dofs_per_cell(), fe_c.n_dofs_per_cell());
          FETools::get_interpolation_matrix(fe_c, fe_f, m);
          local[key] = m;
        }
        const FullMatrix<double>& m = local[key];
        fd.resize(fe_f.n_dofs_per_cell()); cd.resize(fe_c.n_dofs_per_cell());
        cf->get_dof_indices(fd); cc->get_dof_indices(cd);
        for (unsigned int i = 0; i < fd.size(); ++i) {
          if (fine_constr.is_constrained(fd[i])) continue;
          if (pass == 0 && owner[fd[i]] == invalid) owner[fd[i]] = c;
          if (owner[fd[i]] != c) continue;
          for (unsigned int j = 0; j < cd.size(); ++j)
            if (std::abs(m(i, j)) > 1e-13) {
              if (pass == 0) dsp.add(fd[i], cd[j]);
              else lev.prolongation.set(fd[i], cd[j], m(i, j));
            }
        }
      }
    }
  }

  void v_cycle(const unsigned int l) const {
    if (l == 0) {
      amg.vmult(x[0], b[0]);
      return;
    }
    const bool top = (l == levels.size());
    const Level& coarse = levels[l - 1];
    if (top) {                                      // pre-smoothing, residual
      top_smoother.vmult(x[l], b[l]);
      fine_op.vmult(r[l], x[l]);
    } else {
      smoothers[l].vmult(x[l], b[l]);
      levels[l].matrix.vmult(r[l], x[l]);
    }
    r[l].sadd(-1.0, 1.0, b[l]);                     // residual
    coarse.prolongation.Tvmult(b[l - 1], r[l]);     // restriction
    coarse.constraints.condense(b[l - 1]);
    coarse.constraints.set_zero(b[l - 1]);
    v_cycle(l - 1);
    coarse.constraints.distribute(x[l - 1]);        // conforming correction
    coarse.prolongation.vmult_add(x[l], x[l - 1]);  // prolongation
    if (top) top_smoother.step(x[l], b[l]);         // post-smoothing
    else     smoothers[l].step(x[l], b[l]);
  }

  const FineOperator& fine_op;
  TopSmoother top_smoother;
  hp::FECollection<2> fes;
  hp::QCollection<2> quads;
  AffineConstraints<double> fine_constraints;
  std::vector<Level> levels;
  std::vector<Smoother> smoothers;
  TrilinosWrappers::PreconditionAMG amg;
  mutable std::vector<Vector<double>> x, b, r;
};

int main(int argc, char** argv) {
  // Trilinos (used by the cg-amg solver) needs MPI initialized; the run is
  // serial and the thread count is left at deal.II's default.
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string geometry = arg_str(argc, argv, "--geometry", "unit_square");
  const std::string bc       = arg_str(argc, argv, "--bc", "hot_bottom");
  const std::string mode     = arg_str(argc, argv, "--refine-mode", "adaptive");
  const unsigned int min_deg = arg_int(argc, argv, "--min-degree", 2);
  const unsigned int max_deg = arg_int(argc, argv, "--max-degree", 5);
  const unsigned int init_ref = arg_int(argc, argv, "--init-refine", 2);
  const unsigned int cycles  = arg_int(argc, argv, "--cycles", 6);
  const double delta         = arg_dbl(argc, argv, "--delta", 0.05);
  const std::string out_file = arg_str(argc, argv, "--out", "sol.dat");
  const std::string time_file = arg_str(argc, argv, "--timing", "timing.dat");
  const std::string solver   = arg_str(argc, argv, "--solver", "umfpack");
  const double cg_tol        = arg_dbl(argc, argv, "--cg-tol", 1e-10);
  // AMG tuning knobs (defaults are the tuned benchmark configuration).
  const double amg_threshold = arg_dbl(argc, argv, "--amg-threshold", 0.02);
  const unsigned int amg_sweeps = arg_int(argc, argv, "--amg-sweeps", 2);
  const std::string amg_smoother = arg_str(argc, argv, "--amg-smoother", "Chebyshev");
  const bool amg_high_order  = arg_int(argc, argv, "--amg-higher-order", 1) != 0;
  const bool amg_w_cycle     = arg_int(argc, argv, "--amg-w-cycle", 0) != 0;
  const bool warm_start      = arg_int(argc, argv, "--warm-start", 1) != 0;
  // p-multigrid knobs (cg-pmg): Chebyshev smoother degree and range.
  const unsigned int pmg_cheb_degree = arg_int(argc, argv, "--pmg-smoother-degree", 5);
  const double pmg_range     = arg_dbl(argc, argv, "--pmg-smoothing-range", 15.0);
  // bisect: halve the degree per level; direct: one coarse level of linear elements.
  const bool pmg_direct      = arg_str(argc, argv, "--pmg-coarsening", "direct") == "direct";
  if (solver != "umfpack" && solver != "cg-amg" && solver != "cg-pmg" &&
      solver != "mf-pmg")
    throw std::runtime_error("unknown solver: " + solver);
  const bool iterative = (solver != "umfpack");
  const bool matrix_free = (solver == "mf-pmg");

  // In global mode the FE collection is a single element of degree min_deg.
  const unsigned int top_deg = (mode == "global") ? min_deg : max_deg;

  // --- hp collections: degrees min_deg .. top_deg ---
  hp::FECollection<2> fes;
  hp::QCollection<2> quads;
  hp::QCollection<1> face_quads;
  for (unsigned int d = min_deg; d <= top_deg; ++d) {
    fes.push_back(FE_Q<2>(d));
    quads.push_back(QGauss<2>(d + 1));
    face_quads.push_back(QGauss<1>(d + 1));
  }
  FESeries::Fourier<2> fourier =
      SmoothnessEstimator::Fourier::default_fe_series(fes);

  // Time the FULL FEM pipeline (R3.5): mesh generation + adaptation are counted
  // alongside setup/assembly/solve, so the total is the end-to-end cost a user
  // pays -- not just the linear solve.
  double t_mesh = 0, t_setup = 0, t_asm = 0, t_solve = 0, t_refine = 0;
  Timer timer;

  timer.restart();
  Triangulation<2> tria;
  build_mesh(geometry, tria);
  tria.refine_global(init_ref);
  t_mesh += timer.wall_time();

  DoFHandler<2> dof(tria);
  for (const auto& cell : dof.active_cell_iterators())
    cell->set_active_fe_index(0);  // start at lowest degree

  Vector<double> solution;
  AffineConstraints<double> constraints;

  // cg-amg only: carry the solution across refinement as the CG initial guess.
  std::unique_ptr<SolutionTransfer<2>> transfer;
  Vector<double> previous_solution;
  unsigned int cg_iters_total = 0, cg_iters_last = 0, mg_levels = 1;
  // Diagnostics: split of the solve phase, and the final cycle's solve alone.
  double t_amg_setup = 0, t_cg = 0, t_transfer = 0, t_solve_last = 0;
  double t_mf_reinit = 0, t_mf_rhs = 0, t_mf_diag = 0;
  // Matrix-free cell loops: threaded (deal.II default) or serial.
  const bool mf_threads = arg_int(argc, argv, "--mf-threads", 1) != 0;
  // Intermediate cycles only steer refinement, so they may be solved loosely.
  const double cg_tol_mid = arg_dbl(argc, argv, "--cg-tol-intermediate", 1e-6);

  for (unsigned int cycle = 0; cycle < cycles; ++cycle) {
    // --- setup: distribute dofs, constraints (hanging nodes + BC), sparsity ---
    timer.restart();
    dof.distribute_dofs(fes);

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof, constraints);
    for (const auto& [id, value] : bc_values(geometry, bc))
      VectorTools::interpolate_boundary_values(
          dof, id, Functions::ConstantFunction<2>(value), constraints);
    constraints.close();

    SparsityPattern sparsity;
    SparseMatrix<double> system_matrix;
    if (!matrix_free) {  // the matrix-free solver never builds the hp matrix
      DynamicSparsityPattern dsp(dof.n_dofs());
      DoFTools::make_sparsity_pattern(dof, dsp, constraints, /*keep_constrained=*/false);
      sparsity.copy_from(dsp);
      system_matrix.reinit(sparsity);
    }
    Vector<double> system_rhs(dof.n_dofs());
    solution.reinit(dof.n_dofs());
    t_setup += timer.wall_time();

    if (transfer) {  // warm start; counted as part of the iterative solve
      timer.restart();
      transfer->interpolate(previous_solution, solution);
      transfer.reset();
      t_solve += timer.wall_time();
      t_transfer += timer.wall_time();
    }

    // --- assembly (stiffness matrix, zero RHS) via hp::FEValues; the
    //     matrix-free solver instead sets up its operator, right-hand side and
    //     diagonal, which is timed in the same slot ---
    timer.restart();
    MatrixFreeLaplace mf_operator;
    Vector<double> mf_inverse_diagonal;
    if (matrix_free) {
      AffineConstraints<double> homogeneous;
      DoFTools::make_hanging_node_constraints(dof, homogeneous);
      for (const auto& id_value : bc_values(geometry, bc))
        VectorTools::interpolate_boundary_values(
            dof, id_value.first, Functions::ZeroFunction<2>(), homogeneous);
      homogeneous.close();
      Timer part;
      mf_operator.reinit(dof, homogeneous, quads, mf_threads);
      t_mf_reinit += part.wall_time();
      part.restart();
      Vector<double> u_dirichlet(dof.n_dofs());
      constraints.distribute(u_dirichlet);
      mf_operator.dirichlet_rhs(system_rhs, u_dirichlet);
      t_mf_rhs += part.wall_time();
      part.restart();
      mf_operator.inverse_diagonal(mf_inverse_diagonal);
      t_mf_diag += part.wall_time();
    }
    hp::FEValues<2> hp_fe_values(fes, quads,
                                 update_gradients | update_JxW_values);
    for (const auto& cell : dof.active_cell_iterators()) {
      if (matrix_free) break;
      hp_fe_values.reinit(cell);
      const FEValues<2>& fv = hp_fe_values.get_present_fe_values();
      const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double> cell_rhs(dofs_per_cell);  // PDE RHS is 0 (Laplace)
      std::vector<types::global_dof_index> local_dofs(dofs_per_cell);
      cell_matrix = 0;
      cell_rhs = 0;
      for (unsigned int q = 0; q < fv.n_quadrature_points; ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            cell_matrix(i, j) +=
                fv.shape_grad(i, q) * fv.shape_grad(j, q) * fv.JxW(q);
      cell->get_dof_indices(local_dofs);
      // Pass both matrix and rhs so the inhomogeneous Dirichlet constraints
      // (u=1 on the hot edge) are condensed into system_rhs.
      constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dofs,
                                             system_matrix, system_rhs);
    }
    t_asm += timer.wall_time();

    // --- solve (UMFPACK sparse direct, or CG + algebraic multigrid) ---
    timer.restart();
    if (solver == "umfpack") {
      SparseDirectUMFPACK direct;
      direct.initialize(system_matrix);
      direct.vmult(solution, system_rhs);
    } else {
      TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
      amg_data.elliptic = true;
      amg_data.aggregation_threshold = amg_threshold;
      amg_data.smoother_sweeps = amg_sweeps;
      amg_data.smoother_type = amg_smoother.c_str();
      amg_data.w_cycle = amg_w_cycle;
      const double tol = (cycle + 1 == cycles) ? cg_tol : cg_tol_mid;
      SolverControl control(5000, tol * system_rhs.l2_norm());
      SolverCG<Vector<double>> cg(control);
      if (solver == "cg-amg") {
        amg_data.higher_order_elements = amg_high_order;
        TrilinosWrappers::PreconditionAMG amg;
        amg.initialize(system_matrix, amg_data);
        t_amg_setup += timer.wall_time();
        const double t_before_cg = timer.wall_time();
        constraints.set_zero(solution);
        cg.solve(system_matrix, solution, system_rhs, amg);
        t_cg += timer.wall_time() - t_before_cg;
      } else {
        // AMG only ever sees the linear-element coarse problem here.
        amg_data.higher_order_elements = false;
        std::vector<types::boundary_id> ids;
        for (const auto& id_value : bc_values(geometry, bc)) ids.push_back(id_value.first);
        const AssembledOperator assembled(system_matrix);
        Vector<double> inverse_diagonal;
        if (matrix_free) {
          inverse_diagonal = mf_inverse_diagonal;
        } else {
          inverse_diagonal.reinit(system_matrix.m());
          for (unsigned int i = 0; i < system_matrix.m(); ++i)
            inverse_diagonal(i) = 1.0 / system_matrix.diag_element(i);
        }
        const FineOperator& fine_op =
            matrix_free ? static_cast<const FineOperator&>(mf_operator)
                        : static_cast<const FineOperator&>(assembled);
        const PMultigrid pmg(tria, dof, fine_op, inverse_diagonal, ids, top_deg,
                             pmg_cheb_degree, pmg_range, pmg_direct, amg_data);
        mg_levels = pmg.n_levels();
        t_amg_setup += timer.wall_time();
        const double t_before_cg = timer.wall_time();
        constraints.set_zero(solution);
        cg.solve(fine_op, solution, system_rhs, pmg);
        t_cg += timer.wall_time() - t_before_cg;
      }
      cg_iters_last = control.last_step();
      cg_iters_total += cg_iters_last;
    }
    constraints.distribute(solution);
    t_solve += timer.wall_time();
    t_solve_last = timer.wall_time();

    if (cycle + 1 == cycles) break;  // last cycle: keep this solution

    // --- refine (mesh adaptation: error estimation + h/p refinement) ---
    timer.restart();
    if (mode == "global") {
      tria.refine_global(1);
    } else {
      Vector<float> est_error(tria.n_active_cells());
      KellyErrorEstimator<2>::estimate(
          dof, face_quads, std::map<types::boundary_id, const Function<2>*>(),
          solution, est_error);
      Vector<float> smoothness(tria.n_active_cells());
      SmoothnessEstimator::Fourier::coefficient_decay(fourier, dof, solution,
                                                      smoothness);
      GridRefinement::refine_and_coarsen_fixed_number(tria, est_error, 0.3, 0.0);
      hp::Refinement::p_adaptivity_from_relative_threshold(dof, smoothness,
                                                           0.2, 0.2);
      hp::Refinement::choose_p_over_h(dof);
      if (iterative && warm_start) {
        tria.prepare_coarsening_and_refinement();
        previous_solution = solution;
        transfer = std::make_unique<SolutionTransfer<2>>(dof);
        transfer->prepare_for_coarsening_and_refinement(previous_solution);
      }
      tria.execute_coarsening_and_refinement();
    }
    t_refine += timer.wall_time();
  }

  // --- evaluate on the exact Volkov delta-grid ---
  const double xmin = 0.0, ymin = 0.0;
  const double xmax = (geometry == "l_shape") ? 2.0 : 1.0;
  const double ymax = xmax;
  const int nx = static_cast<int>(std::ceil((xmax - xmin) / delta));
  const int ny = static_cast<int>(std::ceil((ymax - ymin) / delta));
  Functions::FEFieldFunction<2> field(dof, solution);

  std::ofstream f(out_file);
  f.precision(12);
  f << std::scientific;
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      const double px = xmin + 0.5 * delta + i * (xmax - xmin) / (nx - 1.0);
      const double py = ymin + 0.5 * delta + j * (ymax - ymin) / (ny - 1.0);
      double u = 0.0;
      if (px > xmin && px < xmax && py > ymin && py < ymax) {
        try {
          u = field.value(Point<2>(px, py));
        } catch (const std::exception&) {
          u = 0.0;
        }
      }
      f << px << " " << py << " " << u << "\n";
    }

  const double t_total = t_mesh + t_setup + t_asm + t_solve + t_refine;
  std::ofstream tf(time_file);
  tf.precision(12);
  tf << std::scientific;
  tf << "mesh " << t_mesh << "\n"
     << "setup " << t_setup << "\n"
     << "assembly " << t_asm << "\n"
     << "solve " << t_solve << "\n"
     << "refine " << t_refine << "\n"
     << "total " << t_total << "\n"
     << "n_dofs " << dof.n_dofs() << "\n"
     << "n_cells " << tria.n_active_cells() << "\n"
     << "refine_mode " << mode << "\n"
     << "min_degree " << min_deg << "\n"
     << "max_degree " << top_deg << "\n"
     << "cycles " << cycles << "\n"
     << "solver "
     << (solver == "umfpack" ? "UMFPACK_sparse_direct"
         : solver == "cg-amg" ? "CG_AMG_TrilinosML"
         : solver == "cg-pmg" ? "CG_pMG_AMGcoarse" : "CG_pMG_matrix_free")
     << "\n";
  tf << "solve_last_cycle " << t_solve_last << "\n";
  if (iterative)
    tf << "mg_levels " << mg_levels << "\n"
       << "mf_reinit " << t_mf_reinit << "\n"
       << "mf_rhs " << t_mf_rhs << "\n"
       << "mf_diagonal " << t_mf_diag << "\n"
       << "pmg_setup_dofs " << PMultigrid::t_setup_parts[0] << "\n"
       << "pmg_setup_assembly " << PMultigrid::t_setup_parts[1] << "\n"
       << "pmg_setup_transfer " << PMultigrid::t_setup_parts[2] << "\n"
       << "pmg_setup_smoother " << PMultigrid::t_setup_parts[3] << "\n"
       << "pmg_setup_amg " << PMultigrid::t_setup_parts[4] << "\n"
       << "amg_setup " << t_amg_setup << "\n"
       << "cg_time " << t_cg << "\n"
       << "warm_start_transfer " << t_transfer << "\n"
       << "cg_tol " << cg_tol << "\n"
       << "cg_iters_total " << cg_iters_total << "\n"
       << "cg_iters_last " << cg_iters_last << "\n";

  std::cout << "mode=" << mode << " bc=" << bc << " deg=[" << min_deg << ","
            << top_deg << "] cycles=" << cycles
            << " n_dofs=" << dof.n_dofs() << " n_cells=" << tria.n_active_cells()
            << " | total=" << t_total << "s\n";
  return 0;
}
