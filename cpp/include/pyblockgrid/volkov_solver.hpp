#pragma once

#include "types.hpp"
#include "polygon.hpp"
#include "block.hpp"
#include "block_covering.hpp"
#include "solution_state.hpp"
#include "carrier_functions.hpp"
#include "poisson_kernels.hpp"
#include "boundary_estimator.hpp"
#include "interior_solver.hpp"
#include <vector>
#include <tuple>

namespace pyblockgrid {

class VolkovSolver {
public:
    // Public state accessible by bindings/visualization
    Polygon poly;
    std::vector<Block> blocks;
    int N = 0, L = 0, M = 0;
    SolutionState state;

    // Parameters
    std::vector<std::vector<double>> boundary_conditions;
    std::vector<bool> is_dirichlet;
    int n_param;
    double delta;
    int max_iter;
    double radial_heuristic;
    double overlap_heuristic;
    double tol;
    bool verify_solution;

    // Grid arrays (created in constructor, like Python)
    int nx, ny;
    double x_min, y_min, x_max, y_max;
    MatXd grid_X, grid_Y;
    MaskedGrid solution_grid;
    MatXd inside_block_ids_grid;
    BoolMat inside_valid_grid;
    MaskedGrid2D cartesian_grid_data;

    // Covering grid (>= solution grid resolution): used only to detect gaps and
    // place third-kind blocks, so the domain is covered completely while the
    // solution is still evaluated on the coarser solution grid.
    int covering_refinement;
    int cov_nx, cov_ny;
    double cov_delta;
    MatXd cov_X, cov_Y;
    BoolMat cov_valid;

    VolkovSolver(Polygon poly,
                 std::vector<std::vector<double>> boundary_conditions,
                 std::vector<bool> is_dirichlet,
                 int n = 10, double delta = 0.01,
                 int max_iter = 100,
                 double radial_heuristic = 0.8,
                 double overlap_heuristic = 0.1,
                 double tolerance = 1e-10,
                 bool verify_solution = false,
                 int covering_refinement = 1);

    // Main solve
    void solve(bool verbose = false);

    // Step-by-step methods (callable separately for testing)
    std::tuple<int, int, int> find_block_covering();
    void initialize_solution();

private:
    BlockCoveringStrategy block_covering_;
    CarrierFunctions carrier_calc_;
    PoissonKernels poisson_calc_;
    BoundaryEstimator boundary_estimator_;
    InteriorSolver interior_solver_;

    // Find uncovered points and assign block ids
    void find_uncovered_points();

    // Create third-kind blocks
    void create_third_kind_blocks();

    // Solution initialization helpers
    void find_tau_block_ids();
    void find_tau_block_polar_coordinates();
    void find_poisson_kernel_values();
    void find_carrier_function_values();
};

} // namespace pyblockgrid
