#include "pyblockgrid/volkov_solver.hpp"
#include "pyblockgrid/coordinate_transforms.hpp"
#include <cmath>
#include <algorithm>
#include <random>
#include <cassert>

namespace pyblockgrid {

VolkovSolver::VolkovSolver(
    Polygon poly,
    std::vector<std::vector<double>> boundary_conditions,
    std::vector<bool> is_dirichlet,
    int n, double delta, int max_iter,
    double radial_heuristic, double overlap_heuristic,
    double tolerance, bool verify_solution, int covering_refinement)
    : poly(std::move(poly)),
      boundary_conditions(std::move(boundary_conditions)),
      is_dirichlet(std::move(is_dirichlet)),
      n_param(n), delta(delta), max_iter(max_iter),
      radial_heuristic(radial_heuristic),
      overlap_heuristic(overlap_heuristic),
      tol(tolerance), verify_solution(verify_solution),
      covering_refinement(covering_refinement),
      block_covering_(radial_heuristic, overlap_heuristic),
      carrier_calc_(tolerance),
      interior_solver_(carrier_calc_, poisson_calc_)
{
    int n_edges = static_cast<int>(this->poly.edges.size());
    assert(static_cast<int>(this->boundary_conditions.size()) == n_edges);
    assert(static_cast<int>(this->is_dirichlet.size()) == n_edges);
    assert(radial_heuristic > 0.707 && radial_heuristic < 1.0);
    assert(overlap_heuristic > 0 && overlap_heuristic < 0.5);

    // Create grid (matching Python __init__ exactly)
    x_min = this->poly.vertices.col(0).minCoeff();
    y_min = this->poly.vertices.col(1).minCoeff();
    x_max = this->poly.vertices.col(0).maxCoeff();
    y_max = this->poly.vertices.col(1).maxCoeff();

    ny = static_cast<int>(std::ceil((y_max - y_min) / delta));
    nx = static_cast<int>(std::ceil((x_max - x_min) / delta));

    // linspace for grid
    VecXd x_vec(nx), y_vec(ny);
    {
        double xs = x_min + 0.5 * delta, xe = x_max + 0.5 * delta;
        for (int i = 0; i < nx; ++i)
            x_vec(i) = nx > 1 ? xs + i * (xe - xs) / (nx - 1) : xs;
    }
    {
        double ys = y_min + 0.5 * delta, ye = y_max + 0.5 * delta;
        for (int i = 0; i < ny; ++i)
            y_vec(i) = ny > 1 ? ys + i * (ye - ys) / (ny - 1) : ys;
    }

    grid_X.resize(ny, nx);
    grid_Y.resize(ny, nx);
    for (int i = 0; i < ny; ++i)
        for (int j = 0; j < nx; ++j) {
            grid_X(i, j) = x_vec(j);
            grid_Y(i, j) = y_vec(i);
        }

    // Create mask using vectorized is_inside
    MatX2d points(ny * nx, 2);
    for (int i = 0; i < ny; ++i)
        for (int j = 0; j < nx; ++j) {
            points(i * nx + j, 0) = grid_X(i, j);
            points(i * nx + j, 1) = grid_Y(i, j);
        }
    BoolVec inside_flat = this->poly.is_inside(points);

    // Set up solution grid
    solution_grid = MaskedGrid(ny, nx);
    inside_block_ids_grid = MatXd::Zero(ny, nx);
    inside_valid_grid = BoolMat::Constant(ny, nx, false);
    cartesian_grid_data = MaskedGrid2D(ny, nx);

    for (int i = 0; i < ny; ++i)
        for (int j = 0; j < nx; ++j) {
            bool v = inside_flat(i * nx + j);
            solution_grid.valid(i, j) = v;
            inside_valid_grid(i, j) = v;
            cartesian_grid_data.valid(i, j) = v;
            cartesian_grid_data.x(i, j) = grid_X(i, j);
            cartesian_grid_data.y(i, j) = grid_Y(i, j);
        }

    // Build the (optionally finer) covering grid used for gap detection.
    assert(covering_refinement >= 1);
    cov_delta = delta / covering_refinement;
    cov_ny = static_cast<int>(std::ceil((y_max - y_min) / cov_delta));
    cov_nx = static_cast<int>(std::ceil((x_max - x_min) / cov_delta));
    cov_X.resize(cov_ny, cov_nx);
    cov_Y.resize(cov_ny, cov_nx);
    {
        double xs = x_min + 0.5 * cov_delta, xe = x_max + 0.5 * cov_delta;
        double ys = y_min + 0.5 * cov_delta, ye = y_max + 0.5 * cov_delta;
        MatX2d cov_points(cov_ny * cov_nx, 2);
        for (int i = 0; i < cov_ny; ++i)
            for (int j = 0; j < cov_nx; ++j) {
                double x = cov_nx > 1 ? xs + j * (xe - xs) / (cov_nx - 1) : xs;
                double y = cov_ny > 1 ? ys + i * (ye - ys) / (cov_ny - 1) : ys;
                cov_X(i, j) = x;
                cov_Y(i, j) = y;
                cov_points(i * cov_nx + j, 0) = x;
                cov_points(i * cov_nx + j, 1) = y;
            }
        BoolVec cov_inside = this->poly.is_inside(cov_points);
        cov_valid = BoolMat::Constant(cov_ny, cov_nx, false);
        for (int i = 0; i < cov_ny; ++i)
            for (int j = 0; j < cov_nx; ++j)
                cov_valid(i, j) = cov_inside(i * cov_nx + j);
    }
}

void VolkovSolver::solve(bool verbose) {
    if (verbose) printf("Step 1: Finding block covering...\n");
    find_block_covering();

    if (verbose) printf("Step 2: Initializing solution...\n");
    initialize_solution();

    if (verbose) printf("Step 3: Boundary estimation...\n");
    boundary_estimator_.estimate(state, max_iter);

    if (verbose) printf("Step 4: Interior solve...\n");
    interior_solver_.solve(state, tol);

    // Copy solution back
    solution_grid = state.solution;

    if (verbose) printf("Done. N=%d, L=%d, M=%d\n", N, L, M);
}

std::tuple<int, int, int> VolkovSolver::find_block_covering() {
    // Create first and second kind blocks
    auto vertex_blocks = block_covering_.create_first_kind_blocks(poly);
    auto edge_blocks = block_covering_.create_second_kind_blocks(
        poly, vertex_blocks, static_cast<int>(vertex_blocks.size()));

    blocks.clear();
    blocks.insert(blocks.end(), vertex_blocks.begin(), vertex_blocks.end());
    blocks.insert(blocks.end(), edge_blocks.begin(), edge_blocks.end());

    N = static_cast<int>(vertex_blocks.size());
    L = N + static_cast<int>(edge_blocks.size());
    M = L;

    // Fill the interior with third-kind blocks using the (finer) covering grid,
    // then assign every solution-grid point to a block for the solve.
    create_third_kind_blocks();
    find_uncovered_points();

    return {N, L, M};
}

void VolkovSolver::find_uncovered_points() {
    // Reset inside_block_ids
    inside_block_ids_grid.setZero();

    // Build flat point array of valid grid points
    int total_valid = 0;
    for (int i = 0; i < ny; ++i)
        for (int j = 0; j < nx; ++j)
            if (solution_grid.valid(i, j)) total_valid++;

    // Track which valid points are covered
    std::vector<bool> covered(ny * nx, false);

    // Check each block
    for (const auto& blk : blocks) {
        for (int i = 0; i < ny; ++i) {
            for (int j = 0; j < nx; ++j) {
                if (!solution_grid.valid(i, j) || covered[i * nx + j])
                    continue;

                Vec2d pt(grid_X(i, j), grid_Y(i, j));
                Vec2d v = pt - blk.center;
                double dist = v.norm();

                if (dist >= blk.length) continue;

                // For third kind, only radial check
                if (blk.block_kind == BlockKind::Third) {
                    covered[i * nx + j] = true;
                    inside_block_ids_grid(i, j) = blk.id_;
                    continue;
                }

                // Angular check
                const std::vector<Vec2d>* edge_list;
                if (blk.boundary_type == BoundaryType::Hole)
                    edge_list = &poly.holes[blk.boundary_id].edges;
                else
                    edge_list = &poly.edges;

                double ref_angle;
                if (blk.block_kind == BlockKind::First)
                    ref_angle = std::atan2((*edge_list)[blk.edge_j_index](1),
                                           (*edge_list)[blk.edge_j_index](0));
                else
                    ref_angle = std::atan2((*edge_list)[blk.edge_i_index](1),
                                           (*edge_list)[blk.edge_i_index](0));

                double point_angle = std::atan2(v(1), v(0));
                double rel = std::fmod(point_angle - ref_angle, 2 * M_PI);
                if (rel < 0) rel += 2 * M_PI;

                if (rel <= blk.angle) {
                    covered[i * nx + j] = true;
                    inside_block_ids_grid(i, j) = blk.id_;
                }
            }
        }
    }
}

void VolkovSolver::create_third_kind_blocks() {
    // Detect gaps and place blocks on the covering grid (>= solution resolution),
    // so nooks between the first/second-kind blocks are filled even when the
    // solution grid is coarse. inside_block_ids is assigned later by
    // find_uncovered_points on the solution grid.
    std::vector<std::pair<int, int>> uncovered;
    std::vector<bool> covered(cov_ny * cov_nx, false);

    for (int i = 0; i < cov_ny; ++i)
        for (int j = 0; j < cov_nx; ++j) {
            if (!cov_valid(i, j)) continue;
            Vec2d pt(cov_X(i, j), cov_Y(i, j));
            bool is_covered = false;
            for (const auto& blk : blocks) {
                Vec2d v = pt - blk.center;
                double dist = v.norm();
                if (dist >= blk.length) continue;
                if (blk.block_kind == BlockKind::Third) { is_covered = true; break; }

                const std::vector<Vec2d>* edge_list =
                    (blk.boundary_type == BoundaryType::Hole)
                        ? &poly.holes[blk.boundary_id].edges : &poly.edges;
                int ei = (blk.block_kind == BlockKind::First) ? blk.edge_j_index
                                                              : blk.edge_i_index;
                double ref_angle = std::atan2((*edge_list)[ei](1), (*edge_list)[ei](0));
                double rel = std::fmod(std::atan2(v(1), v(0)) - ref_angle, 2 * M_PI);
                if (rel < 0) rel += 2 * M_PI;
                if (rel <= blk.angle) { is_covered = true; break; }
            }
            covered[i * cov_nx + j] = is_covered;
            if (!is_covered) uncovered.emplace_back(i, j);
        }

    if (uncovered.empty()) return;

    MatX2d edge_starts, edge_ends;
    block_covering_.get_all_edges_unified(poly, edge_starts, edge_ends);

    std::vector<int> indices(uncovered.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);  // deterministic seed
    std::shuffle(indices.begin(), indices.end(), rng);

    for (int idx : indices) {
        auto [yi, xi] = uncovered[idx];
        if (covered[yi * cov_nx + xi]) continue;

        // Keep the original (uniform-cov_delta) placement convention: at
        // covering_refinement=1 this reproduces the prior behaviour exactly.
        double px = xi * cov_delta + x_min;
        double py = yi * cov_delta + y_min;
        Vec2d current_center(px + 0.5 * cov_delta, py + 0.5 * cov_delta);
        covered[yi * cov_nx + xi] = true;

        if (!poly.point_in_polygon(current_center(0), current_center(1)))
            continue;

        VecXd dists = block_covering_.distances_from_edges(
            edge_starts, edge_ends, current_center);
        double radius = dists.minCoeff();
        double r0 = radial_heuristic * radius;
        double length = radial_heuristic * r0;

        Block new_block(current_center, 2 * M_PI, length, r0,
                        BlockKind::Third, M, -1, -1, BoundaryType::Main, 0);

        // Mark covering-grid points inside this block's basic disk as covered
        for (size_t k = 0; k < uncovered.size(); ++k) {
            auto [yk, xk] = uncovered[k];
            if (covered[yk * cov_nx + xk]) continue;
            Vec2d pt(xk * cov_delta + x_min, yk * cov_delta + y_min);
            if ((pt - current_center).norm() <= length)
                covered[yk * cov_nx + xk] = true;
        }

        blocks.push_back(new_block);
        M++;
    }
}

void VolkovSolver::initialize_solution() {
    state = SolutionState::from_polygon_and_blocks(
        poly, blocks, delta, n_param,
        boundary_conditions, is_dirichlet, tol);

    // Copy over pre-computed grid data
    state.X = grid_X;
    state.Y = grid_Y;
    state.solution = solution_grid;
    state.inside_block_ids_data = inside_block_ids_grid;
    state.inside_valid = inside_valid_grid;
    state.cartesian_grid = cartesian_grid_data;

    // Complete initialization
    find_tau_block_ids();
    find_tau_block_polar_coordinates();
    find_poisson_kernel_values();
    find_carrier_function_values();
}

void VolkovSolver::find_tau_block_ids() {
    // For each boundary point (mu, k), find which basic block contains it
    for (int mu = 0; mu < state.num_blocks; ++mu) {
        int nm = static_cast<int>(state.n_mu(mu));
        for (int k = 0; k < nm; ++k) {
            Vec2d pt = state.quantized_boundary_points[mu].row(k).transpose();

            // Find first block that contains this point
            for (int b = 0; b < static_cast<int>(blocks.size()); ++b) {
                Vec2d v = pt - blocks[b].center;
                double dist = v.norm();
                if (dist > blocks[b].length) continue;

                if (blocks[b].block_kind == BlockKind::Third) {
                    state.tau_block_ids[mu](k) = b;
                    break;
                }

                // Angular check
                const std::vector<Vec2d>* edge_list;
                if (blocks[b].boundary_type == BoundaryType::Hole)
                    edge_list = &poly.holes[blocks[b].boundary_id].edges;
                else
                    edge_list = &poly.edges;

                double ref_angle;
                if (blocks[b].block_kind == BlockKind::First)
                    ref_angle = std::atan2((*edge_list)[blocks[b].edge_j_index](1),
                                           (*edge_list)[blocks[b].edge_j_index](0));
                else
                    ref_angle = std::atan2((*edge_list)[blocks[b].edge_i_index](1),
                                           (*edge_list)[blocks[b].edge_i_index](0));

                double pa = std::atan2(v(1), v(0));
                double rel = std::fmod(pa - ref_angle, 2 * M_PI);
                if (rel < 0) rel += 2 * M_PI;

                if (rel <= blocks[b].angle * M_PI / M_PI) {
                    // Match Python: compare against blk.angle (which is in radians, pi*alpha)
                    state.tau_block_ids[mu](k) = b;
                    break;
                }
            }
        }
    }
}

void VolkovSolver::find_tau_block_polar_coordinates() {
    for (int mu = 0; mu < state.num_blocks; ++mu) {
        int nm = static_cast<int>(state.n_mu(mu));
        for (int k = 0; k < nm; ++k) {
            int tau_id = state.tau_block_ids[mu](k);
            if (tau_id < 0) continue;

            Vec2d pt = state.quantized_boundary_points[mu].row(k).transpose();
            double r, theta;
            cartesian_to_polar(pt, state.center_mu.row(tau_id).transpose(),
                              state.ref_theta(tau_id), r, theta);
            state.tau_polar[mu](k, 0) = r;
            state.tau_polar[mu](k, 1) = theta;
        }
    }
}

void VolkovSolver::find_poisson_kernel_values() {
    int max_bounds = static_cast<int>(state.phi_ij_a[0].size());
    VecXd k_vec(max_bounds);
    for (int i = 0; i < max_bounds; ++i) k_vec(i) = i;

    for (int mu = 0; mu < state.num_blocks; ++mu) {
        int nm = static_cast<int>(state.n_mu(mu));
        for (int k = 0; k < nm; ++k) {
            int tau_id = state.tau_block_ids[mu](k);
            if (tau_id < 0) continue;

            int bk_tau = state.block_type(tau_id);
            auto [nu_i, nu_j] = state.block_nu[tau_id];
            double alpha_tau = state.alpha_mu(tau_id);
            double r0_tau = state.r0_mu(tau_id);
            double r_ = state.tau_polar[mu](k, 0);
            double theta_ = state.tau_polar[mu](k, 1);

            int n_tau = static_cast<int>(state.n_mu(tau_id));
            for (int j = 0; j < n_tau; ++j) {
                double eta_j = state.theta_mu[tau_id](j);
                state.pk_matrix[mu](k, j) = poisson_calc_.calculate(
                    bk_tau, nu_i, nu_j, r_, r0_tau, theta_, eta_j, alpha_tau);
            }
        }
    }
}

void VolkovSolver::find_carrier_function_values() {
    int max_bounds = static_cast<int>(state.phi_ij_a[0].size());
    VecXd k_vec(max_bounds);
    for (int i = 0; i < max_bounds; ++i) k_vec(i) = i;

    for (int mu = 0; mu < state.num_blocks; ++mu) {
        int nm = static_cast<int>(state.n_mu(mu));
        for (int k = 0; k < nm; ++k) {
            // Tau carrier values: carrier at point (mu,k) using tau block parameters
            int tau_id = state.tau_block_ids[mu](k);
            if (tau_id >= 0) {
                double r_ = state.tau_polar[mu](k, 0);
                double theta_ = state.tau_polar[mu](k, 1);
                state.tau_carrier_values[mu](k) = carrier_calc_.calculate(
                    state.block_type(tau_id),
                    state.block_boundary_identifier[tau_id],
                    r_, theta_, k_vec,
                    state.phi_ij_a[tau_id], state.phi_ij_b[tau_id],
                    state.alpha_mu(tau_id));
            }

            // Mu carrier values: carrier at point (mu,k) using mu block's own parameters
            double r_mu = state.r0_mu(mu);
            double theta_mu = state.theta_mu[mu](k);
            state.mu_carrier_values[mu](k) = carrier_calc_.calculate(
                state.block_type(mu),
                state.block_boundary_identifier[mu],
                r_mu, theta_mu, k_vec,
                state.phi_ij_a[mu], state.phi_ij_b[mu],
                state.alpha_mu(mu));
        }
    }
}

} // namespace pyblockgrid
