#pragma once

#include "types.hpp"
#include "block.hpp"
#include "polygon.hpp"
#include <vector>

namespace pyblockgrid {

struct SolutionState {
    // Grid info
    int nx, ny;
    double delta, x_min, y_min, x_max, y_max;
    MatXd X, Y;  // meshgrid

    // Solution grid: valid(i,j)==true means inside domain
    MaskedGrid solution;
    MatXd inside_block_ids_data;  // int values stored as double
    BoolMat inside_valid;
    MaskedGrid2D cartesian_grid;

    // Block parameters (flat arrays, indexed by block id)
    VecXd alpha_mu, r0_mu, r_mu, beta_mu, n_mu, ref_theta;
    MatX2d center_mu;
    VecXi block_type;
    int num_blocks;
    int max_n;  // max of all n_mu values

    // Per-block ragged arrays
    std::vector<VecXd> theta_mu;               // [mu] size n_mu[mu]
    std::vector<VecXd> boundary_estimates;     // [mu] size n_mu[mu]
    std::vector<MatX2d> quantized_boundary_points;  // [mu] size (n_mu[mu], 2)

    // Boundary condition data per block
    std::vector<VecXd> phi_ij_a;  // [mu] edge i BC coeffs
    std::vector<VecXd> phi_ij_b;  // [mu] edge j BC coeffs
    std::vector<std::pair<int, int>> block_nu;  // (nu_i, nu_j), -1 for absent
    std::vector<int> block_boundary_identifier;  // 2*nu_i + nu_j

    // Tau block data per boundary point (ragged)
    std::vector<VecXi> tau_block_ids;     // [mu] size n_mu[mu]
    std::vector<MatX2d> tau_polar;        // [mu] (n_mu[mu], 2) = (r, theta)
    std::vector<MatXd> pk_matrix;         // [mu] (n_mu[mu], max_n) padded with 0
    std::vector<VecXd> tau_carrier_values;  // [mu] size n_mu[mu]
    std::vector<VecXd> mu_carrier_values;   // [mu] size n_mu[mu]

    // Boundary segment info
    struct BoundarySegment {
        BoundaryType btype;
        int boundary_id;
        int edge_id;
    };
    std::vector<BoundarySegment> boundary_segments;
    std::vector<std::vector<double>> boundary_conditions_flat;
    std::vector<bool> is_dirichlet_flat;

    static SolutionState from_polygon_and_blocks(
        const Polygon& poly, const std::vector<Block>& blocks,
        double delta, int n,
        const std::vector<std::vector<double>>& boundary_conditions,
        const std::vector<bool>& is_dirichlet,
        double tolerance = 1e-10);
};

} // namespace pyblockgrid
