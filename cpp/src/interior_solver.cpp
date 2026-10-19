#include "pyblockgrid/interior_solver.hpp"
#include "pyblockgrid/coordinate_transforms.hpp"
#include <cmath>

namespace pyblockgrid {

void InteriorSolver::solve(SolutionState& state, double tolerance) const {
    // For each valid grid point, compute Q (carrier) and R (Poisson kernel vector),
    // then combine with boundary estimates using equation (5.1).

    int max_bounds = static_cast<int>(state.phi_ij_a[0].size());
    VecXd k_vec(max_bounds);
    for (int i = 0; i < max_bounds; ++i)
        k_vec(i) = i;

    for (int i = 0; i < state.ny; ++i) {
        for (int j = 0; j < state.nx; ++j) {
            if (!state.solution.valid(i, j)) continue;

            int block_id = static_cast<int>(state.inside_block_ids_data(i, j));

            // Get polar coordinates of this grid point relative to its block
            Vec2d point(state.X(i, j), state.Y(i, j));
            double r, theta;
            cartesian_to_polar(point, state.center_mu.row(block_id).transpose(),
                              state.ref_theta(block_id), r, theta);

            // Skip points at the pole
            if (r < tolerance) continue;

            int bk = state.block_type(block_id);
            int bi = state.block_boundary_identifier[block_id];
            double alpha = state.alpha_mu(block_id);
            auto [nu_i, nu_j] = state.block_nu[block_id];

            // Q: carrier function value at this grid point
            double Q = carrier_calc_.calculate(
                bk, bi, r, theta, k_vec,
                state.phi_ij_a[block_id], state.phi_ij_b[block_id], alpha);

            // R: Poisson kernel values at this grid point for each boundary point of the block
            int nb = static_cast<int>(state.n_mu(block_id));
            double r0 = state.r0_mu(block_id);

            double sum = 0;
            for (int k = 0; k < nb; ++k) {
                double eta_k = state.theta_mu[block_id](k);
                double R_k = poisson_calc_.calculate(
                    bk, nu_i, nu_j, r, r0, theta, eta_k, alpha);
                sum += (state.boundary_estimates[block_id](k)
                      - state.mu_carrier_values[block_id](k)) * R_k;
            }

            state.solution.data(i, j) = Q + state.beta_mu(block_id) * sum;
        }
    }
}

} // namespace pyblockgrid
