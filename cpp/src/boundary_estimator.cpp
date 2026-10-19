#include "pyblockgrid/boundary_estimator.hpp"
#include <cmath>
#include <algorithm>

namespace pyblockgrid {

void BoundaryEstimator::estimate(SolutionState& state, int max_iter) const {
    // Gauss-Seidel iteration matching Python exactly.
    // Python iterates (i=0..N-1, j=0..M-1) over the rectangular (num_blocks, max_n) array,
    // skipping masked entries. With ragged arrays we iterate (mu=0..num_blocks-1, k=0..n_mu[mu]-1).
    // The iteration order is the same because Python skips j >= n_mu[i] (masked).

    int iter_count = 0;
    while (true) {
        for (int mu = 0; mu < state.num_blocks; ++mu) {
            int nm = static_cast<int>(state.n_mu(mu));
            for (int k = 0; k < nm; ++k) {
                int tau_id = state.tau_block_ids[mu](k);
                if (tau_id < 0) continue;

                double beta_tau = state.beta_mu(tau_id);
                int n_tau = static_cast<int>(state.n_mu(tau_id));

                // Sum pk values for normalization
                double pk_sum = 0;
                for (int j = 0; j < n_tau; ++j)
                    pk_sum += state.pk_matrix[mu](k, j);

                double denom = std::max(1.0, beta_tau * pk_sum);

                // Weighted sum of boundary estimate differences
                double diff_pk_sum = 0;
                for (int j = 0; j < n_tau; ++j) {
                    diff_pk_sum += (state.boundary_estimates[tau_id](j)
                                  - state.mu_carrier_values[tau_id](j))
                                  * state.pk_matrix[mu](k, j);
                }

                state.boundary_estimates[mu](k) =
                    state.tau_carrier_values[mu](k)
                    + beta_tau * diff_pk_sum / denom;
            }
        }
        iter_count++;
        if (iter_count > max_iter) break;
    }
}

} // namespace pyblockgrid
