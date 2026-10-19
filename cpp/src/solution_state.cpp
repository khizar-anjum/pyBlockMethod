#include "pyblockgrid/solution_state.hpp"
#include "pyblockgrid/coordinate_transforms.hpp"
#include <cmath>
#include <algorithm>

namespace pyblockgrid {

namespace {
// Re-expand an edge boundary polynomial into the block-vertex frame.
// Data is u(t) = sum_k c_k t^k, t = arc length from the edge start. For the edge
// that ENDS at a first-kind block's vertex, the block's radial coordinate is
// r = L - t, so the datum reads g(r) = u(L - r) = sum_m d_m r^m with
// d_m = (-1)^m sum_{k>=m} c_k C(k,m) L^(k-m). Constant/symmetric data is
// unchanged; asymmetric data (e.g. a ramp) would otherwise be read backward.
std::vector<double> reverse_bc_coeffs(const std::vector<double>& c, double L) {
    int n = static_cast<int>(c.size());
    std::vector<double> d(n, 0.0);
    for (int m = 0; m < n; ++m) {
        double s = 0.0, binom = 1.0;  // C(m,m)=1, built up as C(k,m)=C(k-1,m)*k/(k-m)
        for (int k = m; k < n; ++k) {
            if (k > m) binom = binom * k / (k - m);
            s += c[k] * binom * std::pow(L, k - m);
        }
        d[m] = (m % 2 == 0 ? 1.0 : -1.0) * s;
    }
    return d;
}
}  // namespace

SolutionState SolutionState::from_polygon_and_blocks(
    const Polygon& poly, const std::vector<Block>& blocks,
    double delta, int n,
    const std::vector<std::vector<double>>& boundary_conditions,
    const std::vector<bool>& is_dirichlet,
    double tolerance) {

    SolutionState s;
    int n_verts = static_cast<int>(poly.vertices.rows());

    // Grid bounds
    s.x_min = poly.vertices.col(0).minCoeff();
    s.y_min = poly.vertices.col(1).minCoeff();
    s.x_max = poly.vertices.col(0).maxCoeff();
    s.y_max = poly.vertices.col(1).maxCoeff();
    s.delta = delta;

    s.ny = static_cast<int>(std::ceil((s.y_max - s.y_min) / delta));
    s.nx = static_cast<int>(std::ceil((s.x_max - s.x_min) / delta));

    // Create grid points: linspace(min + 0.5*delta, max + 0.5*delta, n)
    VecXd x_vec(s.nx), y_vec(s.ny);
    {
        double x_start = s.x_min + 0.5 * delta;
        double x_end = s.x_max + 0.5 * delta;
        for (int i = 0; i < s.nx; ++i)
            x_vec(i) = s.nx > 1 ? x_start + i * (x_end - x_start) / (s.nx - 1) : x_start;
    }
    {
        double y_start = s.y_min + 0.5 * delta;
        double y_end = s.y_max + 0.5 * delta;
        for (int i = 0; i < s.ny; ++i)
            y_vec(i) = s.ny > 1 ? y_start + i * (y_end - y_start) / (s.ny - 1) : y_start;
    }

    // Meshgrid
    s.X.resize(s.ny, s.nx);
    s.Y.resize(s.ny, s.nx);
    for (int i = 0; i < s.ny; ++i)
        for (int j = 0; j < s.nx; ++j) {
            s.X(i, j) = x_vec(j);
            s.Y(i, j) = y_vec(i);
        }

    // Flatten boundary conditions (main + holes)
    for (int i = 0; i < n_verts; ++i) {
        s.boundary_segments.push_back({BoundaryType::Main, 0, i});
        s.boundary_conditions_flat.push_back(boundary_conditions[i]);
        s.is_dirichlet_flat.push_back(is_dirichlet[i]);
    }
    for (int h = 0; h < static_cast<int>(poly.holes.size()); ++h) {
        for (int i = 0; i < poly.holes[h].n_vertices; ++i) {
            s.boundary_segments.push_back({BoundaryType::Hole, h, i});
            s.boundary_conditions_flat.push_back(poly.holes[h].boundary_conditions[i]);
            s.is_dirichlet_flat.push_back(poly.holes[h].is_dirichlet[i]);
        }
    }

    // Block parameters
    s.num_blocks = static_cast<int>(blocks.size());
    s.alpha_mu = VecXd::Zero(s.num_blocks);
    s.r0_mu = VecXd::Zero(s.num_blocks);
    s.r_mu = VecXd::Zero(s.num_blocks);
    s.center_mu = MatX2d::Zero(s.num_blocks, 2);
    s.ref_theta = VecXd::Zero(s.num_blocks);
    s.block_type = VecXi::Zero(s.num_blocks);
    s.block_nu.resize(s.num_blocks, {-1, -1});
    s.block_boundary_identifier.resize(s.num_blocks, 0);

    // Find max boundary condition size
    int max_bounds = 1;
    for (const auto& bc : s.boundary_conditions_flat)
        max_bounds = std::max(max_bounds, static_cast<int>(bc.size()));

    s.phi_ij_a.resize(s.num_blocks, VecXd::Zero(max_bounds));
    s.phi_ij_b.resize(s.num_blocks, VecXd::Zero(max_bounds));

    auto get_flat_idx = [&](BoundaryType bt, int bid, int eid) -> int {
        if (bt == BoundaryType::Main) return eid;
        int offset = n_verts;
        for (int h = 0; h < bid; ++h)
            offset += poly.holes[h].n_vertices;
        return offset + eid;
    };

    for (const auto& blk : blocks) {
        int id = blk.id_;
        s.alpha_mu(id) = blk.angle / M_PI;
        s.r0_mu(id) = blk.r0;
        s.r_mu(id) = blk.length;
        s.center_mu.row(id) = blk.center.transpose();
        s.block_type(id) = static_cast<int>(blk.block_kind);

        if (blk.block_kind == BlockKind::First) {
            const std::vector<Vec2d>* edges;
            if (blk.boundary_type == BoundaryType::Hole)
                edges = &poly.holes[blk.boundary_id].edges;
            else
                edges = &poly.edges;

            s.ref_theta(id) = std::atan2((*edges)[blk.edge_j_index](1),
                                          (*edges)[blk.edge_j_index](0));

            int ei_flat = get_flat_idx(blk.boundary_type, blk.boundary_id, blk.edge_i_index);
            int ej_flat = get_flat_idx(blk.boundary_type, blk.boundary_id, blk.edge_j_index);

            // edge_i ends at this vertex, so its datum runs backward relative to
            // the block's radial coordinate; reverse it into the vertex frame.
            std::vector<double> bc_i = reverse_bc_coeffs(
                s.boundary_conditions_flat[ei_flat],
                (*edges)[blk.edge_i_index].norm());
            const auto& bc_j = s.boundary_conditions_flat[ej_flat];
            for (int k = 0; k < static_cast<int>(bc_i.size()); ++k)
                s.phi_ij_a[id](k) = bc_i[k];
            for (int k = 0; k < static_cast<int>(bc_j.size()); ++k)
                s.phi_ij_b[id](k) = bc_j[k];

            int nu_i = s.is_dirichlet_flat[ei_flat] ? 1 : 0;
            int nu_j = s.is_dirichlet_flat[ej_flat] ? 1 : 0;
            s.block_nu[id] = {nu_i, nu_j};
            s.block_boundary_identifier[id] = 2 * nu_i + nu_j;

        } else if (blk.block_kind == BlockKind::Second) {
            const std::vector<Vec2d>* edges;
            if (blk.boundary_type == BoundaryType::Hole)
                edges = &poly.holes[blk.boundary_id].edges;
            else
                edges = &poly.edges;

            s.ref_theta(id) = std::atan2((*edges)[blk.edge_i_index](1),
                                          (*edges)[blk.edge_i_index](0));

            int ei_flat = get_flat_idx(blk.boundary_type, blk.boundary_id, blk.edge_i_index);
            const auto& bc_i = s.boundary_conditions_flat[ei_flat];
            for (int k = 0; k < static_cast<int>(bc_i.size()); ++k)
                s.phi_ij_a[id](k) = bc_i[k];

            int nu_i = s.is_dirichlet_flat[ei_flat] ? 1 : 0;
            s.block_nu[id] = {nu_i, -1};
            s.block_boundary_identifier[id] = 2 * nu_i;

        } else {
            s.block_nu[id] = {-1, -1};
            s.block_boundary_identifier[id] = 0;
        }
    }

    // Discretization: n_mu, beta_mu, theta_mu
    s.n_mu = VecXd::Zero(s.num_blocks);
    s.beta_mu = VecXd::Zero(s.num_blocks);
    s.max_n = 0;

    for (int i = 0; i < s.num_blocks; ++i) {
        s.n_mu(i) = std::max(4.0, std::floor(n * s.alpha_mu(i)));
        s.beta_mu(i) = M_PI * s.alpha_mu(i) / s.n_mu(i);
        s.max_n = std::max(s.max_n, static_cast<int>(s.n_mu(i)));
    }

    // Ragged arrays
    s.theta_mu.resize(s.num_blocks);
    s.quantized_boundary_points.resize(s.num_blocks);
    s.boundary_estimates.resize(s.num_blocks);
    s.tau_block_ids.resize(s.num_blocks);
    s.pk_matrix.resize(s.num_blocks);
    s.tau_carrier_values.resize(s.num_blocks);
    s.mu_carrier_values.resize(s.num_blocks);
    s.tau_polar.resize(s.num_blocks);

    for (int mu = 0; mu < s.num_blocks; ++mu) {
        int nm = static_cast<int>(s.n_mu(mu));
        s.theta_mu[mu].resize(nm);
        s.boundary_estimates[mu] = VecXd::Zero(nm);
        s.quantized_boundary_points[mu].resize(nm, 2);
        s.tau_block_ids[mu] = VecXi::Constant(nm, -1);
        s.pk_matrix[mu] = MatXd::Zero(nm, s.max_n);
        s.tau_carrier_values[mu] = VecXd::Zero(nm);
        s.mu_carrier_values[mu] = VecXd::Zero(nm);
        s.tau_polar[mu] = MatX2d::Zero(nm, 2);

        for (int k = 0; k < nm; ++k) {
            s.theta_mu[mu](k) = s.beta_mu(mu) * (k + 0.5);
            // Convert polar (r0, theta) to Cartesian
            s.quantized_boundary_points[mu].row(k) =
                polar_to_cartesian(s.r0_mu(mu), s.theta_mu[mu](k),
                                   s.center_mu.row(mu).transpose(),
                                   s.ref_theta(mu)).transpose();
        }
    }

    return s;
}

} // namespace pyblockgrid
