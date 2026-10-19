#include "pyblockgrid/block_covering.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace pyblockgrid {

void BlockCoveringStrategy::get_all_edges_unified(const Polygon& poly,
                                                   MatX2d& starts, MatX2d& ends) const {
    int n_main = static_cast<int>(poly.vertices.rows());
    int total = n_main;
    for (const auto& h : poly.holes)
        total += h.n_vertices;

    starts.resize(total, 2);
    ends.resize(total, 2);

    // Main polygon edges (closed loop)
    for (int i = 0; i < n_main; ++i) {
        starts.row(i) = poly.vertices.row(i);
        ends.row(i) = poly.vertices.row((i + 1) % n_main);
    }

    // Hole edges
    int offset = n_main;
    for (const auto& hole : poly.holes) {
        for (int i = 0; i < hole.n_vertices; ++i) {
            starts.row(offset + i) = hole.vertices.row(i);
            ends.row(offset + i) = hole.vertices.row((i + 1) % hole.n_vertices);
        }
        offset += hole.n_vertices;
    }
}

VecXd BlockCoveringStrategy::distances_from_edges(const MatX2d& v, const MatX2d& w,
                                                    const Vec2d& p) const {
    int n = static_cast<int>(v.rows());
    VecXd dists(n);

    for (int i = 0; i < n; ++i) {
        Vec2d vi = v.row(i).transpose();
        Vec2d wi = w.row(i).transpose();
        Vec2d edge = wi - vi;
        double l2 = edge.squaredNorm();

        if (l2 == 0.0) {
            dists(i) = (p - vi).norm();
        } else {
            double t = (p - vi).dot(edge) / l2;
            t = std::clamp(t, 0.0, 1.0);
            Vec2d proj = vi + t * edge;
            dists(i) = (p - proj).norm();
        }
    }
    return dists;
}

VecXd BlockCoveringStrategy::distances_from_edges_filtered(
    const MatX2d& v, const MatX2d& w, const Vec2d& p,
    const Vec2d& inward_normal, double fov_angle) const {

    double cos_half_fov = std::cos((fov_angle + 4e-2) / 2.0);

    int n = static_cast<int>(v.rows());
    std::vector<int> keep;
    for (int i = 0; i < n; ++i) {
        Vec2d dv = v.row(i).transpose() - p;
        Vec2d dw = w.row(i).transpose() - p;
        double v_dot = dv.dot(inward_normal) / dv.norm();
        double w_dot = dw.dot(inward_normal) / dw.norm();
        if (v_dot >= cos_half_fov || w_dot >= cos_half_fov)
            keep.push_back(i);
    }

    if (keep.empty()) {
        VecXd result(1);
        result(0) = std::numeric_limits<double>::infinity();
        return result;
    }

    MatX2d vf(keep.size(), 2), wf(keep.size(), 2);
    for (int i = 0; i < static_cast<int>(keep.size()); ++i) {
        vf.row(i) = v.row(keep[i]);
        wf.row(i) = w.row(keep[i]);
    }
    return distances_from_edges(vf, wf, p);
}

Vec2d BlockCoveringStrategy::get_edge_inward_normal(const Vec2d& edge_vector) const {
    Vec2d n(-edge_vector(1), edge_vector(0));
    return n / n.norm();
}

Vec2d BlockCoveringStrategy::get_vertex_average_normal(
    const MatX2d& adj_starts, const MatX2d& adj_ends,
    const Vec2d& vertex) const {
    Vec2d e1 = adj_ends.row(0).transpose() - adj_starts.row(0).transpose();
    Vec2d e2 = adj_ends.row(1).transpose() - adj_starts.row(1).transpose();
    Vec2d n1 = get_edge_inward_normal(e1);
    Vec2d n2 = get_edge_inward_normal(e2);
    Vec2d avg = (n1 + n2) / 2.0;
    return avg / avg.norm();
}

std::vector<Block> BlockCoveringStrategy::create_first_kind_blocks(const Polygon& poly) const {
    std::vector<Block> blocks;
    int block_id = 0;

    MatX2d edge_starts, edge_ends;
    get_all_edges_unified(poly, edge_starts, edge_ends);
    int total_edges = static_cast<int>(edge_starts.rows());
    int n_main = static_cast<int>(poly.vertices.rows());

    // Helper: iterate all vertices (main + holes)
    struct VertexInfo {
        Vec2d vertex;
        double angle;
        BoundaryType btype;
        int boundary_id;
        int vertex_id;
    };
    std::vector<VertexInfo> all_vertices;

    for (int i = 0; i < n_main; ++i)
        all_vertices.push_back({poly.vertices.row(i).transpose(),
                                poly.angles[i], BoundaryType::Main, 0, i});

    for (int h = 0; h < static_cast<int>(poly.holes.size()); ++h) {
        const auto& hole = poly.holes[h];
        for (int i = 0; i < hole.n_vertices; ++i)
            all_vertices.push_back({hole.vertices.row(i).transpose(),
                                    hole.interior_angles[i], BoundaryType::Hole, h, i});
    }

    for (const auto& vi : all_vertices) {
        // Build non-adjacent edge mask
        BoolVec adj_mask = BoolVec::Constant(total_edges, true);

        if (vi.btype == BoundaryType::Main) {
            adj_mask(vi.vertex_id) = false;
            adj_mask((vi.vertex_id - 1 + n_main) % n_main) = false;
        } else {
            int hole_start = n_main;
            for (int h = 0; h < vi.boundary_id; ++h)
                hole_start += poly.holes[h].n_vertices;
            int nv = poly.holes[vi.boundary_id].n_vertices;
            adj_mask(hole_start + vi.vertex_id) = false;
            adj_mask(hole_start + (vi.vertex_id - 1 + nv) % nv) = false;
        }

        // Split into adjacent/non-adjacent
        int n_non_adj = adj_mask.cast<int>().sum();
        int n_adj = total_edges - n_non_adj;
        MatX2d non_adj_s(n_non_adj, 2), non_adj_e(n_non_adj, 2);
        MatX2d adj_s(n_adj, 2), adj_e(n_adj, 2);
        int ni = 0, ai = 0;
        for (int i = 0; i < total_edges; ++i) {
            if (adj_mask(i)) {
                non_adj_s.row(ni) = edge_starts.row(i);
                non_adj_e.row(ni) = edge_ends.row(i);
                ni++;
            } else {
                adj_s.row(ai) = edge_starts.row(i);
                adj_e.row(ai) = edge_ends.row(i);
                ai++;
            }
        }

        Vec2d avg_normal = get_vertex_average_normal(adj_s, adj_e, vi.vertex);

        VecXd filtered_dists = distances_from_edges_filtered(
            non_adj_s, non_adj_e, vi.vertex, avg_normal, vi.angle);

        double min_filtered = filtered_dists.minCoeff();
        // Python uses np.linalg.norm(adjacent_starts - vertex) which computes
        // the Frobenius norm of the (2,2) matrix — matching the distance to
        // the far endpoint since one row is always zero (the vertex itself)
        double adj_starts_frobenius = (adj_s.rowwise() - vi.vertex.transpose()).norm();
        double adj_ends_frobenius = (adj_e.rowwise() - vi.vertex.transpose()).norm();

        double r0_val = radial_heuristic * std::min({min_filtered,
                                                      adj_starts_frobenius,
                                                      adj_ends_frobenius});
        double length_val = radial_heuristic * r0_val;

        int ei, ej;
        if (vi.btype == BoundaryType::Main) {
            ei = (vi.vertex_id - 1 + n_main) % n_main;
            ej = vi.vertex_id;
        } else {
            int nv = poly.holes[vi.boundary_id].n_vertices;
            ei = (vi.vertex_id - 1 + nv) % nv;
            ej = vi.vertex_id;
        }

        blocks.emplace_back(vi.vertex, vi.angle, length_val, r0_val,
                            BlockKind::First, block_id,
                            ei, ej, vi.btype, vi.boundary_id, vi.vertex_id);
        block_id++;
    }
    return blocks;
}

std::pair<std::vector<Block>, int> BlockCoveringStrategy::fill_gap_recursive(
    const Vec2d& start, const Vec2d& end,
    double start_radius, double end_radius,
    const MatX2d& edge_starts_filtered, const MatX2d& edge_ends_filtered,
    const Vec2d& edge_normal, int edge_index,
    int block_id_counter, const Vec2d& unit_edge_vector,
    BoundaryType btype, int boundary_id) const {

    double gap_distance = (end - start).norm();
    if (gap_distance < 1e-6)
        return {{}, block_id_counter};

    Vec2d midpoint = start + 0.5 * gap_distance * unit_edge_vector;

    double max_allowed = radial_heuristic * distances_from_edges_filtered(
        edge_starts_filtered, edge_ends_filtered, midpoint, edge_normal, M_PI).minCoeff();
    double actual_radius = radial_heuristic * max_allowed;

    if (2 * actual_radius >= gap_distance) {
        Block blk(midpoint, M_PI, actual_radius, max_allowed,
                  BlockKind::Second, block_id_counter,
                  edge_index, -1, btype, boundary_id);
        return {{blk}, block_id_counter + 1};
    }

    Block mid_block(midpoint, M_PI, actual_radius, max_allowed,
                    BlockKind::Second, block_id_counter,
                    edge_index, -1, btype, boundary_id);
    std::vector<Block> blocks = {mid_block};
    int next_id = block_id_counter + 1;

    // Left gap
    Vec2d left_end = midpoint - (1 - overlap_heuristic) * actual_radius * unit_edge_vector;
    auto [left_blocks, lid] = fill_gap_recursive(
        start, left_end, start_radius, actual_radius,
        edge_starts_filtered, edge_ends_filtered, edge_normal,
        edge_index, next_id, unit_edge_vector, btype, boundary_id);
    blocks.insert(blocks.end(), left_blocks.begin(), left_blocks.end());
    next_id = lid;

    // Right gap
    Vec2d right_start = midpoint + (1 - overlap_heuristic) * actual_radius * unit_edge_vector;
    auto [right_blocks, rid] = fill_gap_recursive(
        right_start, end, actual_radius, end_radius,
        edge_starts_filtered, edge_ends_filtered, edge_normal,
        edge_index, next_id, unit_edge_vector, btype, boundary_id);
    blocks.insert(blocks.end(), right_blocks.begin(), right_blocks.end());
    next_id = rid;

    return {blocks, next_id};
}

std::vector<Block> BlockCoveringStrategy::fill_edge_gap(
    const Polygon& poly, const Vec2d& prev_vertex, const Vec2d& curr_vertex,
    const Block& prev_block, const Block& curr_block,
    int edge_index, int block_id_counter,
    BoundaryType btype, int boundary_id) const {

    double distance = (curr_vertex - prev_vertex).norm();
    double dist_covered = prev_block.length + curr_block.length;
    if (distance <= dist_covered) return {};

    Vec2d unit_edge = (curr_vertex - prev_vertex) / distance;
    Vec2d start_v = prev_vertex + (1 - overlap_heuristic) * prev_block.length * unit_edge;
    Vec2d end_v = curr_vertex - (1 - overlap_heuristic) * curr_block.length * unit_edge;

    MatX2d edge_starts, edge_ends;
    get_all_edges_unified(poly, edge_starts, edge_ends);

    // Calculate global edge index for filtering
    int global_edge_idx = edge_index;
    if (btype == BoundaryType::Hole) {
        global_edge_idx = static_cast<int>(poly.vertices.rows());
        for (int h = 0; h < boundary_id; ++h)
            global_edge_idx += poly.holes[h].n_vertices;
        global_edge_idx += edge_index;
    }

    // Remove current edge from constraint set
    int total = static_cast<int>(edge_starts.rows());
    MatX2d es_filt(total - 1, 2), ee_filt(total - 1, 2);
    int j = 0;
    for (int i = 0; i < total; ++i) {
        if (i != global_edge_idx) {
            es_filt.row(j) = edge_starts.row(i);
            ee_filt.row(j) = edge_ends.row(i);
            j++;
        }
    }

    Vec2d edge_normal = get_edge_inward_normal(unit_edge);

    auto [blocks, _] = fill_gap_recursive(
        start_v, end_v, prev_block.length, curr_block.length,
        es_filt, ee_filt, edge_normal, edge_index,
        block_id_counter, unit_edge, btype, boundary_id);
    return blocks;
}

std::vector<Block> BlockCoveringStrategy::create_second_kind_blocks(
    const Polygon& poly, const std::vector<Block>& vertex_blocks,
    int block_id_counter) const {

    std::vector<Block> edge_blocks;
    int vb_offset = 0;
    int n_main = static_cast<int>(poly.vertices.rows());

    // Main polygon edges
    for (int i = 0; i < n_main; ++i) {
        int prev_idx = (i - 1 + n_main) % n_main;
        Vec2d prev_v = poly.vertices.row(prev_idx).transpose();
        Vec2d curr_v = poly.vertices.row(i).transpose();

        const Block& prev_blk = vertex_blocks[vb_offset + prev_idx];
        const Block& curr_blk = vertex_blocks[vb_offset + i];

        if ((curr_v - prev_v).norm() > prev_blk.length + curr_blk.length) {
            auto gap_blocks = fill_edge_gap(
                poly, prev_v, curr_v, prev_blk, curr_blk,
                prev_idx, block_id_counter, BoundaryType::Main, 0);
            edge_blocks.insert(edge_blocks.end(), gap_blocks.begin(), gap_blocks.end());
            block_id_counter += static_cast<int>(gap_blocks.size());
        }
    }
    vb_offset += n_main;

    // Hole edges
    for (int h = 0; h < static_cast<int>(poly.holes.size()); ++h) {
        int nv = poly.holes[h].n_vertices;
        for (int i = 0; i < nv; ++i) {
            int prev_idx = (i - 1 + nv) % nv;
            Vec2d prev_v = poly.holes[h].vertices.row(prev_idx).transpose();
            Vec2d curr_v = poly.holes[h].vertices.row(i).transpose();

            const Block& prev_blk = vertex_blocks[vb_offset + prev_idx];
            const Block& curr_blk = vertex_blocks[vb_offset + i];

            if ((curr_v - prev_v).norm() > prev_blk.length + curr_blk.length) {
                auto gap_blocks = fill_edge_gap(
                    poly, prev_v, curr_v, prev_blk, curr_blk,
                    prev_idx, block_id_counter, BoundaryType::Hole, h);
                edge_blocks.insert(edge_blocks.end(), gap_blocks.begin(), gap_blocks.end());
                block_id_counter += static_cast<int>(gap_blocks.size());
            }
        }
        vb_offset += nv;
    }

    return edge_blocks;
}

} // namespace pyblockgrid
