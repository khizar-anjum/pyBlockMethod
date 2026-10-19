#pragma once

#include "types.hpp"
#include "block.hpp"
#include "polygon.hpp"
#include <vector>
#include <tuple>

namespace pyblockgrid {

class BlockCoveringStrategy {
public:
    double radial_heuristic;
    double overlap_heuristic;

    BlockCoveringStrategy(double radial = 0.8, double overlap = 0.1)
        : radial_heuristic(radial), overlap_heuristic(overlap) {}

    // Create first-kind blocks (vertex sectors)
    std::vector<Block> create_first_kind_blocks(const Polygon& poly) const;

    // Create second-kind blocks (edge half-disks)
    std::vector<Block> create_second_kind_blocks(
        const Polygon& poly,
        const std::vector<Block>& vertex_blocks,
        int block_id_counter) const;

    // Get all edges as unified arrays (main + holes, closed loops)
    void get_all_edges_unified(const Polygon& poly,
                               MatX2d& starts, MatX2d& ends) const;

    // Distance from point to line segments
    VecXd distances_from_edges(const MatX2d& v, const MatX2d& w,
                                const Vec2d& p) const;

    // Filtered distance (FoV-based)
    VecXd distances_from_edges_filtered(const MatX2d& v, const MatX2d& w,
                                         const Vec2d& p, const Vec2d& inward_normal,
                                         double fov_angle = M_PI) const;

private:
    // Fill edge gap with second-kind blocks
    std::vector<Block> fill_edge_gap(
        const Polygon& poly, const Vec2d& prev_vertex, const Vec2d& curr_vertex,
        const Block& prev_block, const Block& curr_block,
        int edge_index, int block_id_counter,
        BoundaryType btype, int boundary_id) const;

    // Recursive gap fill
    std::pair<std::vector<Block>, int> fill_gap_recursive(
        const Vec2d& start, const Vec2d& end,
        double start_radius, double end_radius,
        const MatX2d& edge_starts_filtered, const MatX2d& edge_ends_filtered,
        const Vec2d& edge_normal, int edge_index,
        int block_id_counter, const Vec2d& unit_edge_vector,
        BoundaryType btype, int boundary_id) const;

    Vec2d get_edge_inward_normal(const Vec2d& edge_vector) const;
    Vec2d get_vertex_average_normal(const MatX2d& adj_starts,
                                    const MatX2d& adj_ends,
                                    const Vec2d& vertex) const;
};

} // namespace pyblockgrid
