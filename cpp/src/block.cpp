#include "pyblockgrid/block.hpp"
#include <cmath>

namespace pyblockgrid {

Block::Block(Vec2d center, double angle, double length, double r0,
             BlockKind kind, int id,
             int edge_i, int edge_j,
             BoundaryType btype, int bid, int vid)
    : center(center), angle(angle), length(length), r0(r0),
      block_kind(kind), id_(id),
      edge_i_index(edge_i), edge_j_index(edge_j),
      boundary_type(btype), boundary_id(bid), vertex_id(vid)
{}

BoolVec Block::is_inside(const MatX2d& points, const BoolVec& point_mask,
                          const Polygon& poly) const {
    int n = static_cast<int>(points.rows());
    BoolVec result(n);

    // Get reference edge for angular check
    const std::vector<Vec2d>* edge_list = nullptr;
    if (boundary_type == BoundaryType::Hole && boundary_id < static_cast<int>(poly.holes.size()))
        edge_list = &poly.holes[boundary_id].edges;
    else
        edge_list = &poly.edges;

    double ref_angle = 0;
    if (block_kind == BlockKind::First) {
        ref_angle = std::atan2((*edge_list)[edge_j_index](1),
                               (*edge_list)[edge_j_index](0));
    } else if (block_kind == BlockKind::Second) {
        ref_angle = std::atan2((*edge_list)[edge_i_index](1),
                               (*edge_list)[edge_i_index](0));
    }

    for (int i = 0; i < n; ++i) {
        if (!point_mask(i)) {
            result(i) = false;
            continue;
        }
        Vec2d v = points.row(i).transpose() - center;
        double dist = v.norm();

        // Radial constraint
        bool inside_radial = dist < length;

        if (block_kind == BlockKind::Third) {
            result(i) = inside_radial;
            continue;
        }

        // Angular constraint
        double point_angle = std::atan2(v(1), v(0));
        double rel_angle = std::fmod(point_angle - ref_angle, 2 * M_PI);
        if (rel_angle < 0) rel_angle += 2 * M_PI;
        bool inside_angular = rel_angle <= angle;

        result(i) = inside_radial && inside_angular;
    }
    return result;
}

} // namespace pyblockgrid
