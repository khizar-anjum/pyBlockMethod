#pragma once

#include "types.hpp"
#include "polygon.hpp"

namespace pyblockgrid {

struct Block {
    Vec2d center;
    double angle;       // angular extent
    double length;      // inner radius (r)
    double r0;          // outer radius
    BlockKind block_kind;
    int id_;
    int edge_i_index;   // -1 if absent
    int edge_j_index;   // -1 if absent
    BoundaryType boundary_type;
    int boundary_id;
    int vertex_id;      // -1 if absent

    Block(Vec2d center, double angle, double length, double r0,
          BlockKind kind, int id,
          int edge_i = -1, int edge_j = -1,
          BoundaryType btype = BoundaryType::Main,
          int bid = 0, int vid = -1);

    // Check which points fall inside this block
    BoolVec is_inside(const MatX2d& points, const BoolVec& point_mask,
                      const Polygon& poly) const;
};

} // namespace pyblockgrid
