#pragma once

#include "types.hpp"
#include <vector>
#include <string>

namespace pyblockgrid {

struct PolygonHole {
    MatX2d vertices;       // CW ordered
    std::vector<std::vector<double>> boundary_conditions;
    std::vector<bool> is_dirichlet;
    int n_vertices;
    std::vector<Vec2d> edges;
    std::vector<double> interior_angles;

    PolygonHole(const MatX2d& verts,
                const std::vector<std::vector<double>>& bc,
                const std::vector<bool>& is_dir);

    double area() const;
    bool point_in_hole(double x, double y) const;

private:
    bool verify_vertex_order() const;
    void compute_edges();
    void compute_interior_angles();
    void validate() const;
};

class Polygon {
public:
    MatX2d vertices;
    std::vector<double> angles;
    std::vector<Vec2d> edges;
    std::vector<std::pair<Vec2d, Vec2d>> inner_connections;
    std::vector<PolygonHole> holes;

    explicit Polygon(const MatX2d& verts);

    double area() const;
    double area_with_holes() const;
    double perimeter() const;
    bool verify_convexity() const;
    bool verify_vertex_order() const;

    // Vectorized point-in-polygon (handles holes)
    BoolVec is_inside(const MatX2d& points, bool ignore_holes = false) const;

    // Single-point check (handles holes)
    bool point_in_polygon(double x, double y, bool ignore_holes = false) const;

    PolygonHole& add_hole(const MatX2d& hole_vertices,
                          const std::vector<std::vector<double>>& bc,
                          const std::vector<bool>& is_dir);

private:
    void calculate_angles();
    void calculate_edges();
    void calculate_inner_connections();
    bool point_in_polygon_original(double x, double y) const;
    BoolVec is_inside_main(const MatX2d& points) const;
    void validate_hole_placement(const PolygonHole& hole) const;
    bool holes_overlap(const PolygonHole& h1, const PolygonHole& h2) const;
};

} // namespace pyblockgrid
