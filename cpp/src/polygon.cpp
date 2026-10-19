#include "pyblockgrid/polygon.hpp"
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace pyblockgrid {

// ============== PolygonHole ==============

PolygonHole::PolygonHole(const MatX2d& verts,
                         const std::vector<std::vector<double>>& bc,
                         const std::vector<bool>& is_dir)
    : vertices(verts), boundary_conditions(bc), is_dirichlet(is_dir),
      n_vertices(static_cast<int>(verts.rows()))
{
    assert(verify_vertex_order() && "Hole vertices must be clockwise");
    compute_edges();
    compute_interior_angles();
    validate();
}

bool PolygonHole::verify_vertex_order() const {
    double signed_area = 0;
    for (int i = 0; i < n_vertices; ++i) {
        int j = (i + 1) % n_vertices;
        signed_area += vertices(i, 0) * vertices(j, 1)
                     - vertices(i, 1) * vertices(j, 0);
    }
    return signed_area < 0; // CW = negative
}

void PolygonHole::compute_edges() {
    edges.resize(n_vertices);
    for (int i = 0; i < n_vertices; ++i) {
        int j = (i + 1) % n_vertices;
        edges[i] = vertices.row(j).transpose() - vertices.row(i).transpose();
    }
}

void PolygonHole::compute_interior_angles() {
    interior_angles.resize(n_vertices);
    for (int i = 0; i < n_vertices; ++i) {
        int prev = (i - 1 + n_vertices) % n_vertices;
        int next = (i + 1) % n_vertices;
        Vec2d v1 = vertices.row(next).transpose() - vertices.row(i).transpose();
        Vec2d v2 = vertices.row(prev).transpose() - vertices.row(i).transpose();
        double cross = v1(0) * v2(1) - v1(1) * v2(0);
        double dot = v1.dot(v2);
        double angle = std::atan2(cross, dot);
        if (angle < 0) angle += 2 * M_PI;
        interior_angles[i] = angle;
    }
}

void PolygonHole::validate() const {
    if (n_vertices < 3)
        throw std::invalid_argument("Hole must have at least 3 vertices");
    if (static_cast<int>(boundary_conditions.size()) != n_vertices)
        throw std::invalid_argument("Boundary conditions count must match vertex count");
    if (static_cast<int>(is_dirichlet.size()) != n_vertices)
        throw std::invalid_argument("Dirichlet flags count must match vertex count");
}

double PolygonHole::area() const {
    double a = 0;
    for (int i = 0; i < n_vertices; ++i) {
        int j = (i + 1) % n_vertices;
        a += vertices(i, 0) * vertices(j, 1);
        a -= vertices(j, 0) * vertices(i, 1);
    }
    return std::abs(a) / 2.0;
}

bool PolygonHole::point_in_hole(double x, double y) const {
    // Ray tracing (matches Python exactly)
    bool inside = false;
    double p1x = vertices(0, 0), p1y = vertices(0, 1);
    for (int i = 1; i <= n_vertices; ++i) {
        double p2x = vertices(i % n_vertices, 0);
        double p2y = vertices(i % n_vertices, 1);
        if (y > std::min(p1y, p2y)) {
            if (y <= std::max(p1y, p2y)) {
                if (x <= std::max(p1x, p2x)) {
                    double xinters = 0;
                    if (p1y != p2y)
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x;
                    if (p1x == p2x || x <= xinters)
                        inside = !inside;
                }
            }
        }
        p1x = p2x;
        p1y = p2y;
    }
    return inside;
}

// ============== Polygon ==============

Polygon::Polygon(const MatX2d& verts) : vertices(verts) {
    assert(verify_vertex_order() && "Vertices must be counterclockwise");
    calculate_angles();
    calculate_edges();
    calculate_inner_connections();
}

void Polygon::calculate_angles() {
    int n = static_cast<int>(vertices.rows());
    angles.resize(n);
    for (int i = 0; i < n; ++i) {
        int prev = (i - 1 + n) % n;
        int next = (i + 1) % n;
        Vec2d v1 = vertices.row(next).transpose() - vertices.row(i).transpose();
        Vec2d v2 = vertices.row(prev).transpose() - vertices.row(i).transpose();
        double cross = v1(0) * v2(1) - v1(1) * v2(0);
        double dot = v1.dot(v2);
        double angle = std::atan2(cross, dot);
        if (angle < 0) angle += 2 * M_PI;
        angles[i] = angle;
    }
}

void Polygon::calculate_edges() {
    int n = static_cast<int>(vertices.rows());
    edges.resize(n);
    for (int i = 0; i < n; ++i) {
        int next = (i + 1) % n;
        edges[i] = vertices.row(next).transpose() - vertices.row(i).transpose();
    }
}

void Polygon::calculate_inner_connections() {
    int n = static_cast<int>(vertices.rows());
    inner_connections.clear();
    for (int i = 0; i < n; ++i) {
        for (int j = i + 2; j < n; ++j) {
            if (!(i == 0 && j == n - 1)) {
                inner_connections.emplace_back(
                    vertices.row(i).transpose(),
                    vertices.row(j).transpose());
            }
        }
    }
}

bool Polygon::verify_vertex_order() const {
    int n = static_cast<int>(vertices.rows());
    double signed_area = 0;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        signed_area += vertices(i, 0) * vertices(j, 1)
                     - vertices(i, 1) * vertices(j, 0);
    }
    return signed_area > 0; // CCW = positive
}

double Polygon::area() const {
    int n = static_cast<int>(vertices.rows());
    double a = 0;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        a += vertices(i, 0) * vertices(j, 1);
        a -= vertices(j, 0) * vertices(i, 1);
    }
    return std::abs(a) / 2.0;
}

double Polygon::area_with_holes() const {
    double total = area();
    for (const auto& hole : holes)
        total -= hole.area();
    return total;
}

double Polygon::perimeter() const {
    double p = 0;
    for (const auto& e : edges)
        p += e.norm();
    return p;
}

bool Polygon::verify_convexity() const {
    int n = static_cast<int>(vertices.rows());
    if (n < 3) return false;

    bool all_pos = true, all_neg = true;
    for (int i = 0; i < n; ++i) {
        int next = (i + 1) % n;
        int next2 = (i + 2) % n;
        Vec2d e1 = vertices.row(next).transpose() - vertices.row(i).transpose();
        Vec2d e2 = vertices.row(next2).transpose() - vertices.row(next).transpose();
        double cross = e1(0) * e2(1) - e1(1) * e2(0);
        if (cross < 0) all_pos = false;
        if (cross > 0) all_neg = false;
    }
    return all_pos || all_neg;
}

BoolVec Polygon::is_inside_main(const MatX2d& points) const {
    // Vectorized ray tracing for main polygon (matches Python exactly)
    int n_points = static_cast<int>(points.rows());
    int n_edges = static_cast<int>(vertices.rows());

    Eigen::VectorXi intersections = Eigen::VectorXi::Zero(n_points);

    for (int e = 0; e < n_edges; ++e) {
        int e_next = (e + 1) % n_edges;
        double es_y = vertices(e, 1);
        double ee_y = vertices(e_next, 1);
        double es_x = vertices(e, 0);
        double ee_x = vertices(e_next, 0);
        double v2_y = ee_y - es_y;
        double v2_x = ee_x - es_x;

        for (int p = 0; p < n_points; ++p) {
            double py = points(p, 1);
            double px = points(p, 0);

            bool above = py >= es_y;
            bool below = py < ee_y;
            bool above_flip = py >= ee_y;
            bool below_flip = py < es_y;

            bool possible = (above && below) || (above_flip && below_flip);
            if (!possible) continue;

            double v1_y = py - es_y;
            if (v2_y == 0) continue;
            double t = v1_y / v2_y;
            double intersect_x = es_x + t * v2_x;
            if (intersect_x > px)
                intersections(p)++;
        }
    }

    BoolVec inside(n_points);
    for (int p = 0; p < n_points; ++p)
        inside(p) = (intersections(p) % 2) == 1;
    return inside;
}

BoolVec Polygon::is_inside(const MatX2d& points, bool ignore_holes) const {
    BoolVec inside = is_inside_main(points);

    if (!ignore_holes) {
        for (const auto& hole : holes) {
            // Check each hole using the same ray tracing
            int n_points = static_cast<int>(points.rows());
            int n_edges = hole.n_vertices;

            Eigen::VectorXi intersections_h = Eigen::VectorXi::Zero(n_points);
            for (int e = 0; e < n_edges; ++e) {
                int e_next = (e + 1) % n_edges;
                double es_y = hole.vertices(e, 1);
                double ee_y = hole.vertices(e_next, 1);
                double es_x = hole.vertices(e, 0);
                double ee_x = hole.vertices(e_next, 0);
                double v2_y = ee_y - es_y;
                double v2_x = ee_x - es_x;

                for (int p = 0; p < n_points; ++p) {
                    if (!inside(p)) continue; // skip already-outside points
                    double py = points(p, 1);
                    double px = points(p, 0);
                    bool above = py >= es_y;
                    bool below = py < ee_y;
                    bool above_flip = py >= ee_y;
                    bool below_flip = py < es_y;
                    bool possible = (above && below) || (above_flip && below_flip);
                    if (!possible) continue;
                    double v1_y = py - es_y;
                    if (v2_y == 0) continue;
                    double t = v1_y / v2_y;
                    double intersect_x = es_x + t * v2_x;
                    if (intersect_x > px)
                        intersections_h(p)++;
                }
            }
            for (int p = 0; p < n_points; ++p) {
                if (inside(p) && (intersections_h(p) % 2 == 1))
                    inside(p) = false;
            }
        }
    }
    return inside;
}

bool Polygon::point_in_polygon_original(double x, double y) const {
    MatX2d pt(1, 2);
    pt(0, 0) = x;
    pt(0, 1) = y;
    return is_inside_main(pt)(0);
}

bool Polygon::point_in_polygon(double x, double y, bool ignore_holes) const {
    if (!point_in_polygon_original(x, y))
        return false;
    if (ignore_holes) return true;
    for (const auto& hole : holes) {
        if (hole.point_in_hole(x, y))
            return false;
    }
    return true;
}

PolygonHole& Polygon::add_hole(const MatX2d& hole_vertices,
                                const std::vector<std::vector<double>>& bc,
                                const std::vector<bool>& is_dir) {
    PolygonHole hole(hole_vertices, bc, is_dir);
    validate_hole_placement(hole);
    holes.push_back(std::move(hole));
    return holes.back();
}

void Polygon::validate_hole_placement(const PolygonHole& hole) const {
    for (int i = 0; i < hole.n_vertices; ++i) {
        if (!point_in_polygon_original(hole.vertices(i, 0), hole.vertices(i, 1)))
            throw std::invalid_argument("Hole vertex is outside main polygon");
    }
    for (const auto& existing : holes) {
        if (holes_overlap(hole, existing))
            throw std::invalid_argument("Holes cannot overlap");
    }
}

bool Polygon::holes_overlap(const PolygonHole& h1, const PolygonHole& h2) const {
    for (int i = 0; i < h1.n_vertices; ++i)
        if (h2.point_in_hole(h1.vertices(i, 0), h1.vertices(i, 1)))
            return true;
    for (int i = 0; i < h2.n_vertices; ++i)
        if (h1.point_in_hole(h2.vertices(i, 0), h2.vertices(i, 1)))
            return true;
    return false;
}

} // namespace pyblockgrid
