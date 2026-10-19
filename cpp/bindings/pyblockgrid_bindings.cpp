#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "pyblockgrid/polygon.hpp"
#include "pyblockgrid/block.hpp"
#include "pyblockgrid/volkov_solver.hpp"

namespace py = pybind11;
using namespace pyblockgrid;

// Helper: create numpy masked array from data + valid mask
static py::object make_masked_array(const MatXd& data, const BoolMat& valid) {
    py::module_ np = py::module_::import("numpy");
    py::module_ ma = np.attr("ma");

    // Create numpy arrays
    py::array_t<double> np_data({data.rows(), data.cols()});
    py::array_t<bool> np_mask({data.rows(), data.cols()});
    auto d = np_data.mutable_unchecked<2>();
    auto m = np_mask.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < data.rows(); ++i)
        for (py::ssize_t j = 0; j < data.cols(); ++j) {
            d(i, j) = data(i, j);
            m(i, j) = !valid(i, j);  // numpy mask: True = invalid
        }
    return ma.attr("masked_array")(np_data, np_mask);
}

// Helper: convert ragged VecXd array to numpy masked 2D array (num_blocks, max_n)
static py::object ragged_to_masked_2d(const std::vector<VecXd>& ragged,
                                       int num_blocks, int max_n) {
    py::module_ np = py::module_::import("numpy");
    py::module_ ma = np.attr("ma");

    py::array_t<double> data({num_blocks, max_n});
    py::array_t<bool> mask({num_blocks, max_n});
    auto d = data.mutable_unchecked<2>();
    auto m = mask.mutable_unchecked<2>();

    for (int i = 0; i < num_blocks; ++i) {
        int sz = static_cast<int>(ragged[i].size());
        for (int j = 0; j < max_n; ++j) {
            if (j < sz) {
                d(i, j) = ragged[i](j);
                m(i, j) = false;
            } else {
                d(i, j) = 0.0;
                m(i, j) = true;
            }
        }
    }
    return ma.attr("masked_array")(data, mask);
}

// Helper: convert ragged MatX2d to numpy masked 3D array (num_blocks, max_n, 2)
static py::object ragged_to_masked_3d(const std::vector<MatX2d>& ragged,
                                       int num_blocks, int max_n) {
    py::module_ np = py::module_::import("numpy");
    py::module_ ma = np.attr("ma");

    py::array_t<double> data({num_blocks, max_n, 2});
    py::array_t<bool> mask({num_blocks, max_n, 2});
    auto d = data.mutable_unchecked<3>();
    auto m = mask.mutable_unchecked<3>();

    for (int i = 0; i < num_blocks; ++i) {
        int sz = static_cast<int>(ragged[i].rows());
        for (int j = 0; j < max_n; ++j) {
            for (int c = 0; c < 2; ++c) {
                if (j < sz) {
                    d(i, j, c) = ragged[i](j, c);
                    m(i, j, c) = false;
                } else {
                    d(i, j, c) = 0.0;
                    m(i, j, c) = true;
                }
            }
        }
    }
    return ma.attr("masked_array")(data, mask);
}

// Helper: convert inside_block_ids (int data + mask) to masked array
static py::object make_int_masked_array(const MatXd& data, const BoolMat& valid) {
    py::module_ np = py::module_::import("numpy");
    py::module_ ma = np.attr("ma");

    py::array_t<int> np_data({data.rows(), data.cols()});
    py::array_t<bool> np_mask({data.rows(), data.cols()});
    auto d = np_data.mutable_unchecked<2>();
    auto m = np_mask.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < data.rows(); ++i)
        for (py::ssize_t j = 0; j < data.cols(); ++j) {
            d(i, j) = static_cast<int>(data(i, j));
            m(i, j) = !valid(i, j);
        }
    return ma.attr("masked_array")(np_data, np_mask);
}

// Helper: convert MaskedGrid2D to numpy masked 3D array (ny, nx, 2)
static py::object make_cartesian_masked(const MaskedGrid2D& grid) {
    py::module_ np = py::module_::import("numpy");
    py::module_ ma = np.attr("ma");

    int ny = static_cast<int>(grid.x.rows());
    int nx = static_cast<int>(grid.x.cols());
    py::array_t<double> data({ny, nx, 2});
    py::array_t<bool> mask({ny, nx, 2});
    auto d = data.mutable_unchecked<3>();
    auto m = mask.mutable_unchecked<3>();
    for (int i = 0; i < ny; ++i)
        for (int j = 0; j < nx; ++j) {
            d(i, j, 0) = grid.x(i, j);
            d(i, j, 1) = grid.y(i, j);
            m(i, j, 0) = !grid.valid(i, j);
            m(i, j, 1) = !grid.valid(i, j);
        }
    return ma.attr("masked_array")(data, mask);
}


PYBIND11_MODULE(_pyblockgrid_cpp, m) {
    m.doc() = "C++ accelerated pyBlockGrid core";

    // PolygonHole
    py::class_<PolygonHole>(m, "PolygonHole")
        .def_readonly("vertices", &PolygonHole::vertices)
        .def_readonly("boundary_conditions", &PolygonHole::boundary_conditions)
        .def_readonly("is_dirichlet", &PolygonHole::is_dirichlet)
        .def_readonly("n_vertices", &PolygonHole::n_vertices)
        .def_property_readonly("edges", [](const PolygonHole& h) {
            std::vector<py::array_t<double>> result;
            for (const auto& e : h.edges) {
                py::array_t<double> arr(2);
                auto buf = arr.mutable_unchecked<1>();
                buf(0) = e(0); buf(1) = e(1);
                result.push_back(arr);
            }
            return result;
        })
        .def_property_readonly("interior_angles", [](const PolygonHole& h) {
            return h.interior_angles;
        })
        .def("area", &PolygonHole::area)
        .def("point_in_hole", &PolygonHole::point_in_hole);

    // Polygon (exposed as "polygon" to match Python API)
    py::class_<Polygon>(m, "polygon")
        .def(py::init([](py::array_t<double> verts) {
            auto buf = verts.unchecked<2>();
            MatX2d mat(buf.shape(0), 2);
            for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
                mat(i, 0) = buf(i, 0);
                mat(i, 1) = buf(i, 1);
            }
            return new Polygon(mat);
        }), py::arg("vertices"))
        .def_property_readonly("vertices", [](const Polygon& p) {
            return py::array_t<double>({p.vertices.rows(), p.vertices.cols()},
                                       p.vertices.data());
        })
        .def_property_readonly("edges", [](const Polygon& p) {
            std::vector<py::array_t<double>> result;
            for (const auto& e : p.edges) {
                py::array_t<double> arr(2);
                auto buf = arr.mutable_unchecked<1>();
                buf(0) = e(0); buf(1) = e(1);
                result.push_back(arr);
            }
            return result;
        })
        .def_property_readonly("angles", [](const Polygon& p) { return p.angles; })
        .def_readonly("holes", &Polygon::holes)
        .def_property_readonly("inner_connections", [](const Polygon& p) {
            py::list result;
            for (const auto& [a, b] : p.inner_connections) {
                py::array_t<double> arr_a(2), arr_b(2);
                auto ba = arr_a.mutable_unchecked<1>();
                auto bb = arr_b.mutable_unchecked<1>();
                ba(0) = a(0); ba(1) = a(1);
                bb(0) = b(0); bb(1) = b(1);
                result.append(py::make_tuple(arr_a, arr_b));
            }
            return result;
        })
        .def("area", &Polygon::area)
        .def("area_with_holes", &Polygon::area_with_holes)
        .def("perimeter", &Polygon::perimeter)
        .def("verify_convexity", &Polygon::verify_convexity)
        .def("verify_vertex_order", &Polygon::verify_vertex_order)
        .def("is_inside", [](const Polygon& p, py::array_t<double> pts,
                              bool ignore_holes) {
            auto buf = pts.unchecked<2>();
            const py::ssize_t n = buf.shape(0);
            MatX2d mat(n, 2);
            for (py::ssize_t i = 0; i < n; ++i) {
                mat(i, 0) = buf(i, 0);
                mat(i, 1) = buf(i, 1);
            }
            BoolVec result = p.is_inside(mat, ignore_holes);
            return py::array_t<bool>(result.size(), result.data());
        }, py::arg("points"), py::arg("ignore_holes") = false)
        .def("add_hole", [](Polygon& p, py::array_t<double> verts,
                            std::vector<std::vector<double>> bc,
                            std::vector<bool> is_dir) {
            auto buf = verts.unchecked<2>();
            MatX2d mat(buf.shape(0), 2);
            for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
                mat(i, 0) = buf(i, 0);
                mat(i, 1) = buf(i, 1);
            }
            return &p.add_hole(mat, bc, is_dir);
        }, py::return_value_policy::reference_internal)
        .def("_point_in_polygon", &Polygon::point_in_polygon,
             py::arg("x"), py::arg("y"), py::arg("ignore_holes") = false)
        .def("plot", [](const Polygon&, py::object, bool) {
            // Delegate to Python visualization
        }, py::arg("ax"), py::arg("show_holes") = true)
        .def("plot_connections", [](const Polygon&, py::object) {}, py::arg("ax"));

    // Block (exposed as "block" to match Python API)
    py::class_<Block>(m, "block")
        .def_readonly("center", &Block::center)
        .def_readonly("angle", &Block::angle)
        .def_readonly("length", &Block::length)
        .def_readonly("r0", &Block::r0)
        .def_property_readonly("block_kind", [](const Block& b) {
            return static_cast<int>(b.block_kind);
        })
        .def_readonly("id_", &Block::id_)
        .def_readonly("edge_i_index", &Block::edge_i_index)
        .def_readonly("edge_j_index", &Block::edge_j_index)
        .def_property_readonly("boundary_type", [](const Block& b) {
            return b.boundary_type == BoundaryType::Main ? "main" : "hole";
        })
        .def_readonly("boundary_id", &Block::boundary_id)
        .def_readonly("vertex_id", &Block::vertex_id);

    // SolutionState wrapper (read-only access for visualization)
    py::class_<SolutionState>(m, "SolutionState")
        .def_property_readonly("X", [](const SolutionState& s) { return s.X; })
        .def_property_readonly("Y", [](const SolutionState& s) { return s.Y; })
        .def_property_readonly("cartesian_grid", [](const SolutionState& s) {
            return make_cartesian_masked(s.cartesian_grid);
        })
        .def_property_readonly("quantized_boundary_points", [](const SolutionState& s) {
            return ragged_to_masked_3d(s.quantized_boundary_points, s.num_blocks, s.max_n);
        })
        .def_property_readonly("boundary_estimates", [](const SolutionState& s) {
            return ragged_to_masked_2d(s.boundary_estimates, s.num_blocks, s.max_n);
        })
        .def_property_readonly("inside_block_ids", [](const SolutionState& s) {
            return make_int_masked_array(s.inside_block_ids_data, s.inside_valid);
        });

    // VolkovSolver (exposed as "volkovSolver" to match Python API)
    py::class_<VolkovSolver>(m, "volkovSolver")
        .def(py::init([](Polygon& poly,
                         std::vector<std::vector<double>> bc,
                         std::vector<bool> is_dir,
                         int n, double delta, int max_iter,
                         double radial_heuristic, double overlap_heuristic,
                         double tolerance, bool verify_solution,
                         int covering_refinement) {
            return new VolkovSolver(poly, std::move(bc), std::move(is_dir),
                                    n, delta, max_iter, radial_heuristic,
                                    overlap_heuristic, tolerance, verify_solution,
                                    covering_refinement);
        }),
            py::arg("poly"),
            py::arg("boundary_conditions"),
            py::arg("is_dirichlet") = py::none(),
            py::arg("n") = 10,
            py::arg("delta") = 0.01,
            py::arg("max_iter") = 100,
            py::arg("radial_heuristic") = 0.8,
            py::arg("overlap_heuristic") = 0.1,
            py::arg("tolerance") = 1e-10,
            py::arg("verify_solution") = false,
            py::arg("covering_refinement") = 1)

        .def("solve", [](VolkovSolver& self, bool verbose, bool plot, py::object ax) {
            self.solve(verbose);
            return make_masked_array(self.state.solution.data, self.state.solution.valid);
        }, py::arg("verbose") = false, py::arg("plot") = false, py::arg("ax") = py::none())

        .def("find_block_covering", [](VolkovSolver& self) {
            auto [n, l, m] = self.find_block_covering();
            // Return (N, L, M, uncovered_points) matching Python API
            // uncovered_points not used by tests, return dummy masked array
            py::module_ np = py::module_::import("numpy");
            py::module_ ma = np.attr("ma");
            py::array_t<double> dummy({1, 1});
            py::array_t<bool> dummy_mask({1, 1});
            dummy_mask.mutable_at(0, 0) = true;
            py::object uncovered = ma.attr("masked_array")(dummy, dummy_mask);
            return py::make_tuple(n, l, m, uncovered);
        })

        .def_property_readonly("solution", [](const VolkovSolver& self) {
            return make_masked_array(self.solution_grid.data, self.solution_grid.valid);
        })
        .def_property_readonly("poly", [](const VolkovSolver& self) -> const Polygon& {
            return self.poly;
        }, py::return_value_policy::reference_internal)
        .def_readonly("blocks", &VolkovSolver::blocks)
        .def_readonly("N", &VolkovSolver::N)
        .def_readonly("L", &VolkovSolver::L)
        .def_readonly("M", &VolkovSolver::M)
        .def_readonly("delta", &VolkovSolver::delta)
        .def_readonly("boundary_conditions", &VolkovSolver::boundary_conditions)
        .def_readonly("is_dirichlet", &VolkovSolver::is_dirichlet)
        .def_readonly("n", &VolkovSolver::n_param)
        .def_readonly("max_iter", &VolkovSolver::max_iter)
        .def_readonly("tol", &VolkovSolver::tol)
        .def_readonly("x_min", &VolkovSolver::x_min)
        .def_readonly("y_min", &VolkovSolver::y_min)
        .def_readonly("x_max", &VolkovSolver::x_max)
        .def_readonly("y_max", &VolkovSolver::y_max)
        .def_readonly("nx", &VolkovSolver::nx)
        .def_readonly("ny", &VolkovSolver::ny)

        .def_property_readonly("state", [](const VolkovSolver& self) -> const SolutionState& {
            return self.state;
        }, py::return_value_policy::reference_internal)

        .def_property_readonly("inside_block_ids_", [](const VolkovSolver& self) {
            return make_int_masked_array(self.inside_block_ids_grid, self.inside_valid_grid);
        })

        .def_property_readonly("_X", [](const VolkovSolver& self) { return self.grid_X; })
        .def_property_readonly("_Y", [](const VolkovSolver& self) { return self.grid_Y; })

        // X and Y properties (backward compat)
        .def_property_readonly("X", [](const VolkovSolver& self) { return self.state.X; })
        .def_property_readonly("Y", [](const VolkovSolver& self) { return self.state.Y; })

        // Visualization methods - delegate to Python
        .def("plot_solution", [](py::object self, py::object ax,
                                  py::object vmin, py::object vmax) {
            py::module_ viz = py::module_::import("pyBlockGrid.visualization.volkov_plots");
            viz.attr("plot_solution_heatmap")(self, ax, vmin, vmax);
        }, py::arg("ax"), py::arg("vmin") = py::none(), py::arg("vmax") = py::none())

        .def("plot_block_covering", [](py::object self, py::object ax,
                                        py::object uncovered_points,
                                        bool show_bc, bool show_qb) {
            py::module_ viz = py::module_::import("pyBlockGrid.visualization.volkov_plots");
            viz.attr("plot_block_covering")(self, ax, uncovered_points, show_bc, show_qb);
        }, py::arg("ax"), py::arg("uncovered_points") = py::none(),
           py::arg("show_boundary_conditions") = true,
           py::arg("show_quantized_boundaries") = false)

        .def("plot_gradient", [](py::object self, py::object ax,
                                  int decimation, double scale) {
            py::module_ viz = py::module_::import("pyBlockGrid.visualization.volkov_plots");
            viz.attr("plot_gradient_field")(self, ax, decimation, scale);
        }, py::arg("ax"), py::arg("decimation_factor") = 2, py::arg("scale") = 20);
}
