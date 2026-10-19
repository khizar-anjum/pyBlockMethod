cmake_minimum_required(VERSION 3.18)
project(pyblockgrid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Eigen3 (system-installed)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Find pybind11. Require >= 2.12: earlier versions corrupt py::array_t<bool>
# outputs under NumPy 2.0 (silently returns all-True), which broke
# polygon.is_inside for batched point queries. Point cmake at a modern pybind11
# with -Dpybind11_DIR=$(python -m pybind11 --cmakedir).
find_package(pybind11 2.12 REQUIRED)

# Core static library (no Python dependency)
add_library(pyblockgrid_core STATIC
    src/polygon.cpp
    src/block.cpp
    src/coordinate_transforms.cpp
    src/carrier_functions.cpp
    src/poisson_kernels.cpp
    src/block_covering.cpp
    src/solution_state.cpp
    src/boundary_estimator.cpp
    src/interior_solver.cpp
    src/volkov_solver.cpp
)
target_include_directories(pyblockgrid_core PUBLIC include)
target_link_libraries(pyblockgrid_core PUBLIC Eigen3::Eigen)

# pybind11 Python extension module
pybind11_add_module(_pyblockgrid_cpp bindings/pyblockgrid_bindings.cpp)
target_link_libraries(_pyblockgrid_cpp PRIVATE pyblockgrid_core)

# Install the extension module next to the Python package
install(TARGETS _pyblockgrid_cpp DESTINATION .)
