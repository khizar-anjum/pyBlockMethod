cmake_minimum_required(VERSION 3.13.4)

project(spectral_bench CXX)

# C++ reimplementations of the Lightning and AAA-LS spectral Laplace solvers so
# the benchmark comparison is like-for-like C++ (reviewer comment R1.4). Eigen
# provides the dense linear algebra; LAPACK provides the complex generalized
# eigensolver (zggev) used by AAA-LS for pole computation.
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(LAPACK REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(lightning_laplace lightning_laplace.cc)
target_link_libraries(lightning_laplace PRIVATE Eigen3::Eigen)

add_executable(aaals_laplace aaals_laplace.cc)
target_link_libraries(aaals_laplace PRIVATE Eigen3::Eigen ${LAPACK_LIBRARIES})
