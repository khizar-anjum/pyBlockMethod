cmake_minimum_required(VERSION 3.13.4)

find_package(deal.II 9.3 REQUIRED
  HINTS ${DEAL_II_DIR} /usr/share/cmake/deal.II)
deal_ii_initialize_cached_variables()

project(dealii_bench CXX)

add_executable(hp_laplace hp_laplace.cc)
deal_ii_setup_target(hp_laplace)
