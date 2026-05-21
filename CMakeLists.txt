cmake_minimum_required(VERSION 3.23)

project(
  external_solvers
  VERSION 0.1.0
  DESCRIPTION "External implementations of the sparse solver interface"
  LANGUAGES CXX
)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(
  SPARSE_SOLVER_INTERFACE_ROOT
  "/home/reidatcheson/sparse_linear_algebra/installs/sparse_solver_interface/0.1.0"
  CACHE PATH
  "Root directory of the sparse_solver_interface installation"
)

find_path(
  SPARSE_SOLVER_INTERFACE_INCLUDE_DIR
  NAMES sparse_solver_interface.hpp sparse_solver_interface_plugin.hpp
  HINTS "${SPARSE_SOLVER_INTERFACE_ROOT}"
  PATH_SUFFIXES include
  REQUIRED
)

add_library(sparse_solver_interface_headers INTERFACE)
add_library(sparse_solver_interface::headers ALIAS sparse_solver_interface_headers)
target_include_directories(
  sparse_solver_interface_headers
  INTERFACE
    "$<BUILD_INTERFACE:${SPARSE_SOLVER_INTERFACE_INCLUDE_DIR}>"
)
target_compile_features(sparse_solver_interface_headers INTERFACE cxx_std_20)

add_subdirectory(solvers)
