option(
  EXTERNAL_SOLVERS_ENABLE_MKL_PARDISO
  "Configure the MKL PARDISO sparse_solver_interface plugin"
  OFF
)

set(
  MKL_ROOT
  "$ENV{HOME}/intel/oneapi/mkl/latest"
  CACHE PATH
  "Root directory of the Intel oneAPI MKL installation"
)

if(EXTERNAL_SOLVERS_ENABLE_MKL_PARDISO)
  find_path(
    MKL_INCLUDE_DIR
    NAMES mkl.h mkl_pardiso.h
    HINTS "${MKL_ROOT}"
    PATH_SUFFIXES include
    REQUIRED
  )
  find_library(
    MKL_RT_LIBRARY
    NAMES mkl_rt
    HINTS "${MKL_ROOT}"
    PATH_SUFFIXES lib
    REQUIRED
  )

  add_library(mkl_pardiso_solver MODULE mkl_pardiso_solver.cpp)
  target_include_directories(mkl_pardiso_solver PRIVATE "${MKL_INCLUDE_DIR}")
  target_link_libraries(
    mkl_pardiso_solver
    PRIVATE
      sparse_solver_interface::headers
      "${MKL_RT_LIBRARY}"
  )
  set_target_properties(
    mkl_pardiso_solver
    PROPERTIES
      PREFIX ""
      OUTPUT_NAME "mkl_pardiso_solver"
      BUILD_RPATH "${MKL_ROOT}/lib"
      INSTALL_RPATH "${MKL_ROOT}/lib"
  )
endif()

add_custom_target(mkl_pardiso_solver_skeleton SOURCES README.md)
