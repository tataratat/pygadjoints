add_subdirectory(pybind11)

set(GISMO_BUILD_EXAMPLES OFF)
set(GISMO_WITH_OPENMP ${PYGADJOINTS_USE_OPENMP})
add_subdirectory(gismo)
