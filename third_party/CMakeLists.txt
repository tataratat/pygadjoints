add_subdirectory(pybind11)

set(CMAKE_BUILD_TYPE Release)
set(GISMO_BUILD_EXAMPLES OFF)
set(GISMO_WITH_OPENMP ${PYGADJOINTS_USE_OPENMP})
add_subdirectory(gismo)
