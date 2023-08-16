add_subdirectory(pybind11)

SET(CMAKE_BUILD_TYPE Release)
SET(GISMO_BUILD_EXAMPLES OFF)
SET(GISMO_WITH_OPENMP ON)
add_subdirectory(gismo)
