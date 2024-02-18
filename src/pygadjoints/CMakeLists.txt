set(PYGADJOINTS_SRCS py_elasticity.cpp py_diffusion.cpp pygadjoints.cpp
                     custom_expression.hpp)

# add target
pybind11_add_module(pygadjoints ${PYGADJOINTS_SRCS})
target_include_directories(pygadjoints PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_definitions(pygadjoints PRIVATE ${PYGADJOINTS_DEFS})

# def
if(PYGADJOINTS_USE_OPENMP)
  if(("x${CMAKE_C_COMPILER_ID}" STREQUAL "xAppleClang"
      OR "x${CMAKE_C_COMPILER_ID}" STREQUAL "xClang"
      AND ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
     OR ("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xAppleClang"
         OR "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xClang"
         AND ${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))
    find_path(
      OpenMP_C_INCLUDE_DIR
      NAMES "omp.h"
      PATHS /usr/local /opt /opt/local/include/libomp /opt/homebrew/opt/libomp
      PATH_SUFFIXES include)
    find_path(
      OpenMP_CXX_INCLUDE_DIR
      NAMES "omp.h"
      PATHS /usr/local /opt /opt/local/include/libomp /opt/homebrew/opt/libomp
      PATH_SUFFIXES include)
    find_library(
      OpenMP_libomp_LIBRARY
      NAMES "omp"
      PATHS /usr/local /opt /opt/local/lib/libomp /opt/homebrew/opt/libomp
      PATH_SUFFIXES lib)
    set(CMAKE_C_FLAGS
        "${CMAKE_C_FLAGS} -Xclang -fopenmp -I${OpenMP_C_INCLUDE_DIR}")
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -Xclang -fopenmp -I${OpenMP_CXX_INCLUDE_DIR} -DPYGADJOINTS_USE_OPENMP"
    )
    set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_libomp_LIBRARY}")
    set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_libomp_LIBRARY}")
  else()
    set(PYGADJOINTS_FLAGS ${PYGADJOINTS_FLAGS} -fopenmp)
    find_package(OpenMP REQUIRED)
    target_compile_definitions(pygadjoints PRIVATE PYGADJOINTS_USE_OPENMP)
    # target_link_libraries(pygadjoints PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(pygadjoints PRIVATE ${OpenMP_C_FLAGS}
                                               ${OpenMP_CXX_FLAGS})
  endif()
endif()

target_compile_options(pygadjoints PRIVATE ${PYGADJOINTS_FLAGS}
                                           ${PYGADJOINTS_OPTIMIZATION_FLAGS})
target_compile_features(pygadjoints PUBLIC cxx_std_17)

# link gismo
target_link_libraries(pygadjoints PRIVATE gismo)
target_include_directories(pygadjoints PRIVATE ${GISMO_INCLUDE_DIRS})
