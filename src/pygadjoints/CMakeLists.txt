set(PYGADJOINTS_SRCS
        py_elasticity.cpp
        pygadjoints.cpp
        custom_expression.hpp)

# add target
pybind11_add_module(pygadjoints ${PYGADJOINTS_SRCS})
target_include_directories(pygadjoints PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_definitions(pygadjoints PRIVATE ${PYGADJOINTS_DEFS})
target_compile_options(pygadjoints
    PRIVATE
        ${PYGADJOINTS_FLAGS}
        ${PYGADJOINTS_OPTIMIZATION_FLAGS})
target_compile_features(pygadjoints PUBLIC cxx_std_17)

# link splinepy
target_link_libraries(pygadjoints PRIVATE gismo)
target_include_directories(pygadjoints PRIVATE ${GISMO_INCLUDE_DIRS})
