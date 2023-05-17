set(PYGADJOINTS_SRCS
        init/py_eleasticiy.cpp
        pygadjoints.cpp)

# add target
pybind11_add_module(pygadjoints ${PYGADJOINTS_SRCS})
target_include_directories(pygadjoints PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_definitions(pygadjoints PRIVATE ${PYGADJOINTS_DEFS})
target_compile_options(pygadjoints
    PRIVATE
        ${PYGADJOINTS_FLAGS}
        ${PYGADJOINTS_OPTIMIZATION_FLAGS})
target_compile_features(pygadjoints PUBLIC cxx_std_14)

# link splinepy
target_link_libraries(pygadjoints PRIVATE gismo)

# configure config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${version_config}" COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/config.cmake.in"
    "${project_config}"
    INSTALL_DESTINATION "${cfg_dest}"
)

install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/src/pygadjoints"
    DESTINATION ${incl_dest}
    FILES_MATCHING PATTERN "*.hpp"
)
