cmake_minimum_required(VERSION 3.1)
project(pygadjoints VERSION 0.0.0 LANGUAGES CXX)

# config
set(exe_dest "bin")
set(incl_dest "include")
set(lib_dest "lib")
set(cfg_dest "${lib_dest}/cmake/${PROJECT_NAME}")
set(gen_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(version_config "${gen_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${gen_dir}/${PROJECT_NAME}Config.cmake")
set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(namespace "${PROJECT_NAME}::")

# flags / defs
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(PYGAJOINTS_FLAGS -fPIC)
  set(PYGAJOINTS_OPTIMIZATION_FLAGS $<$<NOT:$<CONFIG:Debug>>:-O3>)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  set(PYGAJOINTS_FLAGS)
  set(PYGAJOINTS_OPTIMIZATION_FLAGS $<$<NOT:$<CONFIG:Debug>>:/O2>)
endif()

set(PYGAJOINTS_DEFS)

if(CMAKE_BUILD_TYPE MATCHES Debug)

# compiler specific flags
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(PYGAJOINTS_FLAGS ${PYGAJOINTS_FLAGS}
          -Wall -Wextra -Wpedantic -Wzero-as-null-pointer-constant -Wno-unused)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(PYGAJOINTS_FLAGS ${PYGAJOINTS_FLAGS}
          -Wall -Wextra -Wmost -Wextra -Wpedantic -Wunreachable-code
          -Wshadow -Wfloat-equal -Weffc++ -Wno-unused-parameter
          -Wno-unused-variable -Wzero-as-null-pointer-constant)
  else()
    message(WARNING
              "tetgenpy is only tested with GNU and Clang.")
  endif()
endif()

add_subdirectory(third_party)
add_subdirectory(src/pygadjoint)

# configure config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${version_config}" COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
    "cmake/config.cmake.in"
    "${project_config}"
    INSTALL_DESTINATION "${cfg_dest}"
)

install(
    FILES "${project_config}" "${version_config}"
    DESTINATION "${cfg_dest}"
)

install(
    EXPORT "${TARGETS_EXPORT_NAME}"
    NAMESPACE "${namespace}"
    DESTINATION "${cfg_dest}"
)
install(
    FILES ${CXX_HEADERS}
    DESTINATION ${incl_dest}
)
