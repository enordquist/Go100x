################################################################################
#
#        Handles the CMake options
#
################################################################################

include(MacroUtilities)
include(Compilers)

# features
add_feature(CMAKE_BUILD_TYPE "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)")
add_feature(CMAKE_INSTALL_PREFIX "Installation prefix")

# options (always available)
add_option(USE_GPERF "Enable Google perftools profiler" OFF)

# RPATH settings
set(_RPATH_LINK ON)
add_option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Hardcode installation rpath based on link path" ${_RPATH_LINK})

if(USE_GPERF)
    configure_file(${PROJECT_SOURCE_DIR}/tools/gperf-cpu-profile.sh
        ${PROJECT_BINARY_DIR}/gperf-cpu-profile.sh COPYONLY)
    configure_file(${PROJECT_SOURCE_DIR}/tools/gperf-heap-profile.sh
        ${PROJECT_BINARY_DIR}/gperf-heap-profile.sh COPYONLY)
endif()
