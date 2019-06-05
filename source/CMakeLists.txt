##########################################################################################
#                                                                                        #
#                                         General                                        #
#                                                                                        #
##########################################################################################
include(MacroUtilities)
include(BuildSettings)

set(PTL_USE_TBB OFF CACHE BOOL "Enable TBB backend for PTL")
set(PTL_USE_ARCH ${USE_ARCH} CACHE BOOL "" FORCE)
set(PTL_USE_GPERF ${USE_GPERF} CACHE BOOL "" FORCE)
set(PTL_USE_SANITIZER ${USE_SANITIZER} CACHE BOOL "" FORCE)
set(TIMEMORY_BUILD_C OFF CACHE BOOL "Build TiMemory C interface" FORCE)
set(TIMEMORY_BUILD_PYTHON ON CACHE BOOL "Build TiMemory Python interface")
set(TIMEMORY_USE_MPI OFF CACHE BOOL "Build TiMemory with MPI support" FORCE)
set(TIMEMORY_USE_CUPTI OFF CACHE BOOL "Build TiMemory with CUPTI support" FORCE)

if(USE_ARCH)
    set(PTL_USE_AVX512 ${USE_AVX512} CACHE BOOL "" FORCE)
endif()

#----------------------------------------------------------------------------------------#
# Submodule interface library
#
add_interface_library(Go100x-submodules)

#----------------------------------------------------------------------------------------#
# PTL submodule
#
checkout_git_submodule(RECURSIVE TEST_FILE CMakeLists.txt
    RELATIVE_PATH source/PTL WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
add_subdirectory(PTL)
list(APPEND ${PROJECT_NAME}_INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_LIST_DIR}/PTL/source)


#----------------------------------------------------------------------------------------#
# TiMemory submodule
#
checkout_git_submodule(RECURSIVE TEST_FILE CMakeLists.txt
    RELATIVE_PATH source/TiMemory WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
add_subdirectory(TiMemory)


#----------------------------------------------------------------------------------------#
# Submodule interface library
#
target_link_libraries(Go100x-submodules INTERFACE
    ptl-shared timemory-cxx-shared timemory-cupti-shared timemory-cuda)

target_include_directories(Go100x-submodules INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/PTL/source)


set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#----------------------------------------------------------------------------------------#
# Project code
#
add_subdirectory(Go100x)

#----------------------------------------------------------------------------------------#
# Python bindings
#
add_subdirectory(python)
