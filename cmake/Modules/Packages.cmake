#----------------------------------------------------------------------------------------#
#
#                                   CUDA
#
#----------------------------------------------------------------------------------------#
find_package(CUDA QUIET)

add_interface_library(Go100x-cuda)

target_compile_definitions(Go100x-cuda INTERFACE USE_CUDA)
target_include_directories(Go100x-cuda INTERFACE ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(Go100x-cuda PROPERTIES
    INTERFACE_CUDA_STANDARD                 ${CMAKE_CUDA_STANDARD}
    INTERFACE_CUDA_STANDARD_REQUIRED        ${CMAKE_CUDA_STANDARD_REQUIRED}
    INTERFACE_CUDA_RESOLVE_DEVICE_SYMBOLS   ON
    INTERFACE_CUDA_SEPARABLE_COMPILATION    ON)

set(CUDA_GENERIC_ARCH "version")
set(CUDA_ARCHITECTURES version kepler tesla maxwell pascal volta turing)
set(CUDA_ARCH "${CUDA_GENERIC_ARCH}" CACHE STRING "CUDA architecture (options: ${CUDA_ARCHITECTURES})")
add_feature(CUDA_ARCH "CUDA architecture (options: ${CUDA_ARCHITECTURES})")
set_property(CACHE CUDA_ARCH PROPERTY STRINGS ${CUDA_ARCHITECTURES})

set(cuda_kepler_arch    30)
set(cuda_tesla_arch     35)
set(cuda_maxwell_arch   50)
set(cuda_pascal_arch    60)
set(cuda_volta_arch     70)
set(cuda_turing_arch    75)

if(NOT "${CUDA_ARCH}" STREQUAL "${CUDA_GENERIC_ARCH}")
    if(NOT "${CUDA_ARCH}" IN_LIST CUDA_ARCHITECTURES)
        message(WARNING "CUDA architecture \"${CUDA_ARCH}\" not known. Options: ${CUDA_ARCH}")
        unset(CUDA_ARCH CACHE)
        set(CUDA_ARCH "${CUDA_GENERIC_ARCH}")
    else()
        set(_ARCH_NUM ${cuda_${CUDA_ARCH}_arch})
    endif()
endif()

add_interface_library(Go100x-cuda-7)
target_compile_options(Go100x-cuda-7 INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
    $<IF:$<STREQUAL:${CUDA_ARCH},${CUDA_GENERIC_ARCH}>,-arch=sm_30,-arch=sm_${_ARCH_NUM}>
    -gencode=arch=compute_20,code=sm_20
    -gencode=arch=compute_30,code=sm_30
    -gencode=arch=compute_50,code=sm_50
    -gencode=arch=compute_52,code=sm_52
    -gencode=arch=compute_52,code=compute_52
    >)

add_interface_library(Go100x-cuda-8)
target_compile_options(Go100x-cuda-8 INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
    $<IF:$<STREQUAL:${CUDA_ARCH},${CUDA_GENERIC_ARCH}>,-arch=sm_30,-arch=sm_${_ARCH_NUM}>
    -gencode=arch=compute_20,code=sm_20
    -gencode=arch=compute_30,code=sm_30
    -gencode=arch=compute_50,code=sm_50
    -gencode=arch=compute_52,code=sm_52
    -gencode=arch=compute_60,code=sm_60
    -gencode=arch=compute_61,code=sm_61
    -gencode=arch=compute_61,code=compute_61
    >)

add_interface_library(Go100x-cuda-9)
target_compile_options(Go100x-cuda-9 INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
    $<IF:$<STREQUAL:${CUDA_ARCH},${CUDA_GENERIC_ARCH}>,-arch=sm_50,-arch=sm_${_ARCH_NUM}>
    -gencode=arch=compute_50,code=sm_50
    -gencode=arch=compute_52,code=sm_52
    -gencode=arch=compute_60,code=sm_60
    -gencode=arch=compute_61,code=sm_61
    -gencode=arch=compute_70,code=sm_70
    -gencode=arch=compute_70,code=compute_70
    >)

add_interface_library(Go100x-cuda-10)
target_compile_options(Go100x-cuda-10 INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
    $<IF:$<STREQUAL:"${CUDA_ARCH}","${CUDA_GENERIC_ARCH}">,-arch=sm_50,-arch=sm_${_ARCH_NUM}>
    -gencode=arch=compute_50,code=sm_50
    -gencode=arch=compute_52,code=sm_52
    -gencode=arch=compute_60,code=sm_60
    -gencode=arch=compute_61,code=sm_61
    -gencode=arch=compute_70,code=sm_70
    -gencode=arch=compute_75,code=sm_75
    -gencode=arch=compute_75,code=compute_75
    >)

string(REPLACE "." ";" CUDA_MAJOR_VERSION "${CUDA_VERSION}")
list(GET CUDA_MAJOR_VERSION 0 CUDA_MAJOR_VERSION)

if(CUDA_MAJOR_VERSION VERSION_GREATER 10 OR CUDA_MAJOR_VERSION MATCHES 10)
    target_link_libraries(Go100x-cuda INTERFACE Go100x-cuda-10)
elseif(CUDA_MAJOR_VERSION MATCHES 9)
    target_link_libraries(Go100x-cuda INTERFACE Go100x-cuda-9)
elseif(CUDA_MAJOR_VERSION MATCHES 8)
    target_link_libraries(Go100x-cuda INTERFACE Go100x-cuda-8)
elseif(CUDA_MAJOR_VERSION MATCHES 7)
    target_link_libraries(Go100x-cuda INTERFACE Go100x-cuda-7)
else()
    target_link_libraries(Go100x-cuda INTERFACE Go100x-cuda-7)
endif()

#   30, 32      + Kepler support
#               + Unified memory programming
#   35          + Dynamic parallelism support
#   50, 52, 53  + Maxwell support
#   60, 61, 62  + Pascal support
#   70, 72      + Volta support
#   75          + Turing support

target_compile_options(Go100x-cuda INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>)

if(NOT WIN32)
    target_compile_options(Go100x-cuda INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-bindir=${CMAKE_CXX_COMPILER}>)
endif()

target_include_directories(Go100x-cuda INTERFACE ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(Go100x-cuda PROPERTIES
    INTERFACE_CUDA_STANDARD 14
    INTERFACE_CUDA_STANDARD_REQUIRED ON
    INTERFACE_CUDA_RESOLVE_DEVICE_SYMBOLS ON
    INTERFACE_CUDA_SEPARABLE_COMPILATION ON)


#----------------------------------------------------------------------------------------#
#
#                               Google PerfTools
#
#----------------------------------------------------------------------------------------#

add_interface_library(Go100x-gperf)
if(USE_GPERF)
    find_package(GPerfTools COMPONENTS profiler tcmalloc)

    if(GPerfTools_FOUND)
        target_compile_definitions(Go100x-gperf INTERFACE USE_GPERF)
        target_include_directories(Go100x-gperf INTERFACE ${GPerfTools_INCLUDE_DIRS})
        target_link_libraries(Go100x-gperf INTERFACE ${GPerfTools_LIBRARIES})
    else()
        set(USE_GPERF OFF)
        message(WARNING "GPerfTools package not found!")
    endif()
endif()


set(${PROJECT_NAME}_PROPERTIES
    C_STANDARD                  ${CMAKE_C_STANDARD}
    C_STANDARD_REQUIRED         ${CMAKE_C_STANDARD_REQUIRED}
    CXX_STANDARD                ${CMAKE_CXX_STANDARD}
    CXX_STANDARD_REQUIRED       ${CMAKE_CXX_STANDARD_REQUIRED}
    ${CUDA_PROPERTIES}
)