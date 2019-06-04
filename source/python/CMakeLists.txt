
################################################################################
#
#        Go100x (Python)
#
################################################################################

unset(CMAKE_CXX_CLANG_TIDY)

if(cxx_Wno_shadow)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-shadow")
endif()

#----------------------------------------------------------------------------
# PyBind11
#
pybind11_add_module(Go100x-python ${CMAKE_CURRENT_LIST_DIR}/go100x.cpp
    ${CMAKE_CURRENT_LIST_DIR}/go100x.hpp)

target_link_libraries(Go100x-python PRIVATE Go100x-library Go100x-submodules Go100x-compile-options)
target_include_directories(Go100x-python PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(Go100x-python SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/source/TiMemory/source/python/pybind11/include)
