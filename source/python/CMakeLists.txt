
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
#pybind11_add_module(Go100x-python
#    ${CMAKE_CURRENT_LIST_DIR}/go100x.cpp
#    ${CMAKE_CURRENT_LIST_DIR}/go100x.hpp)

add_library(Go100x-python MODULE
    ${CMAKE_CURRENT_LIST_DIR}/go100x.cpp
    ${CMAKE_CURRENT_LIST_DIR}/go100x.hpp
)
add_dependencies(Go100x-python Go100x-library)
target_link_libraries(Go100x-python PUBLIC Go100x-library Go100x-cuda Go100x-submodules Go100x-compile-options)
target_link_libraries(Go100x-python PRIVATE pybind11::module)
target_include_directories(Go100x-python PRIVATE ${CMAKE_CURRENT_LIST_DIR}
    ${PROJECT_SOURCE_DIR}/source)
target_include_directories(Go100x-python SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/source/TiMemory/source/python/pybind11/include)

set_target_properties(Go100x-python PROPERTIES
    OUTPUT_NAME go100x
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/go100x
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/go100x
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/go100x
)

FILE(WRITE ${CMAKE_BINARY_DIR}/go100x/__init__.py
"
from __future__ import absolute_import
import ctypes
from ctypes.util import find_library
lib = ctypes.CDLL('libGo100x.so', mode=ctypes.RTLD_GLOBAL)
from go100x.go100x import *
")


