################################################################################
#
#        Handles the build options
#
################################################################################

include_guard(GLOBAL)

include(GNUInstallDirs)
include(Compilers)

# ---------------------------------------------------------------------------- #
# set the compiler flags
add_cxx_flag_if_avail("-W")
add_cxx_flag_if_avail("-Wall")
add_cxx_flag_if_avail("-Wextra")
add_cxx_flag_if_avail("-Wno-shadow")
add_cxx_flag_if_avail("-Wno-attributes")
add_cxx_flag_if_avail("-Wno-unused-value")
add_cxx_flag_if_avail("-Wno-unknown-pragmas")
add_cxx_flag_if_avail("-Wno-unused-parameter")
add_cxx_flag_if_avail("-Wunused-but-set-parameter")
add_cxx_flag_if_avail("-faligned-new")

add_cxx_flag_if_avail("-fopenmp-simd")
add_cxx_flag_if_avail("-fp-model=precise")

if(USE_SANITIZER)
    add_cxx_flag_if_avail("-fsanitize=${SANITIZER_TYPE}")
endif()

# ---------------------------------------------------------------------------- #
# user customization
to_list(_CXXFLAGS "${CXXFLAGS};$ENV{CXXFLAGS}")
foreach(_FLAG ${_CXXFLAGS})
    add_cxx_flag_if_avail("${_FLAG}")
endforeach()

