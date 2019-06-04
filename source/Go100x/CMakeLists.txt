
glob_files(OUTPUT_VAR headers EXTENSIONS h hpp)
glob_files(OUTPUT_VAR sources DIRECTORIES src EXTENSIONS cpp cu)

CREATE_LIBRARY(
    TARGET_NAME             Go100x-library
    OUTPUT_NAME             Go100x
    TYPE                    SHARED
    HEADERS                 ${headers}
    SOURCES                 ${sources}
    INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR}
    LINK_LIBRARIES          Go100x-compile-options Go100x-submodules Go100x-cuda Go100x-gperf
)
