
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
set_target_properties(Go100x-library PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/go100x
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/go100x
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/go100x
)
