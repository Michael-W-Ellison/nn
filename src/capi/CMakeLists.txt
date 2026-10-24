# File: src/capi/CMakeLists.txt
#
# C API (dpan_c.h): one shared library that contains the agent and every
# module it uses. Only the dpan_* C functions are exported.

add_library(dpan_c SHARED dpan_c.cpp)
set_target_properties(dpan_c PROPERTIES
    EXPORT_NAME c_api
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
)
target_compile_definitions(dpan_c PRIVATE
    DPAN_C_BUILDING
    DPAN_VERSION_STRING="${PROJECT_VERSION}"
)
target_include_directories(dpan_c PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/capi>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/dpan/capi>
)
target_link_libraries(dpan_c PRIVATE dpan_agent)

# The static module libraries are built with default visibility; export
# only the C functions (ELF linkers: GNU ld, gold, lld)
if(UNIX AND NOT APPLE)
    target_link_options(dpan_c PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/dpan_c.map")
    set_property(TARGET dpan_c APPEND PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dpan_c.map)
endif()
add_library(dpan::c_api ALIAS dpan_c)
