# File: python/CMakeLists.txt
#
# Python package dpan: python/dpan/__init__.py plus the extension dpan._core
# (pybind11). Both are placed in <build>/python/dpan, so
#   PYTHONPATH=<build>/python python3 -c "import dpan"
# works without installing. pybind11 is taken from the system if found,
# otherwise fetched.

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

# Let pybind11 use the FindPython result above (not the deprecated FindPythonLibs)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 2.10 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

set(DPAN_PYTHON_PACKAGE_DIR ${CMAKE_BINARY_DIR}/python/dpan)

pybind11_add_module(dpan_python dpan_module.cpp)
set_target_properties(dpan_python PROPERTIES
    OUTPUT_NAME _core
    LIBRARY_OUTPUT_DIRECTORY ${DPAN_PYTHON_PACKAGE_DIR}
)
target_compile_definitions(dpan_python PRIVATE DPAN_VERSION_STRING="${PROJECT_VERSION}")
target_link_libraries(dpan_python PRIVATE dpan_agent)

configure_file(dpan/__init__.py ${DPAN_PYTHON_PACKAGE_DIR}/__init__.py COPYONLY)

if(DPAN_INSTALL)
    set(DPAN_PYTHON_INSTALL_DIR "" CACHE PATH
        "Where to install the dpan Python package (empty: not installed)")
    if(DPAN_PYTHON_INSTALL_DIR)
        install(TARGETS dpan_python LIBRARY DESTINATION ${DPAN_PYTHON_INSTALL_DIR}/dpan)
        install(FILES dpan/__init__.py DESTINATION ${DPAN_PYTHON_INSTALL_DIR}/dpan)
    endif()
endif()

# Reference environments A-F and I and API tests through the bindings
if(DPAN_BUILD_TESTS)
    set(DPAN_PYTHON_TESTS
        test_api
        test_reference.ReferenceTest.test_a_stream_classification_with_drift
        test_reference.ReferenceTest.test_b_stream_regression
        test_reference.ReferenceTest.test_c_time_series_forecasting
        test_reference.ReferenceTest.test_d_anomaly_detection
        test_reference.ReferenceTest.test_e_gridworld
        test_reference.ReferenceTest.test_f_cart_pole
        test_reference.ReferenceTest.test_i_pendulum_swing_up
    )
    foreach(test IN LISTS DPAN_PYTHON_TESTS)
        string(REGEX REPLACE ".*\\.test_" "" short "${test}")
        add_test(NAME python.${short}
                 COMMAND ${Python_EXECUTABLE} -m unittest -v ${test}
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        set_tests_properties(python.${short} PROPERTIES
            ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python;PYTHONDONTWRITEBYTECODE=1"
            TIMEOUT 900)
    endforeach()
    set_tests_properties(python.test_api PROPERTIES TIMEOUT 300)
    foreach(test IN LISTS DPAN_PYTHON_TESTS)
        if(test MATCHES "test_reference")
            string(REGEX REPLACE ".*\\.test_" "" short "${test}")
            set_tests_properties(python.${short} PROPERTIES LABELS "python;reference")
        endif()
    endforeach()
    set_tests_properties(python.test_api PROPERTIES LABELS python)
endif()
