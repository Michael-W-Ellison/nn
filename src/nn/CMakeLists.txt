# File: src/nn/CMakeLists.txt
#
# Neural network module: dense networks trained with backpropagation, and
# their integration with DPAN (learned encoders, similarity and prediction
# scoring)

add_library(dpan_nn
    matrix.cpp
    compute.cpp
    layers.cpp
    convolution.cpp
    loss.cpp
    optimizer.cpp
    sequential.cpp
    neural_encoder.cpp
    neural_prediction_scorer.cpp
)

target_include_directories(dpan_nn PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/dpan>
)

target_link_libraries(dpan_nn PUBLIC
    dpan_core
    dpan_storage
    dpan_similarity
    dpan_association
)

# Optional BLAS for matrix products (cblas interface, e.g. OpenBLAS)
if(DPAN_WITH_BLAS)
    find_package(BLAS REQUIRED)
    find_path(DPAN_CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas)
    if(NOT DPAN_CBLAS_INCLUDE_DIR)
        message(FATAL_ERROR "DPAN_WITH_BLAS: cblas.h not found (install a BLAS with the C interface, e.g. libopenblas-dev)")
    endif()
    target_include_directories(dpan_nn PRIVATE ${DPAN_CBLAS_INCLUDE_DIR})
    target_link_libraries(dpan_nn PRIVATE BLAS::BLAS)
    target_compile_definitions(dpan_nn PRIVATE DPAN_HAS_BLAS=1)
endif()

target_compile_features(dpan_nn PUBLIC cxx_std_17)
