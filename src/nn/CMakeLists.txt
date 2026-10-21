# File: src/nn/CMakeLists.txt
#
# Neural network module: dense networks trained with backpropagation, and
# their integration with DPAN (learned encoders, similarity and prediction
# scoring)

add_library(dpan_nn
    matrix.cpp
    layers.cpp
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

target_compile_features(dpan_nn PUBLIC cxx_std_17)
