# File: src/learning/CMakeLists.txt
#
# Learning module: attention mechanisms and pattern importance

add_library(dpan_learning
    attention_utils.cpp
    basic_attention.cpp
    context_aware_attention.cpp
    multi_head_attention.cpp
    semantic_attention_head.cpp
    temporal_attention_head.cpp
    structural_attention_head.cpp
    association_attention_head.cpp
    self_attention.cpp
    pattern_importance.cpp
)

target_include_directories(dpan_learning PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)

target_link_libraries(dpan_learning PUBLIC
    dpan_core
    dpan_storage
    dpan_similarity
    dpan_association
)

target_compile_features(dpan_learning PUBLIC cxx_std_17)
