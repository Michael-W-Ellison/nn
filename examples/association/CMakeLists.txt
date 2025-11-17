# Association Learning Examples
#
# These examples demonstrate the DPAN association learning system features.

# Example 1: Basic Learning
add_executable(basic_learning basic_learning.cpp)
target_link_libraries(basic_learning
    dpan_association
    dpan_core
)

# Example 2: Custom Formation Rules
add_executable(custom_formation_rules custom_formation_rules.cpp)
target_link_libraries(custom_formation_rules
    dpan_association
    dpan_core
)

# Example 3: Activation Propagation Demo
add_executable(activation_propagation_demo activation_propagation_demo.cpp)
target_link_libraries(activation_propagation_demo
    dpan_association
    dpan_core
)

# Install examples
install(TARGETS
    basic_learning
    custom_formation_rules
    activation_propagation_demo
    RUNTIME DESTINATION bin/examples/association
)

# Install source files for reference
install(FILES
    basic_learning.cpp
    custom_formation_rules.cpp
    activation_propagation_demo.cpp
    DESTINATION share/dpan/examples/association
)
