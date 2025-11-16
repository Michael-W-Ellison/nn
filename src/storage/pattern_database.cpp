// File: src/storage/pattern_database.cpp
#include "storage/pattern_database.hpp"
#include <stdexcept>

namespace dpan {

std::unique_ptr<PatternDatabase> CreatePatternDatabase(const std::string& config_path) {
    // TODO: Implement configuration file parsing and backend selection
    // This will be implemented in Task 2.2.2 (In-Memory Backend) and later tasks
    throw std::runtime_error(
        "CreatePatternDatabase not yet implemented. "
        "This will be completed when backend implementations are added."
    );
}

} // namespace dpan
