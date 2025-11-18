// File: src/cli/cli_config.cpp
//
// YAML Configuration Implementation for DPAN CLI

#include "cli/cli_config.hpp"
#include <yaml.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <vector>

namespace dpan {

// Helper function to read string from YAML scalar
static std::string GetScalarValue(yaml_event_t* event) {
    return std::string(reinterpret_cast<char*>(event->data.scalar.value),
                      event->data.scalar.length);
}

// Helper to convert string to bool
static bool ParseBool(const std::string& value) {
    return (value == "true" || value == "True" || value == "TRUE" ||
            value == "yes" || value == "Yes" || value == "YES" ||
            value == "1" || value == "on" || value == "On" || value == "ON");
}

std::optional<CliConfig> CliConfig::LoadFromFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << filepath << std::endl;
        return std::nullopt;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return LoadFromString(buffer.str());
}

std::optional<CliConfig> CliConfig::LoadFromString(const std::string& yaml_content) {
    yaml_parser_t parser;
    yaml_event_t event;

    if (!yaml_parser_initialize(&parser)) {
        std::cerr << "Failed to initialize YAML parser" << std::endl;
        return std::nullopt;
    }

    // Set input string
    yaml_parser_set_input_string(&parser,
        reinterpret_cast<const unsigned char*>(yaml_content.c_str()),
        yaml_content.size());

    CliConfig config = Default();
    std::string current_section;
    std::string current_key;
    std::string last_key;
    int depth = 0;

    bool done = false;
    while (!done) {
        if (!yaml_parser_parse(&parser, &event)) {
            std::cerr << "YAML parse error" << std::endl;
            yaml_parser_delete(&parser);
            return std::nullopt;
        }

        switch (event.type) {
            case YAML_STREAM_START_EVENT:
            case YAML_DOCUMENT_START_EVENT:
                break;

            case YAML_MAPPING_START_EVENT:
                depth++;
                break;

            case YAML_MAPPING_END_EVENT:
                depth--;
                if (depth == 1) {
                    current_section.clear();
                }
                break;

            case YAML_SCALAR_EVENT: {
                std::string value = GetScalarValue(&event);

                if (depth == 1) {
                    // Top-level key (section name)
                    current_section = value;
                } else if (depth == 2) {
                    if (current_key.empty()) {
                        // This is a key
                        current_key = value;
                        last_key = value;
                    } else {
                        // This is a value
                        // Apply configuration based on section and key
                        if (current_section == "interface") {
                            if (current_key == "prompt") config.interface.prompt = value;
                            else if (current_key == "colors_enabled") config.interface.colors_enabled = ParseBool(value);
                            else if (current_key == "verbose") config.interface.verbose = ParseBool(value);
                            else if (current_key == "session_file") config.interface.session_file = value;
                        }
                        else if (current_section == "learning") {
                            if (current_key == "active_learning") config.learning.active_learning = ParseBool(value);
                            else if (current_key == "attention_enabled") config.learning.attention_enabled = ParseBool(value);
                        }
                        else if (current_section == "pattern_engine") {
                            if (current_key == "similarity_metric") config.learning.pattern_engine.similarity_metric = value;
                            else if (current_key == "enable_auto_refinement") config.learning.pattern_engine.enable_auto_refinement = ParseBool(value);
                            else if (current_key == "enable_indexing") config.learning.pattern_engine.enable_indexing = ParseBool(value);
                            else if (current_key == "feature_dimension") config.learning.pattern_engine.feature_dimension = std::stoul(value);
                            else if (current_key == "min_pattern_size") config.learning.pattern_engine.min_pattern_size = std::stoul(value);
                            else if (current_key == "max_pattern_size") config.learning.pattern_engine.max_pattern_size = std::stoul(value);
                            else if (current_key == "similarity_threshold") config.learning.pattern_engine.similarity_threshold = std::stof(value);
                            else if (current_key == "strong_match_threshold") config.learning.pattern_engine.strong_match_threshold = std::stof(value);
                        }
                        else if (current_section == "association") {
                            if (current_key == "enable_auto_maintenance") config.learning.association.enable_auto_maintenance = ParseBool(value);
                            else if (current_key == "min_co_occurrences") config.learning.association.min_co_occurrences = std::stoul(value);
                            else if (current_key == "decay_rate") config.learning.association.decay_rate = std::stof(value);
                            else if (current_key == "window_size_seconds") config.learning.association.window_size_seconds = std::stoul(value);
                        }
                        else if (current_section == "attention") {
                            if (current_key == "num_heads") config.attention.num_heads = std::stoul(value);
                            else if (current_key == "temperature") config.attention.temperature = std::stof(value);
                            else if (current_key == "use_context") config.attention.use_context = ParseBool(value);
                            else if (current_key == "use_importance") config.attention.use_importance = ParseBool(value);
                            else if (current_key == "attention_type") config.attention.attention_type = value;
                            else if (current_key == "association_weight") config.attention.association_weight = std::stof(value);
                            else if (current_key == "attention_weight") config.attention.attention_weight = std::stof(value);
                            else if (current_key == "enable_caching") config.attention.enable_caching = ParseBool(value);
                            else if (current_key == "cache_size") config.attention.cache_size = std::stoul(value);
                            else if (current_key == "debug_logging") config.attention.debug_logging = ParseBool(value);
                        }
                        else if (current_section == "context") {
                            if (current_key == "decay_rate") config.context.decay_rate = std::stof(value);
                            else if (current_key == "decay_interval") config.context.decay_interval = std::stof(value);
                            else if (current_key == "removal_threshold") config.context.removal_threshold = std::stof(value);
                            else if (current_key == "max_topics") config.context.max_topics = std::stoul(value);
                        }
                        else if (current_section == "performance") {
                            if (current_key == "association_batch_interval") config.performance.association_batch_interval = std::stoul(value);
                            else if (current_key == "association_batch_initial") config.performance.association_batch_initial = std::stoul(value);
                        }

                        current_key.clear();
                    }
                }
                break;
            }

            case YAML_STREAM_END_EVENT:
            case YAML_DOCUMENT_END_EVENT:
                done = true;
                break;

            default:
                break;
        }

        yaml_event_delete(&event);
    }

    yaml_parser_delete(&parser);

    // Validate configuration
    if (!config.Validate()) {
        std::cerr << "Configuration validation failed:" << std::endl;
        for (const auto& error : config.GetValidationErrors()) {
            std::cerr << "  - " << error << std::endl;
        }
        return std::nullopt;
    }

    return config;
}

bool CliConfig::SaveToFile(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        return false;
    }

    file << ToYamlString();
    return true;
}

std::string CliConfig::ToYamlString() const {
    std::ostringstream ss;

    ss << "# DPAN CLI Configuration\n";
    ss << "# Auto-generated configuration file\n\n";

    ss << "interface:\n";
    ss << "  prompt: \"" << interface.prompt << "\"\n";
    ss << "  colors_enabled: " << (interface.colors_enabled ? "true" : "false") << "\n";
    ss << "  verbose: " << (interface.verbose ? "true" : "false") << "\n";
    ss << "  session_file: \"" << interface.session_file << "\"\n\n";

    ss << "learning:\n";
    ss << "  active_learning: " << (learning.active_learning ? "true" : "false") << "\n";
    ss << "  attention_enabled: " << (learning.attention_enabled ? "true" : "false") << "\n\n";

    ss << "pattern_engine:\n";
    ss << "  similarity_metric: \"" << learning.pattern_engine.similarity_metric << "\"\n";
    ss << "  enable_auto_refinement: " << (learning.pattern_engine.enable_auto_refinement ? "true" : "false") << "\n";
    ss << "  enable_indexing: " << (learning.pattern_engine.enable_indexing ? "true" : "false") << "\n";
    ss << "  feature_dimension: " << learning.pattern_engine.feature_dimension << "\n";
    ss << "  min_pattern_size: " << learning.pattern_engine.min_pattern_size << "\n";
    ss << "  max_pattern_size: " << learning.pattern_engine.max_pattern_size << "\n";
    ss << "  similarity_threshold: " << learning.pattern_engine.similarity_threshold << "\n";
    ss << "  strong_match_threshold: " << learning.pattern_engine.strong_match_threshold << "\n\n";

    ss << "association:\n";
    ss << "  enable_auto_maintenance: " << (learning.association.enable_auto_maintenance ? "true" : "false") << "\n";
    ss << "  min_co_occurrences: " << learning.association.min_co_occurrences << "\n";
    ss << "  decay_rate: " << learning.association.decay_rate << "\n";
    ss << "  window_size_seconds: " << learning.association.window_size_seconds << "\n\n";

    ss << "attention:\n";
    ss << "  num_heads: " << attention.num_heads << "\n";
    ss << "  temperature: " << attention.temperature << "\n";
    ss << "  use_context: " << (attention.use_context ? "true" : "false") << "\n";
    ss << "  use_importance: " << (attention.use_importance ? "true" : "false") << "\n";
    ss << "  attention_type: \"" << attention.attention_type << "\"\n";
    ss << "  association_weight: " << attention.association_weight << "\n";
    ss << "  attention_weight: " << attention.attention_weight << "\n";
    ss << "  enable_caching: " << (attention.enable_caching ? "true" : "false") << "\n";
    ss << "  cache_size: " << attention.cache_size << "\n";
    ss << "  debug_logging: " << (attention.debug_logging ? "true" : "false") << "\n\n";

    ss << "context:\n";
    ss << "  decay_rate: " << context.decay_rate << "\n";
    ss << "  decay_interval: " << context.decay_interval << "\n";
    ss << "  removal_threshold: " << context.removal_threshold << "\n";
    ss << "  max_topics: " << context.max_topics << "\n\n";

    ss << "performance:\n";
    ss << "  association_batch_interval: " << performance.association_batch_interval << "\n";
    ss << "  association_batch_initial: " << performance.association_batch_initial << "\n";

    return ss.str();
}

bool CliConfig::Validate() const {
    return GetValidationErrors().empty();
}

std::vector<std::string> CliConfig::GetValidationErrors() const {
    std::vector<std::string> errors;

    // Validate feature dimension
    if (learning.pattern_engine.feature_dimension == 0) {
        errors.push_back("feature_dimension must be greater than 0");
    }

    // Validate pattern sizes
    if (learning.pattern_engine.min_pattern_size == 0) {
        errors.push_back("min_pattern_size must be greater than 0");
    }
    if (learning.pattern_engine.max_pattern_size < learning.pattern_engine.min_pattern_size) {
        errors.push_back("max_pattern_size must be >= min_pattern_size");
    }

    // Validate thresholds
    if (learning.pattern_engine.similarity_threshold < 0.0f || learning.pattern_engine.similarity_threshold > 1.0f) {
        errors.push_back("similarity_threshold must be between 0.0 and 1.0");
    }
    if (learning.pattern_engine.strong_match_threshold < 0.0f || learning.pattern_engine.strong_match_threshold > 1.0f) {
        errors.push_back("strong_match_threshold must be between 0.0 and 1.0");
    }

    // Validate attention
    if (attention.num_heads == 0) {
        errors.push_back("num_heads must be greater than 0");
    }
    if (attention.temperature <= 0.0f) {
        errors.push_back("temperature must be greater than 0");
    }
    if (attention.association_weight < 0.0f || attention.attention_weight < 0.0f) {
        errors.push_back("attention weights must be non-negative");
    }
    if (attention.association_weight + attention.attention_weight == 0.0f) {
        errors.push_back("sum of attention weights must be greater than 0");
    }
    if (attention.attention_type != "dot_product" &&
        attention.attention_type != "additive" &&
        attention.attention_type != "multiplicative") {
        errors.push_back("attention_type must be one of: dot_product, additive, multiplicative");
    }

    // Validate context settings
    if (context.decay_rate < 0.0f || context.decay_rate > 1.0f) {
        errors.push_back("context decay_rate must be between 0.0 and 1.0");
    }
    if (context.decay_interval <= 0.0f) {
        errors.push_back("context decay_interval must be greater than 0");
    }
    if (context.removal_threshold < 0.0f || context.removal_threshold > 1.0f) {
        errors.push_back("context removal_threshold must be between 0.0 and 1.0");
    }

    return errors;
}

CliConfig CliConfig::Default() {
    return CliConfig{};  // Uses default member initializers
}

} // namespace dpan
