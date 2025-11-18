// File: include/cli/cli_config.hpp
//
// YAML Configuration Support for DPAN CLI
// Allows loading CLI settings from YAML configuration files

#ifndef DPAN_CLI_CONFIG_HPP
#define DPAN_CLI_CONFIG_HPP

#include <string>
#include <optional>
#include <map>
#include <vector>

namespace dpan {

/// Configuration structure for DPAN CLI
struct CliConfig {
    // === Interface Settings ===
    struct Interface {
        std::string prompt = "dpan> ";
        bool colors_enabled = true;
        bool verbose = false;
        std::string session_file = "dpan_session.db";
    } interface;

    // === Learning Settings ===
    struct Learning {
        bool active_learning = false;
        bool attention_enabled = false;

        // Pattern engine configuration
        struct PatternEngine {
            std::string similarity_metric = "context";
            bool enable_auto_refinement = true;
            bool enable_indexing = true;
            size_t feature_dimension = 64;
            size_t min_pattern_size = 1;
            size_t max_pattern_size = 1000;
            float similarity_threshold = 0.60f;
            float strong_match_threshold = 0.75f;
        } pattern_engine;

        // Association system configuration
        struct Association {
            bool enable_auto_maintenance = true;
            size_t min_co_occurrences = 2;
            float decay_rate = 0.01f;
            size_t window_size_seconds = 300;  // 5 minutes default
        } association;
    } learning;

    // === Attention Mechanism Settings ===
    struct Attention {
        size_t num_heads = 4;
        float temperature = 1.0f;
        bool use_context = true;
        bool use_importance = true;
        std::string attention_type = "dot_product";
        float association_weight = 0.6f;
        float attention_weight = 0.4f;
        bool enable_caching = true;
        size_t cache_size = 1000;
        bool debug_logging = false;
    } attention;

    // === Context Tracking Settings ===
    struct Context {
        float decay_rate = 0.10f;          // 10% decay
        float decay_interval = 30.0f;      // Every 30 seconds
        float removal_threshold = 0.05f;   // Remove below 5%
        size_t max_topics = 50;            // Maximum tracked topics
    } context;

    // === Performance Settings ===
    struct Performance {
        size_t association_batch_interval = 10;  // Form associations every N inputs
        size_t association_batch_initial = 100;  // Always form during first N inputs
    } performance;

    /// Load configuration from YAML file
    /// @param filepath Path to YAML configuration file
    /// @return CliConfig structure if successful, std::nullopt on error
    static std::optional<CliConfig> LoadFromFile(const std::string& filepath);

    /// Load configuration from YAML string
    /// @param yaml_content YAML content as string
    /// @return CliConfig structure if successful, std::nullopt on error
    static std::optional<CliConfig> LoadFromString(const std::string& yaml_content);

    /// Save configuration to YAML file
    /// @param filepath Path to save YAML file
    /// @return true if successful, false on error
    bool SaveToFile(const std::string& filepath) const;

    /// Convert to YAML string
    /// @return YAML representation of configuration
    std::string ToYamlString() const;

    /// Validate configuration values
    /// @return true if configuration is valid, false otherwise
    bool Validate() const;

    /// Get validation errors (if any)
    /// @return Vector of error messages
    std::vector<std::string> GetValidationErrors() const;

    /// Create default configuration
    static CliConfig Default();
};

} // namespace dpan

#endif // DPAN_CLI_CONFIG_HPP
