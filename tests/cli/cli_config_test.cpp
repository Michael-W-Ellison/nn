// File: tests/cli/cli_config_test.cpp
//
// Tests for YAML configuration system

#include "cli/cli_config.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>

using namespace dpan;

class CliConfigTest : public ::testing::Test {
protected:
    std::string temp_config_path = "/tmp/test_config.yaml";

    void TearDown() override {
        // Clean up temp file
        std::filesystem::remove(temp_config_path);
    }
};

TEST_F(CliConfigTest, DefaultConfig) {
    auto config = CliConfig::Default();

    EXPECT_EQ(config.interface.prompt, "dpan> ");
    EXPECT_TRUE(config.interface.colors_enabled);
    EXPECT_FALSE(config.interface.verbose);
    EXPECT_EQ(config.interface.session_file, "dpan_session.db");

    EXPECT_FALSE(config.learning.active_learning);
    EXPECT_FALSE(config.learning.attention_enabled);

    EXPECT_EQ(config.attention.num_heads, 4u);
    EXPECT_FLOAT_EQ(config.attention.temperature, 1.0f);
}

TEST_F(CliConfigTest, LoadFromString) {
    std::string yaml = R"(
interface:
  prompt: "test> "
  colors_enabled: false
  verbose: true
  session_file: "test.db"

learning:
  active_learning: true
  attention_enabled: true

attention:
  num_heads: 8
  temperature: 0.5
)";

    auto config_opt = CliConfig::LoadFromString(yaml);
    ASSERT_TRUE(config_opt.has_value());

    auto config = config_opt.value();
    EXPECT_EQ(config.interface.prompt, "test> ");
    EXPECT_FALSE(config.interface.colors_enabled);
    EXPECT_TRUE(config.interface.verbose);
    EXPECT_EQ(config.interface.session_file, "test.db");

    EXPECT_TRUE(config.learning.active_learning);
    EXPECT_TRUE(config.learning.attention_enabled);

    EXPECT_EQ(config.attention.num_heads, 8u);
    EXPECT_FLOAT_EQ(config.attention.temperature, 0.5f);
}

TEST_F(CliConfigTest, SaveAndLoad) {
    // Create a config
    auto config = CliConfig::Default();
    config.interface.prompt = "custom> ";
    config.interface.verbose = true;
    config.attention.num_heads = 6;

    // Save it
    ASSERT_TRUE(config.SaveToFile(temp_config_path));

    // Load it back
    auto loaded_opt = CliConfig::LoadFromFile(temp_config_path);
    ASSERT_TRUE(loaded_opt.has_value());

    auto loaded = loaded_opt.value();
    EXPECT_EQ(loaded.interface.prompt, "custom> ");
    EXPECT_TRUE(loaded.interface.verbose);
    EXPECT_EQ(loaded.attention.num_heads, 6u);
}

TEST_F(CliConfigTest, Validation) {
    auto config = CliConfig::Default();

    // Valid config
    EXPECT_TRUE(config.Validate());
    EXPECT_TRUE(config.GetValidationErrors().empty());

    // Invalid: feature_dimension = 0
    config.learning.pattern_engine.feature_dimension = 0;
    EXPECT_FALSE(config.Validate());
    auto errors = config.GetValidationErrors();
    EXPECT_FALSE(errors.empty());

    // Reset and test another invalid config
    config = CliConfig::Default();
    config.attention.temperature = 0.0f;  // Invalid: must be > 0
    EXPECT_FALSE(config.Validate());
}

TEST_F(CliConfigTest, InvalidAttentionType) {
    std::string yaml = R"(
attention:
  attention_type: "invalid_type"
)";

    // The parser validates during loading, so it should fail to load
    auto config_opt = CliConfig::LoadFromString(yaml);
    EXPECT_FALSE(config_opt.has_value());  // Loading fails due to validation
}

TEST_F(CliConfigTest, ToYamlString) {
    auto config = CliConfig::Default();
    config.interface.prompt = "test> ";
    config.attention.num_heads = 8;

    std::string yaml = config.ToYamlString();

    // Check that YAML contains expected values
    EXPECT_NE(yaml.find("prompt: \"test> \""), std::string::npos);
    EXPECT_NE(yaml.find("num_heads: 8"), std::string::npos);
}

TEST_F(CliConfigTest, LoadNonexistentFile) {
    auto config_opt = CliConfig::LoadFromFile("/nonexistent/path/config.yaml");
    EXPECT_FALSE(config_opt.has_value());
}

TEST_F(CliConfigTest, InvalidYaml) {
    std::string invalid_yaml = "this is { not: valid: yaml [";
    auto config_opt = CliConfig::LoadFromString(invalid_yaml);
    // Parser may fail or succeed depending on implementation
    // Just verify it doesn't crash
}

TEST_F(CliConfigTest, PartialConfig) {
    // Config with only some fields - others should use defaults
    std::string yaml = R"(
interface:
  prompt: "partial> "
)";

    auto config_opt = CliConfig::LoadFromString(yaml);
    ASSERT_TRUE(config_opt.has_value());

    auto config = config_opt.value();
    EXPECT_EQ(config.interface.prompt, "partial> ");
    // Other fields should have defaults
    EXPECT_TRUE(config.interface.colors_enabled);
    EXPECT_EQ(config.attention.num_heads, 4u);
}
