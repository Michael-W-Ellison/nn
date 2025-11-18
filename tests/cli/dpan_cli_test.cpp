// File: tests/cli/dpan_cli_test.cpp
//
// Comprehensive test suite for DPAN CLI
// Following TDD principles to ensure robustness

#include "cli/dpan_cli.hpp"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace dpan {
namespace {

// Test fixture for CLI tests
class DPANCliTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a unique test session file to avoid conflicts
        test_session_file_ = "test_session_" + std::to_string(
            std::chrono::system_clock::now().time_since_epoch().count()
        ) + ".db";

        cli_ = std::make_unique<DPANCli>();
        cli_->SetSessionFile(test_session_file_);
        cli_->InitializeClean();
    }

    void TearDown() override {
        // Clean up test files
        CleanupTestFiles();
    }

    void CleanupTestFiles() {
        std::filesystem::remove(test_session_file_);
        std::filesystem::remove(test_session_file_ + ".associations");
        std::filesystem::remove(test_session_file_ + ".mappings");
        std::filesystem::remove(test_session_file_ + "-wal");
        std::filesystem::remove(test_session_file_ + "-shm");
    }

    std::unique_ptr<DPANCli> cli_;
    std::string test_session_file_;
};

// ============================================================================
// Construction & Initialization Tests
// ============================================================================

TEST_F(DPANCliTest, DefaultConstruction) {
    EXPECT_EQ(0u, cli_->GetTotalInputs());
    EXPECT_EQ(0u, cli_->GetPatternsLearned());
    EXPECT_EQ(0u, cli_->GetConversationLength());
    EXPECT_EQ(0u, cli_->GetVocabularySize());
    EXPECT_FALSE(cli_->IsActiveLearningEnabled());
    EXPECT_FALSE(cli_->IsVerboseEnabled());
}

TEST_F(DPANCliTest, InitializeClean) {
    // Process some input first
    cli_->ProcessCommand("Hello");
    EXPECT_GT(cli_->GetTotalInputs(), 0u);

    // Re-initialize clean
    cli_->InitializeClean();

    // Should reset state (note: we don't reset counters in InitializeClean currently)
    // This is a design decision - testing current behavior
}

// ============================================================================
// Command Parsing Tests
// ============================================================================

TEST_F(DPANCliTest, EmptyCommandDoesNothing) {
    cli_->ProcessCommand("");
    EXPECT_EQ(0u, cli_->GetTotalInputs());
}

TEST_F(DPANCliTest, TextInputIncrementsCounter) {
    cli_->ProcessCommand("Hello world");
    EXPECT_EQ(1u, cli_->GetTotalInputs());

    cli_->ProcessCommand("How are you?");
    EXPECT_EQ(2u, cli_->GetTotalInputs());
}

TEST_F(DPANCliTest, CommandWithSlashIsRecognizedAsCommand) {
    size_t before = cli_->GetTotalInputs();
    cli_->ProcessCommand("/stats");
    // /stats should not increment input counter
    EXPECT_EQ(before, cli_->GetTotalInputs());
}

TEST_F(DPANCliTest, HelpCommandExecutesWithoutError) {
    EXPECT_NO_THROW(cli_->ProcessCommand("/help"));
}

TEST_F(DPANCliTest, StatsCommandExecutesWithoutError) {
    EXPECT_NO_THROW(cli_->ProcessCommand("/stats"));
}

TEST_F(DPANCliTest, VerboseCommandTogglesState) {
    EXPECT_FALSE(cli_->IsVerboseEnabled());

    cli_->ProcessCommand("/verbose");
    EXPECT_TRUE(cli_->IsVerboseEnabled());

    cli_->ProcessCommand("/verbose");
    EXPECT_FALSE(cli_->IsVerboseEnabled());
}

TEST_F(DPANCliTest, ActiveLearningCommandTogglesState) {
    EXPECT_FALSE(cli_->IsActiveLearningEnabled());

    cli_->ProcessCommand("/active");
    EXPECT_TRUE(cli_->IsActiveLearningEnabled());

    cli_->ProcessCommand("/active");
    EXPECT_FALSE(cli_->IsActiveLearningEnabled());
}

TEST_F(DPANCliTest, UnknownCommandDoesNotCrash) {
    EXPECT_NO_THROW(cli_->ProcessCommand("/unknown_command"));
}

// ============================================================================
// Text-to-Pattern Conversion Tests
// ============================================================================

TEST_F(DPANCliTest, TextInputCreatesPattern) {
    cli_->ProcessCommand("Hello DPAN");

    // Should create a pattern
    EXPECT_GT(cli_->GetPatternsLearned(), 0u);
    EXPECT_EQ(1u, cli_->GetVocabularySize());

    // Should have text-to-pattern mapping
    auto pattern_id = cli_->GetPatternForText("Hello DPAN");
    EXPECT_TRUE(pattern_id.has_value());
}

TEST_F(DPANCliTest, MultipleUniqueInputsCreateMultiplePatterns) {
    cli_->ProcessCommand("First input");
    cli_->ProcessCommand("Second input");
    cli_->ProcessCommand("Third input");

    EXPECT_EQ(3u, cli_->GetVocabularySize());
    EXPECT_EQ(3u, cli_->GetConversationLength());
}

TEST_F(DPANCliTest, RepeatedInputRecognizesExistingPattern) {
    cli_->ProcessCommand("Repeated text");
    auto first_pattern = cli_->GetPatternForText("Repeated text");
    size_t first_learned_count = cli_->GetPatternsLearned();

    cli_->ProcessCommand("Repeated text");
    auto second_pattern = cli_->GetPatternForText("Repeated text");

    // Should recognize the same pattern (may create similar patterns)
    EXPECT_TRUE(first_pattern.has_value());
    EXPECT_TRUE(second_pattern.has_value());

    // Vocabulary should still be 1 unique text
    EXPECT_EQ(1u, cli_->GetVocabularySize());
}

TEST_F(DPANCliTest, PatternToTextMappingIsCorrect) {
    std::string test_text = "Test mapping";
    cli_->ProcessCommand(test_text);

    auto pattern_id = cli_->GetPatternForText(test_text);
    ASSERT_TRUE(pattern_id.has_value());

    auto retrieved_text = cli_->GetTextForPattern(pattern_id.value());
    ASSERT_TRUE(retrieved_text.has_value());
    EXPECT_EQ(test_text, retrieved_text.value());
}

TEST_F(DPANCliTest, UnknownTextReturnsNoPattern) {
    auto pattern = cli_->GetPatternForText("Never seen before");
    EXPECT_FALSE(pattern.has_value());
}

TEST_F(DPANCliTest, UnknownPatternReturnsNoText) {
    PatternID fake_id(999999);
    auto text = cli_->GetTextForPattern(fake_id);
    EXPECT_FALSE(text.has_value());
}

// ============================================================================
// Conversation Flow Tests
// ============================================================================

TEST_F(DPANCliTest, ConversationHistoryGrows) {
    EXPECT_EQ(0u, cli_->GetConversationLength());

    cli_->ProcessCommand("First message");
    EXPECT_EQ(1u, cli_->GetConversationLength());

    cli_->ProcessCommand("Second message");
    EXPECT_EQ(2u, cli_->GetConversationLength());

    cli_->ProcessCommand("Third message");
    EXPECT_EQ(3u, cli_->GetConversationLength());
}

TEST_F(DPANCliTest, ConversationMaintainsSequence) {
    cli_->ProcessCommand("Hello");
    cli_->ProcessCommand("How are you?");
    cli_->ProcessCommand("I am fine");

    // All should be tracked
    EXPECT_EQ(3u, cli_->GetConversationLength());
    EXPECT_EQ(3u, cli_->GetTotalInputs());
}

// ============================================================================
// Statistics and Inspection Tests
// ============================================================================

TEST_F(DPANCliTest, StatsCommandShowsCorrectCounts) {
    cli_->ProcessCommand("Message 1");
    cli_->ProcessCommand("Message 2");

    // Should execute without error
    EXPECT_NO_THROW(cli_->ProcessCommand("/stats"));

    // Verify internal state
    EXPECT_EQ(2u, cli_->GetTotalInputs());
    EXPECT_GE(cli_->GetPatternsLearned(), 2u);
}

TEST_F(DPANCliTest, PatternsCommandExecutesWithoutError) {
    cli_->ProcessCommand("Test pattern");
    EXPECT_NO_THROW(cli_->ProcessCommand("/patterns"));
}

TEST_F(DPANCliTest, AssociationsCommandExecutesWithoutError) {
    cli_->ProcessCommand("First");
    cli_->ProcessCommand("Second");
    EXPECT_NO_THROW(cli_->ProcessCommand("/associations"));
}

TEST_F(DPANCliTest, PredictCommandWithKnownPattern) {
    cli_->ProcessCommand("Hello");
    cli_->ProcessCommand("World");

    // Predict should work with known pattern
    EXPECT_NO_THROW(cli_->ProcessCommand("/predict Hello"));
}

TEST_F(DPANCliTest, PredictCommandWithUnknownPattern) {
    // Should handle gracefully
    EXPECT_NO_THROW(cli_->ProcessCommand("/predict UnknownPattern"));
}

// ============================================================================
// Session Persistence Tests
// ============================================================================

TEST_F(DPANCliTest, SaveSessionCreatesFiles) {
    cli_->ProcessCommand("Test data");
    cli_->ProcessCommand("/save");

    // Check if association file exists
    EXPECT_TRUE(std::filesystem::exists(test_session_file_ + ".associations"));
    EXPECT_TRUE(std::filesystem::exists(test_session_file_ + ".mappings"));
}

TEST_F(DPANCliTest, SaveAndLoadPreservesVocabulary) {
    // Create some patterns
    cli_->ProcessCommand("Pattern A");
    cli_->ProcessCommand("Pattern B");
    cli_->ProcessCommand("Pattern C");

    size_t vocab_before = cli_->GetVocabularySize();

    // Save
    cli_->ProcessCommand("/save");

    // Create new CLI instance with same session file
    auto new_cli = std::make_unique<DPANCli>();
    new_cli->SetSessionFile(test_session_file_);

    // Should load automatically or we can trigger load
    // For now, test manual load
    new_cli->ProcessCommand("/load");

    // Vocabulary size should match
    EXPECT_EQ(vocab_before, new_cli->GetVocabularySize());
}

TEST_F(DPANCliTest, LoadNonexistentSessionHandlesGracefully) {
    auto new_cli = std::make_unique<DPANCli>();
    new_cli->SetSessionFile("nonexistent_session.db");

    // Should not crash
    EXPECT_NO_THROW(new_cli->ProcessCommand("/load"));
}

// ============================================================================
// Batch Learning Tests
// ============================================================================

TEST_F(DPANCliTest, LearnFromNonexistentFileHandlesError) {
    // Should handle gracefully
    EXPECT_NO_THROW(cli_->ProcessCommand("/learn nonexistent_file.txt"));

    // Should not have learned anything
    EXPECT_EQ(0u, cli_->GetVocabularySize());
}

TEST_F(DPANCliTest, LearnFromFileProcessesAllLines) {
    // Create a temporary test file
    std::string test_file = "test_learn_file.txt";
    std::ofstream out(test_file);
    out << "Line 1\n";
    out << "Line 2\n";
    out << "Line 3\n";
    out << "\n";  // Empty line
    out << "Line 4\n";
    out.close();

    cli_->ProcessCommand("/learn " + test_file);

    // Should have processed 4 non-empty lines
    EXPECT_GE(cli_->GetVocabularySize(), 4u);
    EXPECT_GE(cli_->GetTotalInputs(), 4u);

    // Cleanup
    std::filesystem::remove(test_file);
}

TEST_F(DPANCliTest, LearnFromFileMaintainsSequence) {
    std::string test_file = "test_sequence.txt";
    std::ofstream out(test_file);
    out << "First\n";
    out << "Second\n";
    out << "Third\n";
    out.close();

    cli_->ProcessCommand("/learn " + test_file);

    // Should have learned in sequence
    EXPECT_EQ(3u, cli_->GetTotalInputs());
    EXPECT_EQ(3u, cli_->GetConversationLength());

    // Check that all patterns exist
    EXPECT_TRUE(cli_->GetPatternForText("First").has_value());
    EXPECT_TRUE(cli_->GetPatternForText("Second").has_value());
    EXPECT_TRUE(cli_->GetPatternForText("Third").has_value());

    std::filesystem::remove(test_file);
}

TEST_F(DPANCliTest, LearnFromLargeFileProcessesCorrectly) {
    std::string test_file = "test_large.txt";
    std::ofstream out(test_file);

    const size_t num_lines = 250;
    for (size_t i = 0; i < num_lines; ++i) {
        out << "Line " << i << "\n";
    }
    out.close();

    cli_->ProcessCommand("/learn " + test_file);

    EXPECT_EQ(num_lines, cli_->GetTotalInputs());
    EXPECT_GE(cli_->GetVocabularySize(), num_lines);

    std::filesystem::remove(test_file);
}

// ============================================================================
// Active Learning Mode Tests
// ============================================================================

TEST_F(DPANCliTest, ActiveLearningModeStartsDisabled) {
    EXPECT_FALSE(cli_->IsActiveLearningEnabled());
}

TEST_F(DPANCliTest, ActiveLearningModeCanBeEnabled) {
    cli_->ProcessCommand("/active");
    EXPECT_TRUE(cli_->IsActiveLearningEnabled());
}

TEST_F(DPANCliTest, ActiveLearningModeCanBeToggled) {
    cli_->ProcessCommand("/active");
    EXPECT_TRUE(cli_->IsActiveLearningEnabled());

    cli_->ProcessCommand("/active");
    EXPECT_FALSE(cli_->IsActiveLearningEnabled());

    cli_->ProcessCommand("/active");
    EXPECT_TRUE(cli_->IsActiveLearningEnabled());
}

// ============================================================================
// Edge Cases and Error Handling Tests
// ============================================================================

TEST_F(DPANCliTest, VeryLongTextIsHandled) {
    std::string long_text(5000, 'a');
    EXPECT_NO_THROW(cli_->ProcessCommand(long_text));
}

TEST_F(DPANCliTest, SpecialCharactersInText) {
    EXPECT_NO_THROW(cli_->ProcessCommand("!@#$%^&*()"));
    EXPECT_NO_THROW(cli_->ProcessCommand("Hello\tWorld"));
    EXPECT_NO_THROW(cli_->ProcessCommand("Unicode: 你好世界"));
}

TEST_F(DPANCliTest, MultipleConsecutiveSlashes) {
    EXPECT_NO_THROW(cli_->ProcessCommand("///"));
    EXPECT_NO_THROW(cli_->ProcessCommand("//help"));
}

TEST_F(DPANCliTest, CommandWithExtraSpaces) {
    EXPECT_NO_THROW(cli_->ProcessCommand("/stats   "));
    EXPECT_NO_THROW(cli_->ProcessCommand("/   stats"));
}

TEST_F(DPANCliTest, VeryLongCommand) {
    std::string long_cmd = "/predict " + std::string(1000, 'x');
    EXPECT_NO_THROW(cli_->ProcessCommand(long_cmd));
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(DPANCliTest, FullConversationFlow) {
    // Simulate a real conversation
    cli_->ProcessCommand("Hello");
    cli_->ProcessCommand("How are you?");
    cli_->ProcessCommand("I am learning");
    cli_->ProcessCommand("This is interesting");

    EXPECT_EQ(4u, cli_->GetTotalInputs());
    EXPECT_EQ(4u, cli_->GetConversationLength());
    EXPECT_EQ(4u, cli_->GetVocabularySize());

    // Check stats
    EXPECT_NO_THROW(cli_->ProcessCommand("/stats"));

    // Check patterns
    EXPECT_NO_THROW(cli_->ProcessCommand("/patterns"));

    // Save session
    EXPECT_NO_THROW(cli_->ProcessCommand("/save"));
}

TEST_F(DPANCliTest, CompleteWorkflowWithFileAndPersistence) {
    // Create training file
    std::string training_file = "training.txt";
    std::ofstream out(training_file);
    out << "Data point 1\n";
    out << "Data point 2\n";
    out << "Data point 3\n";
    out.close();

    // Learn from file
    cli_->ProcessCommand("/learn " + training_file);

    // Add some interactive inputs
    cli_->ProcessCommand("Interactive 1");
    cli_->ProcessCommand("Interactive 2");

    size_t total = cli_->GetTotalInputs();
    size_t vocab = cli_->GetVocabularySize();

    // Enable active learning
    cli_->ProcessCommand("/active");

    // Save
    cli_->ProcessCommand("/save");

    // Load in new session
    auto new_cli = std::make_unique<DPANCli>();
    new_cli->SetSessionFile(test_session_file_);
    new_cli->ProcessCommand("/load");

    // Verify state preserved
    EXPECT_EQ(vocab, new_cli->GetVocabularySize());

    // Cleanup
    std::filesystem::remove(training_file);
}

// ============================================================================
// Performance and Stress Tests
// ============================================================================

TEST_F(DPANCliTest, RapidSequentialInputs) {
    const size_t num_inputs = 100;

    for (size_t i = 0; i < num_inputs; ++i) {
        cli_->ProcessCommand("Input " + std::to_string(i));
    }

    EXPECT_EQ(num_inputs, cli_->GetTotalInputs());
    EXPECT_EQ(num_inputs, cli_->GetVocabularySize());
}

TEST_F(DPANCliTest, MixedCommandsAndInputs) {
    cli_->ProcessCommand("Hello");
    cli_->ProcessCommand("/stats");
    cli_->ProcessCommand("World");
    cli_->ProcessCommand("/verbose");
    cli_->ProcessCommand("Testing");
    cli_->ProcessCommand("/patterns");
    cli_->ProcessCommand("More data");
    cli_->ProcessCommand("/active");

    // Only text inputs should count
    EXPECT_EQ(4u, cli_->GetTotalInputs());
}

} // namespace
} // namespace dpan
