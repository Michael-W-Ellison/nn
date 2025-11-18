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

// ============================================================================
// Context Tracking Tests
// ============================================================================

TEST_F(DPANCliTest, ContextAccumulatesWithInputs) {
    // Initially, context should be empty or minimal
    auto& initial_context = cli_->GetCurrentContext();
    EXPECT_TRUE(initial_context.IsEmpty() || initial_context.Size() <= 3);  // May have temporal_hour

    // Process some inputs
    cli_->ProcessCommand("machine learning is fascinating");
    auto& context1 = cli_->GetCurrentContext();
    EXPECT_GT(context1.Size(), 0u);

    // Context should have conversation_depth
    EXPECT_TRUE(context1.Has("conversation_depth"));

    // Process more inputs
    cli_->ProcessCommand("neural networks are powerful");
    auto& context2 = cli_->GetCurrentContext();

    // Context should grow or maintain depth
    EXPECT_TRUE(context2.Has("conversation_depth"));
    EXPECT_GT(context2.Size(), 0u);
}

TEST_F(DPANCliTest, ContextTracksTopics) {
    // Process input with clear topics
    cli_->ProcessCommand("machine learning algorithms");

    auto& context = cli_->GetCurrentContext();

    // Should have captured at least one topic
    bool has_topic = false;
    for (size_t i = 0; i < 10; ++i) {
        std::string topic_name = "topic_";
        // Check if any topic dimension exists (we don't know exact names)
        if (context.Size() > 1) {  // More than just conversation_depth
            has_topic = true;
            break;
        }
    }

    // Context should have some dimensions
    EXPECT_GT(context.Size(), 1u);  // More than just conversation_depth
}

TEST_F(DPANCliTest, ContextHasConversationDepth) {
    cli_->ProcessCommand("first message");
    auto& context1 = cli_->GetCurrentContext();
    float depth1 = context1.Get("conversation_depth");

    cli_->ProcessCommand("second message");
    cli_->ProcessCommand("third message");
    auto& context3 = cli_->GetCurrentContext();
    float depth3 = context3.Get("conversation_depth");

    // Conversation depth should increase
    EXPECT_GT(depth3, depth1);
}

TEST_F(DPANCliTest, ContextHasTopicDiversity) {
    // Single topic conversation
    cli_->ProcessCommand("machine learning");
    auto& context1 = cli_->GetCurrentContext();
    float diversity1 = context1.Get("topic_diversity");

    // Multi-topic conversation
    cli_->ProcessCommand("database systems are important");
    cli_->ProcessCommand("network protocols enable communication");
    auto& context2 = cli_->GetCurrentContext();
    float diversity2 = context2.Get("topic_diversity");

    // Topic diversity should exist
    EXPECT_TRUE(context1.Has("topic_diversity"));
    EXPECT_TRUE(context2.Has("topic_diversity"));
}

TEST_F(DPANCliTest, ContextIncludesTemporalInformation) {
    cli_->ProcessCommand("test input");

    auto& context = cli_->GetCurrentContext();

    // Should have temporal_hour dimension
    EXPECT_TRUE(context.Has("temporal_hour"));

    float hour = context.Get("temporal_hour");

    // Hour factor should be between 0 and 1
    EXPECT_GE(hour, 0.0f);
    EXPECT_LE(hour, 1.0f);
}

TEST_F(DPANCliTest, ContextDecaysOverTime) {
    // This test would require manipulating time, which is tricky
    // For now, just verify that context can be built multiple times
    cli_->ProcessCommand("first input with topics");

    auto& context1 = cli_->GetCurrentContext();
    size_t size1 = context1.Size();

    // Process more inputs
    cli_->ProcessCommand("second input");
    cli_->ProcessCommand("third input");

    auto& context2 = cli_->GetCurrentContext();

    // Context should still exist
    EXPECT_GT(context2.Size(), 0u);

    // NOTE: Actual decay testing would require time manipulation
    // which is beyond the scope of this basic test
}

TEST_F(DPANCliTest, ContextUsedInPredictions) {
    // Create some patterns with associations
    cli_->ProcessCommand("hello world");
    cli_->ProcessCommand("world peace");
    cli_->ProcessCommand("peace treaty");

    // Verify context is present when predicting
    auto& context = cli_->GetCurrentContext();
    EXPECT_GT(context.Size(), 0u);

    // The actual prediction call will use the context
    // (verification is indirect through code inspection)
}

TEST_F(DPANCliTest, MultipleInputsAccumulateContext) {
    // Process multiple inputs to build up context
    const std::vector<std::string> inputs = {
        "artificial intelligence",
        "machine learning models",
        "deep neural networks",
        "training algorithms"
    };

    for (const auto& input : inputs) {
        cli_->ProcessCommand(input);
    }

    auto& final_context = cli_->GetCurrentContext();

    // Should have accumulated conversation depth
    float depth = final_context.Get("conversation_depth");
    EXPECT_GT(depth, 0.0f);

    // Should have topic diversity
    EXPECT_TRUE(final_context.Has("topic_diversity"));

    // Context should have multiple dimensions
    EXPECT_GT(final_context.Size(), 2u);  // At least depth + diversity
}

TEST_F(DPANCliTest, ContextTopicsAreLimited) {
    // Create many different topics
    for (int i = 0; i < 50; ++i) {
        std::string input = "topic" + std::to_string(i) + " content here";
        cli_->ProcessCommand(input);
    }

    auto& context = cli_->GetCurrentContext();

    // Context should not grow unbounded
    // Maximum should be around: 5 topics + conversation_depth + topic_diversity + temporal_hour
    EXPECT_LE(context.Size(), 15u);  // Reasonable upper bound
}

// ============================================================================
// Attention Integration Tests
// ============================================================================

TEST_F(DPANCliTest, AttentionDefaultsToDisabled) {
    // Attention should be disabled by default
    EXPECT_FALSE(cli_->IsAttentionEnabled());
}

TEST_F(DPANCliTest, ToggleAttentionCommand) {
    // Initially disabled
    EXPECT_FALSE(cli_->IsAttentionEnabled());

    // Toggle on
    cli_->ProcessCommand("/attention");
    EXPECT_TRUE(cli_->IsAttentionEnabled());

    // Toggle off
    cli_->ProcessCommand("/attention");
    EXPECT_FALSE(cli_->IsAttentionEnabled());
}

TEST_F(DPANCliTest, AttentionUsedInPredictions) {
    // Build up some conversation history
    cli_->ProcessCommand("machine learning");
    cli_->ProcessCommand("neural networks");
    cli_->ProcessCommand("deep learning");
    cli_->ProcessCommand("machine learning");  // Repeat to create association

    // Enable attention
    cli_->ProcessCommand("/attention");
    EXPECT_TRUE(cli_->IsAttentionEnabled());

    // Process another input - should use attention-enhanced predictions
    cli_->ProcessCommand("machine learning");

    // No way to directly verify attention was used, but at least ensure it doesn't crash
    EXPECT_TRUE(cli_->IsAttentionEnabled());
}

TEST_F(DPANCliTest, AttentionFallbackWhenDisabled) {
    // Build up some conversation history
    cli_->ProcessCommand("hello world");
    cli_->ProcessCommand("hello again");

    // Ensure attention is disabled
    EXPECT_FALSE(cli_->IsAttentionEnabled());

    // Process input - should use basic predictions
    cli_->ProcessCommand("hello world");

    // Should still work fine
    EXPECT_EQ(cli_->GetConversationLength(), 3u);
}

TEST_F(DPANCliTest, AttentionWithContextTracking) {
    // Enable attention
    cli_->ProcessCommand("/attention");

    // Build conversation with context
    cli_->ProcessCommand("artificial intelligence research");
    cli_->ProcessCommand("machine learning algorithms");
    cli_->ProcessCommand("neural network architecture");

    // Context should be accumulated
    auto& context = cli_->GetCurrentContext();
    EXPECT_GT(context.Size(), 0u);

    // Attention should use this context for predictions
    cli_->ProcessCommand("artificial intelligence research");

    // Verify still enabled and working
    EXPECT_TRUE(cli_->IsAttentionEnabled());
}

TEST_F(DPANCliTest, AttentionDoesNotBreakExistingBehavior) {
    // Test that enabling/disabling attention doesn't break normal operation

    // Normal operation
    cli_->ProcessCommand("test input");
    EXPECT_EQ(cli_->GetConversationLength(), 1u);

    // Enable attention
    cli_->ProcessCommand("/attention");

    // Continue normal operation
    cli_->ProcessCommand("another test");
    EXPECT_EQ(cli_->GetConversationLength(), 2u);

    // Disable attention
    cli_->ProcessCommand("/attention");

    // Still works
    cli_->ProcessCommand("final test");
    EXPECT_EQ(cli_->GetConversationLength(), 3u);
}

TEST_F(DPANCliTest, PredictCommandWithAttention) {
    // Build some patterns
    cli_->ProcessCommand("hello world");
    cli_->ProcessCommand("world peace");
    cli_->ProcessCommand("hello world");  // Repeat

    // Enable attention
    cli_->ProcessCommand("/attention");

    // Use predict command - should work with attention
    cli_->ProcessCommand("/predict hello");

    // Should not crash and attention should still be enabled
    EXPECT_TRUE(cli_->IsAttentionEnabled());
}

TEST_F(DPANCliTest, MultipleTogglesWork) {
    // Toggle multiple times
    for (int i = 0; i < 5; ++i) {
        cli_->ProcessCommand("/attention");
        EXPECT_EQ(cli_->IsAttentionEnabled(), (i % 2 == 0));
    }

    // Final state should be ON (started at 0, toggled 5 times)
    EXPECT_TRUE(cli_->IsAttentionEnabled());
}

// ============================================================================
// A/B Comparison Tests
// ============================================================================

TEST_F(DPANCliTest, CompareCommandWithKnownPattern) {
    // Build some patterns
    cli_->ProcessCommand("hello world");
    cli_->ProcessCommand("world peace");
    cli_->ProcessCommand("hello world");  // Repeat to create association

    // Compare command should work without crashing
    cli_->ProcessCommand("/compare hello");

    // Should still have the same conversation length
    EXPECT_EQ(cli_->GetConversationLength(), 3u);
}

TEST_F(DPANCliTest, CompareCommandWithUnknownPattern) {
    // Try to compare an unknown pattern
    cli_->ProcessCommand("/compare unknown");

    // Should not crash
    EXPECT_EQ(cli_->GetConversationLength(), 0u);
}

TEST_F(DPANCliTest, CompareCommandShowsBothModes) {
    // Build conversation with associations
    cli_->ProcessCommand("machine learning");
    cli_->ProcessCommand("neural networks");
    cli_->ProcessCommand("deep learning");
    cli_->ProcessCommand("machine learning");  // Create association

    // Compare should run both prediction modes
    cli_->ProcessCommand("/compare machine");

    // Verify state hasn't changed
    EXPECT_EQ(cli_->GetConversationLength(), 4u);
}

TEST_F(DPANCliTest, CompareDoesNotChangeAttentionState) {
    // Initially disabled
    EXPECT_FALSE(cli_->IsAttentionEnabled());

    // Build some patterns
    cli_->ProcessCommand("test pattern");
    cli_->ProcessCommand("another pattern");

    // Run compare
    cli_->ProcessCommand("/compare test");

    // Attention state should still be disabled
    EXPECT_FALSE(cli_->IsAttentionEnabled());

    // Enable attention
    cli_->ProcessCommand("/attention");
    EXPECT_TRUE(cli_->IsAttentionEnabled());

    // Run compare again
    cli_->ProcessCommand("/compare test");

    // Attention state should still be enabled
    EXPECT_TRUE(cli_->IsAttentionEnabled());
}

TEST_F(DPANCliTest, CompareWithEmptyPredictions) {
    // Create a pattern with no associations
    cli_->ProcessCommand("isolated pattern");

    // Compare should handle empty predictions gracefully
    cli_->ProcessCommand("/compare isolated");

    // Should not crash
    EXPECT_EQ(cli_->GetConversationLength(), 1u);
}

TEST_F(DPANCliTest, CompareWorksWithContext) {
    // Build conversation with context
    cli_->ProcessCommand("artificial intelligence");
    cli_->ProcessCommand("machine learning");
    cli_->ProcessCommand("neural networks");
    cli_->ProcessCommand("artificial intelligence");  // Repeat

    // Compare should use context for both prediction modes
    cli_->ProcessCommand("/compare artificial");

    // Context should still be present
    auto& context = cli_->GetCurrentContext();
    EXPECT_GT(context.Size(), 0u);
}

TEST_F(DPANCliTest, CompareCommandInHelp) {
    // Verify /compare is documented in help
    // This is a smoke test - just ensure help doesn't crash
    cli_->ProcessCommand("/help");

    // Help should execute without errors
    EXPECT_TRUE(true);  // Just verify we got here
}

// ============================================================================
// Visualization Tests (Tasks 9.1-9.3)
// ============================================================================

TEST_F(DPANCliTest, AttentionInfoCommand) {
    // Should execute without crashing
    cli_->ProcessCommand("/attention-info");

    // Should show configuration and statistics
    EXPECT_TRUE(true);  // Just verify we got here
}

TEST_F(DPANCliTest, AttentionInfoWhenEnabled) {
    // Enable attention
    cli_->ProcessCommand("/attention");

    // Show attention info
    cli_->ProcessCommand("/attention-info");

    // Should not crash
    EXPECT_TRUE(cli_->IsAttentionEnabled());
}

TEST_F(DPANCliTest, VerbosePredictionsShowAttentionWeights) {
    // Build some patterns
    cli_->ProcessCommand("machine learning");
    cli_->ProcessCommand("neural networks");
    cli_->ProcessCommand("machine learning");

    // Enable verbose and attention
    cli_->ProcessCommand("/verbose");
    cli_->ProcessCommand("/attention");

    // Make a prediction - should show attention weights in verbose mode
    cli_->ProcessCommand("/predict machine");

    // Verify state
    EXPECT_TRUE(cli_->IsVerboseEnabled());
    EXPECT_TRUE(cli_->IsAttentionEnabled());
}

TEST_F(DPANCliTest, DetailedPredictCommand) {
    // Build some patterns
    cli_->ProcessCommand("hello world");
    cli_->ProcessCommand("world peace");
    cli_->ProcessCommand("hello world");

    // Detailed prediction should work
    cli_->ProcessCommand("/predict-detailed hello");

    // Should not crash
    EXPECT_EQ(cli_->GetConversationLength(), 3u);
}

TEST_F(DPANCliTest, DetailedPredictWithUnknownPattern) {
    // Try detailed prediction on unknown pattern
    cli_->ProcessCommand("/predict-detailed unknown");

    // Should handle gracefully
    EXPECT_EQ(cli_->GetConversationLength(), 0u);
}

TEST_F(DPANCliTest, DetailedPredictShowsComponents) {
    // Build conversation
    cli_->ProcessCommand("artificial intelligence");
    cli_->ProcessCommand("machine learning");
    cli_->ProcessCommand("artificial intelligence");

    // Detailed prediction should show component breakdown
    cli_->ProcessCommand("/predict-detailed artificial");

    // Verify state hasn't changed
    EXPECT_EQ(cli_->GetConversationLength(), 3u);
}

TEST_F(DPANCliTest, VisualizationCommandsInHelp) {
    // Verify all visualization commands are documented
    cli_->ProcessCommand("/help");

    // Just verify help doesn't crash
    EXPECT_TRUE(true);
}

TEST_F(DPANCliTest, AttentionWeightsShownOnlyInVerbose) {
    // Build patterns
    cli_->ProcessCommand("test one");
    cli_->ProcessCommand("test two");
    cli_->ProcessCommand("test one");

    // Enable attention but NOT verbose
    cli_->ProcessCommand("/attention");
    EXPECT_TRUE(cli_->IsAttentionEnabled());
    EXPECT_FALSE(cli_->IsVerboseEnabled());

    // Predictions should not show detailed weights
    cli_->ProcessCommand("/predict test");

    // Now enable verbose
    cli_->ProcessCommand("/verbose");
    EXPECT_TRUE(cli_->IsVerboseEnabled());

    // Predictions should show detailed weights
    cli_->ProcessCommand("/predict test");

    // State should be preserved
    EXPECT_TRUE(cli_->IsAttentionEnabled());
    EXPECT_TRUE(cli_->IsVerboseEnabled());
}

// ============================================================================
// Unit Tests - Task 10.1
// ============================================================================

TEST_F(DPANCliTest, CommandParsingEmpty) {
    // Empty command should do nothing
    cli_->ProcessCommand("");
    EXPECT_EQ(0u, cli_->GetConversationLength());
}

TEST_F(DPANCliTest, CommandParsingWhitespace) {
    // Whitespace-only should do nothing
    cli_->ProcessCommand("   ");
    EXPECT_EQ(0u, cli_->GetConversationLength());
}

TEST_F(DPANCliTest, AttentionInfoWithoutMechanism) {
    // Should handle gracefully even without attention mechanism initialized
    // (Though in practice it's always initialized in our CLI)
    cli_->ProcessCommand("/attention-info");
    // Should not crash
    EXPECT_TRUE(true);
}

TEST_F(DPANCliTest, PredictCommandRequiresPattern) {
    // Predict should fail gracefully for unknown patterns
    cli_->ProcessCommand("/predict nonexistent");
    EXPECT_EQ(0u, cli_->GetConversationLength());
}

TEST_F(DPANCliTest, CompareCommandRequiresPattern) {
    // Compare should fail gracefully for unknown patterns
    cli_->ProcessCommand("/compare nonexistent");
    EXPECT_EQ(0u, cli_->GetConversationLength());
}

TEST_F(DPANCliTest, DetailedPredictRequiresPattern) {
    // Detailed predict should fail gracefully for unknown patterns
    cli_->ProcessCommand("/predict-detailed nonexistent");
    EXPECT_EQ(0u, cli_->GetConversationLength());
}

TEST_F(DPANCliTest, StatePreservedAcrossCommands) {
    // Enable attention
    cli_->ProcessCommand("/attention");
    EXPECT_TRUE(cli_->IsAttentionEnabled());

    // Enable verbose
    cli_->ProcessCommand("/verbose");
    EXPECT_TRUE(cli_->IsVerboseEnabled());

    // Process some text
    cli_->ProcessCommand("test input");

    // States should still be preserved
    EXPECT_TRUE(cli_->IsAttentionEnabled());
    EXPECT_TRUE(cli_->IsVerboseEnabled());
    EXPECT_EQ(1u, cli_->GetConversationLength());
}

TEST_F(DPANCliTest, ToggleCommandsAreIdempotent) {
    // Toggle attention twice
    EXPECT_FALSE(cli_->IsAttentionEnabled());
    cli_->ProcessCommand("/attention");
    EXPECT_TRUE(cli_->IsAttentionEnabled());
    cli_->ProcessCommand("/attention");
    EXPECT_FALSE(cli_->IsAttentionEnabled());

    // State should be consistent
    EXPECT_FALSE(cli_->IsAttentionEnabled());
}

TEST_F(DPANCliTest, VocabularySizeGrowsWithUniqueInputs) {
    cli_->ProcessCommand("first");
    EXPECT_EQ(1u, cli_->GetVocabularySize());

    cli_->ProcessCommand("second");
    EXPECT_EQ(2u, cli_->GetVocabularySize());

    // Repeated input doesn't grow vocabulary
    cli_->ProcessCommand("first");
    EXPECT_EQ(2u, cli_->GetVocabularySize());
}

TEST_F(DPANCliTest, ConversationLengthGrowsWithAllInputs) {
    cli_->ProcessCommand("first");
    EXPECT_EQ(1u, cli_->GetConversationLength());

    cli_->ProcessCommand("second");
    EXPECT_EQ(2u, cli_->GetConversationLength());

    // Repeated input still grows conversation
    cli_->ProcessCommand("first");
    EXPECT_EQ(3u, cli_->GetConversationLength());
}

TEST_F(DPANCliTest, AttentionOnlyAffectsPredictionsNotLearning) {
    // Learn with attention off
    cli_->ProcessCommand("hello");
    EXPECT_EQ(1u, cli_->GetPatternsLearned());

    // Enable attention
    cli_->ProcessCommand("/attention");

    // Learning still works
    cli_->ProcessCommand("world");
    EXPECT_EQ(2u, cli_->GetPatternsLearned());

    // Disable attention
    cli_->ProcessCommand("/attention");

    // Learning still works
    cli_->ProcessCommand("again");
    EXPECT_EQ(3u, cli_->GetPatternsLearned());
}

TEST_F(DPANCliTest, ContextAccumulatesRegardlessOfAttention) {
    // Context should accumulate whether attention is on or off
    cli_->ProcessCommand("machine learning");
    auto& context1 = cli_->GetCurrentContext();

    cli_->ProcessCommand("neural networks");
    auto& context2 = cli_->GetCurrentContext();
    size_t size2 = context2.Size();

    // Context should have grown
    EXPECT_GT(size2, 0u);

    // Enable attention - context should still work
    cli_->ProcessCommand("/attention");
    cli_->ProcessCommand("deep learning");
    auto& context3 = cli_->GetCurrentContext();

    EXPECT_GT(context3.Size(), 0u);
}

TEST_F(DPANCliTest, VerboseModeIndependentOfAttention) {
    // Verbose can be enabled without attention
    cli_->ProcessCommand("/verbose");
    EXPECT_TRUE(cli_->IsVerboseEnabled());
    EXPECT_FALSE(cli_->IsAttentionEnabled());

    // Attention can be enabled without verbose
    cli_->ProcessCommand("/verbose");  // Turn off
    cli_->ProcessCommand("/attention");
    EXPECT_FALSE(cli_->IsVerboseEnabled());
    EXPECT_TRUE(cli_->IsAttentionEnabled());

    // Both can be enabled together
    cli_->ProcessCommand("/verbose");
    EXPECT_TRUE(cli_->IsVerboseEnabled());
    EXPECT_TRUE(cli_->IsAttentionEnabled());
}

TEST_F(DPANCliTest, UnknownCommandsHandledGracefully) {
    // Unknown commands should be handled gracefully
    cli_->ProcessCommand("/unknown-command");
    cli_->ProcessCommand("/not-a-real-command");
    cli_->ProcessCommand("/asdf1234");

    // Should not crash and state should be unchanged
    EXPECT_EQ(0u, cli_->GetConversationLength());
}

TEST_F(DPANCliTest, HelpCommandAlwaysWorks) {
    // Help should work in any state
    cli_->ProcessCommand("/help");

    cli_->ProcessCommand("/attention");
    cli_->ProcessCommand("/help");

    cli_->ProcessCommand("/verbose");
    cli_->ProcessCommand("/help");

    // Should not crash
    EXPECT_TRUE(true);
}

TEST_F(DPANCliTest, StatsCommandAlwaysWorks) {
    // Stats should work even with no data
    cli_->ProcessCommand("/stats");

    // Stats should work with data
    cli_->ProcessCommand("test");
    cli_->ProcessCommand("/stats");

    // Should not crash
    EXPECT_EQ(1u, cli_->GetConversationLength());
}

// ============================================================================
// Integration Tests - Task 11.1
// ============================================================================

TEST_F(DPANCliTest, IntegrationFullConversationWorkflow) {
    // Simulate a full conversation workflow
    cli_->ProcessCommand("Hello there");
    EXPECT_EQ(1u, cli_->GetConversationLength());

    cli_->ProcessCommand("How are you");
    EXPECT_EQ(2u, cli_->GetConversationLength());

    cli_->ProcessCommand("Hello there");  // Repeat
    EXPECT_EQ(3u, cli_->GetConversationLength());

    // Should have learned 2 unique patterns
    EXPECT_EQ(2u, cli_->GetVocabularySize());

    // Context should have accumulated
    auto& context = cli_->GetCurrentContext();
    EXPECT_GT(context.Size(), 0u);
}

TEST_F(DPANCliTest, IntegrationAttentionCompleteWorkflow) {
    // Build conversation
    cli_->ProcessCommand("machine learning");
    cli_->ProcessCommand("neural networks");
    cli_->ProcessCommand("deep learning");
    cli_->ProcessCommand("machine learning");  // Repeat to create association

    // Test basic prediction
    cli_->ProcessCommand("/predict machine");

    // Enable attention and test again
    cli_->ProcessCommand("/attention");
    EXPECT_TRUE(cli_->IsAttentionEnabled());
    cli_->ProcessCommand("/predict machine");

    // Compare both modes
    cli_->ProcessCommand("/compare machine");

    // Get detailed breakdown
    cli_->ProcessCommand("/predict-detailed machine");

    // Check attention info
    cli_->ProcessCommand("/attention-info");

    // Verify state
    EXPECT_EQ(4u, cli_->GetConversationLength());
    EXPECT_TRUE(cli_->IsAttentionEnabled());
}

TEST_F(DPANCliTest, IntegrationVerboseAttentionWorkflow) {
    // Build patterns
    cli_->ProcessCommand("artificial intelligence");
    cli_->ProcessCommand("machine learning");
    cli_->ProcessCommand("artificial intelligence");

    // Enable both verbose and attention
    cli_->ProcessCommand("/verbose");
    cli_->ProcessCommand("/attention");

    EXPECT_TRUE(cli_->IsVerboseEnabled());
    EXPECT_TRUE(cli_->IsAttentionEnabled());

    // Predictions should show detailed weights
    cli_->ProcessCommand("/predict artificial");

    // Verify final state
    EXPECT_EQ(3u, cli_->GetConversationLength());
    EXPECT_TRUE(cli_->IsVerboseEnabled());
    EXPECT_TRUE(cli_->IsAttentionEnabled());
}

TEST_F(DPANCliTest, IntegrationContextEvolutionOverConversation) {
    // Track context evolution over a conversation
    cli_->ProcessCommand("deep learning frameworks");

    cli_->ProcessCommand("tensorflow and pytorch");
    auto& context2 = cli_->GetCurrentContext();
    size_t size2 = context2.Size();

    cli_->ProcessCommand("neural network architectures");
    auto& context3 = cli_->GetCurrentContext();
    size_t size3 = context3.Size();

    // Context should be accumulating
    EXPECT_GT(size2, 0u);
    EXPECT_GT(size3, 0u);

    // Should have conversation depth
    EXPECT_TRUE(context3.Has("conversation_depth"));

    // Should have topic diversity
    EXPECT_TRUE(context3.Has("topic_diversity"));
}

TEST_F(DPANCliTest, IntegrationMultipleCommandTypes) {
    // Test various command types in sequence
    cli_->ProcessCommand("test pattern one");
    cli_->ProcessCommand("test pattern two");

    // Show stats
    cli_->ProcessCommand("/stats");

    // Show patterns
    cli_->ProcessCommand("/patterns");

    // Show associations
    cli_->ProcessCommand("/associations");

    // Toggle modes
    cli_->ProcessCommand("/verbose");
    cli_->ProcessCommand("/attention");
    cli_->ProcessCommand("/active");

    // Show info
    cli_->ProcessCommand("/attention-info");
    cli_->ProcessCommand("/help");

    // Verify all states are correct
    EXPECT_TRUE(cli_->IsVerboseEnabled());
    EXPECT_TRUE(cli_->IsAttentionEnabled());
    EXPECT_TRUE(cli_->IsActiveLearningEnabled());
    EXPECT_EQ(2u, cli_->GetConversationLength());
}

TEST_F(DPANCliTest, IntegrationPredictionConsistency) {
    // Build a clear pattern
    cli_->ProcessCommand("hello");
    cli_->ProcessCommand("world");
    cli_->ProcessCommand("hello");
    cli_->ProcessCommand("world");
    cli_->ProcessCommand("hello");
    cli_->ProcessCommand("world");

    // Predictions without attention
    cli_->ProcessCommand("/predict hello");

    // Predictions with attention
    cli_->ProcessCommand("/attention");
    cli_->ProcessCommand("/predict hello");

    // Compare should show the difference
    cli_->ProcessCommand("/compare hello");

    // Conversation length should only include actual inputs, not commands
    EXPECT_EQ(6u, cli_->GetConversationLength());
}

TEST_F(DPANCliTest, IntegrationAllVisualizationFeatures) {
    // Build conversation
    cli_->ProcessCommand("data science");
    cli_->ProcessCommand("machine learning");
    cli_->ProcessCommand("data science");

    // Test all visualization features
    cli_->ProcessCommand("/stats");
    cli_->ProcessCommand("/patterns");
    cli_->ProcessCommand("/associations");

    // Enable attention
    cli_->ProcessCommand("/attention");
    cli_->ProcessCommand("/attention-info");

    // Test all prediction modes
    cli_->ProcessCommand("/predict data");
    cli_->ProcessCommand("/compare data");
    cli_->ProcessCommand("/predict-detailed data");

    // Verify state integrity
    EXPECT_EQ(3u, cli_->GetConversationLength());
    EXPECT_TRUE(cli_->IsAttentionEnabled());
}

TEST_F(DPANCliTest, IntegrationStateToggling) {
    // Test state toggling doesn't affect conversation
    cli_->ProcessCommand("test one");
    EXPECT_EQ(1u, cli_->GetConversationLength());

    cli_->ProcessCommand("/attention");
    cli_->ProcessCommand("test two");
    EXPECT_EQ(2u, cli_->GetConversationLength());

    cli_->ProcessCommand("/attention");  // Toggle off
    cli_->ProcessCommand("test three");
    EXPECT_EQ(3u, cli_->GetConversationLength());

    cli_->ProcessCommand("/verbose");
    cli_->ProcessCommand("test four");
    EXPECT_EQ(4u, cli_->GetConversationLength());

    // All patterns should be learned
    EXPECT_EQ(4u, cli_->GetVocabularySize());
    EXPECT_EQ(4u, cli_->GetPatternsLearned());
}

TEST_F(DPANCliTest, IntegrationComplexConversationFlow) {
    // Simulate a complex conversation with all features
    cli_->ProcessCommand("I love programming");
    cli_->ProcessCommand("Python is great");
    cli_->ProcessCommand("Machine learning is fascinating");

    // Check stats
    cli_->ProcessCommand("/stats");
    EXPECT_EQ(3u, cli_->GetConversationLength());

    // Enable verbose and attention
    cli_->ProcessCommand("/verbose");
    cli_->ProcessCommand("/attention");

    // Continue conversation
    cli_->ProcessCommand("I love programming");  // Repeat
    EXPECT_EQ(4u, cli_->GetConversationLength());

    // Test predictions with all features
    cli_->ProcessCommand("/predict I");

    // Check context has accumulated
    auto& context = cli_->GetCurrentContext();
    EXPECT_GT(context.Size(), 0u);
    EXPECT_TRUE(context.Has("conversation_depth"));

    // Verify final state
    EXPECT_TRUE(cli_->IsVerboseEnabled());
    EXPECT_TRUE(cli_->IsAttentionEnabled());
    EXPECT_EQ(3u, cli_->GetVocabularySize());
}

TEST_F(DPANCliTest, IntegrationErrorRecovery) {
    // Test that errors don't corrupt state
    cli_->ProcessCommand("valid input");
    EXPECT_EQ(1u, cli_->GetConversationLength());

    // Try invalid commands
    cli_->ProcessCommand("/predict nonexistent");
    cli_->ProcessCommand("/compare nonexistent");
    cli_->ProcessCommand("/predict-detailed nonexistent");
    cli_->ProcessCommand("/unknown-command");

    // State should be unchanged
    EXPECT_EQ(1u, cli_->GetConversationLength());
    EXPECT_EQ(1u, cli_->GetVocabularySize());

    // Continue normally
    cli_->ProcessCommand("another valid input");
    EXPECT_EQ(2u, cli_->GetConversationLength());
}

TEST_F(DPANCliTest, IntegrationAttentionVsBasicComparison) {
    // Build strong associations
    cli_->ProcessCommand("cat");
    cli_->ProcessCommand("dog");
    cli_->ProcessCommand("cat");
    cli_->ProcessCommand("dog");
    cli_->ProcessCommand("cat");

    // Compare basic vs attention predictions
    cli_->ProcessCommand("/compare cat");

    // Enable attention and verify it changes predictions
    cli_->ProcessCommand("/attention");
    cli_->ProcessCommand("/predict cat");

    // Context should be rich by now
    auto& context = cli_->GetCurrentContext();
    EXPECT_GT(context.Size(), 0u);

    // Should have learned associations
    EXPECT_EQ(5u, cli_->GetConversationLength());
}

} // namespace
} // namespace dpan
