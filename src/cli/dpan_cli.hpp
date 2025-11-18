// File: src/cli/dpan_cli.hpp
//
// DPAN CLI class definition
// Extracted for testability

#ifndef DPAN_CLI_HPP
#define DPAN_CLI_HPP

#include "core/pattern_engine.hpp"
#include "association/association_learning_system.hpp"
#include "storage/persistent_backend.hpp"
#include "core/types.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>

namespace dpan {

// Forward declarations
class AttentionMechanism;

/// ANSI color codes for terminal output
namespace Color {
    // Reset
    inline const char* RESET = "\033[0m";

    // Regular colors
    inline const char* BLACK = "\033[30m";
    inline const char* RED = "\033[31m";
    inline const char* GREEN = "\033[32m";
    inline const char* YELLOW = "\033[33m";
    inline const char* BLUE = "\033[34m";
    inline const char* MAGENTA = "\033[35m";
    inline const char* CYAN = "\033[36m";
    inline const char* WHITE = "\033[37m";

    // Bold colors
    inline const char* BOLD_RED = "\033[1;31m";
    inline const char* BOLD_GREEN = "\033[1;32m";
    inline const char* BOLD_YELLOW = "\033[1;33m";
    inline const char* BOLD_BLUE = "\033[1;34m";
    inline const char* BOLD_MAGENTA = "\033[1;35m";
    inline const char* BOLD_CYAN = "\033[1;36m";
    inline const char* BOLD_WHITE = "\033[1;37m";

    // Background colors
    inline const char* BG_RED = "\033[41m";
    inline const char* BG_GREEN = "\033[42m";
    inline const char* BG_YELLOW = "\033[43m";

    // Styles
    inline const char* BOLD = "\033[1m";
    inline const char* DIM = "\033[2m";
    inline const char* ITALIC = "\033[3m";
    inline const char* UNDERLINE = "\033[4m";
}

/// Interactive CLI interface for DPAN
/// Allows direct communication, active learning, and pattern exploration
class DPANCli {
public:
    DPANCli();
    ~DPANCli();

    /// Main run loop - interactive mode
    void Run();

    /// Process a single command (for testing)
    void ProcessCommand(const std::string& input);

    /// Get statistics for verification
    size_t GetTotalInputs() const { return total_inputs_; }
    size_t GetPatternsLearned() const { return patterns_learned_; }
    size_t GetConversationLength() const { return conversation_history_.size(); }
    size_t GetVocabularySize() const { return text_to_pattern_.size(); }
    bool IsActiveLearningEnabled() const { return active_learning_mode_; }
    bool IsAttentionEnabled() const { return attention_enabled_; }
    bool IsVerboseEnabled() const { return verbose_; }

    /// Get pattern for text (for testing)
    std::optional<PatternID> GetPatternForText(const std::string& text) const;

    /// Get text for pattern (for testing)
    std::optional<std::string> GetTextForPattern(PatternID pattern_id) const;

    /// Set session file path (for testing with temp files)
    void SetSessionFile(const std::string& path) { session_file_ = path; }

    /// Initialize without loading previous session (for testing)
    void InitializeClean();

    /// Get current context (for testing/inspection)
    const ContextVector& GetCurrentContext() const { return current_context_; }

private:
    // Engine and system state
    std::unique_ptr<PatternEngine> engine_;
    std::unique_ptr<AssociationLearningSystem> assoc_system_;
    std::shared_ptr<PersistentBackend> storage_;
    std::unique_ptr<AttentionMechanism> attention_mechanism_;

    bool running_ = true;
    bool active_learning_mode_ = false;
    bool attention_enabled_ = false;  // Enable attention-based predictions
    bool verbose_ = false;
    bool colors_enabled_ = true;  // Enable colors by default
    std::string prompt_ = "dpan> ";
    std::string session_file_ = "dpan_session.db";

    // Learning state
    size_t total_inputs_ = 0;
    size_t patterns_learned_ = 0;
    std::vector<PatternID> conversation_history_;
    std::map<std::string, PatternID> text_to_pattern_;
    std::map<PatternID, std::string> pattern_to_text_;

    // Context tracking
    ContextVector current_context_;
    std::chrono::steady_clock::time_point last_interaction_time_;
    std::map<std::string, float> recent_topics_;  // Topic -> recency weight

    // Initialization
    void InitializeEngine();
    void InitializeAssociations();
    void PrintWelcome();

    // Command handling
    void HandleCommand(const std::string& cmd);
    void HandleConversation(const std::string& text);

    // Commands
    void ShowHelp();
    void ShowStatistics();
    void ShowPatterns();
    void ShowAssociations();
    void PredictNext(const std::string& text);
    void CompareMode(const std::string& text);
    void LearnFromFile(const std::string& filepath);
    void ToggleActiveLearning();
    void ToggleAttention();
    void SaveSession();
    void LoadSession();
    void LoadSessionIfExists();
    void ResetSession();
    void Shutdown();

    // Response generation
    void GenerateResponse(PatternID input_pattern);
    bool ShouldRequestMoreData(const PatternEngine::ProcessResult& result);
    void RequestMoreData(const std::string& context);

    // Context tracking
    void UpdateContext(const std::string& input_text);
    void ApplyContextDecay();
    void ExtractTopicsFromText(const std::string& text, std::vector<std::string>& topics);
    void UpdateRecentTopics(const std::vector<std::string>& topics);
    void BuildContextVector();

    // Utilities
    std::vector<uint8_t> TextToBytes(const std::string& text);

    // Color helpers
    const char* C(const char* color) const { return colors_enabled_ ? color : ""; }
};

} // namespace dpan

#endif // DPAN_CLI_HPP
