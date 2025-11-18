// File: src/cli/dpan_cli.hpp
//
// DPAN CLI class definition
// Extracted for testability

#ifndef DPAN_CLI_HPP
#define DPAN_CLI_HPP

#include "core/pattern_engine.hpp"
#include "association/association_learning_system.hpp"
#include "storage/persistent_backend.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace dpan {

/// Interactive CLI interface for DPAN
/// Allows direct communication, active learning, and pattern exploration
class DPANCli {
public:
    DPANCli();

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
    bool IsVerboseEnabled() const { return verbose_; }

    /// Get pattern for text (for testing)
    std::optional<PatternID> GetPatternForText(const std::string& text) const;

    /// Get text for pattern (for testing)
    std::optional<std::string> GetTextForPattern(PatternID pattern_id) const;

    /// Set session file path (for testing with temp files)
    void SetSessionFile(const std::string& path) { session_file_ = path; }

    /// Initialize without loading previous session (for testing)
    void InitializeClean();

private:
    // Engine and system state
    std::unique_ptr<PatternEngine> engine_;
    std::unique_ptr<AssociationLearningSystem> assoc_system_;
    std::shared_ptr<PersistentBackend> storage_;

    bool running_ = true;
    bool active_learning_mode_ = false;
    bool verbose_ = false;
    std::string prompt_ = "dpan> ";
    std::string session_file_ = "dpan_session.db";

    // Learning state
    size_t total_inputs_ = 0;
    size_t patterns_learned_ = 0;
    std::vector<PatternID> conversation_history_;
    std::map<std::string, PatternID> text_to_pattern_;
    std::map<PatternID, std::string> pattern_to_text_;

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
    void LearnFromFile(const std::string& filepath);
    void ToggleActiveLearning();
    void SaveSession();
    void LoadSession();
    void LoadSessionIfExists();
    void ResetSession();
    void Shutdown();

    // Response generation
    void GenerateResponse(PatternID input_pattern);
    bool ShouldRequestMoreData(const PatternEngine::ProcessResult& result);
    void RequestMoreData(const std::string& context);

    // Utilities
    std::vector<uint8_t> TextToBytes(const std::string& text);
};

} // namespace dpan

#endif // DPAN_CLI_HPP
