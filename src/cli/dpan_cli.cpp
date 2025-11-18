// File: src/cli/dpan_cli.cpp
//
// Interactive CLI interface for DPAN
// Allows direct communication, active learning, and pattern exploration
//
// Features:
// - Interactive conversation mode
// - File upload and batch processing
// - Active learning (DPAN requests data when uncertain)
// - Pattern inspection and statistics
// - Session persistence

#include "core/pattern_engine.hpp"
#include "association/association_learning_system.hpp"
#include "storage/persistent_backend.hpp"
#include "storage/memory_backend.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <map>
#include <filesystem>
#include <chrono>
#include <thread>

using namespace dpan;

/// CLI State and Configuration
class DPANCli {
public:
    DPANCli() {
        InitializeEngine();
        InitializeAssociations();
    }

    void Run() {
        PrintWelcome();

        std::string line;
        while (running_) {
            std::cout << prompt_;
            std::getline(std::cin, line);

            if (std::cin.eof() || line == "exit" || line == "quit") {
                break;
            }

            ProcessCommand(line);
        }

        Shutdown();
    }

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

    void InitializeEngine() {
        // Use persistent storage for learning sessions
        PersistentBackend::Config storage_config;
        storage_config.db_path = session_file_;
        storage_ = std::make_shared<PersistentBackend>(storage_config);

        PatternEngine::Config engine_config;
        engine_config.similarity_metric = "context";
        engine_config.enable_auto_refinement = true;
        engine_config.enable_indexing = true;

        // Configure for text/sequence learning
        engine_config.extraction_config.modality = DataModality::TEXT;
        engine_config.extraction_config.min_pattern_size = 1;
        engine_config.extraction_config.max_pattern_size = 1000;
        engine_config.extraction_config.feature_dimension = 64;

        // Lower thresholds for better learning
        engine_config.matching_config.similarity_threshold = 0.60f;
        engine_config.matching_config.strong_match_threshold = 0.75f;

        engine_ = std::make_unique<PatternEngine>(engine_config);
    }

    void InitializeAssociations() {
        AssociationLearningSystem::Config assoc_config;

        // Configure for conversation learning
        assoc_config.co_occurrence.window_size = std::chrono::seconds(30);

        assoc_config.formation.min_co_occurrences = 2;
        assoc_config.formation.initial_strength = 0.3f;

        assoc_config.competition.competition_factor = 0.3f;

        assoc_config.enable_auto_maintenance = true;
        assoc_config.prune_threshold = 0.1f;

        assoc_system_ = std::make_unique<AssociationLearningSystem>(assoc_config);
    }

    void PrintWelcome() {
        std::cout << R"(
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   DPAN Interactive Learning Interface                       ║
║   Dynamic Pattern Association Network                       ║
║                                                              ║
║   A neural network that learns and grows from interaction   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

Type 'help' for available commands, or just start talking!
The system will learn from everything you say.

)";
        LoadSessionIfExists();
    }

    void ProcessCommand(const std::string& input) {
        if (input.empty()) return;

        // Check if it's a command (starts with /)
        if (input[0] == '/') {
            HandleCommand(input.substr(1));
        } else {
            // It's conversational input - learn from it
            HandleConversation(input);
        }
    }

    void HandleCommand(const std::string& cmd) {
        std::istringstream iss(cmd);
        std::string command;
        iss >> command;

        if (command == "help") {
            ShowHelp();
        } else if (command == "stats") {
            ShowStatistics();
        } else if (command == "learn") {
            std::string filepath;
            iss >> filepath;
            LearnFromFile(filepath);
        } else if (command == "active") {
            ToggleActiveLearning();
        } else if (command == "save") {
            SaveSession();
        } else if (command == "load") {
            LoadSession();
        } else if (command == "patterns") {
            ShowPatterns();
        } else if (command == "associations") {
            ShowAssociations();
        } else if (command == "predict") {
            std::string text;
            std::getline(iss, text);
            PredictNext(text);
        } else if (command == "verbose") {
            verbose_ = !verbose_;
            std::cout << "Verbose mode: " << (verbose_ ? "ON" : "OFF") << "\n";
        } else if (command == "reset") {
            ResetSession();
        } else if (command == "clear") {
            std::cout << "\033[2J\033[1;1H"; // Clear screen
        } else {
            std::cout << "Unknown command: /" << command << "\n";
            std::cout << "Type '/help' for available commands.\n";
        }
    }

    void HandleConversation(const std::string& text) {
        total_inputs_++;

        if (verbose_) {
            std::cout << "[Processing: \"" << text << "\"]\n";
        }

        // Convert text to bytes
        auto bytes = TextToBytes(text);

        // Process with engine
        auto result = engine_->ProcessInput(bytes, DataModality::TEXT);

        // Track patterns
        PatternID primary_pattern;
        if (!result.created_patterns.empty()) {
            primary_pattern = result.created_patterns[0];
            patterns_learned_ += result.created_patterns.size();

            // Store text mapping
            text_to_pattern_[text] = primary_pattern;
            pattern_to_text_[primary_pattern] = text;

            if (verbose_) {
                std::cout << "[Created " << result.created_patterns.size()
                         << " new pattern(s)]\n";
            }
        } else if (!result.activated_patterns.empty()) {
            primary_pattern = result.activated_patterns[0];

            if (verbose_) {
                std::cout << "[Activated existing pattern]\n";
            }
        } else {
            std::cout << "[No patterns matched or created - learning...]\n";
            return;
        }

        // Record in association system
        ContextVector context;
        assoc_system_->RecordPatternActivation(primary_pattern, context);
        conversation_history_.push_back(primary_pattern);

        // Form associations with recent patterns
        if (conversation_history_.size() > 1) {
            // Use the storage backend as PatternDatabase
            assoc_system_->FormAssociationsForPattern(primary_pattern, *storage_);
        }

        // Generate response
        GenerateResponse(primary_pattern);

        // Active learning check
        if (active_learning_mode_ && ShouldRequestMoreData(result)) {
            RequestMoreData(text);
        }
    }

    void GenerateResponse(PatternID input_pattern) {
        // Use associations to predict next patterns
        auto predictions = assoc_system_->Predict(input_pattern, 3);

        if (predictions.empty()) {
            std::cout << "→ [Learning... I don't have enough context yet to respond.]\n";
            return;
        }

        // Try to generate text response from predicted patterns
        std::vector<std::string> response_candidates;
        for (auto pred : predictions) {
            if (pattern_to_text_.count(pred)) {
                response_candidates.push_back(pattern_to_text_[pred]);
            }
        }

        if (!response_candidates.empty()) {
            // Get association strengths
            const auto& matrix = assoc_system_->GetAssociationMatrix();
            auto* edge = matrix.GetAssociation(input_pattern, predictions[0]);
            float confidence = edge ? edge->GetStrength() : 0.0f;

            std::cout << "→ " << response_candidates[0];
            if (verbose_) {
                std::cout << " [confidence: " << std::fixed << std::setprecision(2)
                         << confidence << "]";
            }
            std::cout << "\n";

            if (response_candidates.size() > 1 && verbose_) {
                std::cout << "   Other possibilities: ";
                for (size_t i = 1; i < std::min<size_t>(3, response_candidates.size()); ++i) {
                    std::cout << "\"" << response_candidates[i] << "\" ";
                }
                std::cout << "\n";
            }
        } else {
            std::cout << "→ [I predicted " << predictions.size()
                     << " pattern(s), but haven't learned text for them yet.]\n";
        }
    }

    bool ShouldRequestMoreData(const PatternEngine::ProcessResult& result) {
        // Request more data if:
        // 1. No patterns were created or activated
        // 2. Low confidence matches
        // 3. Predictions are weak

        if (result.created_patterns.empty() && result.activated_patterns.empty()) {
            return true;
        }

        if (!result.activated_patterns.empty()) {
            auto pattern = engine_->GetPattern(result.activated_patterns[0]);
            if (pattern.has_value() && pattern->GetConfidenceScore() < 0.6f) {
                return true;
            }
        }

        return false;
    }

    void RequestMoreData(const std::string& context) {
        std::cout << "\n[ACTIVE LEARNING] I'm not confident about that. ";
        std::cout << "Can you tell me more or rephrase?\n";
    }

    void LearnFromFile(const std::string& filepath) {
        if (!std::filesystem::exists(filepath)) {
            std::cout << "Error: File not found: " << filepath << "\n";
            return;
        }

        std::cout << "Learning from file: " << filepath << "\n";

        std::ifstream file(filepath);
        std::string line;
        size_t lines_processed = 0;

        auto start = std::chrono::steady_clock::now();

        while (std::getline(file, line)) {
            if (!line.empty()) {
                HandleConversation(line);
                lines_processed++;

                if (lines_processed % 100 == 0) {
                    std::cout << "\r  Processed " << lines_processed << " lines...";
                    std::cout.flush();
                }
            }
        }

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "\n✓ Learned from " << lines_processed << " lines in "
                 << duration.count() << " ms\n";
        std::cout << "  Patterns created: " << patterns_learned_ << "\n";
    }

    void ShowHelp() {
        std::cout << R"(
Available Commands:
===================

Conversation:
  <text>              Learn from and respond to text input
  /predict <text>     Show what the system predicts will follow

Learning:
  /learn <file>       Learn from a text file (one line = one input)
  /active             Toggle active learning mode (DPAN asks questions)

Information:
  /stats              Show learning statistics
  /patterns           List learned patterns
  /associations       Show association graph statistics
  /verbose            Toggle verbose output

Session Management:
  /save               Save current session
  /load               Load previous session
  /reset              Reset session (clear all learned data)

Utility:
  /clear              Clear screen
  /help               Show this help
  exit, quit          Exit the program

Examples:
  Hello world
  /learn conversation.txt
  /predict The cat sat on the
  /stats

)";
    }

    void ShowStatistics() {
        auto stats = engine_->GetStatistics();
        auto storage_stats = storage_->GetStats();

        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════╗\n";
        std::cout << "║         DPAN Learning Statistics         ║\n";
        std::cout << "╚══════════════════════════════════════════╝\n\n";

        std::cout << "Session:\n";
        std::cout << "  Inputs processed: " << total_inputs_ << "\n";
        std::cout << "  Patterns learned: " << patterns_learned_ << "\n";
        std::cout << "  Conversation length: " << conversation_history_.size() << "\n";
        std::cout << "  Vocabulary size: " << text_to_pattern_.size() << " unique inputs\n\n";

        std::cout << "Patterns:\n";
        std::cout << "  Total patterns: " << stats.total_patterns << "\n";
        std::cout << "  Atomic: " << stats.atomic_patterns << "\n";
        std::cout << "  Composite: " << stats.composite_patterns << "\n";
        std::cout << "  Average confidence: " << std::fixed << std::setprecision(2)
                 << stats.avg_confidence << "\n\n";

        std::cout << "Associations:\n";
        auto assoc_stats = assoc_system_->GetStatistics();
        std::cout << "  Total associations: " << assoc_stats.total_associations << "\n";
        std::cout << "  Average strength: " << std::fixed << std::setprecision(2)
                 << assoc_stats.average_strength << "\n";
        std::cout << "  Strongest association: " << std::fixed << std::setprecision(2)
                 << assoc_stats.max_strength << "\n\n";

        std::cout << "Storage:\n";
        std::cout << "  Database: " << session_file_ << "\n";
        std::cout << "  Size: " << storage_stats.disk_usage_bytes / 1024 << " KB\n";
        std::cout << "  Active learning: " << (active_learning_mode_ ? "ON" : "OFF") << "\n\n";
    }

    void ShowPatterns() {
        std::cout << "\nLearned Patterns (text mappings):\n";
        std::cout << "================================\n\n";

        size_t count = 0;
        for (const auto& [text, pattern_id] : text_to_pattern_) {
            auto pattern = engine_->GetPattern(pattern_id);
            if (pattern.has_value()) {
                std::cout << std::setw(4) << ++count << ". \"" << text << "\"\n";
                std::cout << "      Pattern ID: " << pattern_id.value() << "\n";
                std::cout << "      Confidence: " << std::fixed << std::setprecision(2)
                         << pattern->GetConfidenceScore() << "\n";

                // Show associations
                auto predictions = assoc_system_->Predict(pattern_id, 3);
                if (!predictions.empty()) {
                    std::cout << "      Leads to: ";
                    for (size_t i = 0; i < std::min<size_t>(3, predictions.size()); ++i) {
                        if (pattern_to_text_.count(predictions[i])) {
                            std::cout << "\"" << pattern_to_text_[predictions[i]] << "\" ";
                        }
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";

                if (count >= 20 && !verbose_) {
                    std::cout << "... (" << (text_to_pattern_.size() - 20)
                             << " more patterns)\n";
                    std::cout << "Use /verbose to see all patterns\n";
                    break;
                }
            }
        }

        if (count == 0) {
            std::cout << "No patterns learned yet. Start a conversation!\n";
        }
    }

    void ShowAssociations() {
        auto stats = assoc_system_->GetStatistics();
        const auto& matrix = assoc_system_->GetAssociationMatrix();

        std::cout << "\nAssociation Graph:\n";
        std::cout << "==================\n\n";
        std::cout << "Total associations: " << stats.total_associations << "\n";
        std::cout << "Average strength: " << std::fixed << std::setprecision(2)
                 << stats.average_strength << "\n\n";

        if (stats.total_associations == 0) {
            std::cout << "No associations formed yet. Keep learning!\n";
            return;
        }

        std::cout << "Strongest associations:\n";

        // Find strongest associations
        std::vector<std::tuple<PatternID, PatternID, float>> strong_assocs;

        for (const auto& [text, pattern_id] : text_to_pattern_) {
            auto outgoing = matrix.GetOutgoingAssociations(pattern_id);
            for (const auto* edge : outgoing) {
                if (edge) {
                    strong_assocs.push_back({edge->GetSource(),
                                            edge->GetTarget(),
                                            edge->GetStrength()});
                }
            }
        }

        // Sort by strength
        std::sort(strong_assocs.begin(), strong_assocs.end(),
                 [](const auto& a, const auto& b) {
                     return std::get<2>(a) > std::get<2>(b);
                 });

        // Display top 10
        size_t display_count = std::min<size_t>(10, strong_assocs.size());
        for (size_t i = 0; i < display_count; ++i) {
            auto [source, target, strength] = strong_assocs[i];

            std::string source_text = pattern_to_text_.count(source) ?
                                     pattern_to_text_[source] : "<unknown>";
            std::string target_text = pattern_to_text_.count(target) ?
                                     pattern_to_text_[target] : "<unknown>";

            std::cout << "  " << (i+1) << ". \"" << source_text << "\" → \""
                     << target_text << "\" [" << std::fixed << std::setprecision(3)
                     << strength << "]\n";
        }
    }

    void PredictNext(const std::string& text) {
        std::string query = text;
        // Trim leading space if present
        if (!query.empty() && query[0] == ' ') {
            query = query.substr(1);
        }

        if (text_to_pattern_.count(query) == 0) {
            std::cout << "Unknown input: \"" << query << "\"\n";
            std::cout << "I haven't learned this pattern yet.\n";
            return;
        }

        PatternID pattern = text_to_pattern_[query];
        auto predictions = assoc_system_->Predict(pattern, 5);

        if (predictions.empty()) {
            std::cout << "No predictions available for: \"" << query << "\"\n";
            return;
        }

        std::cout << "\nPredictions for \"" << query << "\":\n";

        const auto& matrix = assoc_system_->GetAssociationMatrix();
        for (size_t i = 0; i < predictions.size(); ++i) {
            auto* edge = matrix.GetAssociation(pattern, predictions[i]);
            float strength = edge ? edge->GetStrength() : 0.0f;

            std::string pred_text = pattern_to_text_.count(predictions[i]) ?
                                   pattern_to_text_[predictions[i]] : "<unknown>";

            std::cout << "  " << (i+1) << ". \"" << pred_text << "\" ["
                     << std::fixed << std::setprecision(3) << strength << "]\n";
        }
    }

    void ToggleActiveLearning() {
        active_learning_mode_ = !active_learning_mode_;
        std::cout << "Active learning mode: "
                 << (active_learning_mode_ ? "ON" : "OFF") << "\n";

        if (active_learning_mode_) {
            std::cout << "DPAN will now ask for clarification when uncertain.\n";
        }
    }

    void SaveSession() {
        std::cout << "Saving session to " << session_file_ << "...\n";

        // Storage is already persistent, but save associations
        std::string assoc_file = session_file_ + ".associations";
        if (assoc_system_->Save(assoc_file)) {
            std::cout << "✓ Session saved successfully\n";
        } else {
            std::cout << "✗ Failed to save associations\n";
        }

        // Save text mappings
        std::string mapping_file = session_file_ + ".mappings";
        std::ofstream out(mapping_file, std::ios::binary);

        size_t count = text_to_pattern_.size();
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));

        for (const auto& [text, pattern_id] : text_to_pattern_) {
            size_t text_len = text.size();
            out.write(reinterpret_cast<const char*>(&text_len), sizeof(text_len));
            out.write(text.data(), text_len);

            uint64_t id_val = pattern_id.value();
            out.write(reinterpret_cast<const char*>(&id_val), sizeof(id_val));
        }

        std::cout << "✓ Saved " << count << " text mappings\n";
    }

    void LoadSession() {
        LoadSessionIfExists();
    }

    void LoadSessionIfExists() {
        if (!std::filesystem::exists(session_file_)) {
            std::cout << "No previous session found. Starting fresh.\n";
            return;
        }

        std::cout << "Loading previous session...\n";

        // Load associations
        std::string assoc_file = session_file_ + ".associations";
        if (std::filesystem::exists(assoc_file)) {
            if (assoc_system_->Load(assoc_file)) {
                std::cout << "✓ Loaded associations\n";
            }
        }

        // Load text mappings
        std::string mapping_file = session_file_ + ".mappings";
        if (std::filesystem::exists(mapping_file)) {
            std::ifstream in(mapping_file, std::ios::binary);

            size_t count;
            in.read(reinterpret_cast<char*>(&count), sizeof(count));

            for (size_t i = 0; i < count; ++i) {
                size_t text_len;
                in.read(reinterpret_cast<char*>(&text_len), sizeof(text_len));

                std::string text(text_len, '\0');
                in.read(&text[0], text_len);

                uint64_t id_val;
                in.read(reinterpret_cast<char*>(&id_val), sizeof(id_val));

                PatternID pattern_id(id_val);
                text_to_pattern_[text] = pattern_id;
                pattern_to_text_[pattern_id] = text;
            }

            std::cout << "✓ Loaded " << count << " text mappings\n";
        }

        auto stats = engine_->GetStatistics();
        std::cout << "Session loaded: " << stats.total_patterns << " patterns\n\n";
    }

    void ResetSession() {
        std::cout << "Are you sure you want to reset? This will erase all learning. (y/N): ";
        std::string confirm;
        std::getline(std::cin, confirm);

        if (confirm != "y" && confirm != "Y") {
            std::cout << "Reset cancelled.\n";
            return;
        }

        // Delete session files
        std::filesystem::remove(session_file_);
        std::filesystem::remove(session_file_ + ".associations");
        std::filesystem::remove(session_file_ + ".mappings");
        std::filesystem::remove(session_file_ + "-wal");
        std::filesystem::remove(session_file_ + "-shm");

        // Reinitialize
        text_to_pattern_.clear();
        pattern_to_text_.clear();
        conversation_history_.clear();
        total_inputs_ = 0;
        patterns_learned_ = 0;

        InitializeEngine();
        InitializeAssociations();

        std::cout << "✓ Session reset. Starting fresh.\n";
    }

    void Shutdown() {
        std::cout << "\nShutting down...\n";
        SaveSession();

        std::cout << "\nSession Summary:\n";
        std::cout << "  Inputs processed: " << total_inputs_ << "\n";
        std::cout << "  Patterns learned: " << patterns_learned_ << "\n";
        std::cout << "  Conversation length: " << conversation_history_.size() << "\n";
        std::cout << "\nThank you for teaching me! Goodbye.\n";
    }

    std::vector<uint8_t> TextToBytes(const std::string& text) {
        return std::vector<uint8_t>(text.begin(), text.end());
    }
};

int main(int argc, char** argv) {
    std::cout << "Initializing DPAN...\n";

    try {
        DPANCli cli;
        cli.Run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
