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

#include "dpan_cli.hpp"
#include "storage/memory_backend.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <thread>

namespace dpan {

DPANCli::DPANCli() {
    InitializeEngine();
    InitializeAssociations();
}

void DPANCli::Run() {
    PrintWelcome();

    std::string line;
    while (running_) {
        std::cout << C(Color::BOLD_CYAN) << prompt_ << C(Color::RESET);
        std::getline(std::cin, line);

        if (std::cin.eof() || line == "exit" || line == "quit") {
            break;
        }

        ProcessCommand(line);
    }

    Shutdown();
}

std::optional<PatternID> DPANCli::GetPatternForText(const std::string& text) const {
    auto it = text_to_pattern_.find(text);
    if (it != text_to_pattern_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::optional<std::string> DPANCli::GetTextForPattern(PatternID pattern_id) const {
    auto it = pattern_to_text_.find(pattern_id);
    if (it != pattern_to_text_.end()) {
        return it->second;
    }
    return std::nullopt;
}

void DPANCli::InitializeClean() {
    InitializeEngine();
    InitializeAssociations();
    // Don't load previous session
}

void DPANCli::InitializeEngine() {
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

void DPANCli::InitializeAssociations() {
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

void DPANCli::PrintWelcome() {
    std::cout << C(Color::BOLD_CYAN) << R"(
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   DPAN Interactive Learning Interface                       ║
║   Dynamic Pattern Association Network                       ║
║                                                              ║
║   A neural network that learns and grows from interaction   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
)" << C(Color::RESET);
    std::cout << C(Color::DIM) << "Type 'help' for available commands, or just start talking!\n";
    std::cout << "The system will learn from everything you say.\n" << C(Color::RESET) << "\n";
    LoadSessionIfExists();
}

void DPANCli::ProcessCommand(const std::string& input) {
        if (input.empty()) return;

        // Check if it's a command (starts with /)
        if (input[0] == '/') {
            HandleCommand(input.substr(1));
        } else {
            // It's conversational input - learn from it
            HandleConversation(input);
        }
    }

void DPANCli::HandleCommand(const std::string& cmd) {
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
            std::cout << "Verbose mode: " << (verbose_ ? C(Color::GREEN) : C(Color::DIM))
                     << (verbose_ ? "ON" : "OFF") << C(Color::RESET) << "\n";
        } else if (command == "color" || command == "colors") {
            colors_enabled_ = !colors_enabled_;
            std::cout << (colors_enabled_ ? C(Color::GREEN) : "") << "Colors: "
                     << (colors_enabled_ ? "ON" : "OFF")
                     << (colors_enabled_ ? C(Color::RESET) : "") << "\n";
        } else if (command == "reset") {
            ResetSession();
        } else if (command == "clear") {
            std::cout << "\033[2J\033[1;1H"; // Clear screen
        } else {
            std::cout << C(Color::RED) << "✗ " << C(Color::RESET)
                     << "Unknown command: /" << command << "\n";
            std::cout << C(Color::DIM) << "Type '/help' for available commands.\n" << C(Color::RESET);
        }
    }

void DPANCli::HandleConversation(const std::string& text) {
        total_inputs_++;

        if (verbose_) {
            std::cout << C(Color::DIM) << "[Processing: \"" << text << "\"]\n" << C(Color::RESET);
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
                std::cout << C(Color::GREEN) << "[Created " << result.created_patterns.size()
                         << " new pattern(s)]\n" << C(Color::RESET);
            }
        } else if (!result.activated_patterns.empty()) {
            primary_pattern = result.activated_patterns[0];

            if (verbose_) {
                std::cout << C(Color::BLUE) << "[Activated existing pattern]\n" << C(Color::RESET);
            }
        } else {
            std::cout << C(Color::YELLOW) << "[No patterns matched or created - learning...]\n" << C(Color::RESET);
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

void DPANCli::GenerateResponse(PatternID input_pattern) {
        // Use associations to predict next patterns
        auto predictions = assoc_system_->Predict(input_pattern, 3);

        if (predictions.empty()) {
            std::cout << C(Color::CYAN) << "→ " << C(Color::DIM)
                     << "[Learning... I don't have enough context yet to respond.]\n" << C(Color::RESET);
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

            std::cout << C(Color::CYAN) << "→ " << C(Color::BOLD_MAGENTA)
                     << response_candidates[0] << C(Color::RESET);
            if (verbose_) {
                std::cout << C(Color::DIM) << " [confidence: " << std::fixed << std::setprecision(2)
                         << confidence << "]" << C(Color::RESET);
            }
            std::cout << "\n";

            if (response_candidates.size() > 1 && verbose_) {
                std::cout << C(Color::DIM) << "   Other possibilities: ";
                for (size_t i = 1; i < std::min<size_t>(3, response_candidates.size()); ++i) {
                    std::cout << "\"" << response_candidates[i] << "\" ";
                }
                std::cout << C(Color::RESET) << "\n";
            }
        } else {
            std::cout << C(Color::CYAN) << "→ " << C(Color::DIM)
                     << "[I predicted " << predictions.size()
                     << " pattern(s), but haven't learned text for them yet.]\n" << C(Color::RESET);
        }
    }

bool DPANCli::ShouldRequestMoreData(const PatternEngine::ProcessResult& result) {
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

void DPANCli::RequestMoreData(const std::string& context) {
        std::cout << "\n" << C(Color::BOLD_YELLOW) << "[ACTIVE LEARNING] " << C(Color::YELLOW)
                 << "I'm not confident about that. ";
        std::cout << "Can you tell me more or rephrase?\n" << C(Color::RESET);
    }

void DPANCli::LearnFromFile(const std::string& filepath) {
        if (!std::filesystem::exists(filepath)) {
            std::cout << C(Color::RED) << "✗ Error: " << C(Color::RESET)
                     << "File not found: " << filepath << "\n";
            return;
        }

        std::cout << C(Color::BLUE) << "Learning from file: " << C(Color::RESET)
                 << filepath << "\n";

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

        std::cout << "\n" << C(Color::GREEN) << "✓ " << C(Color::RESET)
                 << "Learned from " << C(Color::BOLD) << lines_processed << C(Color::RESET)
                 << " lines in " << duration.count() << " ms\n";
        std::cout << C(Color::DIM) << "  Patterns created: " << patterns_learned_
                 << C(Color::RESET) << "\n";
    }

void DPANCli::ShowHelp() {
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
  /color              Toggle colorized output

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

void DPANCli::ShowStatistics() {
        auto stats = engine_->GetStatistics();
        auto storage_stats = storage_->GetStats();

        std::cout << "\n" << C(Color::BOLD_CYAN);
        std::cout << "╔══════════════════════════════════════════╗\n";
        std::cout << "║         DPAN Learning Statistics         ║\n";
        std::cout << "╚══════════════════════════════════════════╝\n" << C(Color::RESET) << "\n";

        std::cout << C(Color::BOLD) << "Session:\n" << C(Color::RESET);
        std::cout << "  Inputs processed: " << C(Color::CYAN) << total_inputs_ << C(Color::RESET) << "\n";
        std::cout << "  Patterns learned: " << C(Color::CYAN) << patterns_learned_ << C(Color::RESET) << "\n";
        std::cout << "  Conversation length: " << C(Color::CYAN) << conversation_history_.size() << C(Color::RESET) << "\n";
        std::cout << "  Vocabulary size: " << C(Color::CYAN) << text_to_pattern_.size() << C(Color::RESET) << " unique inputs\n\n";

        std::cout << C(Color::BOLD) << "Patterns:\n" << C(Color::RESET);
        std::cout << "  Total patterns: " << C(Color::CYAN) << stats.total_patterns << C(Color::RESET) << "\n";
        std::cout << "  Atomic: " << C(Color::CYAN) << stats.atomic_patterns << C(Color::RESET) << "\n";
        std::cout << "  Composite: " << C(Color::CYAN) << stats.composite_patterns << C(Color::RESET) << "\n";
        std::cout << "  Average confidence: " << C(Color::CYAN) << std::fixed << std::setprecision(2)
                 << stats.avg_confidence << C(Color::RESET) << "\n\n";

        std::cout << C(Color::BOLD) << "Associations:\n" << C(Color::RESET);
        auto assoc_stats = assoc_system_->GetStatistics();
        std::cout << "  Total associations: " << C(Color::CYAN) << assoc_stats.total_associations << C(Color::RESET) << "\n";
        std::cout << "  Average strength: " << C(Color::CYAN) << std::fixed << std::setprecision(2)
                 << assoc_stats.average_strength << C(Color::RESET) << "\n";
        std::cout << "  Strongest association: " << C(Color::CYAN) << std::fixed << std::setprecision(2)
                 << assoc_stats.max_strength << C(Color::RESET) << "\n\n";

        std::cout << C(Color::BOLD) << "Storage:\n" << C(Color::RESET);
        std::cout << "  Database: " << C(Color::DIM) << session_file_ << C(Color::RESET) << "\n";
        std::cout << "  Size: " << C(Color::CYAN) << storage_stats.disk_usage_bytes / 1024 << C(Color::RESET) << " KB\n";
        std::cout << "  Active learning: " << (active_learning_mode_ ?
                 C(Color::GREEN) : C(Color::DIM)) << (active_learning_mode_ ? "ON" : "OFF")
                 << C(Color::RESET) << "\n\n";
    }

void DPANCli::ShowPatterns() {
        std::cout << "\n" << C(Color::BOLD_CYAN) << "Learned Patterns (text mappings):\n";
        std::cout << "================================\n" << C(Color::RESET) << "\n";

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
            std::cout << C(Color::YELLOW) << "No patterns learned yet. Start a conversation!\n"
                     << C(Color::RESET);
        }
    }

void DPANCli::ShowAssociations() {
        auto stats = assoc_system_->GetStatistics();
        const auto& matrix = assoc_system_->GetAssociationMatrix();

        std::cout << "\n" << C(Color::BOLD_CYAN) << "Association Graph:\n";
        std::cout << "==================\n" << C(Color::RESET) << "\n";
        std::cout << "Total associations: " << stats.total_associations << "\n";
        std::cout << "Average strength: " << std::fixed << std::setprecision(2)
                 << stats.average_strength << "\n\n";

        if (stats.total_associations == 0) {
            std::cout << C(Color::YELLOW) << "No associations formed yet. Keep learning!\n"
                     << C(Color::RESET);
            return;
        }

        std::cout << C(Color::BOLD) << "Strongest associations:\n" << C(Color::RESET);

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

void DPANCli::PredictNext(const std::string& text) {
        std::string query = text;
        // Trim leading space if present
        if (!query.empty() && query[0] == ' ') {
            query = query.substr(1);
        }

        if (text_to_pattern_.count(query) == 0) {
            std::cout << C(Color::YELLOW) << "Unknown input: " << C(Color::RESET)
                     << "\"" << query << "\"\n";
            std::cout << C(Color::DIM) << "I haven't learned this pattern yet.\n" << C(Color::RESET);
            return;
        }

        PatternID pattern = text_to_pattern_[query];
        auto predictions = assoc_system_->Predict(pattern, 5);

        if (predictions.empty()) {
            std::cout << C(Color::YELLOW) << "No predictions available for: "
                     << C(Color::RESET) << "\"" << query << "\"\n";
            return;
        }

        std::cout << "\n" << C(Color::BOLD) << "Predictions for \"" << query << "\":\n"
                 << C(Color::RESET);

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

void DPANCli::ToggleActiveLearning() {
        active_learning_mode_ = !active_learning_mode_;
        std::cout << "Active learning mode: " << (active_learning_mode_ ?
                 C(Color::BOLD_GREEN) : C(Color::DIM))
                 << (active_learning_mode_ ? "ON" : "OFF") << C(Color::RESET) << "\n";

        if (active_learning_mode_) {
            std::cout << C(Color::DIM) << "DPAN will now ask for clarification when uncertain.\n"
                     << C(Color::RESET);
        }
    }

void DPANCli::SaveSession() {
        std::cout << C(Color::BLUE) << "Saving session to " << C(Color::RESET)
                 << session_file_ << "...\n";

        // Storage is already persistent, but save associations
        std::string assoc_file = session_file_ + ".associations";
        if (assoc_system_->Save(assoc_file)) {
            std::cout << C(Color::GREEN) << "✓ " << C(Color::RESET)
                     << "Session saved successfully\n";
        } else {
            std::cout << C(Color::RED) << "✗ " << C(Color::RESET)
                     << "Failed to save associations\n";
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

        std::cout << C(Color::GREEN) << "✓ " << C(Color::RESET)
                 << "Saved " << count << " text mappings\n";
    }

void DPANCli::LoadSession() {
        LoadSessionIfExists();
    }

void DPANCli::LoadSessionIfExists() {
        if (!std::filesystem::exists(session_file_)) {
            std::cout << C(Color::DIM) << "No previous session found. Starting fresh.\n"
                     << C(Color::RESET);
            return;
        }

        std::cout << C(Color::BLUE) << "Loading previous session...\n" << C(Color::RESET);

        // Load associations
        std::string assoc_file = session_file_ + ".associations";
        if (std::filesystem::exists(assoc_file)) {
            if (assoc_system_->Load(assoc_file)) {
                std::cout << C(Color::GREEN) << "✓ " << C(Color::RESET)
                         << "Loaded associations\n";
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

            std::cout << C(Color::GREEN) << "✓ " << C(Color::RESET)
                     << "Loaded " << count << " text mappings\n";
        }

        auto stats = engine_->GetStatistics();
        std::cout << C(Color::GREEN) << "Session loaded: " << C(Color::CYAN)
                 << stats.total_patterns << C(Color::RESET) << " patterns\n\n";
    }

void DPANCli::ResetSession() {
        std::cout << "Are you sure you want to reset? This will erase all learning. (y/N): ";
        std::string confirm;
        std::getline(std::cin, confirm);

        if (confirm != "y" && confirm != "Y") {
            std::cout << C(Color::YELLOW) << "Reset cancelled.\n" << C(Color::RESET);
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

        std::cout << C(Color::GREEN) << "✓ " << C(Color::RESET)
                 << "Session reset. Starting fresh.\n";
    }

void DPANCli::Shutdown() {
        std::cout << "\n" << C(Color::BLUE) << "Shutting down...\n" << C(Color::RESET);
        SaveSession();

        std::cout << "\n" << C(Color::BOLD) << "Session Summary:\n" << C(Color::RESET);
        std::cout << "  Inputs processed: " << C(Color::CYAN) << total_inputs_ << C(Color::RESET) << "\n";
        std::cout << "  Patterns learned: " << C(Color::CYAN) << patterns_learned_ << C(Color::RESET) << "\n";
        std::cout << "  Conversation length: " << C(Color::CYAN) << conversation_history_.size()
                 << C(Color::RESET) << "\n";
        std::cout << "\n" << C(Color::BOLD_CYAN) << "Thank you for teaching me! Goodbye.\n"
                 << C(Color::RESET);
    }

std::vector<uint8_t> DPANCli::TextToBytes(const std::string& text) {
    return std::vector<uint8_t>(text.begin(), text.end());
}

} // namespace dpan

#ifndef DPAN_CLI_TEST_BUILD
int main(int argc, char** argv) {
    std::cout << "Initializing DPAN...\n";

    try {
        dpan::DPANCli cli;
        cli.Run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
#endif
