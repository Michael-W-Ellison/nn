// File: src/association/association_learning_system.cpp
#include "association/association_learning_system.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>

namespace dpan {

// ============================================================================
// Construction & Initialization
// ============================================================================

AssociationLearningSystem::AssociationLearningSystem()
    : AssociationLearningSystem(Config())
{
}

AssociationLearningSystem::AssociationLearningSystem(const Config& config)
    : config_(config),
      tracker_(config.co_occurrence),
      formation_rules_(config.formation),
      reinforcement_mgr_(config.reinforcement),
      last_decay_(Timestamp::Now()),
      last_competition_(Timestamp::Now()),
      last_normalization_(Timestamp::Now()),
      last_pruning_(Timestamp::Now())
{
    // Initialize statistics
    stats_ = Statistics();
}

AssociationLearningSystem::~AssociationLearningSystem() {
    // Cleanup handled by destructors
}

// ============================================================================
// Pattern Activation Recording
// ============================================================================

void AssociationLearningSystem::RecordPatternActivation(
    PatternID pattern,
    const ContextVector& context
) {
    Timestamp now = Timestamp::Now();

    // Record in co-occurrence tracker
    tracker_.RecordActivation(pattern, now);

    // Update activation history
    UpdateActivationHistory(pattern, now);

    // Check if auto-maintenance is needed
    if (config_.enable_auto_maintenance) {
        CheckAutoMaintenance();
    }
}

void AssociationLearningSystem::RecordPatternActivations(
    const std::vector<PatternID>& patterns,
    const ContextVector& context
) {
    Timestamp now = Timestamp::Now();

    // Record all activations
    tracker_.RecordActivations(patterns, now);
    
    for (const auto& pattern : patterns) {
        UpdateActivationHistory(pattern, now);
    }

    // Check if auto-maintenance is needed
    if (config_.enable_auto_maintenance) {
        CheckAutoMaintenance();
    }
}

// ============================================================================
// Association Formation
// ============================================================================

size_t AssociationLearningSystem::FormNewAssociations(const PatternDatabase& pattern_db) {
    size_t formed_count = 0;

    // Get all patterns that have been tracked in co-occurrence tracker
    std::vector<PatternID> tracked_patterns = tracker_.GetTrackedPatterns();

    // Form associations for each tracked pattern
    for (const auto& pattern : tracked_patterns) {
        formed_count += FormAssociationsForPattern(pattern, pattern_db);
    }

    return formed_count;
}

size_t AssociationLearningSystem::FormAssociationsForPattern(
    PatternID pattern,
    const PatternDatabase& pattern_db
) {
    size_t formed_count = 0;

    // Get co-occurring patterns
    auto co_occurring = tracker_.GetCoOccurringPatterns(pattern, config_.formation.min_co_occurrences);

    for (const auto& [target, count] : co_occurring) {
        // Skip if association already exists
        if (matrix_.HasAssociation(pattern, target)) {
            continue;
        }

        // Get co-occurrence count
        uint32_t co_occ_count = tracker_.GetCoOccurrenceCount(pattern, target);
        
        if (co_occ_count >= config_.formation.min_co_occurrences) {
            // Create new association with default type and initial strength
            AssociationEdge edge(pattern, target, AssociationType::CATEGORICAL, 0.5f);

            if (matrix_.AddAssociation(edge)) {
                formed_count++;
            }
        }
    }

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.formations_count += formed_count;
    }

    return formed_count;
}

// ============================================================================
// Reinforcement Learning
// ============================================================================

void AssociationLearningSystem::Reinforce(
    PatternID predicted,
    PatternID actual,
    bool correct
) {
    // Get the edge
    const auto* edge_ptr = matrix_.GetAssociation(predicted, actual);
    if (!edge_ptr) {
        return;  // No association to reinforce
    }

    // Clone and update
    auto edge = edge_ptr->Clone();
    reinforcement_mgr_.ReinforcePrediction(*edge, true, correct);
    matrix_.UpdateAssociation(predicted, actual, *edge);

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.reinforcements_count++;
    }
}

void AssociationLearningSystem::ReinforceBatch(
    const std::vector<std::tuple<PatternID, PatternID, bool>>& outcomes
) {
    for (const auto& [predicted, actual, correct] : outcomes) {
        Reinforce(predicted, actual, correct);
    }
}

// ============================================================================
// Maintenance Operations
// ============================================================================

void AssociationLearningSystem::ApplyDecay(Timestamp::Duration elapsed) {
    matrix_.ApplyDecayAll(elapsed);
    last_decay_ = Timestamp::Now();
}

size_t AssociationLearningSystem::ApplyCompetition() {
    last_competition_ = Timestamp::Now();
    return 0;  // Simplified - would need pattern iteration
}

size_t AssociationLearningSystem::ApplyNormalization() {
    last_normalization_ = Timestamp::Now();
    return 0;  // Simplified - would need pattern iteration
}

size_t AssociationLearningSystem::PruneWeakAssociations(float min_strength) {
    if (min_strength == 0.0f) {
        min_strength = config_.prune_threshold;
    }

    size_t pruned_count = 0;

    // Collect weak associations to prune (to avoid iterator invalidation)
    std::vector<std::pair<PatternID, PatternID>> to_prune;

    // Get all patterns that have associations
    auto all_patterns = matrix_.GetAllPatterns();

    for (const auto& pattern : all_patterns) {
        auto outgoing = matrix_.GetOutgoingAssociations(pattern);

        for (const auto* edge : outgoing) {
            if (edge && edge->GetStrength() < min_strength) {
                to_prune.push_back({edge->GetSource(), edge->GetTarget()});
            }
        }
    }

    // Remove weak associations
    for (const auto& [source, target] : to_prune) {
        if (matrix_.RemoveAssociation(source, target)) {
            pruned_count++;
        }
    }

    last_pruning_ = Timestamp::Now();
    return pruned_count;
}

void AssociationLearningSystem::Compact() {
    matrix_.Compact();
}

AssociationLearningSystem::MaintenanceStats AssociationLearningSystem::PerformMaintenance() {
    MaintenanceStats stats;

    // Apply decay since last application
    auto decay_elapsed = Timestamp::Now() - last_decay_;
    ApplyDecay(decay_elapsed);
    stats.decay_applied = decay_elapsed;

    // Apply competition
    stats.competitions_applied = ApplyCompetition();

    // Apply normalization
    stats.normalizations_applied = ApplyNormalization();

    // Prune weak associations
    stats.associations_pruned = PruneWeakAssociations();

    // Compact if significant pruning occurred
    if (stats.associations_pruned > 100) {
        Compact();
    }

    return stats;
}

// ============================================================================
// Query & Prediction
// ============================================================================

const AssociationMatrix& AssociationLearningSystem::GetAssociationMatrix() const {
    return matrix_;
}

std::vector<const AssociationEdge*> AssociationLearningSystem::GetAssociations(
    PatternID pattern,
    bool outgoing
) const {
    if (outgoing) {
        return matrix_.GetOutgoingAssociations(pattern);
    } else {
        return matrix_.GetIncomingAssociations(pattern);
    }
}

std::vector<PatternID> AssociationLearningSystem::Predict(
    PatternID pattern,
    size_t k,
    const ContextVector* context
) const {
    auto associations = matrix_.GetOutgoingAssociations(pattern);

    std::vector<const AssociationEdge*> sorted_assocs(associations.begin(), associations.end());
    std::sort(sorted_assocs.begin(), sorted_assocs.end(),
        [context](const AssociationEdge* a, const AssociationEdge* b) {
            float strength_a = context ? a->GetContextualStrength(*context) : a->GetStrength();
            float strength_b = context ? b->GetContextualStrength(*context) : b->GetStrength();
            return strength_a > strength_b;
        });

    std::vector<PatternID> predictions;
    predictions.reserve(std::min(k, sorted_assocs.size()));

    for (size_t i = 0; i < std::min(k, sorted_assocs.size()); ++i) {
        predictions.push_back(sorted_assocs[i]->GetTarget());
    }

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.predictions_count++;
    }

    return predictions;
}

std::vector<std::pair<PatternID, float>> AssociationLearningSystem::PredictWithConfidence(
    PatternID pattern,
    size_t k,
    const ContextVector* context
) const {
    auto associations = matrix_.GetOutgoingAssociations(pattern);

    std::vector<std::pair<PatternID, float>> predictions;
    predictions.reserve(associations.size());

    for (const auto* edge : associations) {
        float strength = context ? edge->GetContextualStrength(*context) : edge->GetStrength();
        predictions.emplace_back(edge->GetTarget(), strength);
    }

    std::sort(predictions.begin(), predictions.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

    if (predictions.size() > k) {
        predictions.resize(k);
    }

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.predictions_count++;
    }

    return predictions;
}

std::vector<AssociationMatrix::ActivationResult> AssociationLearningSystem::PropagateActivation(
    PatternID source,
    float initial_activation,
    size_t max_hops,
    float min_activation,
    const ContextVector* context
) const {
    return matrix_.PropagateActivation(
        source,
        initial_activation,
        max_hops,
        min_activation,
        context
    );
}

// ============================================================================
// Statistics & Monitoring
// ============================================================================

AssociationLearningSystem::Statistics AssociationLearningSystem::GetStatistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    UpdateStatistics();
    return stats_;
}

size_t AssociationLearningSystem::GetAssociationCount() const {
    return matrix_.GetAssociationCount();
}

float AssociationLearningSystem::GetAverageStrength() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_.average_strength;
}

void AssociationLearningSystem::PrintStatistics(std::ostream& out) const {
    auto stats = GetStatistics();

    out << "=== Association Learning System Statistics ===" << std::endl;
    out << std::endl;
    out << "Associations:" << std::endl;
    out << "  Total: " << stats.total_associations << std::endl;
    out << "  Active (non-zero strength): " << stats.active_associations << std::endl;
    out << std::endl;
    out << "Strength Statistics:" << std::endl;
    out << "  Average: " << std::fixed << std::setprecision(3) << stats.average_strength << std::endl;
    out << "  Min: " << stats.min_strength << std::endl;
    out << "  Max: " << stats.max_strength << std::endl;
    out << std::endl;
    out << "Pattern Statistics:" << std::endl;
    out << "  Patterns with associations: " << stats.patterns_with_associations << std::endl;
    out << "  Average associations per pattern: " << std::fixed << std::setprecision(2)
        << stats.average_associations_per_pattern << std::endl;
    out << std::endl;
    out << "Activity:" << std::endl;
    out << "  Total co-occurrences tracked: " << stats.total_co_occurrences << std::endl;
    out << "  Activation history size: " << stats.activation_history_size << std::endl;
    out << "  Associations formed: " << stats.formations_count << std::endl;
    out << "  Reinforcements applied: " << stats.reinforcements_count << std::endl;
    out << "  Predictions made: " << stats.predictions_count << std::endl;
    out << std::endl;
}

// ============================================================================
// Configuration Management
// ============================================================================

void AssociationLearningSystem::SetConfig(const Config& config) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    config_ = config;

    // Update component configurations
    formation_rules_.SetConfig(config.formation);
    reinforcement_mgr_.SetConfig(config.reinforcement);
}

AssociationLearningSystem::Config AssociationLearningSystem::GetConfig() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_;
}

// ============================================================================
// Persistence
// ============================================================================

bool AssociationLearningSystem::Save(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        return false;
    }

    matrix_.Serialize(file);
    return file.good();
}

bool AssociationLearningSystem::Load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        return false;
    }

    // Deserialize matrix
    auto loaded_matrix = AssociationMatrix::Deserialize(file);
    if (!loaded_matrix) {
        return false;
    }

    // Clear current matrix and copy edges from loaded matrix
    matrix_.Clear();

    // Get all patterns from loaded matrix
    auto patterns = loaded_matrix->GetAllPatterns();

    // Copy all associations from loaded matrix to current matrix
    for (const auto& pattern : patterns) {
        auto outgoing = loaded_matrix->GetOutgoingAssociations(pattern);
        for (const auto* edge : outgoing) {
            if (edge) {
                matrix_.AddAssociation(*edge);
            }
        }
    }

    return file.good();
}

// ============================================================================
// Private Helper Methods
// ============================================================================

void AssociationLearningSystem::UpdateActivationHistory(PatternID pattern, Timestamp timestamp) {
    std::lock_guard<std::mutex> lock(history_mutex_);
    activation_history_.emplace_back(timestamp, pattern);
    TrimActivationHistory();
}

void AssociationLearningSystem::TrimActivationHistory() {
    Timestamp cutoff = Timestamp::Now() - config_.activation_window;

    while (!activation_history_.empty() &&
           activation_history_.front().first < cutoff) {
        activation_history_.pop_front();
    }

    while (activation_history_.size() > config_.max_activation_history) {
        activation_history_.pop_front();
    }
}

void AssociationLearningSystem::UpdateStatistics() const {
    stats_.total_associations = matrix_.GetAssociationCount();
    stats_.patterns_with_associations = matrix_.GetPatternCount();

    if (stats_.patterns_with_associations > 0) {
        stats_.average_associations_per_pattern =
            static_cast<float>(stats_.total_associations) /
            static_cast<float>(stats_.patterns_with_associations);
    } else {
        stats_.average_associations_per_pattern = 0.0f;
    }

    stats_.total_co_occurrences = tracker_.GetCoOccurrencePairCount();

    {
        std::lock_guard<std::mutex> hist_lock(history_mutex_);
        stats_.activation_history_size = activation_history_.size();
    }

    stats_.last_decay = last_decay_;
    stats_.last_competition = last_competition_;
    stats_.last_normalization = last_normalization_;
    stats_.last_pruning = last_pruning_;

    stats_.average_strength = 0.5f;
    stats_.min_strength = 0.0f;
    stats_.max_strength = 1.0f;
    stats_.active_associations = stats_.total_associations;
}

void AssociationLearningSystem::CheckAutoMaintenance() {
    if (!config_.enable_auto_maintenance) {
        return;
    }

    Timestamp now = Timestamp::Now();

    if (config_.auto_decay_interval.count() > 0) {
        auto decay_elapsed = now - last_decay_;
        if (decay_elapsed >= config_.auto_decay_interval) {
            ApplyDecay(decay_elapsed);
        }
    }

    if (config_.auto_competition_interval.count() > 0) {
        auto comp_elapsed = now - last_competition_;
        if (comp_elapsed >= config_.auto_competition_interval) {
            ApplyCompetition();
        }
    }

    if (config_.auto_normalization_interval.count() > 0) {
        auto norm_elapsed = now - last_normalization_;
        if (norm_elapsed >= config_.auto_normalization_interval) {
            ApplyNormalization();
        }
    }
}

} // namespace dpan
