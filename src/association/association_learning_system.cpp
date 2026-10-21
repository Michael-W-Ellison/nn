// File: src/association/association_learning_system.cpp
#include "association/association_learning_system.hpp"
#include "learning/attention_mechanism.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <unordered_set>

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
      attention_mechanism_(nullptr),  // Default to nullptr (backwards compatible)
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

    // Associations into this pattern learn the context it occurred in
    UpdateContextProfiles({pattern}, context, now);

    // Update activation history
    UpdateActivationHistory(pattern, now);

    // Check if auto-maintenance is needed
    CheckAutoMaintenance();
}

void AssociationLearningSystem::RecordPatternActivations(
    const std::vector<PatternID>& patterns,
    const ContextVector& context
) {
    Timestamp now = Timestamp::Now();

    // Record all activations
    tracker_.RecordActivations(patterns, now);

    UpdateContextProfiles(patterns, context, now);

    for (const auto& pattern : patterns) {
        UpdateActivationHistory(pattern, now);
    }

    // Check if auto-maintenance is needed
    CheckAutoMaintenance();
}

// ============================================================================
// Association Formation
// ============================================================================

size_t AssociationLearningSystem::FormNewAssociations(PatternDatabase& pattern_db) {
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
    PatternDatabase& pattern_db
) {
    size_t formed_count = 0;

    AssociationFormationRules rules;
    bool require_significance;
    {
        std::lock_guard<std::mutex> lock(config_mutex_);
        rules = formation_rules_;
        require_significance = config_.require_significance;
    }
    uint32_t min_co_occurrences = rules.GetConfig().min_co_occurrences;

    // Get co-occurring patterns
    auto co_occurring = tracker_.GetCoOccurringPatterns(pattern, min_co_occurrences);
    if (co_occurring.empty()) {
        return 0;
    }

    auto source_node = pattern_db.Retrieve(pattern);
    std::vector<std::pair<Timestamp, PatternID>> history;
    {
        std::lock_guard<std::mutex> lock(history_mutex_);
        history.assign(activation_history_.begin(), activation_history_.end());
    }

    for (const auto& [target, count] : co_occurring) {
        // Skip if association already exists
        if (matrix_.HasAssociation(pattern, target)) {
            continue;
        }

        if (count < min_co_occurrences) {
            continue;
        }
        if (require_significance && !rules.ShouldFormAssociation(tracker_, pattern, target)) {
            continue;
        }

        // Classify from pattern data when both patterns are stored
        AssociationType type = AssociationType::CATEGORICAL;
        if (source_node) {
            if (auto target_node = pattern_db.Retrieve(target)) {
                type = rules.ClassifyAssociationType(*source_node, *target_node, history);
            }
        }

        float strength = rules.CalculateInitialStrength(tracker_, pattern, target, type);
        AssociationEdge edge(pattern, target, type, strength);
        for (uint32_t i = 0; i < count; ++i) {
            edge.IncrementCoOccurrence();
        }

        if (matrix_.AddAssociation(edge)) {
            formed_count++;
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
    {
        std::lock_guard<std::mutex> lock(reinforcement_mutex_);

        // Get the edge
        const auto* edge_ptr = matrix_.GetAssociation(predicted, actual);
        if (!edge_ptr) {
            return;  // No association to reinforce
        }

        // Clone and update
        auto edge = edge_ptr->Clone();
        reinforcement_mgr_.ReinforcePrediction(*edge, true, correct);
        matrix_.UpdateAssociation(predicted, actual, *edge);
    }

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
    std::lock_guard<std::mutex> lock(maintenance_mutex_);
    ApplyDecayLocked(elapsed);
}

size_t AssociationLearningSystem::ApplyCompetition() {
    std::lock_guard<std::mutex> lock(maintenance_mutex_);
    return ApplyCompetitionLocked();
}

size_t AssociationLearningSystem::ApplyNormalization() {
    std::lock_guard<std::mutex> lock(maintenance_mutex_);
    return ApplyNormalizationLocked();
}

size_t AssociationLearningSystem::PruneWeakAssociations(float min_strength) {
    std::lock_guard<std::mutex> lock(maintenance_mutex_);
    return PruneWeakAssociationsLocked(min_strength);
}

void AssociationLearningSystem::ApplyDecayLocked(Timestamp::Duration elapsed) {
    matrix_.ApplyDecayAll(elapsed);
    last_decay_ = Timestamp::Now();
}

size_t AssociationLearningSystem::ApplyCompetitionLocked() {
    CompetitiveLearner::Config competition;
    {
        std::lock_guard<std::mutex> lock(config_mutex_);
        competition = config_.competition;
    }

    size_t applied = CompetitiveLearner::ApplyCompetitionAll(matrix_, competition);
    last_competition_ = Timestamp::Now();
    return applied;
}

size_t AssociationLearningSystem::ApplyNormalizationLocked() {
    StrengthNormalizer::Config normalization;
    {
        std::lock_guard<std::mutex> lock(config_mutex_);
        normalization = config_.normalization;
    }

    size_t normalized = StrengthNormalizer::NormalizeAll(matrix_, normalization);
    last_normalization_ = Timestamp::Now();
    return normalized;
}

size_t AssociationLearningSystem::PruneWeakAssociationsLocked(float min_strength) {
    if (min_strength == 0.0f) {
        std::lock_guard<std::mutex> lock(config_mutex_);
        min_strength = config_.prune_threshold;
    }

    size_t pruned_count = 0;

    // Collect weak associations to prune (to avoid iterator invalidation)
    std::vector<std::pair<PatternID, PatternID>> to_prune;

    for (const auto& [source, target] : matrix_.GetAllAssociationKeys()) {
        const auto* edge = matrix_.GetAssociation(source, target);
        if (edge && edge->GetStrength() < min_strength) {
            to_prune.emplace_back(source, target);
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
    std::lock_guard<std::mutex> lock(maintenance_mutex_);

    // Apply decay since last application
    auto decay_elapsed = Timestamp::Now() - last_decay_;
    ApplyDecayLocked(decay_elapsed);
    stats.decay_applied = decay_elapsed;

    // Apply competition
    stats.competitions_applied = ApplyCompetitionLocked();

    // Apply normalization
    stats.normalizations_applied = ApplyNormalizationLocked();

    // Prune weak associations
    stats.associations_pruned = PruneWeakAssociationsLocked(0.0f);

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

AssociationMatrix& AssociationLearningSystem::GetAssociationMatrix() {
    return matrix_;
}

bool AssociationLearningSystem::AddAssociation(const AssociationEdge& edge) {
    return matrix_.AddAssociation(edge);
}

bool AssociationLearningSystem::AddAssociation(PatternID source, PatternID target,
                                               AssociationType type, float strength) {
    return matrix_.AddAssociation(AssociationEdge(source, target, type, strength));
}

bool AssociationLearningSystem::SetAssociationStrength(PatternID source, PatternID target,
                                                       float strength) {
    std::lock_guard<std::mutex> lock(reinforcement_mutex_);
    const auto* edge = matrix_.GetAssociation(source, target);
    if (!edge) {
        return false;
    }
    auto updated = edge->Clone();
    updated->SetStrength(strength);
    return matrix_.UpdateAssociation(source, target, *updated);
}

bool AssociationLearningSystem::RemoveAssociation(PatternID source, PatternID target) {
    return matrix_.RemoveAssociation(source, target);
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

std::vector<std::pair<PatternID, float>> AssociationLearningSystem::RankCandidates(
    PatternID pattern,
    const ContextVector* context
) const {
    std::shared_ptr<const PredictionScorer> scorer;
    float weight;
    {
        std::lock_guard<std::mutex> lock(scorer_mutex_);
        scorer = prediction_scorer_;
        weight = prediction_scorer_weight_;
    }

    auto associations = matrix_.GetOutgoingAssociations(pattern);

    std::vector<std::pair<PatternID, float>> ranked;
    ranked.reserve(associations.size());

    for (const auto* edge : associations) {
        float strength = context ? edge->GetContextualStrength(*context) : edge->GetStrength();
        float score = strength;
        if (scorer) {
            float learned = std::clamp(
                scorer->Score(pattern, edge->GetTarget(), strength, context), 0.0f, 1.0f);
            score = (1.0f - weight) * strength + weight * learned;
        }
        ranked.emplace_back(edge->GetTarget(), score);
    }

    std::sort(ranked.begin(), ranked.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

    return ranked;
}

std::vector<PatternID> AssociationLearningSystem::Predict(
    PatternID pattern,
    size_t k,
    const ContextVector* context
) const {
    auto ranked = RankCandidates(pattern, context);

    std::vector<PatternID> predictions;
    predictions.reserve(std::min(k, ranked.size()));
    for (size_t i = 0; i < std::min(k, ranked.size()); ++i) {
        predictions.push_back(ranked[i].first);
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
    auto predictions = RankCandidates(pattern, context);

    if (predictions.size() > k) {
        predictions.resize(k);
    }

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.predictions_count++;
    }

    return predictions;
}

std::vector<std::pair<PatternID, float>> AssociationLearningSystem::PredictWithAttention(
    PatternID source,
    size_t k,
    const ContextVector& context
) const {
    // Get the attention mechanism (thread-safe)
    AttentionMechanism* attention = nullptr;
    {
        std::lock_guard<std::mutex> lock(attention_mutex_);
        attention = attention_mechanism_;
    }

    // If no attention mechanism is set, fall back to PredictWithConfidence
    if (attention == nullptr) {
        return PredictWithConfidence(source, k, &context);
    }

    // Get outgoing associations from the source pattern
    auto associations = matrix_.GetOutgoingAssociations(source);

    if (associations.empty()) {
        return {};  // No associations, no predictions
    }

    // Extract candidate patterns and their association strengths
    std::vector<PatternID> candidates;
    std::map<PatternID, float> association_strengths;
    candidates.reserve(associations.size());

    for (const auto* edge : associations) {
        PatternID target = edge->GetTarget();
        float strength = edge->GetContextualStrength(context);

        candidates.push_back(target);
        association_strengths[target] = strength;
    }

    // Combine association strengths with attention weights:
    //   score = association_weight * strength + attention_weight * attention
    auto attention_weights = attention->ComputeAttention(source, candidates, context);
    const AttentionConfig& attention_config = attention->GetConfig();

    std::vector<std::pair<PatternID, float>> weighted_predictions;
    weighted_predictions.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        auto it = attention_weights.find(candidate);
        float attention_weight = it != attention_weights.end() ? it->second : 0.0f;
        float score = attention_config.association_weight * association_strengths[candidate] +
                      attention_config.attention_weight * attention_weight;
        weighted_predictions.emplace_back(candidate, score);
    }

    // Sort by combined score (descending)
    std::sort(weighted_predictions.begin(), weighted_predictions.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

    // Return top-k predictions
    if (weighted_predictions.size() > k) {
        weighted_predictions.resize(k);
    }

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.predictions_count++;
    }

    return weighted_predictions;
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
    std::lock_guard<std::mutex> reinforcement_lock(reinforcement_mutex_);
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
// Attention Mechanism Integration
// ============================================================================

void AssociationLearningSystem::SetAttentionMechanism(AttentionMechanism* attention) {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    attention_mechanism_ = attention;
}

void AssociationLearningSystem::SetPredictionScorer(
    std::shared_ptr<const PredictionScorer> scorer, float weight) {
    if (weight < 0.0f || weight > 1.0f) {
        throw std::invalid_argument("SetPredictionScorer: weight must be in [0, 1]");
    }
    std::lock_guard<std::mutex> lock(scorer_mutex_);
    prediction_scorer_ = std::move(scorer);
    prediction_scorer_weight_ = weight;
}

std::shared_ptr<const PredictionScorer> AssociationLearningSystem::GetPredictionScorer() const {
    std::lock_guard<std::mutex> lock(scorer_mutex_);
    return prediction_scorer_;
}

AttentionMechanism* AssociationLearningSystem::GetAttentionMechanism() const {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    return attention_mechanism_;
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

void AssociationLearningSystem::UpdateContextProfiles(
    const std::vector<PatternID>& activated,
    const ContextVector& context,
    Timestamp timestamp) {

    if (context.IsEmpty()) {
        return;
    }

    Timestamp::Duration window;
    float learning_rate;
    {
        std::lock_guard<std::mutex> lock(config_mutex_);
        window = config_.activation_window;
        learning_rate = config_.reinforcement.learning_rate;
    }

    // Sources: patterns active in the window before this activation, plus the
    // other patterns activated together with it
    std::unordered_set<PatternID> sources(activated.begin(), activated.end());
    {
        std::lock_guard<std::mutex> lock(history_mutex_);
        Timestamp cutoff = timestamp - window;
        for (auto it = activation_history_.rbegin();
             it != activation_history_.rend() && !(it->first < cutoff); ++it) {
            sources.insert(it->second);
        }
    }

    std::lock_guard<std::mutex> lock(reinforcement_mutex_);
    for (const auto& target : activated) {
        for (const auto& source : sources) {
            if (source == target) {
                continue;
            }
            const auto* edge = matrix_.GetAssociation(source, target);
            if (!edge) {
                continue;
            }
            auto updated = edge->Clone();
            updated->UpdateContextProfile(context, learning_rate);
            matrix_.UpdateAssociation(source, target, *updated);
        }
    }
}

void AssociationLearningSystem::UpdateActivationHistory(PatternID pattern, Timestamp timestamp) {
    Timestamp::Duration window;
    size_t max_size;
    {
        std::lock_guard<std::mutex> lock(config_mutex_);
        window = config_.activation_window;
        max_size = config_.max_activation_history;
    }

    std::lock_guard<std::mutex> lock(history_mutex_);
    activation_history_.emplace_back(timestamp, pattern);
    TrimActivationHistory(window, max_size);
}

void AssociationLearningSystem::TrimActivationHistory(Timestamp::Duration window, size_t max_size) {
    Timestamp cutoff = Timestamp::Now() - window;

    while (!activation_history_.empty() &&
           activation_history_.front().first < cutoff) {
        activation_history_.pop_front();
    }

    while (activation_history_.size() > max_size) {
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

    {
        std::lock_guard<std::mutex> maintenance_lock(maintenance_mutex_);
        stats_.last_decay = last_decay_;
        stats_.last_competition = last_competition_;
        stats_.last_normalization = last_normalization_;
        stats_.last_pruning = last_pruning_;
    }

    // Active = non-zero strength
    auto summary = matrix_.GetStrengthSummary(std::numeric_limits<float>::min());
    stats_.average_strength = summary.average;
    stats_.min_strength = summary.min;
    stats_.max_strength = summary.max;
    stats_.active_associations = summary.count_at_or_above;
}

void AssociationLearningSystem::CheckAutoMaintenance() {
    Timestamp::Duration decay_interval;
    Timestamp::Duration competition_interval;
    Timestamp::Duration normalization_interval;
    {
        std::lock_guard<std::mutex> lock(config_mutex_);
        if (!config_.enable_auto_maintenance) {
            return;
        }
        decay_interval = config_.auto_decay_interval;
        competition_interval = config_.auto_competition_interval;
        normalization_interval = config_.auto_normalization_interval;
    }

    // Skip if another thread is already running maintenance
    std::unique_lock<std::mutex> lock(maintenance_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
        return;
    }

    Timestamp now = Timestamp::Now();

    if (decay_interval.count() > 0) {
        auto decay_elapsed = now - last_decay_;
        if (decay_elapsed >= decay_interval) {
            ApplyDecayLocked(decay_elapsed);
        }
    }

    if (competition_interval.count() > 0) {
        auto comp_elapsed = now - last_competition_;
        if (comp_elapsed >= competition_interval) {
            ApplyCompetitionLocked();
        }
    }

    if (normalization_interval.count() > 0) {
        auto norm_elapsed = now - last_normalization_;
        if (norm_elapsed >= normalization_interval) {
            ApplyNormalizationLocked();
        }
    }
}

} // namespace dpan
