// File: src/association/association_matrix.cpp
#include "association/association_matrix.hpp"
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <sstream>

namespace dpan {

// ============================================================================
// Construction
// ============================================================================

AssociationMatrix::AssociationMatrix()
    : AssociationMatrix(Config())
{
}

AssociationMatrix::AssociationMatrix(const Config& config)
    : config_(config)
{
    edges_.reserve(config_.initial_capacity);
}

// ============================================================================
// Helper Methods
// ============================================================================

size_t AssociationMatrix::AllocateEdgeIndex() {
    if (!deleted_indices_.empty()) {
        size_t index = deleted_indices_.back();
        deleted_indices_.pop_back();
        return index;
    }

    size_t index = edges_.size();
    edges_.push_back(nullptr);  // Add placeholder (will be assigned immediately)
    return index;
}

void AssociationMatrix::ReleaseEdgeIndex(size_t index) {
    deleted_indices_.push_back(index);
}

void AssociationMatrix::UpdateIndices(size_t edge_index, bool add) {
    const AssociationEdge& edge = *edges_[edge_index];
    PatternID source = edge.GetSource();
    PatternID target = edge.GetTarget();
    AssociationType type = edge.GetType();

    if (add) {
        // Add to outgoing index
        outgoing_index_[source].push_back(edge_index);

        // Add to incoming index (if enabled)
        if (config_.enable_reverse_lookup) {
            incoming_index_[target].push_back(edge_index);
        }

        // Add to type index (if enabled)
        if (config_.enable_type_index) {
            type_index_[type].push_back(edge_index);
        }
    } else {
        // Remove from outgoing index
        auto& outgoing = outgoing_index_[source];
        outgoing.erase(std::remove(outgoing.begin(), outgoing.end(), edge_index), outgoing.end());
        if (outgoing.empty()) {
            outgoing_index_.erase(source);
        }

        // Remove from incoming index
        if (config_.enable_reverse_lookup) {
            auto& incoming = incoming_index_[target];
            incoming.erase(std::remove(incoming.begin(), incoming.end(), edge_index), incoming.end());
            if (incoming.empty()) {
                incoming_index_.erase(target);
            }
        }

        // Remove from type index
        if (config_.enable_type_index) {
            auto& type_edges = type_index_[type];
            type_edges.erase(std::remove(type_edges.begin(), type_edges.end(), edge_index), type_edges.end());
            if (type_edges.empty()) {
                type_index_.erase(type);
            }
        }
    }
}

void AssociationMatrix::RebuildIndices() {
    outgoing_index_.clear();
    incoming_index_.clear();
    type_index_.clear();

    for (size_t i = 0; i < edges_.size(); ++i) {
        // Skip deleted edges (check if it's in the lookup map)
        bool is_deleted = true;
        for (const auto& [key, index] : edge_lookup_) {
            if (index == i) {
                is_deleted = false;
                break;
            }
        }
        if (!is_deleted) {
            UpdateIndices(i, true);
        }
    }
}

// ============================================================================
// Add/Update/Remove Operations
// ============================================================================

bool AssociationMatrix::AddAssociation(const AssociationEdge& edge) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    PatternID source = edge.GetSource();
    PatternID target = edge.GetTarget();

    auto key = std::make_pair(source, target);

    // Check if association already exists
    if (edge_lookup_.find(key) != edge_lookup_.end()) {
        return false;  // Already exists
    }

    // Allocate index and store edge (clone it)
    size_t index = AllocateEdgeIndex();
    edges_[index] = edge.Clone();

    // Update lookup
    edge_lookup_[key] = index;

    // Update indices
    UpdateIndices(index, true);

    return true;
}

bool AssociationMatrix::UpdateAssociation(
    PatternID source,
    PatternID target,
    const AssociationEdge& edge
) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto key = std::make_pair(source, target);
    auto it = edge_lookup_.find(key);

    if (it == edge_lookup_.end()) {
        return false;  // Doesn't exist
    }

    edges_[it->second] = edge.Clone();
    return true;
}

bool AssociationMatrix::RemoveAssociation(PatternID source, PatternID target) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto key = std::make_pair(source, target);
    auto it = edge_lookup_.find(key);

    if (it == edge_lookup_.end()) {
        return false;  // Doesn't exist
    }

    size_t index = it->second;

    // Update indices before deletion
    UpdateIndices(index, false);

    // Remove from lookup
    edge_lookup_.erase(it);

    // Mark for reuse
    ReleaseEdgeIndex(index);

    return true;
}

// ============================================================================
// Lookup Operations
// ============================================================================

const AssociationEdge* AssociationMatrix::GetAssociation(
    PatternID source,
    PatternID target
) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    auto key = std::make_pair(source, target);
    auto it = edge_lookup_.find(key);

    if (it == edge_lookup_.end()) {
        return nullptr;
    }

    return edges_[it->second].get();
}

bool AssociationMatrix::HasAssociation(PatternID source, PatternID target) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    auto key = std::make_pair(source, target);
    return edge_lookup_.find(key) != edge_lookup_.end();
}

std::vector<const AssociationEdge*> AssociationMatrix::GetOutgoingAssociations(PatternID source) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::vector<const AssociationEdge*> result;

    auto it = outgoing_index_.find(source);
    if (it != outgoing_index_.end()) {
        result.reserve(it->second.size());
        for (size_t index : it->second) {
            result.push_back(edges_[index].get());
        }
    }

    return result;
}

std::vector<const AssociationEdge*> AssociationMatrix::GetIncomingAssociations(PatternID target) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::vector<const AssociationEdge*> result;

    if (!config_.enable_reverse_lookup) {
        // Fallback: linear scan through edge_lookup (slower)
        for (const auto& [key, index] : edge_lookup_) {
            if (key.second == target) {
                result.push_back(edges_[index].get());
            }
        }
    } else {
        auto it = incoming_index_.find(target);
        if (it != incoming_index_.end()) {
            result.reserve(it->second.size());
            for (size_t index : it->second) {
                result.push_back(edges_[index].get());
            }
        }
    }

    return result;
}

std::vector<const AssociationEdge*> AssociationMatrix::GetAssociationsByType(AssociationType type) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::vector<const AssociationEdge*> result;

    if (!config_.enable_type_index) {
        // Fallback: scan all edges
        for (const auto& [key, index] : edge_lookup_) {
            const AssociationEdge* edge = edges_[index].get();
            if (edge->GetType() == type) {
                result.push_back(edge);
            }
        }
    } else {
        auto it = type_index_.find(type);
        if (it != type_index_.end()) {
            result.reserve(it->second.size());
            for (size_t index : it->second) {
                result.push_back(edges_[index].get());
            }
        }
    }

    return result;
}

std::vector<PatternID> AssociationMatrix::GetNeighbors(PatternID pattern, bool outgoing) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::vector<PatternID> neighbors;

    const auto& index_map = outgoing ? outgoing_index_ : incoming_index_;
    auto it = index_map.find(pattern);

    if (it != index_map.end()) {
        neighbors.reserve(it->second.size());
        for (size_t edge_index : it->second) {
            const AssociationEdge& edge = *edges_[edge_index];
            PatternID neighbor = outgoing ? edge.GetTarget() : edge.GetSource();
            neighbors.push_back(neighbor);
        }
    }

    return neighbors;
}

std::vector<PatternID> AssociationMatrix::GetMutualNeighbors(PatternID pattern) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::vector<PatternID> result;

    // Get outgoing neighbors
    auto outgoing_it = outgoing_index_.find(pattern);
    if (outgoing_it == outgoing_index_.end()) {
        return result;
    }

    // Build set of outgoing neighbors
    std::unordered_set<PatternID> outgoing_set;
    for (size_t edge_index : outgoing_it->second) {
        outgoing_set.insert(edges_[edge_index]->GetTarget());
    }

    // Check incoming neighbors
    auto incoming_it = incoming_index_.find(pattern);
    if (incoming_it != incoming_index_.end()) {
        for (size_t edge_index : incoming_it->second) {
            PatternID source = edges_[edge_index]->GetSource();
            if (outgoing_set.find(source) != outgoing_set.end()) {
                result.push_back(source);
            }
        }
    }

    return result;
}

// ============================================================================
// Strength Operations
// ============================================================================

bool AssociationMatrix::StrengthenAssociation(PatternID source, PatternID target, float amount) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto key = std::make_pair(source, target);
    auto it = edge_lookup_.find(key);

    if (it == edge_lookup_.end()) {
        return false;
    }

    edges_[it->second]->AdjustStrength(amount);
    return true;
}

bool AssociationMatrix::WeakenAssociation(PatternID source, PatternID target, float amount) {
    return StrengthenAssociation(source, target, -amount);
}

void AssociationMatrix::ApplyDecayAll(Timestamp::Duration elapsed_time) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    for (const auto& [key, index] : edge_lookup_) {
        edges_[index]->ApplyDecay(elapsed_time);
    }
}

void AssociationMatrix::ApplyDecayPattern(PatternID pattern, Timestamp::Duration elapsed_time) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    // Apply decay to outgoing edges
    auto out_it = outgoing_index_.find(pattern);
    if (out_it != outgoing_index_.end()) {
        for (size_t index : out_it->second) {
            edges_[index]->ApplyDecay(elapsed_time);
        }
    }

    // Apply decay to incoming edges
    auto in_it = incoming_index_.find(pattern);
    if (in_it != incoming_index_.end()) {
        for (size_t index : in_it->second) {
            edges_[index]->ApplyDecay(elapsed_time);
        }
    }
}

// ============================================================================
// Statistics
// ============================================================================

size_t AssociationMatrix::GetAssociationCount() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return edge_lookup_.size();
}

size_t AssociationMatrix::GetPatternCount() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::unordered_set<PatternID> unique_patterns;
    for (const auto& [key, index] : edge_lookup_) {
        unique_patterns.insert(key.first);
        unique_patterns.insert(key.second);
    }

    return unique_patterns.size();
}

float AssociationMatrix::GetAverageDegree() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    if (outgoing_index_.empty()) return 0.0f;

    size_t total_degree = 0;
    for (const auto& [pattern, indices] : outgoing_index_) {
        total_degree += indices.size();
    }

    return static_cast<float>(total_degree) / outgoing_index_.size();
}

float AssociationMatrix::GetAverageStrength() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    if (edge_lookup_.empty()) return 0.0f;

    float total_strength = 0.0f;
    for (const auto& [key, index] : edge_lookup_) {
        total_strength += edges_[index]->GetStrength();
    }

    return total_strength / edge_lookup_.size();
}

float AssociationMatrix::GetDensity() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    size_t pattern_count = GetPatternCount();
    if (pattern_count <= 1) return 0.0f;

    size_t possible_edges = pattern_count * (pattern_count - 1);  // Directed graph
    return static_cast<float>(edge_lookup_.size()) / possible_edges;
}

// ============================================================================
// Graph Properties
// ============================================================================

size_t AssociationMatrix::GetDegree(PatternID pattern, bool outgoing) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    const auto& index_map = outgoing ? outgoing_index_ : incoming_index_;
    auto it = index_map.find(pattern);

    if (it == index_map.end()) {
        return 0;
    }

    return it->second.size();
}

std::vector<PatternID> AssociationMatrix::GetIsolatedPatterns() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::vector<PatternID> isolated;

    // Get all patterns
    std::unordered_set<PatternID> all_patterns;
    for (const auto& [key, index] : edge_lookup_) {
        all_patterns.insert(key.first);
        all_patterns.insert(key.second);
    }

    // Check which patterns have no edges
    for (const PatternID& pattern : all_patterns) {
        bool has_outgoing = outgoing_index_.find(pattern) != outgoing_index_.end();
        bool has_incoming = incoming_index_.find(pattern) != incoming_index_.end();

        if (!has_outgoing && !has_incoming) {
            isolated.push_back(pattern);
        }
    }

    return isolated;
}

std::vector<PatternID> AssociationMatrix::GetAllPatterns() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    // Get all patterns (both source and target)
    std::unordered_set<PatternID> all_patterns;
    for (const auto& [key, index] : edge_lookup_) {
        all_patterns.insert(key.first);  // source
        all_patterns.insert(key.second); // target
    }

    return std::vector<PatternID>(all_patterns.begin(), all_patterns.end());
}

// ============================================================================
// Activation Propagation
// ============================================================================

std::vector<AssociationMatrix::ActivationResult> AssociationMatrix::PropagateActivation(
    PatternID source,
    float initial_activation,
    size_t max_hops,
    float min_activation,
    const ContextVector* context
) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    // Breadth-first propagation with activation accumulation
    std::unordered_map<PatternID, float> activations;
    std::queue<std::pair<PatternID, size_t>> queue;  // (pattern, hop_count)
    std::unordered_set<PatternID> visited;

    activations[source] = initial_activation;
    queue.push({source, 0});
    visited.insert(source);

    while (!queue.empty()) {
        auto [current, hops] = queue.front();
        queue.pop();

        if (hops >= max_hops) continue;

        float current_activation = activations[current];

        // Get outgoing associations
        auto it = outgoing_index_.find(current);
        if (it == outgoing_index_.end()) continue;

        for (size_t edge_index : it->second) {
            const AssociationEdge& edge = *edges_[edge_index];
            PatternID target = edge.GetTarget();

            // Compute propagated activation
            float strength = context ?
                edge.GetContextualStrength(*context) :
                edge.GetStrength();

            float propagated = current_activation * strength;

            // Accumulate activation at target
            activations[target] += propagated;

            // Continue propagation if significant and not visited
            if (propagated >= min_activation && visited.find(target) == visited.end()) {
                queue.push({target, hops + 1});
                visited.insert(target);
            }
        }
    }

    // Convert to result vector (exclude source)
    std::vector<ActivationResult> results;
    results.reserve(activations.size());

    for (const auto& [pattern, activation] : activations) {
        if (pattern != source && activation >= min_activation) {
            results.push_back({pattern, activation});
        }
    }

    // Sort by activation (descending)
    std::sort(results.begin(), results.end(),
        [](const ActivationResult& a, const ActivationResult& b) {
            return a.activation > b.activation;
        });

    return results;
}

// ============================================================================
// Serialization
// ============================================================================

void AssociationMatrix::Serialize(std::ostream& out) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    // Write count
    size_t count = edge_lookup_.size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));

    // Write each edge
    for (const auto& [key, index] : edge_lookup_) {
        edges_[index]->Serialize(out);
    }
}

std::unique_ptr<AssociationMatrix> AssociationMatrix::Deserialize(std::istream& in) {
    auto matrix = std::make_unique<AssociationMatrix>();

    size_t count;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));

    for (size_t i = 0; i < count; ++i) {
        auto edge = AssociationEdge::Deserialize(in);
        matrix->AddAssociation(*edge);
    }

    return matrix;
}

// ============================================================================
// Memory Management
// ============================================================================

void AssociationMatrix::Compact() {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    if (deleted_indices_.empty()) return;

    // Rebuild edge vector without deleted entries
    std::vector<std::unique_ptr<AssociationEdge>> new_edges;
    std::unordered_map<std::pair<PatternID, PatternID>, size_t, PatternPairHash> new_lookup;

    new_edges.reserve(edge_lookup_.size());

    for (const auto& [key, old_index] : edge_lookup_) {
        size_t new_index = new_edges.size();
        new_edges.push_back(edges_[old_index]->Clone());
        new_lookup[key] = new_index;
    }

    edges_ = std::move(new_edges);
    edge_lookup_ = std::move(new_lookup);
    deleted_indices_.clear();

    // Rebuild all indices
    RebuildIndices();
}

void AssociationMatrix::Clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    edges_.clear();
    outgoing_index_.clear();
    incoming_index_.clear();
    edge_lookup_.clear();
    type_index_.clear();
    deleted_indices_.clear();
}

size_t AssociationMatrix::EstimateMemoryUsage() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    size_t total = 0;

    // Edges
    total += edges_.capacity() * sizeof(AssociationEdge);

    // Indices (approximate)
    total += outgoing_index_.size() * (sizeof(PatternID) + sizeof(std::vector<size_t>) + 16);
    total += incoming_index_.size() * (sizeof(PatternID) + sizeof(std::vector<size_t>) + 16);
    total += type_index_.size() * (sizeof(AssociationType) + sizeof(std::vector<size_t>) + 16);

    // Lookup map
    total += edge_lookup_.size() * (sizeof(std::pair<PatternID, PatternID>) + sizeof(size_t) + 16);

    // Deleted indices
    total += deleted_indices_.capacity() * sizeof(size_t);

    return total;
}

// ============================================================================
// Debugging
// ============================================================================

void AssociationMatrix::PrintStatistics(std::ostream& out) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    out << "AssociationMatrix Statistics:\n";
    out << "  Association Count: " << edge_lookup_.size() << "\n";
    out << "  Pattern Count: " << GetPatternCount() << "\n";
    out << "  Average Degree: " << GetAverageDegree() << "\n";
    out << "  Average Strength: " << GetAverageStrength() << "\n";
    out << "  Density: " << GetDensity() << "\n";
    out << "  Memory Usage: " << EstimateMemoryUsage() << " bytes\n";
    out << "  Deleted Indices: " << deleted_indices_.size() << "\n";
}

std::string AssociationMatrix::ToString() const {
    std::ostringstream oss;
    oss << "AssociationMatrix{";
    oss << "count=" << GetAssociationCount();
    oss << ", patterns=" << GetPatternCount();
    oss << ", avg_deg=" << GetAverageDegree();
    oss << "}";
    return oss.str();
}

} // namespace dpan
