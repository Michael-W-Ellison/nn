// File: src/agent/word_sequence.hpp
//
// The agent network's reader of a TEXT channel's word sequence
// (docs/AGENT_PLAN.md, Phase 10): word ids -> word vectors -> a GRU over the
// words in order (and, bidirectional, a second GRU from the last word to the
// first). The output is, per GRU, its state after its last step and the mean
// of its states over the words; then the mean of the word vectors.
//
// Ids: 0 = no word (the sequence ends at the first 0); 1..P = the P
// pretrained vectors (shared between copies of the model, fixed unless
// trainable); P+1..P+L = L learned vectors.
#pragma once

#include "nn/layers.hpp"
#include "nn/matrix.hpp"
#include "nn/recurrent.hpp"

#include <cstdint>
#include <random>
#include <iosfwd>
#include <memory>
#include <vector>

namespace dpan {
namespace agent {

class WordSequenceModel {
public:
    /// @param pretrained Vectors of ids 1..rows (nullptr = none); their
    ///        width sets the dimension if given
    /// @param train_pretrained Learn the pretrained vectors too (the model
    ///        then keeps its own copy)
    WordSequenceModel(size_t length, size_t learned_words, size_t dimension, size_t units, bool bidirectional,
                      std::shared_ptr<const nn::Matrix> pretrained, bool train_pretrained, uint32_t seed);

    /// Training only: each word is read as an empty vector with probability
    /// `word_dropout`, each output value is zeroed with probability `dropout`
    /// (the others scaled by 1 / (1 - dropout))
    void SetDropout(float word_dropout, float dropout);

    size_t Length() const { return length_; }
    size_t OutputSize() const { return grus_.size() * 2 * units_ + dimension_; }

    /// ids: rows x Length() word ids (as floats). Forward keeps what Backward needs.
    nn::Matrix Forward(const nn::Matrix& ids, bool training);
    nn::Matrix Infer(const nn::Matrix& ids) const;

    /// Adds to the gradients of the learned vectors and the GRU
    void Backward(const nn::Matrix& grad_output);

    std::vector<nn::Parameter> Parameters();
    void ZeroGradients();
    size_t ParameterCount() const;

    /// The pretrained vectors are not serialized; a deserialized model needs
    /// them attached (rows and width must match)
    bool NeedsPretrained() const { return pretrained_rows_ > 0 && !train_pretrained_ && !pretrained_; }
    /// @throws std::invalid_argument if the shape does not match
    void AttachPretrained(std::shared_ptr<const nn::Matrix> pretrained);

    void Serialize(std::ostream& out) const;
    static WordSequenceModel Deserialize(std::istream& in);

private:
    WordSequenceModel() = default;

    void BackwardMasked(const nn::Matrix& grad_output);
    /// Vector of an id (nullptr for 0 or an id beyond the table)
    const float* Vector(uint32_t id) const;
    /// Ids of each row and their count (up to the first 0)
    void Parse(const nn::Matrix& ids, std::vector<std::vector<uint32_t>>& rows, std::vector<size_t>& lengths) const;
    /// Word index read at `step` by GRU `direction` (0: first to last, 1: last to first)
    static size_t Position(size_t direction, size_t step, size_t length);
    std::vector<nn::Matrix> Inputs(const std::vector<std::vector<uint32_t>>& rows, const std::vector<size_t>& lengths,
                                   size_t direction) const;
    nn::Matrix Output(const std::vector<std::vector<nn::Matrix>>& states,
                      const std::vector<std::vector<uint32_t>>& rows, const std::vector<size_t>& lengths) const;

    size_t length_{0};
    size_t dimension_{0};
    size_t units_{0};
    size_t pretrained_rows_{0};
    bool train_pretrained_{false};
    std::shared_ptr<const nn::Matrix> pretrained_;  // fixed, shared
    nn::Matrix learned_, learned_grad_;              // trainable rows (with train_pretrained: pretrained first)
    std::vector<std::unique_ptr<nn::GRU>> grus_;     // one per direction

    float word_dropout_{0.0f};
    float dropout_{0.0f};
    std::mt19937 rng_;
    nn::Matrix cache_mask_;  // output dropout of the last Forward (empty = none)

    std::vector<std::vector<uint32_t>> cache_ids_;
    std::vector<size_t> cache_lengths_;

public:
    WordSequenceModel(const WordSequenceModel& other);
    WordSequenceModel& operator=(const WordSequenceModel&) = delete;
    WordSequenceModel(WordSequenceModel&&) = default;
};

}  // namespace agent
}  // namespace dpan
