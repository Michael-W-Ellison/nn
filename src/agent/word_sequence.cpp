// File: src/agent/word_sequence.cpp
#include "agent/word_sequence.hpp"

#include "core/binary_io.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <stdexcept>

namespace dpan {
namespace agent {

using nn::Matrix;

WordSequenceModel::WordSequenceModel(size_t length, size_t learned_words, size_t dimension, size_t units,
                                     bool bidirectional, std::shared_ptr<const Matrix> pretrained,
                                     bool train_pretrained, uint32_t seed)
    : length_(length),
      dimension_(pretrained ? pretrained->Cols() : dimension),
      units_(units),
      pretrained_rows_(pretrained ? pretrained->Rows() : 0),
      train_pretrained_(train_pretrained && pretrained),
      rng_(seed + 7) {
    if (length_ == 0 || dimension_ == 0 || units_ == 0 || learned_words == 0) {
        throw std::invalid_argument("WordSequenceModel: sizes must be positive");
    }
    const size_t own = learned_words + (train_pretrained_ ? pretrained_rows_ : 0);
    learned_ = Matrix(own, dimension_);
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 0.1f);
    for (size_t i = 0; i < learned_.Size(); ++i) learned_.Data()[i] = normal(rng);
    if (train_pretrained_) {
        std::copy(pretrained->Data(), pretrained->Data() + pretrained->Size(), learned_.Data());
    } else {
        pretrained_ = std::move(pretrained);
    }
    learned_grad_ = Matrix(own, dimension_);
    for (size_t d = 0; d < (bidirectional ? 2u : 1u); ++d) {
        grus_.push_back(std::make_unique<nn::GRU>(dimension_, units_, seed + 1 + static_cast<uint32_t>(d),
                                                  static_cast<float>(length_)));
    }
}

WordSequenceModel::WordSequenceModel(const WordSequenceModel& other)
    : length_(other.length_),
      dimension_(other.dimension_),
      units_(other.units_),
      pretrained_rows_(other.pretrained_rows_),
      train_pretrained_(other.train_pretrained_),
      pretrained_(other.pretrained_),
      learned_(other.learned_),
      learned_grad_(other.learned_grad_),
      word_dropout_(other.word_dropout_),
      dropout_(other.dropout_),
      rng_(other.rng_) {
    for (const auto& gru : other.grus_) grus_.push_back(std::make_unique<nn::GRU>(*gru));
}

void WordSequenceModel::SetDropout(float word_dropout, float dropout) {
    if (!(word_dropout >= 0.0f && word_dropout < 1.0f) || !(dropout >= 0.0f && dropout < 1.0f)) {
        throw std::invalid_argument("WordSequenceModel: dropout must be in [0, 1)");
    }
    word_dropout_ = word_dropout;
    dropout_ = dropout;
}

const float* WordSequenceModel::Vector(uint32_t id) const {
    if (id == 0) return nullptr;
    size_t index = id - 1;
    if (!train_pretrained_ && pretrained_rows_ > 0) {
        if (index < pretrained_rows_) return pretrained_ ? pretrained_->Data() + index * dimension_ : nullptr;
        index -= pretrained_rows_;
    }
    return index < learned_.Rows() ? learned_.Data() + index * dimension_ : nullptr;
}

void WordSequenceModel::Parse(const Matrix& ids, std::vector<std::vector<uint32_t>>& rows,
                              std::vector<size_t>& lengths) const {
    if (ids.Cols() != length_) {
        throw std::invalid_argument("WordSequenceModel: expected " + std::to_string(length_) + " word ids per row");
    }
    rows.assign(ids.Rows(), std::vector<uint32_t>(length_, 0));
    lengths.assign(ids.Rows(), 0);
    for (size_t r = 0; r < ids.Rows(); ++r) {
        for (size_t t = 0; t < length_; ++t) {
            float id = ids(r, t);
            if (!(id >= 1.0f)) break;
            rows[r][t] = static_cast<uint32_t>(id);
            lengths[r] = t + 1;
        }
    }
}

size_t WordSequenceModel::Position(size_t direction, size_t step, size_t length) {
    return direction == 0 ? step : length - 1 - step;
}

std::vector<Matrix> WordSequenceModel::Inputs(const std::vector<std::vector<uint32_t>>& rows,
                                              const std::vector<size_t>& lengths, size_t direction) const {
    std::vector<Matrix> xs(length_, Matrix(rows.size(), dimension_));
    for (size_t r = 0; r < rows.size(); ++r) {
        for (size_t t = 0; t < lengths[r]; ++t) {
            if (const float* v = Vector(rows[r][Position(direction, t, lengths[r])])) {
                std::copy(v, v + dimension_, xs[t].Data() + r * dimension_);
            }
        }
    }
    return xs;
}

Matrix WordSequenceModel::Output(const std::vector<std::vector<Matrix>>& states,
                                 const std::vector<std::vector<uint32_t>>& rows,
                                 const std::vector<size_t>& lengths) const {
    Matrix out(rows.size(), OutputSize());
    for (size_t r = 0; r < rows.size(); ++r) {
        const size_t n = lengths[r];
        if (n == 0) continue;
        float* o = out.Data() + r * OutputSize();
        for (size_t d = 0; d < states.size(); ++d, o += 2 * units_) {
            std::copy(states[d][n - 1].Data() + r * units_, states[d][n - 1].Data() + (r + 1) * units_, o);
            for (size_t t = 0; t < n; ++t) {
                const float* h = states[d][t].Data() + r * units_;
                for (size_t j = 0; j < units_; ++j) o[units_ + j] += h[j] / static_cast<float>(n);
            }
        }
        for (size_t t = 0; t < n; ++t) {
            if (const float* v = Vector(rows[r][t])) {
                for (size_t j = 0; j < dimension_; ++j) o[j] += v[j];
            }
        }
        for (size_t j = 0; j < dimension_; ++j) o[j] /= static_cast<float>(n);
    }
    return out;
}

Matrix WordSequenceModel::Forward(const Matrix& ids, bool training) {
    if (!training) return Infer(ids);
    if (NeedsPretrained()) throw std::logic_error("WordSequenceModel: pretrained vectors not attached");
    Parse(ids, cache_ids_, cache_lengths_);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    if (word_dropout_ > 0.0f) {
        // A dropped word has an id beyond the table: an empty vector, no gradient
        for (size_t r = 0; r < cache_ids_.size(); ++r) {
            for (size_t t = 0; t < cache_lengths_[r]; ++t) {
                if (uniform(rng_) < word_dropout_) cache_ids_[r][t] = UINT32_MAX;
            }
        }
    }
    std::vector<std::vector<Matrix>> states;
    for (size_t d = 0; d < grus_.size(); ++d) {
        states.push_back(grus_[d]->Forward(Inputs(cache_ids_, cache_lengths_, d), Matrix(ids.Rows(), units_), {}));
    }
    Matrix out = Output(states, cache_ids_, cache_lengths_);
    cache_mask_ = Matrix();
    if (dropout_ > 0.0f) {
        cache_mask_ = Matrix(out.Rows(), out.Cols());
        const float keep = 1.0f / (1.0f - dropout_);
        for (size_t i = 0; i < out.Size(); ++i) {
            cache_mask_.Data()[i] = uniform(rng_) < dropout_ ? 0.0f : keep;
            out.Data()[i] *= cache_mask_.Data()[i];
        }
    }
    return out;
}

Matrix WordSequenceModel::Infer(const Matrix& ids) const {
    if (NeedsPretrained()) throw std::logic_error("WordSequenceModel: pretrained vectors not attached");
    std::vector<std::vector<uint32_t>> rows;
    std::vector<size_t> lengths;
    Parse(ids, rows, lengths);
    size_t longest = lengths.empty() ? 0 : *std::max_element(lengths.begin(), lengths.end());
    std::vector<std::vector<Matrix>> states(grus_.size());
    for (size_t d = 0; d < grus_.size(); ++d) {
        auto xs = Inputs(rows, lengths, d);
        Matrix state(ids.Rows(), units_);
        for (size_t t = 0; t < longest; ++t) {
            state = grus_[d]->Step(xs[t], state);
            states[d].push_back(state);
        }
    }
    return Output(states, rows, lengths);
}

void WordSequenceModel::Backward(const Matrix& grad_output) {
    const size_t rows = cache_ids_.size();
    if (grad_output.Rows() != rows || grad_output.Cols() != OutputSize()) {
        throw std::invalid_argument("WordSequenceModel::Backward: gradient does not match the last Forward");
    }
    if (cache_mask_.Empty()) {
        BackwardMasked(grad_output);
    } else {
        BackwardMasked(Hadamard(grad_output, cache_mask_));
    }
}

void WordSequenceModel::BackwardMasked(const Matrix& grad_output) {
    const size_t rows = cache_ids_.size();
    // Gradient of each word's vector: from every direction's GRU input and
    // from the mean vector
    auto add = [&](size_t r, size_t t, const float* g, float scale) {
        size_t index = cache_ids_[r][t] - 1;
        if (!train_pretrained_ && pretrained_rows_ > 0) {
            if (index < pretrained_rows_) return;  // fixed
            index -= pretrained_rows_;
        }
        if (index >= learned_.Rows()) return;
        float* to = learned_grad_.Data() + index * dimension_;
        for (size_t j = 0; j < dimension_; ++j) to[j] += scale * g[j];
    };
    for (size_t d = 0; d < grus_.size(); ++d) {
        std::vector<Matrix> grad_states(length_, Matrix(rows, units_));
        for (size_t r = 0; r < rows; ++r) {
            const size_t n = cache_lengths_[r];
            if (n == 0) continue;
            const float* g = grad_output.Data() + r * OutputSize() + d * 2 * units_;
            std::copy(g, g + units_, grad_states[n - 1].Data() + r * units_);
            for (size_t t = 0; t < n; ++t) {
                float* to = grad_states[t].Data() + r * units_;
                for (size_t j = 0; j < units_; ++j) to[j] += g[units_ + j] / static_cast<float>(n);
            }
        }
        auto grad_inputs = grus_[d]->Backward(grad_states);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t t = 0; t < cache_lengths_[r]; ++t) {
                add(r, Position(d, t, cache_lengths_[r]), grad_inputs[t].Data() + r * dimension_, 1.0f);
            }
        }
    }
    for (size_t r = 0; r < rows; ++r) {
        const size_t n = cache_lengths_[r];
        const float* grad_mean = grad_output.Data() + r * OutputSize() + grus_.size() * 2 * units_;
        for (size_t t = 0; t < n; ++t) add(r, t, grad_mean, 1.0f / static_cast<float>(n));
    }
}

std::vector<nn::Parameter> WordSequenceModel::Parameters() {
    std::vector<nn::Parameter> parameters{{&learned_, &learned_grad_}};
    for (auto& gru : grus_) {
        auto p = gru->Parameters();
        parameters.insert(parameters.end(), p.begin(), p.end());
    }
    return parameters;
}

void WordSequenceModel::ZeroGradients() {
    learned_grad_.Fill(0.0f);
    for (auto& gru : grus_) gru->ZeroGradients();
}

size_t WordSequenceModel::ParameterCount() const {
    size_t count = learned_.Size();
    for (const auto& gru : grus_) count += gru->ParameterCount();
    return count;
}

void WordSequenceModel::AttachPretrained(std::shared_ptr<const Matrix> pretrained) {
    if (!pretrained || pretrained->Rows() != pretrained_rows_ || pretrained->Cols() != dimension_) {
        throw std::invalid_argument("WordSequenceModel: pretrained vectors do not match");
    }
    if (!train_pretrained_) pretrained_ = std::move(pretrained);
}

void WordSequenceModel::Serialize(std::ostream& out) const {
    io::WriteSize(out, length_);
    io::WriteSize(out, dimension_);
    io::WriteSize(out, units_);
    io::WriteSize(out, pretrained_rows_);
    io::Write<uint8_t>(out, train_pretrained_ ? 1 : 0);
    learned_.Serialize(out);
    io::Write<uint8_t>(out, static_cast<uint8_t>(grus_.size()));
    for (const auto& gru : grus_) gru->Serialize(out);
    io::Write<float>(out, word_dropout_);
    io::Write<float>(out, dropout_);
    std::ostringstream rng;
    rng << rng_;
    io::WriteString(out, rng.str());
}

WordSequenceModel WordSequenceModel::Deserialize(std::istream& in) {
    WordSequenceModel model;
    model.length_ = io::ReadSize(in, 1u << 16);
    model.dimension_ = io::ReadSize(in, 1u << 16);
    model.units_ = io::ReadSize(in, 1u << 16);
    model.pretrained_rows_ = io::ReadSize(in, 1u << 26);
    model.train_pretrained_ = io::Read<uint8_t>(in) != 0;
    model.learned_ = Matrix::Deserialize(in);
    model.learned_grad_ = Matrix(model.learned_.Rows(), model.learned_.Cols());
    const size_t directions = io::Read<uint8_t>(in);
    if (directions < 1 || directions > 2) throw std::runtime_error("WordSequenceModel: malformed data");
    for (size_t d = 0; d < directions; ++d) {
        model.grus_.push_back(std::make_unique<nn::GRU>(nn::GRU::Deserialize(in)));
        if (model.grus_.back()->InputSize() != model.dimension_ || model.grus_.back()->HiddenSize() != model.units_) {
            throw std::runtime_error("WordSequenceModel: malformed data");
        }
    }
    if (model.learned_.Cols() != model.dimension_ || model.length_ == 0) {
        throw std::runtime_error("WordSequenceModel: malformed data");
    }
    model.word_dropout_ = io::Read<float>(in);
    model.dropout_ = io::Read<float>(in);
    std::istringstream rng(io::ReadString(in, 1u << 16));
    rng >> model.rng_;
    if (!rng || !(model.word_dropout_ >= 0.0f && model.word_dropout_ < 1.0f) ||
        !(model.dropout_ >= 0.0f && model.dropout_ < 1.0f)) {
        throw std::runtime_error("WordSequenceModel: malformed data");
    }
    return model;
}

}  // namespace agent
}  // namespace dpan
