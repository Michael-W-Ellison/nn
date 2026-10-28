// File: src/agent/word_sequence.cpp
#include "agent/word_sequence.hpp"

#include "core/binary_io.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace dpan {
namespace agent {

using nn::Matrix;

WordSequenceModel::WordSequenceModel(size_t length, size_t learned_words, size_t dimension, size_t units,
                                     std::shared_ptr<const Matrix> pretrained, bool train_pretrained, uint32_t seed)
    : length_(length),
      dimension_(pretrained ? pretrained->Cols() : dimension),
      units_(units),
      pretrained_rows_(pretrained ? pretrained->Rows() : 0),
      train_pretrained_(train_pretrained && pretrained) {
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
    gru_ = std::make_unique<nn::GRU>(dimension_, units_, seed + 1, static_cast<float>(length_));
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
      gru_(other.gru_ ? std::make_unique<nn::GRU>(*other.gru_) : nullptr) {}

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

std::vector<Matrix> WordSequenceModel::Inputs(const std::vector<std::vector<uint32_t>>& rows) const {
    std::vector<Matrix> xs(length_, Matrix(rows.size(), dimension_));
    for (size_t r = 0; r < rows.size(); ++r) {
        for (size_t t = 0; t < length_; ++t) {
            if (const float* v = Vector(rows[r][t])) std::copy(v, v + dimension_, xs[t].Data() + r * dimension_);
        }
    }
    return xs;
}

Matrix WordSequenceModel::Output(const std::vector<Matrix>& states, const std::vector<std::vector<uint32_t>>& rows,
                                 const std::vector<size_t>& lengths) const {
    Matrix out(rows.size(), OutputSize());
    for (size_t r = 0; r < rows.size(); ++r) {
        if (lengths[r] == 0) continue;
        std::copy(states[lengths[r] - 1].Data() + r * units_, states[lengths[r] - 1].Data() + (r + 1) * units_,
                  out.Data() + r * OutputSize());
        float* mean_state = out.Data() + r * OutputSize() + units_;
        for (size_t t = 0; t < lengths[r]; ++t) {
            const float* h = states[t].Data() + r * units_;
            for (size_t j = 0; j < units_; ++j) mean_state[j] += h[j] / static_cast<float>(lengths[r]);
        }
        float* mean = mean_state + units_;
        for (size_t t = 0; t < lengths[r]; ++t) {
            if (const float* v = Vector(rows[r][t])) {
                for (size_t j = 0; j < dimension_; ++j) mean[j] += v[j];
            }
        }
        for (size_t j = 0; j < dimension_; ++j) mean[j] /= static_cast<float>(lengths[r]);
    }
    return out;
}

Matrix WordSequenceModel::Forward(const Matrix& ids, bool training) {
    if (!training) return Infer(ids);
    if (NeedsPretrained()) throw std::logic_error("WordSequenceModel: pretrained vectors not attached");
    Parse(ids, cache_ids_, cache_lengths_);
    auto states = gru_->Forward(Inputs(cache_ids_), Matrix(ids.Rows(), units_), {});
    return Output(states, cache_ids_, cache_lengths_);
}

Matrix WordSequenceModel::Infer(const Matrix& ids) const {
    if (NeedsPretrained()) throw std::logic_error("WordSequenceModel: pretrained vectors not attached");
    std::vector<std::vector<uint32_t>> rows;
    std::vector<size_t> lengths;
    Parse(ids, rows, lengths);
    auto xs = Inputs(rows);
    std::vector<Matrix> states;
    Matrix state(ids.Rows(), units_);
    size_t longest = lengths.empty() ? 0 : *std::max_element(lengths.begin(), lengths.end());
    for (size_t t = 0; t < longest; ++t) {
        state = gru_->Step(xs[t], state);
        states.push_back(state);
    }
    return Output(states, rows, lengths);
}

void WordSequenceModel::Backward(const Matrix& grad_output) {
    const size_t rows = cache_ids_.size();
    if (grad_output.Rows() != rows || grad_output.Cols() != OutputSize()) {
        throw std::invalid_argument("WordSequenceModel::Backward: gradient does not match the last Forward");
    }
    std::vector<Matrix> grad_states(length_, Matrix(rows, units_));
    for (size_t r = 0; r < rows; ++r) {
        if (cache_lengths_[r] == 0) continue;
        std::copy(grad_output.Data() + r * OutputSize(), grad_output.Data() + r * OutputSize() + units_,
                  grad_states[cache_lengths_[r] - 1].Data() + r * units_);
        const float* grad_mean_state = grad_output.Data() + r * OutputSize() + units_;
        for (size_t t = 0; t < cache_lengths_[r]; ++t) {
            float* g = grad_states[t].Data() + r * units_;
            for (size_t j = 0; j < units_; ++j) g[j] += grad_mean_state[j] / static_cast<float>(cache_lengths_[r]);
        }
    }
    auto grad_inputs = gru_->Backward(grad_states);
    for (size_t r = 0; r < rows; ++r) {
        const size_t n = cache_lengths_[r];
        const float* grad_mean = grad_output.Data() + r * OutputSize() + 2 * units_;
        for (size_t t = 0; t < n; ++t) {
            uint32_t id = cache_ids_[r][t];
            size_t index = id - 1;
            if (!train_pretrained_ && pretrained_rows_ > 0) {
                if (index < pretrained_rows_) continue;  // fixed
                index -= pretrained_rows_;
            }
            if (index >= learned_.Rows()) continue;
            float* g = learned_grad_.Data() + index * dimension_;
            const float* from_gru = grad_inputs[t].Data() + r * dimension_;
            for (size_t j = 0; j < dimension_; ++j) g[j] += from_gru[j] + grad_mean[j] / static_cast<float>(n);
        }
    }
}

std::vector<nn::Parameter> WordSequenceModel::Parameters() {
    std::vector<nn::Parameter> parameters{{&learned_, &learned_grad_}};
    auto gru = gru_->Parameters();
    parameters.insert(parameters.end(), gru.begin(), gru.end());
    return parameters;
}

void WordSequenceModel::ZeroGradients() {
    learned_grad_.Fill(0.0f);
    gru_->ZeroGradients();
}

size_t WordSequenceModel::ParameterCount() const { return learned_.Size() + gru_->ParameterCount(); }

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
    gru_->Serialize(out);
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
    model.gru_ = std::make_unique<nn::GRU>(nn::GRU::Deserialize(in));
    if (model.learned_.Cols() != model.dimension_ || model.gru_->InputSize() != model.dimension_ ||
        model.gru_->HiddenSize() != model.units_ || model.length_ == 0) {
        throw std::runtime_error("WordSequenceModel: malformed data");
    }
    return model;
}

}  // namespace agent
}  // namespace dpan
