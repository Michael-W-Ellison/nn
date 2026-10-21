// File: src/nn/loss.cpp
#include "nn/loss.hpp"
#include "nn/layers.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dpan {
namespace nn {

namespace {

constexpr float kProbabilityEpsilon = 1e-7f;

void RequireMatchingBatch(const Matrix& prediction, const Matrix& target, const std::string& loss) {
    if (!prediction.SameShape(target)) {
        throw std::invalid_argument(loss + ": prediction " + prediction.ShapeString() +
                                    " and target " + target.ShapeString() + " differ");
    }
    if (prediction.Rows() == 0) {
        throw std::invalid_argument(loss + ": empty batch");
    }
}

}  // namespace

float MeanSquaredError::Value(const Matrix& prediction, const Matrix& target) const {
    RequireMatchingBatch(prediction, target, Name());
    double sum = 0.0;
    for (size_t i = 0; i < prediction.Size(); ++i) {
        double diff = prediction.Data()[i] - target.Data()[i];
        sum += diff * diff;
    }
    return static_cast<float>(0.5 * sum / static_cast<double>(prediction.Rows()));
}

Matrix MeanSquaredError::Gradient(const Matrix& prediction, const Matrix& target) const {
    RequireMatchingBatch(prediction, target, Name());
    return (prediction - target) * (1.0f / static_cast<float>(prediction.Rows()));
}

float BinaryCrossEntropy::Value(const Matrix& prediction, const Matrix& target) const {
    RequireMatchingBatch(prediction, target, Name());
    double sum = 0.0;
    for (size_t i = 0; i < prediction.Size(); ++i) {
        double p = std::clamp(prediction.Data()[i], kProbabilityEpsilon, 1.0f - kProbabilityEpsilon);
        double t = target.Data()[i];
        sum -= t * std::log(p) + (1.0 - t) * std::log(1.0 - p);
    }
    return static_cast<float>(sum / static_cast<double>(prediction.Rows()));
}

Matrix BinaryCrossEntropy::Gradient(const Matrix& prediction, const Matrix& target) const {
    RequireMatchingBatch(prediction, target, Name());
    Matrix grad(prediction.Rows(), prediction.Cols());
    float scale = 1.0f / static_cast<float>(prediction.Rows());
    for (size_t i = 0; i < prediction.Size(); ++i) {
        float p = std::clamp(prediction.Data()[i], kProbabilityEpsilon, 1.0f - kProbabilityEpsilon);
        float t = target.Data()[i];
        grad.Data()[i] = scale * (p - t) / (p * (1.0f - p));
    }
    return grad;
}

float SoftmaxCrossEntropy::Value(const Matrix& logits, const Matrix& target) const {
    RequireMatchingBatch(logits, target, Name());
    double sum = 0.0;
    for (size_t r = 0; r < logits.Rows(); ++r) {
        const float* row = logits.Data() + r * logits.Cols();
        float max_value = *std::max_element(row, row + logits.Cols());
        double log_sum = 0.0;
        for (size_t c = 0; c < logits.Cols(); ++c) {
            log_sum += std::exp(static_cast<double>(row[c] - max_value));
        }
        log_sum = std::log(log_sum) + max_value;
        for (size_t c = 0; c < logits.Cols(); ++c) {
            // -sum t * log softmax = -sum t * (x - logsumexp)
            sum -= target(r, c) * (row[c] - log_sum);
        }
    }
    return static_cast<float>(sum / static_cast<double>(logits.Rows()));
}

Matrix SoftmaxCrossEntropy::Gradient(const Matrix& logits, const Matrix& target) const {
    RequireMatchingBatch(logits, target, Name());
    return (Softmax::Compute(logits) - target) * (1.0f / static_cast<float>(logits.Rows()));
}

float Accuracy(const Matrix& prediction, const Matrix& target) {
    RequireMatchingBatch(prediction, target, "Accuracy");
    auto predicted = ArgMaxRows(prediction);
    auto expected = ArgMaxRows(target);
    size_t correct = 0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        if (predicted[i] == expected[i]) {
            correct++;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(predicted.size());
}

float BinaryAccuracy(const Matrix& prediction, const Matrix& target) {
    RequireMatchingBatch(prediction, target, "BinaryAccuracy");
    size_t correct = 0;
    for (size_t i = 0; i < prediction.Size(); ++i) {
        bool predicted = prediction.Data()[i] >= 0.5f;
        bool expected = target.Data()[i] >= 0.5f;
        if (predicted == expected) {
            correct++;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(prediction.Size());
}

}  // namespace nn
}  // namespace dpan
