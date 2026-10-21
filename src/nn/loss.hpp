// File: src/nn/loss.hpp
//
// Loss functions. Value() is the mean loss over the batch; Gradient() is
// dLoss/dPrediction for that mean (so it already includes the 1/N factor).
#pragma once

#include "nn/matrix.hpp"

#include <memory>
#include <string>

namespace dpan {
namespace nn {

class Loss {
public:
    virtual ~Loss() = default;
    virtual float Value(const Matrix& prediction, const Matrix& target) const = 0;
    virtual Matrix Gradient(const Matrix& prediction, const Matrix& target) const = 0;
    virtual std::string Name() const = 0;
};

/// Mean squared error: mean over samples of sum over outputs of (p - t)^2 / 2
class MeanSquaredError : public Loss {
public:
    float Value(const Matrix& prediction, const Matrix& target) const override;
    Matrix Gradient(const Matrix& prediction, const Matrix& target) const override;
    std::string Name() const override { return "MeanSquaredError"; }
};

/// Binary cross-entropy on probabilities in (0, 1) (e.g. Sigmoid outputs);
/// targets are 0 or 1 per output. Predictions are clamped away from 0 and 1.
class BinaryCrossEntropy : public Loss {
public:
    float Value(const Matrix& prediction, const Matrix& target) const override;
    Matrix Gradient(const Matrix& prediction, const Matrix& target) const override;
    std::string Name() const override { return "BinaryCrossEntropy"; }
};

/// Softmax + categorical cross-entropy on logits (do not end the network
/// with a Softmax layer when using this loss). Targets are one-hot or
/// probability rows. Gradient: (softmax(logits) - target) / N.
class SoftmaxCrossEntropy : public Loss {
public:
    float Value(const Matrix& logits, const Matrix& target) const override;
    Matrix Gradient(const Matrix& logits, const Matrix& target) const override;
    std::string Name() const override { return "SoftmaxCrossEntropy"; }
};

/// Fraction of rows whose arg-max matches the target's arg-max
float Accuracy(const Matrix& prediction, const Matrix& target);

/// Fraction of outputs on the correct side of 0.5 (binary targets)
float BinaryAccuracy(const Matrix& prediction, const Matrix& target);

}  // namespace nn
}  // namespace dpan
