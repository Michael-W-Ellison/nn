// File: src/nn/sequential.hpp
//
// Feed-forward network built from a stack of layers, trained with
// mini-batch backpropagation.
//
// Thread safety: Predict() and Evaluate() are const and may run
// concurrently with each other. Training methods (Fit, TrainBatch, Forward,
// Backward) modify the model and need external synchronization.
#pragma once

#include "nn/layers.hpp"
#include "nn/loss.hpp"
#include "nn/optimizer.hpp"

#include <functional>
#include <initializer_list>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace dpan {
namespace nn {

class Sequential {
public:
    Sequential() = default;
    Sequential(const Sequential& other);
    Sequential& operator=(const Sequential& other);
    Sequential(Sequential&&) noexcept = default;
    Sequential& operator=(Sequential&&) noexcept = default;

    /// Append a layer; returns *this for chaining
    Sequential& Add(std::unique_ptr<Layer> layer);

    /// Construct and append a layer: model.Emplace<Dense>(4, 8)
    template <typename LayerType, typename... Args>
    Sequential& Emplace(Args&&... args) {
        return Add(std::make_unique<LayerType>(std::forward<Args>(args)...));
    }

    size_t LayerCount() const { return layers_.size(); }
    Layer& GetLayer(size_t index) { return *layers_.at(index); }
    const Layer& GetLayer(size_t index) const { return *layers_.at(index); }

    /// Copy of layers [begin, end) as a new model (e.g. the encoder half of
    /// an autoencoder)
    Sequential Slice(size_t begin, size_t end) const;

    // ------------------------------------------------------------------
    // Inference
    // ------------------------------------------------------------------

    /// Outputs for a batch (one sample per row)
    Matrix Predict(const Matrix& input) const;

    /// Output for one sample
    std::vector<float> Predict(const std::vector<float>& input) const;
    std::vector<float> Predict(std::initializer_list<float> input) const {
        return Predict(std::vector<float>(input));
    }

    /// Mean loss over a dataset
    float Evaluate(const Matrix& input, const Matrix& target, const Loss& loss) const;

    // ------------------------------------------------------------------
    // Training
    // ------------------------------------------------------------------

    /// Forward pass that caches activations for Backward()
    Matrix Forward(const Matrix& input, bool training = true);

    /// Backpropagate dLoss/dOutput through all layers, accumulating gradients
    void Backward(const Matrix& grad_output);

    void ZeroGradients();

    /// All trainable parameters, in layer order
    std::vector<Parameter> Parameters();

    size_t ParameterCount() const;

    /// One optimization step on a batch; returns the batch loss
    float TrainBatch(const Matrix& input, const Matrix& target, const Loss& loss,
                     Optimizer& optimizer);

    struct FitOptions {
        size_t epochs{100};
        size_t batch_size{32};
        bool shuffle{true};
        uint32_t seed{42};

        /// Fraction of samples held out for validation (taken from the end)
        float validation_split{0.0f};

        /// Stop when validation (or training) loss has not improved by
        /// min_delta for this many epochs (0 = disabled)
        size_t early_stopping_patience{0};
        float min_delta{0.0f};

        /// Called after each epoch with (epoch, train_loss, validation_loss);
        /// return false to stop training. validation_loss is NaN without a
        /// validation split.
        std::function<bool(size_t, float, float)> on_epoch_end;
    };

    struct History {
        std::vector<float> train_loss;
        std::vector<float> validation_loss;
        size_t epochs_run{0};
        bool stopped_early{false};
    };

    /// Train on a dataset (one sample per row of input/target)
    History Fit(const Matrix& input, const Matrix& target, const Loss& loss,
                Optimizer& optimizer, const FitOptions& options);
    History Fit(const Matrix& input, const Matrix& target, const Loss& loss,
                Optimizer& optimizer);

    // ------------------------------------------------------------------
    // Persistence
    // ------------------------------------------------------------------

    void Serialize(std::ostream& out) const;
    static Sequential Deserialize(std::istream& in);

    /// Write to / read from a file; Save returns false on I/O failure,
    /// Load throws std::runtime_error on missing or malformed files
    bool Save(const std::string& path) const;
    static Sequential Load(const std::string& path);

    /// Multi-line description of layers and parameter counts
    std::string Summary() const;

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};

}  // namespace nn
}  // namespace dpan
