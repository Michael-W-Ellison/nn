// File: src/nn/convolution.hpp
//
// Image layers. A batch of images is a matrix with one image per row,
// stored height x width x channels: row[(y * width + x) * channels + c]
// (the layout of agent::ImageEncoder).
//
// Conv2D uses im2col: the patches of the batch form one matrix, so the
// convolution is a single matrix product (accelerated by BLAS when DPAN is
// built with it). Its output rows are again height x width x filters.
#pragma once

#include "nn/layers.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dpan {
namespace nn {

struct ImageShape {
    size_t height{0};
    size_t width{0};
    size_t channels{1};

    size_t Size() const { return height * width * channels; }
    bool operator==(const ImageShape& other) const {
        return height == other.height && width == other.width && channels == other.channels;
    }
};

/// 2-D convolution with `filters` kernels of kernel x kernel x channels,
/// zero padding and a bias per filter
class Conv2D : public Layer {
public:
    /// @param padding Zero padding on each side (kernel / 2 keeps the size at stride 1)
    /// @throws std::invalid_argument for empty shapes or a kernel larger than the padded input
    Conv2D(ImageShape input, size_t filters, size_t kernel = 3, size_t stride = 1, size_t padding = 1,
           Initialization init = Initialization::HE, uint32_t seed = 42);

    Matrix Forward(const Matrix& input, bool training) override;
    Matrix Backward(const Matrix& grad_output) override;
    Matrix Infer(const Matrix& input) const override;
    std::vector<Parameter> Parameters() override;
    std::string Type() const override { return "Conv2D"; }
    void SerializeParameters(std::ostream& out) const override;
    std::unique_ptr<Layer> Clone() const override;
    static std::unique_ptr<Conv2D> DeserializeParameters(std::istream& in);

    const ImageShape& InputShape() const { return input_; }
    const ImageShape& OutputShape() const { return output_; }

    /// kernel * kernel * channels x filters
    Matrix& Weights() { return weights_; }
    Matrix& Bias() { return bias_; }

private:
    Conv2D() = default;
    void Configure();

    /// One row per output position (sample, y, x), one column per kernel value
    Matrix Im2Col(const Matrix& input) const;
    Matrix Col2Im(const Matrix& columns, size_t batch) const;

    ImageShape input_;
    ImageShape output_;
    size_t kernel_{3};
    size_t stride_{1};
    size_t padding_{1};
    Matrix weights_;
    Matrix bias_;
    Matrix weight_grad_;
    Matrix bias_grad_;
    Matrix columns_cache_;
    size_t batch_cache_{0};
};

/// Maximum over pool x pool windows of each channel (no padding; positions
/// that do not fill a window are dropped)
class MaxPool2D : public Layer {
public:
    /// @param stride 0 = pool (non-overlapping windows)
    MaxPool2D(ImageShape input, size_t pool = 2, size_t stride = 0);

    Matrix Forward(const Matrix& input, bool training) override;
    Matrix Backward(const Matrix& grad_output) override;
    Matrix Infer(const Matrix& input) const override;
    std::string Type() const override { return "MaxPool2D"; }
    void SerializeParameters(std::ostream& out) const override;
    std::unique_ptr<Layer> Clone() const override { return std::make_unique<MaxPool2D>(*this); }
    static std::unique_ptr<MaxPool2D> DeserializeParameters(std::istream& in);

    const ImageShape& InputShape() const { return input_; }
    const ImageShape& OutputShape() const { return output_; }

private:
    Matrix Pool(const Matrix& input, std::vector<uint32_t>* winners) const;

    ImageShape input_;
    ImageShape output_;
    size_t pool_;
    size_t stride_;
    std::vector<uint32_t> winners_;  // input index of each output's maximum (last Forward)
    size_t batch_cache_{0};
};

/// Marks the end of the image layers. Rows are already flat, so this is an
/// identity; it checks the row length and documents the shape.
class Flatten : public Layer {
public:
    explicit Flatten(ImageShape input) : input_(input) {}

    Matrix Forward(const Matrix& input, bool) override { return Infer(input); }
    Matrix Backward(const Matrix& grad_output) override { return grad_output; }
    Matrix Infer(const Matrix& input) const override;
    std::string Type() const override { return "Flatten"; }
    void SerializeParameters(std::ostream& out) const override;
    std::unique_ptr<Layer> Clone() const override { return std::make_unique<Flatten>(*this); }
    static std::unique_ptr<Flatten> DeserializeParameters(std::istream& in);

    const ImageShape& InputShape() const { return input_; }

private:
    ImageShape input_;
};

}  // namespace nn
}  // namespace dpan
