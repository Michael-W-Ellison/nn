// File: src/nn/convolution.cpp
#include "nn/convolution.hpp"

#include "core/binary_io.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace dpan {
namespace nn {

namespace {

void WriteShape(std::ostream& out, const ImageShape& shape) {
    io::WriteSize(out, shape.height);
    io::WriteSize(out, shape.width);
    io::WriteSize(out, shape.channels);
}

ImageShape ReadShape(std::istream& in) {
    ImageShape shape;
    shape.height = io::ReadSize(in, 1u << 16);
    shape.width = io::ReadSize(in, 1u << 16);
    shape.channels = io::ReadSize(in, 1u << 16);
    return shape;
}

void CheckRows(const Matrix& input, const ImageShape& shape, const char* layer) {
    if (input.Cols() != shape.Size()) {
        throw std::invalid_argument(std::string(layer) + ": expected rows of " + std::to_string(shape.height) + "x" +
                                    std::to_string(shape.width) + "x" + std::to_string(shape.channels) + " = " +
                                    std::to_string(shape.Size()) + " values, got " + std::to_string(input.Cols()));
    }
}

/// Matrix with the same values and another shape
Matrix Reshape(const Matrix& m, size_t rows, size_t cols) {
    return Matrix(rows, cols, std::vector<float>(m.Data(), m.Data() + m.Size()));
}

}  // namespace

// ============================================================================
// Conv2D
// ============================================================================

Conv2D::Conv2D(ImageShape input, size_t filters, size_t kernel, size_t stride, size_t padding,
               Initialization init, uint32_t seed)
    : input_(input), kernel_(kernel), stride_(stride), padding_(padding) {
    if (input.Size() == 0 || filters == 0 || kernel == 0 || stride == 0) {
        throw std::invalid_argument("Conv2D: shapes, filters, kernel and stride must be positive");
    }
    output_.channels = filters;
    Configure();
    size_t fan_in = kernel * kernel * input.channels;
    weights_ = Matrix(fan_in, filters);
    bias_ = Matrix(1, filters);
    std::mt19937 rng(seed);
    if (init == Initialization::HE) {
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / static_cast<float>(fan_in)));
        for (size_t i = 0; i < weights_.Size(); ++i) weights_.Data()[i] = dist(rng);
    } else if (init == Initialization::XAVIER) {
        float limit = std::sqrt(6.0f / static_cast<float>(fan_in + filters));
        std::uniform_real_distribution<float> dist(-limit, limit);
        for (size_t i = 0; i < weights_.Size(); ++i) weights_.Data()[i] = dist(rng);
    }
    weight_grad_ = Matrix(fan_in, filters);
    bias_grad_ = Matrix(1, filters);
}

void Conv2D::Configure() {
    size_t padded_h = input_.height + 2 * padding_, padded_w = input_.width + 2 * padding_;
    if (kernel_ > padded_h || kernel_ > padded_w) {
        throw std::invalid_argument("Conv2D: kernel larger than the padded input");
    }
    output_.height = (padded_h - kernel_) / stride_ + 1;
    output_.width = (padded_w - kernel_) / stride_ + 1;
}

Matrix Conv2D::Im2Col(const Matrix& input) const {
    const size_t n = input.Rows(), c = input_.channels, k = kernel_;
    Matrix columns(n * output_.height * output_.width, k * k * c);
    for (size_t s = 0; s < n; ++s) {
        const float* image = input.Data() + s * input_.Size();
        for (size_t oy = 0; oy < output_.height; ++oy) {
            for (size_t ox = 0; ox < output_.width; ++ox) {
                float* row = columns.Data() + ((s * output_.height + oy) * output_.width + ox) * columns.Cols();
                for (size_t ky = 0; ky < k; ++ky) {
                    long iy = static_cast<long>(oy * stride_ + ky) - static_cast<long>(padding_);
                    if (iy < 0 || iy >= static_cast<long>(input_.height)) continue;
                    for (size_t kx = 0; kx < k; ++kx) {
                        long ix = static_cast<long>(ox * stride_ + kx) - static_cast<long>(padding_);
                        if (ix < 0 || ix >= static_cast<long>(input_.width)) continue;
                        const float* pixel = image + (static_cast<size_t>(iy) * input_.width + static_cast<size_t>(ix)) * c;
                        std::copy(pixel, pixel + c, row + (ky * k + kx) * c);
                    }
                }
            }
        }
    }
    return columns;
}

Matrix Conv2D::Col2Im(const Matrix& columns, size_t batch) const {
    const size_t c = input_.channels, k = kernel_;
    Matrix images(batch, input_.Size());
    for (size_t s = 0; s < batch; ++s) {
        float* image = images.Data() + s * input_.Size();
        for (size_t oy = 0; oy < output_.height; ++oy) {
            for (size_t ox = 0; ox < output_.width; ++ox) {
                const float* row = columns.Data() + ((s * output_.height + oy) * output_.width + ox) * columns.Cols();
                for (size_t ky = 0; ky < k; ++ky) {
                    long iy = static_cast<long>(oy * stride_ + ky) - static_cast<long>(padding_);
                    if (iy < 0 || iy >= static_cast<long>(input_.height)) continue;
                    for (size_t kx = 0; kx < k; ++kx) {
                        long ix = static_cast<long>(ox * stride_ + kx) - static_cast<long>(padding_);
                        if (ix < 0 || ix >= static_cast<long>(input_.width)) continue;
                        float* pixel = image + (static_cast<size_t>(iy) * input_.width + static_cast<size_t>(ix)) * c;
                        const float* value = row + (ky * k + kx) * c;
                        for (size_t i = 0; i < c; ++i) pixel[i] += value[i];
                    }
                }
            }
        }
    }
    return images;
}

Matrix Conv2D::Infer(const Matrix& input) const {
    CheckRows(input, input_, "Conv2D");
    Matrix product = AddRowVector(MatMul(Im2Col(input), weights_), bias_);
    return Reshape(product, input.Rows(), output_.Size());
}

Matrix Conv2D::Forward(const Matrix& input, bool) {
    CheckRows(input, input_, "Conv2D");
    columns_cache_ = Im2Col(input);
    batch_cache_ = input.Rows();
    Matrix product = AddRowVector(MatMul(columns_cache_, weights_), bias_);
    return Reshape(product, input.Rows(), output_.Size());
}

Matrix Conv2D::Backward(const Matrix& grad_output) {
    if (columns_cache_.Empty()) throw std::logic_error("Conv2D::Backward called before Forward");
    if (grad_output.Rows() != batch_cache_ || grad_output.Cols() != output_.Size()) {
        throw std::invalid_argument("Conv2D::Backward: gradient shape does not match the last Forward");
    }
    // One row per output position, as in the forward product
    Matrix grad = Reshape(grad_output, batch_cache_ * output_.height * output_.width, output_.channels);
    weight_grad_ += MatMulTransposeA(columns_cache_, grad);
    bias_grad_ += SumRows(grad);
    return Col2Im(MatMulTransposeB(grad, weights_), batch_cache_);
}

std::vector<Parameter> Conv2D::Parameters() {
    return {{&weights_, &weight_grad_}, {&bias_, &bias_grad_}};
}

void Conv2D::SerializeParameters(std::ostream& out) const {
    WriteShape(out, input_);
    io::WriteSize(out, kernel_);
    io::WriteSize(out, stride_);
    io::WriteSize(out, padding_);
    weights_.Serialize(out);
    bias_.Serialize(out);
}

std::unique_ptr<Conv2D> Conv2D::DeserializeParameters(std::istream& in) {
    std::unique_ptr<Conv2D> layer(new Conv2D());
    layer->input_ = ReadShape(in);
    layer->kernel_ = io::ReadSize(in, 1u << 10);
    layer->stride_ = io::ReadSize(in, 1u << 10);
    layer->padding_ = io::ReadSize(in, 1u << 10);
    layer->weights_ = Matrix::Deserialize(in);
    layer->bias_ = Matrix::Deserialize(in);
    if (layer->input_.Size() == 0 || layer->kernel_ == 0 || layer->stride_ == 0 ||
        layer->weights_.Rows() != layer->kernel_ * layer->kernel_ * layer->input_.channels ||
        layer->bias_.Rows() != 1 || layer->bias_.Cols() != layer->weights_.Cols()) {
        throw std::runtime_error("Conv2D::Deserialize: inconsistent shapes");
    }
    layer->output_.channels = layer->weights_.Cols();
    layer->Configure();
    layer->weight_grad_ = Matrix(layer->weights_.Rows(), layer->weights_.Cols());
    layer->bias_grad_ = Matrix(1, layer->bias_.Cols());
    return layer;
}

std::unique_ptr<Layer> Conv2D::Clone() const {
    auto copy = std::make_unique<Conv2D>(*this);
    copy->columns_cache_ = Matrix();
    return copy;
}

// ============================================================================
// MaxPool2D
// ============================================================================

MaxPool2D::MaxPool2D(ImageShape input, size_t pool, size_t stride)
    : input_(input), pool_(pool), stride_(stride == 0 ? pool : stride) {
    if (input.Size() == 0 || pool == 0 || pool > input.height || pool > input.width) {
        throw std::invalid_argument("MaxPool2D: pool must be positive and fit the input");
    }
    output_ = {(input.height - pool) / stride_ + 1, (input.width - pool) / stride_ + 1, input.channels};
}

Matrix MaxPool2D::Pool(const Matrix& input, std::vector<uint32_t>* winners) const {
    CheckRows(input, input_, "MaxPool2D");
    const size_t c = input_.channels;
    Matrix out(input.Rows(), output_.Size());
    if (winners) winners->assign(input.Rows() * output_.Size(), 0);
    for (size_t s = 0; s < input.Rows(); ++s) {
        const float* image = input.Data() + s * input_.Size();
        for (size_t oy = 0; oy < output_.height; ++oy) {
            for (size_t ox = 0; ox < output_.width; ++ox) {
                for (size_t ch = 0; ch < c; ++ch) {
                    float best = -std::numeric_limits<float>::infinity();
                    size_t best_index = 0;
                    for (size_t py = 0; py < pool_; ++py) {
                        for (size_t px = 0; px < pool_; ++px) {
                            size_t index = ((oy * stride_ + py) * input_.width + ox * stride_ + px) * c + ch;
                            if (image[index] > best) {
                                best = image[index];
                                best_index = index;
                            }
                        }
                    }
                    size_t o = (oy * output_.width + ox) * c + ch;
                    out(s, o) = best;
                    if (winners) (*winners)[s * output_.Size() + o] = static_cast<uint32_t>(best_index);
                }
            }
        }
    }
    return out;
}

Matrix MaxPool2D::Infer(const Matrix& input) const { return Pool(input, nullptr); }

Matrix MaxPool2D::Forward(const Matrix& input, bool) {
    batch_cache_ = input.Rows();
    return Pool(input, &winners_);
}

Matrix MaxPool2D::Backward(const Matrix& grad_output) {
    if (grad_output.Rows() != batch_cache_ || grad_output.Cols() != output_.Size() || winners_.empty()) {
        throw std::logic_error("MaxPool2D::Backward: no matching Forward");
    }
    Matrix grad(batch_cache_, input_.Size());
    for (size_t s = 0; s < batch_cache_; ++s) {
        for (size_t o = 0; o < output_.Size(); ++o) {
            grad(s, winners_[s * output_.Size() + o]) += grad_output(s, o);
        }
    }
    return grad;
}

void MaxPool2D::SerializeParameters(std::ostream& out) const {
    WriteShape(out, input_);
    io::WriteSize(out, pool_);
    io::WriteSize(out, stride_);
}

std::unique_ptr<MaxPool2D> MaxPool2D::DeserializeParameters(std::istream& in) {
    ImageShape shape = ReadShape(in);
    size_t pool = io::ReadSize(in, 1u << 10);
    size_t stride = io::ReadSize(in, 1u << 10);
    try {
        return std::make_unique<MaxPool2D>(shape, pool, stride);
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error(std::string("MaxPool2D::Deserialize: ") + e.what());
    }
}

// ============================================================================
// Flatten
// ============================================================================

Matrix Flatten::Infer(const Matrix& input) const {
    CheckRows(input, input_, "Flatten");
    return input;
}

void Flatten::SerializeParameters(std::ostream& out) const { WriteShape(out, input_); }

std::unique_ptr<Flatten> Flatten::DeserializeParameters(std::istream& in) {
    return std::make_unique<Flatten>(ReadShape(in));
}

}  // namespace nn
}  // namespace dpan
