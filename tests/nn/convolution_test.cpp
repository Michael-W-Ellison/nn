// File: tests/nn/convolution_test.cpp
#include "nn/convolution.hpp"
#include "nn/loss.hpp"
#include "nn/optimizer.hpp"
#include "nn/sequential.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <sstream>

namespace dpan {
namespace nn {
namespace {

Matrix Random(size_t rows, size_t cols, uint32_t seed, float scale = 1.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, scale);
    Matrix m(rows, cols);
    for (size_t i = 0; i < m.Size(); ++i) m.Data()[i] = normal(rng);
    return m;
}

/// L = sum(layer(x) * weights): dL/doutput = weights
double Loss(const Layer& layer, const Matrix& x, const Matrix& weights) {
    Matrix y = layer.Infer(x);
    double sum = 0.0;
    for (size_t i = 0; i < y.Size(); ++i) sum += static_cast<double>(y.Data()[i]) * weights.Data()[i];
    return sum;
}

/// Compares Backward() with central differences for the input and every parameter
void CheckGradients(Layer& layer, Matrix x, float epsilon, float tolerance) {
    Matrix out = layer.Forward(x, true);
    Matrix weights = Random(out.Rows(), out.Cols(), 7);
    layer.ZeroGradients();
    Matrix grad_input = layer.Backward(weights);

    for (size_t i = 0; i < x.Size(); ++i) {
        float saved = x.Data()[i];
        x.Data()[i] = saved + epsilon;
        double plus = Loss(layer, x, weights);
        x.Data()[i] = saved - epsilon;
        double minus = Loss(layer, x, weights);
        x.Data()[i] = saved;
        double numeric = (plus - minus) / (2.0 * epsilon);
        ASSERT_NEAR(numeric, grad_input.Data()[i], tolerance * std::max(1.0, std::fabs(numeric))) << "input " << i;
    }
    for (auto& parameter : layer.Parameters()) {
        for (size_t i = 0; i < parameter.value->Size(); ++i) {
            float saved = parameter.value->Data()[i];
            parameter.value->Data()[i] = saved + epsilon;
            double plus = Loss(layer, x, weights);
            parameter.value->Data()[i] = saved - epsilon;
            double minus = Loss(layer, x, weights);
            parameter.value->Data()[i] = saved;
            double numeric = (plus - minus) / (2.0 * epsilon);
            ASSERT_NEAR(numeric, parameter.gradient->Data()[i], tolerance * std::max(1.0, std::fabs(numeric)))
                << "parameter value " << i;
        }
    }
}

TEST(Conv2DTest, OutputShapes) {
    Conv2D same({8, 6, 3}, 4);  // 3x3, stride 1, padding 1
    EXPECT_EQ((ImageShape{8, 6, 4}), same.OutputShape());
    Conv2D strided({8, 6, 3}, 5, 3, 2, 1);
    EXPECT_EQ((ImageShape{4, 3, 5}), strided.OutputShape());
    Conv2D valid({5, 5, 1}, 2, 3, 1, 0);
    EXPECT_EQ((ImageShape{3, 3, 2}), valid.OutputShape());
    EXPECT_EQ(9u * 3u, same.Weights().Rows());
    EXPECT_THROW(Conv2D({2, 2, 1}, 1, 5, 1, 0), std::invalid_argument);
    EXPECT_THROW(same.Infer(Matrix(1, 10)), std::invalid_argument);
}

TEST(Conv2DTest, ComputesAKnownConvolution) {
    // 3x3 single-channel image, one 3x3 kernel with a 1 in the centre-right:
    // each output is the pixel to its right (0 past the border)
    Conv2D conv({3, 3, 1}, 1, 3, 1, 1, Initialization::ZEROS);
    conv.Weights()(5, 0) = 1.0f;  // (ky=1, kx=2)
    conv.Bias()(0, 0) = 0.5f;
    Matrix image = Matrix::FromRows({{1, 2, 3, 4, 5, 6, 7, 8, 9}});
    Matrix out = conv.Infer(image);
    std::vector<float> expected = {2, 3, 0, 5, 6, 0, 8, 9, 0};
    for (size_t i = 0; i < 9; ++i) EXPECT_FLOAT_EQ(expected[i] + 0.5f, out(0, i)) << i;
}

TEST(Conv2DTest, GradientsMatchFiniteDifferences) {
    Conv2D same({5, 4, 2}, 3, 3, 1, 1, Initialization::HE, 3);
    CheckGradients(same, Random(2, 5 * 4 * 2, 1), 1e-2f, 2e-2f);
    Conv2D strided({6, 5, 2}, 2, 3, 2, 1, Initialization::HE, 4);
    CheckGradients(strided, Random(3, 6 * 5 * 2, 2), 1e-2f, 2e-2f);
}

TEST(MaxPool2DTest, PoolsAndRoutesGradients) {
    MaxPool2D pool({4, 4, 1}, 2);
    EXPECT_EQ((ImageShape{2, 2, 1}), pool.OutputShape());
    Matrix image = Matrix::FromRows({{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15}});
    Matrix out = pool.Forward(image, true);
    EXPECT_EQ((std::vector<float>{6, 8, 14, 16}), out.Row(0));
    Matrix grad = pool.Backward(Matrix::FromRows({{1, 2, 3, 4}}));
    EXPECT_FLOAT_EQ(1.0f, grad(0, 5));
    EXPECT_FLOAT_EQ(2.0f, grad(0, 7));
    EXPECT_FLOAT_EQ(3.0f, grad(0, 13));
    EXPECT_FLOAT_EQ(4.0f, grad(0, 14));
    EXPECT_FLOAT_EQ(0.0f, grad(0, 0));
}

TEST(MaxPool2DTest, GradientsMatchFiniteDifferences) {
    // Values spread apart so small perturbations keep the same maxima
    std::mt19937 rng(5);
    Matrix x(2, 6 * 6 * 2);
    std::vector<float> values(x.Size());
    for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i) * 0.1f;
    std::shuffle(values.begin(), values.end(), rng);
    std::copy(values.begin(), values.end(), x.Data());
    MaxPool2D pool({6, 6, 2}, 2);
    CheckGradients(pool, x, 1e-3f, 1e-2f);
    MaxPool2D overlapping({6, 6, 2}, 3, 1);
    CheckGradients(overlapping, x, 1e-3f, 1e-2f);
}

TEST(FlattenTest, IsAnIdentityWithAShapeCheck) {
    Flatten flatten({2, 2, 3});
    Matrix x = Random(2, 12, 6);
    EXPECT_EQ(x.Values(), flatten.Forward(x, true).Values());
    EXPECT_EQ(x.Values(), flatten.Backward(x).Values());
    EXPECT_THROW(flatten.Infer(Matrix(1, 11)), std::invalid_argument);
}

TEST(ConvolutionTest, SequentialSavesAndLoadsImageLayers) {
    Sequential model;
    model.Emplace<Conv2D>(ImageShape{6, 6, 1}, 4)
        .Emplace<ReLU>()
        .Emplace<MaxPool2D>(ImageShape{6, 6, 4}, 2)
        .Emplace<Flatten>(ImageShape{3, 3, 4})
        .Emplace<Dense>(36, 2);
    Matrix x = Random(3, 36, 8);
    std::stringstream buffer;
    model.Serialize(buffer);
    Sequential restored = Sequential::Deserialize(buffer);
    EXPECT_EQ(model.Predict(x).Values(), restored.Predict(x).Values());
    Sequential copy = model;  // Clone of every layer
    EXPECT_EQ(model.Predict(x).Values(), copy.Predict(x).Values());
}

TEST(ConvolutionTest, SmallNetworkLearnsShapes) {
    // 8x8 images with a horizontal or a vertical bar at a random place
    std::mt19937 rng(9);
    std::uniform_int_distribution<int> where(1, 6);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    auto make = [&](size_t count, Matrix& x, Matrix& y) {
        x = Matrix(count, 64);
        y = Matrix(count, 2);
        for (size_t s = 0; s < count; ++s) {
            bool horizontal = s % 2 == 0;
            int line = where(rng);
            for (int i = 0; i < 64; ++i) x(s, i) = noise(rng);
            for (int t = 1; t < 7; ++t) x(s, horizontal ? line * 8 + t : t * 8 + line) += 1.0f;
            y(s, horizontal ? 0 : 1) = 1.0f;
        }
    };
    Matrix x, y, test_x, test_y;
    make(400, x, y);
    make(200, test_x, test_y);
    Sequential model;
    model.Emplace<Conv2D>(ImageShape{8, 8, 1}, 6)
        .Emplace<ReLU>()
        .Emplace<MaxPool2D>(ImageShape{8, 8, 6}, 2)
        .Emplace<Flatten>(ImageShape{4, 4, 6})
        .Emplace<Dense>(96, 2);
    Adam optimizer(0.01f);
    Sequential::FitOptions options;
    options.epochs = 30;
    options.batch_size = 32;
    model.Fit(x, y, SoftmaxCrossEntropy(), optimizer, options);
    EXPECT_GE(Accuracy(model.Predict(test_x), test_y), 0.95f);
}

}  // namespace
}  // namespace nn
}  // namespace dpan
