// File: tests/nn/recurrent_test.cpp
#include "nn/optimizer.hpp"
#include "nn/recurrent.hpp"

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

/// L = sum over steps of sum(h_t * weights_t), computed with Step()
double Loss(const GRU& gru, const std::vector<Matrix>& xs, const Matrix& h0,
            const std::vector<std::vector<bool>>& resets, const std::vector<Matrix>& weights) {
    Matrix h = h0;
    double sum = 0.0;
    for (size_t t = 0; t < xs.size(); ++t) {
        for (size_t row = 0; row < h.Rows(); ++row) {
            if (!resets[t][row]) continue;
            for (size_t c = 0; c < h.Cols(); ++c) h(row, c) = 0.0f;
        }
        h = gru.Step(xs[t], h);
        for (size_t i = 0; i < h.Size(); ++i) sum += static_cast<double>(h.Data()[i]) * weights[t].Data()[i];
    }
    return sum;
}

TEST(GRUTest, GradientsMatchFiniteDifferences) {
    const size_t rows = 3, inputs = 4, hidden = 5, steps = 6;
    GRU gru(inputs, hidden, 3);
    std::vector<Matrix> xs, weights;
    for (size_t t = 0; t < steps; ++t) {
        xs.push_back(Random(rows, inputs, 10 + static_cast<uint32_t>(t)));
        weights.push_back(Random(rows, hidden, 30 + static_cast<uint32_t>(t)));
    }
    Matrix h0 = Random(rows, hidden, 50, 0.5f);
    // Row 1 starts a new episode at step 3
    std::vector<std::vector<bool>> resets(steps, std::vector<bool>(rows, false));
    resets[3][1] = true;

    auto states = gru.Forward(xs, h0, resets);
    ASSERT_EQ(steps, states.size());
    gru.ZeroGradients();
    auto grad_xs = gru.Backward(weights);

    const float epsilon = 1e-3f, tolerance = 2e-2f;
    for (size_t t = 0; t < steps; ++t) {
        for (size_t i = 0; i < xs[t].Size(); ++i) {
            float saved = xs[t].Data()[i];
            xs[t].Data()[i] = saved + epsilon;
            double plus = Loss(gru, xs, h0, resets, weights);
            xs[t].Data()[i] = saved - epsilon;
            double minus = Loss(gru, xs, h0, resets, weights);
            xs[t].Data()[i] = saved;
            double numeric = (plus - minus) / (2.0 * epsilon);
            ASSERT_NEAR(numeric, grad_xs[t].Data()[i], tolerance * std::max(1.0, std::fabs(numeric)))
                << "step " << t << " input " << i;
        }
    }
    size_t index = 0;
    for (auto& parameter : gru.Parameters()) {
        for (size_t i = 0; i < parameter.value->Size(); ++i) {
            float saved = parameter.value->Data()[i];
            parameter.value->Data()[i] = saved + epsilon;
            double plus = Loss(gru, xs, h0, resets, weights);
            parameter.value->Data()[i] = saved - epsilon;
            double minus = Loss(gru, xs, h0, resets, weights);
            parameter.value->Data()[i] = saved;
            double numeric = (plus - minus) / (2.0 * epsilon);
            ASSERT_NEAR(numeric, parameter.gradient->Data()[i], tolerance * std::max(1.0, std::fabs(numeric)))
                << "parameter " << index << " value " << i;
        }
        ++index;
    }
}

TEST(GRUTest, ForwardMatchesStepAndResets) {
    GRU gru(2, 3, 4);
    std::vector<Matrix> xs = {Random(2, 2, 1), Random(2, 2, 2), Random(2, 2, 3)};
    Matrix h0 = Random(2, 3, 5);
    std::vector<std::vector<bool>> resets = {{false, false}, {true, false}, {false, false}};
    auto states = gru.Forward(xs, h0, resets);
    Matrix h = gru.Step(xs[0], h0);
    EXPECT_EQ(h.Values(), states[0].Values());
    Matrix reset_row = h;
    for (size_t c = 0; c < 3; ++c) reset_row(0, c) = 0.0f;
    h = gru.Step(xs[1], reset_row);
    EXPECT_EQ(h.Values(), states[1].Values());
    // A reset row does not depend on the state before it
    Matrix other = h0;
    other(0, 0) += 1.0f;
    auto changed = gru.Forward(xs, other, resets);
    for (size_t c = 0; c < 3; ++c) EXPECT_EQ(states[2](0, c), changed[2](0, c));
    EXPECT_NE(states[0](0, 0), changed[0](0, 0));
}

TEST(GRUTest, LearnsToRememberTheFirstInput) {
    // The sign of the first input decides the target at the last step; the
    // inputs in between are noise
    GRU gru(1, 8, 6);
    Matrix readout = Random(8, 1, 7, 0.1f);
    Adam optimizer(0.01f);
    std::mt19937 rng(8);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    const size_t steps = 10, rows = 32;
    double last_loss = 0.0;
    for (int iteration = 0; iteration < 600; ++iteration) {
        std::vector<Matrix> xs(steps, Matrix(rows, 1));
        std::vector<float> target(rows);
        for (size_t row = 0; row < rows; ++row) {
            for (size_t t = 0; t < steps; ++t) xs[t](row, 0) = noise(rng);
            target[row] = xs[0](row, 0) > 0.0f ? 1.0f : -1.0f;
        }
        auto states = gru.Forward(xs, Matrix(rows, 8), {});
        Matrix y = MatMul(states.back(), readout);
        Matrix grad_y(rows, 1);
        double loss = 0.0;
        for (size_t row = 0; row < rows; ++row) {
            float diff = y(row, 0) - target[row];
            loss += diff * diff / rows;
            grad_y(row, 0) = 2.0f * diff / rows;
        }
        std::vector<Matrix> grad_hs(steps, Matrix(rows, 8));
        grad_hs.back() = MatMulTransposeB(grad_y, readout);
        Matrix grad_readout = MatMulTransposeA(states.back(), grad_y);
        gru.ZeroGradients();
        gru.Backward(grad_hs);
        auto parameters = gru.Parameters();
        parameters.push_back({&readout, &grad_readout});
        optimizer.Step(parameters);
        last_loss = loss;
    }
    EXPECT_LT(last_loss, 0.2);  // 1.0 = no memory of the first input
}

TEST(GRUTest, SerializesExactly) {
    GRU gru(3, 4, 9);
    std::stringstream buffer;
    gru.Serialize(buffer);
    GRU loaded = GRU::Deserialize(buffer);
    Matrix x = Random(2, 3, 1), h = Random(2, 4, 2);
    EXPECT_EQ(gru.Step(x, h).Values(), loaded.Step(x, h).Values());
    EXPECT_EQ(gru.ParameterCount(), loaded.ParameterCount());
    EXPECT_THROW(GRU(0, 4), std::invalid_argument);
}

}  // namespace
}  // namespace nn
}  // namespace dpan
