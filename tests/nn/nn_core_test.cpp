// File: tests/nn/nn_core_test.cpp
//
// Tests for the neural network core: matrix operations, analytic gradients
// (checked against finite differences), training convergence and
// persistence.
#include "nn/sequential.hpp"

#include <gtest/gtest.h>
#include "../test_temp_path.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <thread>

namespace dpan {
namespace nn {
namespace {

Matrix RandomMatrix(size_t rows, size_t cols, uint32_t seed, float scale = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    Matrix m(rows, cols);
    for (size_t i = 0; i < m.Size(); ++i) {
        m.Data()[i] = dist(rng);
    }
    return m;
}

void ExpectMatrixNear(const Matrix& expected, const Matrix& actual, float tolerance) {
    ASSERT_TRUE(expected.SameShape(actual)) << expected.ShapeString() << " vs " << actual.ShapeString();
    for (size_t i = 0; i < expected.Size(); ++i) {
        EXPECT_NEAR(expected.Data()[i], actual.Data()[i], tolerance) << "at index " << i;
    }
}

/// Compare analytic gradients of every parameter (and of the input) with
/// central finite differences of the loss
void CheckGradients(Sequential& model, const Matrix& input, const Matrix& target, const Loss& loss,
                    float tolerance = 2e-3f) {
    model.ZeroGradients();
    Matrix prediction = model.Forward(input, false);
    model.Backward(loss.Gradient(prediction, target));

    const double h = 1e-3;
    auto loss_at = [&]() { return static_cast<double>(loss.Value(model.Predict(input), target)); };

    for (auto& param : model.Parameters()) {
        for (size_t i = 0; i < param.value->Size(); ++i) {
            float original = param.value->Data()[i];
            param.value->Data()[i] = original + static_cast<float>(h);
            double plus = loss_at();
            param.value->Data()[i] = original - static_cast<float>(h);
            double minus = loss_at();
            param.value->Data()[i] = original;

            double numeric = (plus - minus) / (2.0 * h);
            double analytic = param.gradient->Data()[i];
            double scale = std::max(1.0, std::abs(numeric) + std::abs(analytic));
            EXPECT_NEAR(numeric / scale, analytic / scale, tolerance)
                << "parameter index " << i << " numeric " << numeric << " analytic " << analytic;
        }
    }
}

// ============================================================================
// Matrix
// ============================================================================

TEST(NNMatrixTest, MatMulVariantsAgreeWithExplicitTranspose) {
    Matrix a = RandomMatrix(3, 4, 1);
    Matrix b = RandomMatrix(4, 2, 2);
    Matrix c = RandomMatrix(3, 2, 3);

    Matrix ab = MatMul(a, b);
    ASSERT_EQ(3u, ab.Rows());
    ASSERT_EQ(2u, ab.Cols());
    float expected = 0.0f;
    for (size_t k = 0; k < 4; ++k) {
        expected += a(1, k) * b(k, 0);
    }
    EXPECT_NEAR(expected, ab(1, 0), 1e-5f);

    ExpectMatrixNear(MatMul(a.Transpose(), c), MatMulTransposeA(a, c), 1e-5f);
    ExpectMatrixNear(MatMul(c, b.Transpose()), MatMulTransposeB(c, b), 1e-5f);
}

TEST(NNMatrixTest, ShapeMismatchThrows) {
    EXPECT_THROW(MatMul(Matrix(2, 3), Matrix(2, 3)), std::invalid_argument);
    EXPECT_THROW(Matrix(2, 2) + Matrix(2, 3), std::invalid_argument);
    EXPECT_THROW(AddRowVector(Matrix(2, 3), Matrix(1, 2)), std::invalid_argument);
    EXPECT_THROW(Matrix(2, 2, std::vector<float>{1.0f}), std::invalid_argument);
    EXPECT_THROW(Matrix::FromRows({{1.0f, 2.0f}, {3.0f}}), std::invalid_argument);
}

TEST(NNMatrixTest, RowHelpers) {
    Matrix m = Matrix::FromRows({{1, 2, 3}, {4, 5, 6}});
    ExpectMatrixNear(Matrix::FromRows({{11, 22, 33}, {14, 25, 36}}),
                     AddRowVector(m, Matrix::RowVector({10, 20, 30})), 0.0f);
    ExpectMatrixNear(Matrix::RowVector({5, 7, 9}), SumRows(m), 0.0f);
    EXPECT_EQ((std::vector<size_t>{2, 2}), ArgMaxRows(m));
    ExpectMatrixNear(Matrix::FromRows({{4, 5, 6}}), m.SelectRows({1}), 0.0f);
    ExpectMatrixNear(Matrix::FromRows({{2, 3}, {5, 6}}), m.SliceCols(1, 3), 0.0f);
    ExpectMatrixNear(Matrix::FromRows({{0, 1}, {1, 0}}), OneHot({1, 0}, 2), 0.0f);
    ExpectMatrixNear(Matrix::FromRows({{1, 2, 3, 1, 2, 3}, {4, 5, 6, 4, 5, 6}}), ConcatCols(m, m), 0.0f);
}

TEST(NNMatrixTest, SerializationRoundTrip) {
    Matrix m = RandomMatrix(5, 3, 7);
    std::stringstream ss;
    m.Serialize(ss);
    ExpectMatrixNear(m, Matrix::Deserialize(ss), 0.0f);
}

// ============================================================================
// Gradients
// ============================================================================

TEST(NNGradientTest, DenseTanhMeanSquaredError) {
    Sequential model;
    model.Emplace<Dense>(3, 4, Initialization::XAVIER, 1).Emplace<Tanh>().Emplace<Dense>(4, 2, Initialization::XAVIER, 2);
    CheckGradients(model, RandomMatrix(5, 3, 10), RandomMatrix(5, 2, 11), MeanSquaredError());
}

TEST(NNGradientTest, SigmoidBinaryCrossEntropy) {
    Sequential model;
    model.Emplace<Dense>(4, 3, Initialization::XAVIER, 3).Emplace<Sigmoid>();
    Matrix target = Matrix::FromRows({{1, 0, 1}, {0, 0, 1}, {1, 1, 0}});
    CheckGradients(model, RandomMatrix(3, 4, 12), target, BinaryCrossEntropy());
}

TEST(NNGradientTest, SoftmaxCrossEntropyOnLogits) {
    Sequential model;
    model.Emplace<Dense>(4, 5, Initialization::HE, 4).Emplace<LeakyReLU>(0.1f).Emplace<Dense>(5, 3, Initialization::XAVIER, 5);
    CheckGradients(model, RandomMatrix(4, 4, 13), OneHot({0, 2, 1, 2}, 3), SoftmaxCrossEntropy());
}

TEST(NNGradientTest, SoftmaxLayerBackward) {
    Sequential model;
    model.Emplace<Dense>(3, 4, Initialization::XAVIER, 6).Emplace<Softmax>();
    CheckGradients(model, RandomMatrix(4, 3, 14), RandomMatrix(4, 4, 15), MeanSquaredError());
}

TEST(NNGradientTest, ReLUAwayFromKink) {
    // Inputs scaled so no pre-activation sits near zero
    Sequential model;
    auto dense = std::make_unique<Dense>(2, 3, Initialization::ZEROS);
    dense->Weights() = Matrix::FromRows({{1.0f, -1.0f, 0.5f}, {0.5f, 1.0f, -1.0f}});
    dense->Bias() = Matrix::RowVector({0.3f, 0.2f, -0.4f});
    model.Add(std::move(dense)).Emplace<ReLU>().Emplace<Dense>(3, 1, Initialization::XAVIER, 7);
    Matrix input = Matrix::FromRows({{1.0f, 2.0f}, {-2.0f, 1.0f}, {2.0f, -1.5f}});
    CheckGradients(model, input, Matrix::FromRows({{1.0f}, {0.0f}, {-1.0f}}), MeanSquaredError());
}

// ============================================================================
// Training
// ============================================================================

TEST(NNTrainingTest, LearnsXor) {
    Matrix x = Matrix::FromRows({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
    Matrix y = Matrix::FromRows({{0}, {1}, {1}, {0}});

    Sequential model;
    model.Emplace<Dense>(2, 8, Initialization::XAVIER, 3).Emplace<Tanh>()
         .Emplace<Dense>(8, 1, Initialization::XAVIER, 4).Emplace<Sigmoid>();

    Adam adam(0.05f);
    Sequential::FitOptions options;
    options.epochs = 1000;
    options.batch_size = 4;
    auto history = model.Fit(x, y, BinaryCrossEntropy(), adam, options);

    EXPECT_LT(history.train_loss.back(), history.train_loss.front());
    EXPECT_LT(model.Evaluate(x, y, BinaryCrossEntropy()), 0.05f);
    EXPECT_FLOAT_EQ(1.0f, BinaryAccuracy(model.Predict(x), y));
}

TEST(NNTrainingTest, LinearRegressionRecoversCoefficients) {
    // y = 2 x0 - 3 x1 + 1
    Matrix x = RandomMatrix(200, 2, 20);
    Matrix y(200, 1);
    for (size_t i = 0; i < 200; ++i) {
        y(i, 0) = 2.0f * x(i, 0) - 3.0f * x(i, 1) + 1.0f;
    }

    Sequential model;
    model.Emplace<Dense>(2, 1, Initialization::ZEROS);
    SGD sgd(0.1f, 0.9f);
    Sequential::FitOptions options;
    options.epochs = 200;
    options.batch_size = 20;
    model.Fit(x, y, MeanSquaredError(), sgd, options);

    auto& dense = static_cast<Dense&>(model.GetLayer(0));
    EXPECT_NEAR(2.0f, dense.Weights()(0, 0), 1e-2f);
    EXPECT_NEAR(-3.0f, dense.Weights()(1, 0), 1e-2f);
    EXPECT_NEAR(1.0f, dense.Bias()(0, 0), 1e-2f);
}

TEST(NNTrainingTest, MultiClassClassification) {
    // Three Gaussian blobs
    std::mt19937 rng(5);
    std::normal_distribution<float> noise(0.0f, 0.3f);
    const float centers[3][2] = {{0, 2}, {2, -1}, {-2, -1}};
    std::vector<std::vector<float>> rows;
    std::vector<size_t> labels;
    for (int i = 0; i < 300; ++i) {
        size_t label = static_cast<size_t>(i % 3);
        rows.push_back({centers[label][0] + noise(rng), centers[label][1] + noise(rng)});
        labels.push_back(label);
    }
    Matrix x = Matrix::FromRows(rows);
    Matrix y = OneHot(labels, 3);

    Sequential model;
    model.Emplace<Dense>(2, 16, Initialization::HE, 8).Emplace<ReLU>()
         .Emplace<Dense>(16, 3, Initialization::XAVIER, 9);
    Adam adam(0.01f);
    Sequential::FitOptions options;
    options.epochs = 50;
    options.batch_size = 32;
    options.validation_split = 0.2f;
    auto history = model.Fit(x, y, SoftmaxCrossEntropy(), adam, options);

    EXPECT_EQ(50u, history.validation_loss.size());
    EXPECT_GT(Accuracy(model.Predict(x), y), 0.97f);
    // Probabilities: apply softmax to the logits
    Matrix probabilities = Softmax::Compute(model.Predict(x.SelectRows({0})));
    float sum = 0.0f;
    for (size_t c = 0; c < 3; ++c) {
        sum += probabilities(0, c);
    }
    EXPECT_NEAR(1.0f, sum, 1e-5f);
}

TEST(NNTrainingTest, EarlyStoppingAndEpochCallback) {
    Matrix x = RandomMatrix(20, 2, 30);
    Matrix y = RandomMatrix(20, 1, 31);  // Unlearnable noise

    Sequential model;
    model.Emplace<Dense>(2, 1, Initialization::XAVIER, 1);
    SGD sgd(0.01f);
    Sequential::FitOptions options;
    options.epochs = 10000;
    options.early_stopping_patience = 5;
    options.min_delta = 1e-4f;
    auto history = model.Fit(x, y, MeanSquaredError(), sgd, options);
    EXPECT_TRUE(history.stopped_early);
    EXPECT_LT(history.epochs_run, 10000u);

    Sequential::FitOptions callback_options;
    callback_options.epochs = 100;
    size_t calls = 0;
    callback_options.on_epoch_end = [&calls](size_t epoch, float, float validation) {
        calls++;
        EXPECT_TRUE(std::isnan(validation));
        return epoch < 2;  // Stop after the third epoch
    };
    history = model.Fit(x, y, MeanSquaredError(), sgd, callback_options);
    EXPECT_EQ(3u, calls);
    EXPECT_EQ(3u, history.epochs_run);
}

TEST(NNTrainingTest, FitRejectsInvalidInput) {
    Sequential model;
    model.Emplace<Dense>(2, 1);
    SGD sgd(0.1f);
    EXPECT_THROW(model.Fit(Matrix(4, 2), Matrix(3, 1), MeanSquaredError(), sgd), std::invalid_argument);
    EXPECT_THROW(model.Fit(Matrix(0, 2), Matrix(0, 1), MeanSquaredError(), sgd), std::invalid_argument);
    EXPECT_THROW(model.Predict(Matrix(1, 3)), std::invalid_argument);  // Wrong feature count

    Sequential::FitOptions options;
    options.batch_size = 0;
    EXPECT_THROW(model.Fit(Matrix(4, 2), Matrix(4, 1), MeanSquaredError(), sgd, options),
                 std::invalid_argument);
}

// ============================================================================
// Layers
// ============================================================================

TEST(NNLayerTest, DropoutScalesDuringTrainingOnly) {
    Dropout dropout(0.5f, 3);
    Matrix input(100, 100, 1.0f);

    Matrix trained = dropout.Forward(input, true);
    size_t zeros = 0;
    double sum = 0.0;
    for (size_t i = 0; i < trained.Size(); ++i) {
        zeros += trained.Data()[i] == 0.0f ? 1 : 0;
        sum += trained.Data()[i];
    }
    EXPECT_NEAR(0.5, static_cast<double>(zeros) / trained.Size(), 0.03);
    EXPECT_NEAR(1.0, sum / trained.Size(), 0.05);  // Expected value preserved

    ExpectMatrixNear(input, dropout.Forward(input, false), 0.0f);
    ExpectMatrixNear(input, dropout.Infer(input), 0.0f);
    EXPECT_THROW(Dropout(1.0f), std::invalid_argument);
}

TEST(NNLayerTest, SigmoidIsStableForLargeInputs) {
    Sigmoid sigmoid;
    Matrix out = sigmoid.Infer(Matrix::RowVector({-1000.0f, 0.0f, 1000.0f}));
    EXPECT_FLOAT_EQ(0.0f, out(0, 0));
    EXPECT_FLOAT_EQ(0.5f, out(0, 1));
    EXPECT_FLOAT_EQ(1.0f, out(0, 2));
    EXPECT_FALSE(std::isnan(BinaryCrossEntropy().Value(out, Matrix::RowVector({0, 1, 1}))));
}

// ============================================================================
// Model utilities and persistence
// ============================================================================

TEST(NNModelTest, SaveAndLoadPreservePredictions) {
    Sequential model;
    model.Emplace<Dense>(3, 5, Initialization::HE, 1).Emplace<LeakyReLU>(0.2f).Emplace<Dropout>(0.3f)
         .Emplace<Dense>(5, 4, Initialization::XAVIER, 2).Emplace<Tanh>()
         .Emplace<Dense>(4, 2, Initialization::XAVIER, 3).Emplace<Sigmoid>()
         .Emplace<Dense>(2, 3, Initialization::XAVIER, 4).Emplace<Softmax>().Emplace<ReLU>();
    Matrix input = RandomMatrix(6, 3, 40);

    std::string path = test::UniqueTempPath("dpan_nn_model") + ".bin";
    ASSERT_TRUE(model.Save(path));
    Sequential loaded = Sequential::Load(path);
    std::filesystem::remove(path);

    EXPECT_EQ(model.LayerCount(), loaded.LayerCount());
    EXPECT_EQ(model.ParameterCount(), loaded.ParameterCount());
    ExpectMatrixNear(model.Predict(input), loaded.Predict(input), 0.0f);
    EXPECT_EQ(model.Summary(), loaded.Summary());
}

TEST(NNModelTest, LoadRejectsInvalidFiles) {
    EXPECT_THROW(Sequential::Load("/nonexistent/model.bin"), std::runtime_error);

    std::string path = test::UniqueTempPath("dpan_nn_bad") + ".bin";
    std::ofstream(path) << "not a model";
    EXPECT_THROW(Sequential::Load(path), std::runtime_error);
    std::filesystem::remove(path);
}

TEST(NNModelTest, CopyAndSliceAreIndependent) {
    Sequential model;
    model.Emplace<Dense>(2, 3, Initialization::XAVIER, 1).Emplace<Tanh>().Emplace<Dense>(3, 1, Initialization::XAVIER, 2);
    Matrix input = RandomMatrix(4, 2, 50);
    Matrix before = model.Predict(input);

    Sequential copy = model;
    static_cast<Dense&>(copy.GetLayer(0)).Weights().Fill(0.0f);
    ExpectMatrixNear(before, model.Predict(input), 0.0f);

    Sequential encoder = model.Slice(0, 2);
    EXPECT_EQ(2u, encoder.LayerCount());
    EXPECT_EQ(3u, encoder.Predict(input).Cols());
    EXPECT_THROW(model.Slice(2, 5), std::out_of_range);
}

TEST(NNModelTest, ConcurrentPredictMatchesSequential) {
    Sequential model;
    model.Emplace<Dense>(8, 16, Initialization::HE, 1).Emplace<ReLU>().Emplace<Dense>(16, 4, Initialization::XAVIER, 2);
    Matrix input = RandomMatrix(32, 8, 60);
    Matrix expected = model.Predict(input);

    std::vector<std::thread> threads;
    std::vector<Matrix> results(8);
    for (size_t t = 0; t < results.size(); ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 50; ++i) {
                results[t] = model.Predict(input);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    for (const auto& result : results) {
        ExpectMatrixNear(expected, result, 0.0f);
    }
}

TEST(NNOptimizerTest, RejectsInvalidHyperparametersAndMismatchedParameters) {
    EXPECT_THROW(SGD(0.0f), std::invalid_argument);
    EXPECT_THROW(SGD(0.1f, 1.0f), std::invalid_argument);
    EXPECT_THROW(Adam(0.1f, 1.0f), std::invalid_argument);

    Sequential a;
    a.Emplace<Dense>(2, 2);
    Sequential b;
    b.Emplace<Dense>(2, 2).Emplace<Dense>(2, 2);
    Adam adam(0.01f);
    adam.Step(a.Parameters());
    EXPECT_THROW(adam.Step(b.Parameters()), std::logic_error);
    adam.Reset();
    EXPECT_NO_THROW(adam.Step(b.Parameters()));
}

// ============================================================================
// Optimizer state, gradient clipping, learning-rate schedules
// ============================================================================

template <typename OptimizerType>
void ExpectResumeIsExact(OptimizerType first_optimizer, OptimizerType fresh_optimizer) {
    Matrix x = RandomMatrix(16, 3, 70);
    Matrix y = RandomMatrix(16, 2, 71);
    MeanSquaredError mse;

    Sequential model;
    model.Emplace<Dense>(3, 4, Initialization::XAVIER, 1).Emplace<Tanh>().Emplace<Dense>(4, 2, Initialization::XAVIER, 2);
    for (int i = 0; i < 5; ++i) {
        model.TrainBatch(x, y, mse, first_optimizer);
    }

    // Snapshot model and optimizer mid-training
    std::stringstream model_state, optimizer_state;
    model.Serialize(model_state);
    first_optimizer.SaveState(optimizer_state);

    // Continue the original
    for (int i = 0; i < 5; ++i) {
        model.TrainBatch(x, y, mse, first_optimizer);
    }

    // Resume from the snapshot with a differently configured optimizer
    Sequential resumed = Sequential::Deserialize(model_state);
    fresh_optimizer.LoadState(optimizer_state);
    for (int i = 0; i < 5; ++i) {
        resumed.TrainBatch(x, y, mse, fresh_optimizer);
    }

    ExpectMatrixNear(model.Predict(x), resumed.Predict(x), 0.0f);
    EXPECT_EQ(first_optimizer.LearningRate(), fresh_optimizer.LearningRate());
}

TEST(NNOptimizerTest, AdamResumesExactlyFromSavedState) {
    ExpectResumeIsExact(Adam(0.01f, 0.8f, 0.99f), Adam(0.5f));
}

TEST(NNOptimizerTest, SgdResumesExactlyFromSavedState) {
    ExpectResumeIsExact(SGD(0.05f, 0.9f, 0.001f), SGD(0.5f));
}

TEST(NNOptimizerTest, LoadStateRejectsOtherOptimizerType) {
    std::stringstream state;
    SGD(0.1f).SaveState(state);
    Adam adam;
    EXPECT_THROW(adam.LoadState(state), std::runtime_error);

    std::stringstream truncated(std::string("\x04\0\0\0\0\0\0\0Adam", 12));
    EXPECT_THROW(adam.LoadState(truncated), std::runtime_error);
}

TEST(NNOptimizerTest, ClipGradientNormScalesToLimit) {
    Sequential model;
    model.Emplace<Dense>(2, 2, Initialization::XAVIER, 1);
    auto params = model.Parameters();
    params[0].gradient->Fill(3.0f);  // 4 values
    params[1].gradient->Fill(4.0f);  // 2 values: norm = sqrt(4*9 + 2*16) = sqrt(68)

    EXPECT_NEAR(std::sqrt(68.0f), GradientNorm(params), 1e-5f);
    EXPECT_NEAR(std::sqrt(68.0f), ClipGradientNorm(params, 2.0f), 1e-5f);
    EXPECT_NEAR(2.0f, GradientNorm(params), 1e-5f);
    // Direction is preserved
    EXPECT_NEAR(3.0f / 4.0f, (*params[0].gradient)(0, 0) / (*params[1].gradient)(0, 0), 1e-5f);

    // Below the limit, or disabled: unchanged
    ClipGradientNorm(params, 10.0f);
    EXPECT_NEAR(2.0f, GradientNorm(params), 1e-5f);
    ClipGradientNorm(params, 0.0f);
    EXPECT_NEAR(2.0f, GradientNorm(params), 1e-5f);
}

TEST(NNOptimizerTest, ClippingKeepsAggressiveTrainingFinite) {
    Matrix x = RandomMatrix(32, 4, 80, 10.0f);
    Matrix y = RandomMatrix(32, 1, 81, 100.0f);

    auto final_loss = [&](float clip) {
        Sequential model;
        model.Emplace<Dense>(4, 8, Initialization::XAVIER, 1).Emplace<Dense>(8, 1, Initialization::XAVIER, 2);
        SGD sgd(0.05f);
        Sequential::FitOptions options;
        options.epochs = 30;
        options.batch_size = 8;
        options.gradient_clip_norm = clip;
        return model.Fit(x, y, MeanSquaredError(), sgd, options).train_loss.back();
    };

    EXPECT_FALSE(std::isfinite(final_loss(0.0f)));  // Diverges without clipping
    EXPECT_TRUE(std::isfinite(final_loss(1.0f)));
}

TEST(NNScheduleTest, ScheduleFormulas) {
    StepDecay step(10, 0.5f);
    EXPECT_FLOAT_EQ(1.0f, step.NextRate(9, 1.0f, 1.0f, 0.0f));
    EXPECT_FLOAT_EQ(0.5f, step.NextRate(10, 1.0f, 1.0f, 0.0f));
    EXPECT_FLOAT_EQ(0.25f, step.NextRate(25, 1.0f, 1.0f, 0.0f));

    ExponentialDecay exponential(0.9f);
    EXPECT_NEAR(0.81f, exponential.NextRate(2, 1.0f, 1.0f, 0.0f), 1e-6f);

    CosineAnnealing cosine(100, 0.1f);
    EXPECT_NEAR(1.0f, cosine.NextRate(0, 1.0f, 1.0f, 0.0f), 1e-6f);
    EXPECT_NEAR(0.55f, cosine.NextRate(50, 1.0f, 1.0f, 0.0f), 1e-6f);
    EXPECT_NEAR(0.1f, cosine.NextRate(100, 1.0f, 1.0f, 0.0f), 1e-6f);

    EXPECT_THROW(StepDecay(0, 0.5f), std::invalid_argument);
    EXPECT_THROW(ExponentialDecay(1.5f), std::invalid_argument);
    EXPECT_THROW(CosineAnnealing(0), std::invalid_argument);
    EXPECT_THROW(ReduceOnPlateau(1.0f), std::invalid_argument);
}

TEST(NNScheduleTest, ReduceOnPlateauWaitsForPatience) {
    ReduceOnPlateau plateau(0.5f, 2, 0.1f);
    float rate = 1.0f;
    rate = plateau.NextRate(1, 1.0f, rate, 1.0f);   // Improvement
    EXPECT_FLOAT_EQ(1.0f, rate);
    rate = plateau.NextRate(2, 1.0f, rate, 1.0f);   // No improvement (1)
    EXPECT_FLOAT_EQ(1.0f, rate);
    rate = plateau.NextRate(3, 1.0f, rate, 1.0f);   // No improvement (2): reduce
    EXPECT_FLOAT_EQ(0.5f, rate);
    rate = plateau.NextRate(4, 1.0f, rate, 0.5f);   // Improvement resets
    rate = plateau.NextRate(5, 1.0f, rate, 0.6f);
    rate = plateau.NextRate(6, 1.0f, rate, 0.6f);
    EXPECT_FLOAT_EQ(0.25f, rate);
    for (int i = 0; i < 10; ++i) {
        rate = plateau.NextRate(7 + i, 1.0f, rate, 0.6f);
    }
    EXPECT_FLOAT_EQ(0.1f, rate);  // Floor
}

TEST(NNScheduleTest, FitAppliesScheduleEachEpoch) {
    Matrix x = RandomMatrix(16, 2, 90);
    Matrix y = RandomMatrix(16, 1, 91);
    Sequential model;
    model.Emplace<Dense>(2, 1, Initialization::XAVIER, 1);
    SGD sgd(0.1f);
    Sequential::FitOptions options;
    options.epochs = 4;
    options.learning_rate_schedule = std::make_shared<ExponentialDecay>(0.5f);
    auto history = model.Fit(x, y, MeanSquaredError(), sgd, options);

    ASSERT_EQ(4u, history.learning_rate.size());
    EXPECT_FLOAT_EQ(0.1f, history.learning_rate[0]);
    EXPECT_FLOAT_EQ(0.05f, history.learning_rate[1]);
    EXPECT_FLOAT_EQ(0.025f, history.learning_rate[2]);
    EXPECT_FLOAT_EQ(0.0125f, history.learning_rate[3]);
    EXPECT_FLOAT_EQ(0.00625f, sgd.LearningRate());
}

}  // namespace
}  // namespace nn
}  // namespace dpan
