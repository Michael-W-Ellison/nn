// File: tests/nn/compute_test.cpp
#include "nn/compute.hpp"
#include "nn/matrix.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>

namespace dpan {
namespace nn {
namespace {

Matrix Random(size_t rows, size_t cols, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    Matrix m(rows, cols);
    for (size_t i = 0; i < m.Size(); ++i) m.Data()[i] = normal(rng);
    return m;
}

/// Restores the global compute settings after a test
struct Settings {
    bool blas = UsingBlas() || !BlasAvailable();
    size_t threads = ThreadCount();
    ~Settings() {
        SetUseBlas(blas);
        SetThreadCount(threads);
    }
};

struct Products {
    Matrix ab, atb, abt;
};

Products Compute(const Matrix& a, const Matrix& b, const Matrix& c) {
    return {MatMul(a, b), MatMulTransposeA(a, c), MatMulTransposeB(a, a)};
}

TEST(ComputeTest, ThreadCountDoesNotChangeResults) {
    Settings restore;
    SetUseBlas(false);
    // Large enough to be split across threads
    Matrix a = Random(300, 200, 1), b = Random(200, 150, 2), c = Random(300, 120, 3);
    SetThreadCount(1);
    Products serial = Compute(a, b, c);
    for (size_t threads : {2, 3, 7}) {
        SetThreadCount(threads);
        Products parallel = Compute(a, b, c);
        EXPECT_EQ(serial.ab.Values(), parallel.ab.Values()) << threads;
        EXPECT_EQ(serial.atb.Values(), parallel.atb.Values()) << threads;
        EXPECT_EQ(serial.abt.Values(), parallel.abt.Values()) << threads;
    }
}

TEST(ComputeTest, ProductsMatchDefinition) {
    Settings restore;
    SetUseBlas(false);
    Matrix a = Random(5, 4, 4), b = Random(4, 3, 5);
    Matrix ab = MatMul(a, b);
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (size_t p = 0; p < 4; ++p) sum += static_cast<double>(a(i, p)) * b(p, j);
            EXPECT_NEAR(sum, ab(i, j), 1e-5);
        }
    }
    EXPECT_EQ(0u, MatMul(Matrix(0, 4), b).Rows());
}

TEST(ComputeTest, BlasAgreesWithReference) {
    if (!BlasAvailable()) GTEST_SKIP() << "built without BLAS (DPAN_WITH_BLAS=OFF)";
    Settings restore;
    Matrix a = Random(300, 200, 6), b = Random(200, 150, 7), c = Random(300, 120, 8);
    SetUseBlas(false);
    Products reference = Compute(a, b, c);
    SetUseBlas(true);
    ASSERT_TRUE(UsingBlas());
    Products blas = Compute(a, b, c);
    auto close = [](const Matrix& x, const Matrix& y) {
        ASSERT_TRUE(x.SameShape(y));
        for (size_t i = 0; i < x.Size(); ++i) ASSERT_NEAR(x.Data()[i], y.Data()[i], 1e-3f * (1.0f + std::fabs(x.Data()[i])));
    };
    close(reference.ab, blas.ab);
    close(reference.atb, blas.atb);
    close(reference.abt, blas.abt);
}

}  // namespace
}  // namespace nn
}  // namespace dpan
