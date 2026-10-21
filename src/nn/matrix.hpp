// File: src/nn/matrix.hpp
//
// Dense row-major float matrix used by the neural network module.
// A batch of N samples with D features is an N x D matrix (one row per sample).
#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace dpan {
namespace nn {

class Matrix {
public:
    Matrix() = default;
    // explicit: a brace list such as {1.0f, 2.0f} must not silently become
    // a rows x cols matrix
    explicit Matrix(size_t rows, size_t cols, float fill = 0.0f);
    explicit Matrix(size_t rows, size_t cols, std::vector<float> data);

    /// Build from rows of equal length
    static Matrix FromRows(const std::vector<std::vector<float>>& rows);

    /// Single-row matrix (1 x values.size())
    static Matrix RowVector(const std::vector<float>& values);

    size_t Rows() const { return rows_; }
    size_t Cols() const { return cols_; }
    size_t Size() const { return data_.size(); }
    bool Empty() const { return data_.empty(); }

    float& operator()(size_t row, size_t col) { return data_[row * cols_ + col]; }
    float operator()(size_t row, size_t col) const { return data_[row * cols_ + col]; }

    float* Data() { return data_.data(); }
    const float* Data() const { return data_.data(); }
    const std::vector<float>& Values() const { return data_; }

    /// Copy of one row
    std::vector<float> Row(size_t row) const;

    /// Rows selected by index (used for mini-batches)
    Matrix SelectRows(const std::vector<size_t>& indices) const;

    /// Columns [begin, end)
    Matrix SliceCols(size_t begin, size_t end) const;

    Matrix Transpose() const;

    void Fill(float value);

    /// Element-wise in-place operations (shapes must match)
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(float scalar);

    bool SameShape(const Matrix& other) const {
        return rows_ == other.rows_ && cols_ == other.cols_;
    }

    void Serialize(std::ostream& out) const;
    static Matrix Deserialize(std::istream& in);

    std::string ShapeString() const;

private:
    size_t rows_{0};
    size_t cols_{0};
    std::vector<float> data_;
};

// Arithmetic (shapes are checked; mismatches throw std::invalid_argument)
Matrix operator+(const Matrix& a, const Matrix& b);
Matrix operator-(const Matrix& a, const Matrix& b);
Matrix operator*(const Matrix& a, float scalar);

/// Element-wise product
Matrix Hadamard(const Matrix& a, const Matrix& b);

/// a (n x k) * b (k x m)
Matrix MatMul(const Matrix& a, const Matrix& b);

/// a^T (k x n)^T * b (k x m) -> n x m, without forming the transpose
Matrix MatMulTransposeA(const Matrix& a, const Matrix& b);

/// a (n x k) * b^T (m x k)^T -> n x m, without forming the transpose
Matrix MatMulTransposeB(const Matrix& a, const Matrix& b);

/// Add a 1 x cols row vector to every row
Matrix AddRowVector(const Matrix& a, const Matrix& row);

/// Column sums as a 1 x cols matrix
Matrix SumRows(const Matrix& a);

/// Index of the largest value in each row
std::vector<size_t> ArgMaxRows(const Matrix& a);

/// One-hot rows (labels.size() x num_classes)
Matrix OneHot(const std::vector<size_t>& labels, size_t num_classes);

/// Horizontal concatenation (row counts must match)
Matrix ConcatCols(const Matrix& a, const Matrix& b);

}  // namespace nn
}  // namespace dpan
