// File: src/nn/matrix.cpp
#include "nn/matrix.hpp"
#include "core/binary_io.hpp"
#include "nn/compute.hpp"

#ifdef DPAN_HAS_BLAS
#include <cblas.h>
#endif

#include <algorithm>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace dpan {
namespace nn {

namespace {

void RequireSameShape(const Matrix& a, const Matrix& b, const char* op) {
    if (!a.SameShape(b)) {
        throw std::invalid_argument(std::string(op) + ": shape mismatch " +
                                    a.ShapeString() + " vs " + b.ShapeString());
    }
}

}  // namespace

Matrix::Matrix(size_t rows, size_t cols, float fill)
    : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

Matrix::Matrix(size_t rows, size_t cols, std::vector<float> data)
    : rows_(rows), cols_(cols), data_(std::move(data)) {
    if (data_.size() != rows * cols) {
        throw std::invalid_argument("Matrix: data size does not match shape");
    }
}

Matrix Matrix::FromRows(const std::vector<std::vector<float>>& rows) {
    if (rows.empty()) {
        return Matrix();
    }
    size_t cols = rows.front().size();
    Matrix m(rows.size(), cols);
    for (size_t r = 0; r < rows.size(); ++r) {
        if (rows[r].size() != cols) {
            throw std::invalid_argument("Matrix::FromRows: rows have different lengths");
        }
        std::copy(rows[r].begin(), rows[r].end(), m.data_.begin() + r * cols);
    }
    return m;
}

Matrix Matrix::RowVector(const std::vector<float>& values) {
    return Matrix(1, values.size(), values);
}

std::vector<float> Matrix::Row(size_t row) const {
    if (row >= rows_) {
        throw std::out_of_range("Matrix::Row: index out of range");
    }
    return std::vector<float>(data_.begin() + row * cols_, data_.begin() + (row + 1) * cols_);
}

Matrix Matrix::SelectRows(const std::vector<size_t>& indices) const {
    Matrix out(indices.size(), cols_);
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= rows_) {
            throw std::out_of_range("Matrix::SelectRows: index out of range");
        }
        std::copy(data_.begin() + indices[i] * cols_, data_.begin() + (indices[i] + 1) * cols_,
                  out.data_.begin() + i * cols_);
    }
    return out;
}

Matrix Matrix::SliceCols(size_t begin, size_t end) const {
    if (begin > end || end > cols_) {
        throw std::out_of_range("Matrix::SliceCols: invalid range");
    }
    Matrix out(rows_, end - begin);
    for (size_t r = 0; r < rows_; ++r) {
        for (size_t c = begin; c < end; ++c) {
            out(r, c - begin) = (*this)(r, c);
        }
    }
    return out;
}

Matrix Matrix::Transpose() const {
    Matrix out(cols_, rows_);
    for (size_t r = 0; r < rows_; ++r) {
        for (size_t c = 0; c < cols_; ++c) {
            out(c, r) = (*this)(r, c);
        }
    }
    return out;
}

void Matrix::Fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

Matrix& Matrix::operator+=(const Matrix& other) {
    RequireSameShape(*this, other, "operator+=");
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    RequireSameShape(*this, other, "operator-=");
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

Matrix& Matrix::operator*=(float scalar) {
    for (auto& v : data_) {
        v *= scalar;
    }
    return *this;
}

void Matrix::Serialize(std::ostream& out) const {
    io::Write<uint64_t>(out, rows_);
    io::Write<uint64_t>(out, cols_);
    io::WriteArray<float>(out, data_.data(), data_.size());
}

Matrix Matrix::Deserialize(std::istream& in) {
    uint64_t rows = io::Read<uint64_t>(in);
    uint64_t cols = io::Read<uint64_t>(in);
    if (cols != 0 && rows > (uint64_t{1} << 32) / cols) {
        throw std::runtime_error("Matrix::Deserialize: invalid header");
    }
    std::vector<float> data(static_cast<size_t>(rows * cols));
    io::ReadArray<float>(in, data.data(), data.size());
    return Matrix(static_cast<size_t>(rows), static_cast<size_t>(cols), std::move(data));
}

std::string Matrix::ShapeString() const {
    return "(" + std::to_string(rows_) + "x" + std::to_string(cols_) + ")";
}

Matrix operator+(const Matrix& a, const Matrix& b) {
    Matrix out = a;
    out += b;
    return out;
}

Matrix operator-(const Matrix& a, const Matrix& b) {
    Matrix out = a;
    out -= b;
    return out;
}

Matrix operator*(const Matrix& a, float scalar) {
    Matrix out = a;
    out *= scalar;
    return out;
}

Matrix Hadamard(const Matrix& a, const Matrix& b) {
    RequireSameShape(a, b, "Hadamard");
    Matrix out(a.Rows(), a.Cols());
    const float* pa = a.Data();
    const float* pb = b.Data();
    float* po = out.Data();
    for (size_t i = 0; i < a.Size(); ++i) {
        po[i] = pa[i] * pb[i];
    }
    return out;
}

Matrix MatMul(const Matrix& a, const Matrix& b) {
    if (a.Cols() != b.Rows()) {
        throw std::invalid_argument("MatMul: shape mismatch " + a.ShapeString() + " * " +
                                    b.ShapeString());
    }
    Matrix out(a.Rows(), b.Cols());
    const size_t n = a.Rows(), k = a.Cols(), m = b.Cols();
    if (out.Empty() || k == 0) return out;
#ifdef DPAN_HAS_BLAS
    if (UsingBlas()) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n), static_cast<int>(m),
                    static_cast<int>(k), 1.0f, a.Data(), static_cast<int>(k), b.Data(), static_cast<int>(m), 0.0f,
                    out.Data(), static_cast<int>(m));
        return out;
    }
#endif
    // i-k-j loop order keeps the inner loop contiguous in b and out; rows of
    // the output are independent, so they may be split across threads
    ParallelFor(n, n * k * m, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            float* out_row = out.Data() + i * m;
            for (size_t p = 0; p < k; ++p) {
                float av = a(i, p);
                if (av == 0.0f) {
                    continue;
                }
                const float* b_row = b.Data() + p * m;
                for (size_t j = 0; j < m; ++j) {
                    out_row[j] += av * b_row[j];
                }
            }
        }
    });
    return out;
}

Matrix MatMulTransposeA(const Matrix& a, const Matrix& b) {
    if (a.Rows() != b.Rows()) {
        throw std::invalid_argument("MatMulTransposeA: shape mismatch " + a.ShapeString() +
                                    "^T * " + b.ShapeString());
    }
    Matrix out(a.Cols(), b.Cols());
    const size_t k = a.Rows(), n = a.Cols(), m = b.Cols();
    if (out.Empty() || k == 0) return out;
#ifdef DPAN_HAS_BLAS
    if (UsingBlas()) {
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(n), static_cast<int>(m),
                    static_cast<int>(k), 1.0f, a.Data(), static_cast<int>(n), b.Data(), static_cast<int>(m), 0.0f,
                    out.Data(), static_cast<int>(m));
        return out;
    }
#endif
    // Each output row i sums over p in ascending order, whichever thread runs it
    ParallelFor(n, n * k * m, [&](size_t begin, size_t end) {
        for (size_t p = 0; p < k; ++p) {
            const float* a_row = a.Data() + p * n;
            const float* b_row = b.Data() + p * m;
            for (size_t i = begin; i < end; ++i) {
                float av = a_row[i];
                if (av == 0.0f) {
                    continue;
                }
                float* out_row = out.Data() + i * m;
                for (size_t j = 0; j < m; ++j) {
                    out_row[j] += av * b_row[j];
                }
            }
        }
    });
    return out;
}

Matrix MatMulTransposeB(const Matrix& a, const Matrix& b) {
    if (a.Cols() != b.Cols()) {
        throw std::invalid_argument("MatMulTransposeB: shape mismatch " + a.ShapeString() +
                                    " * " + b.ShapeString() + "^T");
    }
    Matrix out(a.Rows(), b.Rows());
    const size_t n = a.Rows(), k = a.Cols(), m = b.Rows();
    if (out.Empty() || k == 0) return out;
#ifdef DPAN_HAS_BLAS
    if (UsingBlas()) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(n), static_cast<int>(m),
                    static_cast<int>(k), 1.0f, a.Data(), static_cast<int>(k), b.Data(), static_cast<int>(k), 0.0f,
                    out.Data(), static_cast<int>(m));
        return out;
    }
#endif
    ParallelFor(n, n * k * m, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            const float* a_row = a.Data() + i * k;
            for (size_t j = 0; j < m; ++j) {
                const float* b_row = b.Data() + j * k;
                float sum = 0.0f;
                for (size_t p = 0; p < k; ++p) {
                    sum += a_row[p] * b_row[p];
                }
                out(i, j) = sum;
            }
        }
    });
    return out;
}

Matrix AddRowVector(const Matrix& a, const Matrix& row) {
    if (row.Rows() != 1 || row.Cols() != a.Cols()) {
        throw std::invalid_argument("AddRowVector: shape mismatch " + a.ShapeString() + " + " +
                                    row.ShapeString());
    }
    Matrix out = a;
    for (size_t r = 0; r < a.Rows(); ++r) {
        float* out_row = out.Data() + r * a.Cols();
        for (size_t c = 0; c < a.Cols(); ++c) {
            out_row[c] += row(0, c);
        }
    }
    return out;
}

Matrix SumRows(const Matrix& a) {
    Matrix out(1, a.Cols());
    for (size_t r = 0; r < a.Rows(); ++r) {
        const float* row = a.Data() + r * a.Cols();
        for (size_t c = 0; c < a.Cols(); ++c) {
            out(0, c) += row[c];
        }
    }
    return out;
}

std::vector<size_t> ArgMaxRows(const Matrix& a) {
    std::vector<size_t> result(a.Rows(), 0);
    for (size_t r = 0; r < a.Rows(); ++r) {
        const float* row = a.Data() + r * a.Cols();
        result[r] = static_cast<size_t>(std::max_element(row, row + a.Cols()) - row);
    }
    return result;
}

Matrix OneHot(const std::vector<size_t>& labels, size_t num_classes) {
    Matrix out(labels.size(), num_classes);
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] >= num_classes) {
            throw std::out_of_range("OneHot: label out of range");
        }
        out(i, labels[i]) = 1.0f;
    }
    return out;
}

Matrix ConcatCols(const Matrix& a, const Matrix& b) {
    if (a.Rows() != b.Rows()) {
        throw std::invalid_argument("ConcatCols: row count mismatch " + a.ShapeString() + ", " +
                                    b.ShapeString());
    }
    Matrix out(a.Rows(), a.Cols() + b.Cols());
    for (size_t r = 0; r < a.Rows(); ++r) {
        for (size_t c = 0; c < a.Cols(); ++c) {
            out(r, c) = a(r, c);
        }
        for (size_t c = 0; c < b.Cols(); ++c) {
            out(r, a.Cols() + c) = b(r, c);
        }
    }
    return out;
}

}  // namespace nn
}  // namespace dpan
