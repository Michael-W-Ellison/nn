// File: src/nn/recurrent.cpp
#include "nn/recurrent.hpp"

#include "core/binary_io.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace dpan {
namespace nn {

namespace {

constexpr size_t kZ = 0, kR = 1, kN = 2;

float SigmoidOf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/// Rows of `h` whose reset flag is set become 0
Matrix MaskRows(Matrix h, const std::vector<bool>& reset) {
    for (size_t row = 0; row < reset.size() && row < h.Rows(); ++row) {
        if (!reset[row]) continue;
        for (size_t c = 0; c < h.Cols(); ++c) h(row, c) = 0.0f;
    }
    return h;
}

}  // namespace

GRU::GRU(size_t input_size, size_t hidden_size, uint32_t seed, float max_timescale)
    : input_size_(input_size), hidden_size_(hidden_size) {
    if (input_size == 0 || hidden_size == 0) throw std::invalid_argument("GRU: sizes must be positive");
    std::mt19937 rng(seed);
    float input_limit = std::sqrt(6.0f / static_cast<float>(input_size + hidden_size));
    float recurrent_limit = 1.0f / std::sqrt(static_cast<float>(hidden_size));
    std::uniform_real_distribution<float> input_dist(-input_limit, input_limit);
    std::uniform_real_distribution<float> recurrent_dist(-recurrent_limit, recurrent_limit);
    for (size_t g = 0; g < 3; ++g) {
        w_[g] = Matrix(input_size, hidden_size);
        u_[g] = Matrix(hidden_size, hidden_size);
        b_[g] = Matrix(1, hidden_size, g == kZ ? 1.0f : 0.0f);
        for (size_t i = 0; i < w_[g].Size(); ++i) w_[g].Data()[i] = input_dist(rng);
        for (size_t i = 0; i < u_[g].Size(); ++i) u_[g].Data()[i] = recurrent_dist(rng);
    }
    if (max_timescale > 2.0f) {
        std::uniform_real_distribution<float> timescale(1.0f, max_timescale - 1.0f);
        for (size_t i = 0; i < hidden_size; ++i) b_[kZ].Data()[i] = std::log(timescale(rng));
    }
    ZeroGradients();
}

Matrix GRU::Step(const Matrix& x, const Matrix& h) const {
    if (x.Cols() != input_size_ || h.Cols() != hidden_size_ || x.Rows() != h.Rows()) {
        throw std::invalid_argument("GRU::Step: expected rows x " + std::to_string(input_size_) + " and rows x " +
                                    std::to_string(hidden_size_) + ", got " + x.ShapeString() + " and " +
                                    h.ShapeString());
    }
    Matrix z = AddRowVector(MatMul(x, w_[kZ]) + MatMul(h, u_[kZ]), b_[kZ]);
    Matrix r = AddRowVector(MatMul(x, w_[kR]) + MatMul(h, u_[kR]), b_[kR]);
    for (size_t i = 0; i < z.Size(); ++i) {
        z.Data()[i] = SigmoidOf(z.Data()[i]);
        r.Data()[i] = SigmoidOf(r.Data()[i]);
    }
    Matrix n = AddRowVector(MatMul(x, w_[kN]) + MatMul(Hadamard(r, h), u_[kN]), b_[kN]);
    Matrix next(h.Rows(), hidden_size_);
    for (size_t i = 0; i < n.Size(); ++i) {
        float candidate = std::tanh(n.Data()[i]);
        next.Data()[i] = (1.0f - z.Data()[i]) * candidate + z.Data()[i] * h.Data()[i];
    }
    return next;
}

std::vector<Matrix> GRU::Forward(const std::vector<Matrix>& xs, const Matrix& h0,
                                 const std::vector<std::vector<bool>>& resets) {
    if (!resets.empty() && resets.size() != xs.size()) {
        throw std::invalid_argument("GRU::Forward: one reset list per step");
    }
    cache_.clear();
    std::vector<Matrix> states;
    Matrix h = h0;
    for (size_t t = 0; t < xs.size(); ++t) {
        StepCache step;
        step.reset = resets.empty() ? std::vector<bool>() : resets[t];
        step.x = xs[t];
        step.h = MaskRows(h, step.reset);
        step.z = AddRowVector(MatMul(step.x, w_[kZ]) + MatMul(step.h, u_[kZ]), b_[kZ]);
        step.r = AddRowVector(MatMul(step.x, w_[kR]) + MatMul(step.h, u_[kR]), b_[kR]);
        for (size_t i = 0; i < step.z.Size(); ++i) {
            step.z.Data()[i] = SigmoidOf(step.z.Data()[i]);
            step.r.Data()[i] = SigmoidOf(step.r.Data()[i]);
        }
        step.rh = Hadamard(step.r, step.h);
        step.n = AddRowVector(MatMul(step.x, w_[kN]) + MatMul(step.rh, u_[kN]), b_[kN]);
        Matrix next(step.h.Rows(), hidden_size_);
        for (size_t i = 0; i < step.n.Size(); ++i) {
            step.n.Data()[i] = std::tanh(step.n.Data()[i]);
            next.Data()[i] = (1.0f - step.z.Data()[i]) * step.n.Data()[i] + step.z.Data()[i] * step.h.Data()[i];
        }
        states.push_back(next);
        h = std::move(next);
        cache_.push_back(std::move(step));
    }
    return states;
}

std::vector<Matrix> GRU::Backward(const std::vector<Matrix>& grad_hs) {
    if (grad_hs.size() != cache_.size()) throw std::invalid_argument("GRU::Backward: one gradient per step");
    std::vector<Matrix> grad_xs(cache_.size());
    Matrix carry;  // gradient flowing into the state after step t from step t + 1
    for (size_t k = cache_.size(); k-- > 0;) {
        const StepCache& s = cache_[k];
        Matrix dh_next = grad_hs[k];
        if (!carry.Empty()) dh_next += carry;
        const size_t size = dh_next.Size();
        Matrix dz_pre(s.z.Rows(), hidden_size_), dn_pre(s.n.Rows(), hidden_size_);
        Matrix dh(s.h.Rows(), hidden_size_);
        for (size_t i = 0; i < size; ++i) {
            float g = dh_next.Data()[i], z = s.z.Data()[i], n = s.n.Data()[i];
            dz_pre.Data()[i] = g * (s.h.Data()[i] - n) * z * (1.0f - z);
            dn_pre.Data()[i] = g * (1.0f - z) * (1.0f - n * n);
            dh.Data()[i] = g * z;
        }
        Matrix d_rh = MatMulTransposeB(dn_pre, u_[kN]);
        Matrix dr_pre(s.r.Rows(), hidden_size_);
        for (size_t i = 0; i < size; ++i) {
            float r = s.r.Data()[i];
            dr_pre.Data()[i] = d_rh.Data()[i] * s.h.Data()[i] * r * (1.0f - r);
            dh.Data()[i] += d_rh.Data()[i] * r;
        }
        const Matrix* pre[3] = {&dz_pre, &dr_pre, &dn_pre};
        for (size_t g = 0; g < 3; ++g) {
            dw_[g] += MatMulTransposeA(s.x, *pre[g]);
            du_[g] += MatMulTransposeA(g == kN ? s.rh : s.h, *pre[g]);
            db_[g] += SumRows(*pre[g]);
        }
        grad_xs[k] = MatMulTransposeB(dz_pre, w_[kZ]) + MatMulTransposeB(dr_pre, w_[kR]) +
                     MatMulTransposeB(dn_pre, w_[kN]);
        dh += MatMulTransposeB(dz_pre, u_[kZ]) + MatMulTransposeB(dr_pre, u_[kR]);
        carry = MaskRows(std::move(dh), s.reset);  // a reset state does not depend on the one before
    }
    return grad_xs;
}

std::vector<Parameter> GRU::Parameters() {
    std::vector<Parameter> parameters;
    for (size_t g = 0; g < 3; ++g) parameters.push_back({&w_[g], &dw_[g]});
    for (size_t g = 0; g < 3; ++g) parameters.push_back({&u_[g], &du_[g]});
    for (size_t g = 0; g < 3; ++g) parameters.push_back({&b_[g], &db_[g]});
    return parameters;
}

void GRU::ZeroGradients() {
    for (size_t g = 0; g < 3; ++g) {
        dw_[g] = Matrix(w_[g].Rows(), w_[g].Cols());
        du_[g] = Matrix(u_[g].Rows(), u_[g].Cols());
        db_[g] = Matrix(1, hidden_size_);
    }
}

size_t GRU::ParameterCount() const { return 3 * (input_size_ * hidden_size_ + hidden_size_ * hidden_size_ + hidden_size_); }

void GRU::SetInputWeights(std::vector<Matrix> weights) {
    if (weights.size() != 3) throw std::invalid_argument("GRU::SetInputWeights: three matrices");
    size_t rows = weights[0].Rows();
    for (const auto& w : weights) {
        if (w.Rows() != rows || w.Cols() != hidden_size_) throw std::invalid_argument("GRU::SetInputWeights: bad shape");
    }
    for (size_t g = 0; g < 3; ++g) w_[g] = std::move(weights[g]);
    input_size_ = rows;
    ZeroGradients();
    cache_.clear();
}

void GRU::Serialize(std::ostream& out) const {
    io::WriteSize(out, input_size_);
    io::WriteSize(out, hidden_size_);
    for (size_t g = 0; g < 3; ++g) {
        w_[g].Serialize(out);
        u_[g].Serialize(out);
        b_[g].Serialize(out);
    }
}

GRU GRU::Deserialize(std::istream& in) {
    GRU gru;
    gru.input_size_ = io::ReadSize(in, 1u << 20);
    gru.hidden_size_ = io::ReadSize(in, 1u << 20);
    for (size_t g = 0; g < 3; ++g) {
        gru.w_[g] = Matrix::Deserialize(in);
        gru.u_[g] = Matrix::Deserialize(in);
        gru.b_[g] = Matrix::Deserialize(in);
        if (gru.w_[g].Rows() != gru.input_size_ || gru.w_[g].Cols() != gru.hidden_size_ ||
            gru.u_[g].Rows() != gru.hidden_size_ || gru.u_[g].Cols() != gru.hidden_size_ ||
            gru.b_[g].Rows() != 1 || gru.b_[g].Cols() != gru.hidden_size_) {
            throw std::runtime_error("GRU: malformed data");
        }
    }
    gru.ZeroGradients();
    return gru;
}

}  // namespace nn
}  // namespace dpan
