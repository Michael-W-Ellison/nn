// File: src/nn/recurrent.hpp
//
// Gated recurrent unit (Cho et al. 2014), one layer:
//
//   z = sigmoid(x Wz + h Uz + bz)          update gate (1 keeps the state)
//   r = sigmoid(x Wr + h Ur + br)          reset gate
//   n = tanh(x Wn + (r * h) Un + bn)       candidate state
//   h' = (1 - z) * n + z * h
//
// for a batch of rows. Step() advances the state for inference; Forward()
// runs a whole sequence and keeps what Backward() needs to propagate
// gradients through time. A row whose reset flag is set at a step starts
// from state 0 there (an episode boundary).
#pragma once

#include "nn/layers.hpp"
#include "nn/matrix.hpp"

#include <cstdint>
#include <iosfwd>
#include <vector>

namespace dpan {
namespace nn {

class GRU {
public:
    /// Input weights Xavier-uniform, recurrent weights uniform(+-1/sqrt(hidden)),
    /// reset and candidate biases 0. Update-gate bias 1 (the state is kept
    /// by default), or with max_timescale > 1 "chrono" initialization (Tallec
    /// & Ollivier 2018): log(u) with u uniform in [1, max_timescale - 1], so
    /// the units start out keeping their state for 2 to max_timescale steps
    /// and gradients reach back that far.
    GRU(size_t input_size, size_t hidden_size, uint32_t seed = 42, float max_timescale = 0.0f);

    size_t InputSize() const { return input_size_; }
    size_t HiddenSize() const { return hidden_size_; }

    /// Next state of each row: x is rows x input, h rows x hidden
    Matrix Step(const Matrix& x, const Matrix& h) const;

    /// Run a sequence: xs[t] is rows x input; resets[t][row] sets the state
    /// before step t to 0 for that row (empty = no resets). Returns the state
    /// after each step and keeps the activations for Backward().
    std::vector<Matrix> Forward(const std::vector<Matrix>& xs, const Matrix& h0,
                                const std::vector<std::vector<bool>>& resets);

    /// grad_hs[t]: gradient of the loss with respect to the state after step
    /// t (from the layers that read it). Adds to the parameter gradients and
    /// returns the gradient with respect to each input xs[t].
    std::vector<Matrix> Backward(const std::vector<Matrix>& grad_hs);

    /// Wz, Wr, Wn, Uz, Ur, Un, bz, br, bn
    std::vector<Parameter> Parameters();
    void ZeroGradients();
    size_t ParameterCount() const;

    /// Input weights of the three gates (input x hidden each), e.g. to widen
    /// the layer below
    Matrix& InputWeights(size_t gate) { return w_[gate]; }

    /// Replace the input weights (the input size may change); gradients are reset
    void SetInputWeights(std::vector<Matrix> weights);

    void Serialize(std::ostream& out) const;
    static GRU Deserialize(std::istream& in);

private:
    GRU() = default;

    struct StepCache {
        Matrix x, h, z, r, n, rh;  // h: state before the step (after any reset)
        std::vector<bool> reset;
    };

    size_t input_size_{0};
    size_t hidden_size_{0};
    Matrix w_[3], u_[3], b_[3];
    Matrix dw_[3], du_[3], db_[3];
    std::vector<StepCache> cache_;
};

}  // namespace nn
}  // namespace dpan
