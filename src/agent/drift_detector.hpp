// File: src/agent/drift_detector.hpp
//
// Page-Hinkley test for an increase in a stream of losses (docs/AGENT_PLAN.md,
// section 4, rule 5). Each loss is standardized with the mean and standard
// deviation of the losses since the last detection and clipped to
// +-clip, so rare large losses (an occasional error of an accurate model)
// cannot trigger it on their own. The test accumulates
//   m += z - delta,  M = min(m),
// and reports a drift when m - M > threshold, then starts over.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace dpan {
namespace agent {

class PageHinkley {
public:
    struct Config {
        double delta{0.5};      ///< Increase tolerated, in standard deviations
        double threshold{20.0}; ///< Accumulated increase that signals a drift
        double clip{3.0};       ///< Bound on each standardized loss
        uint64_t warmup{50};    ///< Losses observed before testing starts
    };

    PageHinkley() = default;
    explicit PageHinkley(const Config& config) : config_(config) {}

    /// Add a loss; true if a drift is detected (the test then restarts)
    bool Add(double loss) {
        if (!std::isfinite(loss)) return false;
        ++count_;
        double delta = loss - mean_;
        mean_ += delta / static_cast<double>(count_);
        m2_ += delta * (loss - mean_);
        if (count_ <= config_.warmup) return false;

        double sd = std::sqrt(m2_ / static_cast<double>(count_));
        double z = sd > 1e-12 ? (loss - mean_) / sd : 0.0;
        z = std::max(-config_.clip, std::min(config_.clip, z));
        sum_ += z - config_.delta;
        minimum_ = std::min(minimum_, sum_);
        if (sum_ - minimum_ > config_.threshold) {
            last_mean_ = mean_;
            Reset();
            return true;
        }
        return false;
    }

    void Reset() {
        count_ = 0;
        mean_ = m2_ = sum_ = minimum_ = 0.0;
    }

    /// Mean loss since the last detection
    double Mean() const { return mean_; }

    /// Mean loss up to the last detection (the level before the drift)
    double MeanBeforeDrift() const { return last_mean_; }

    uint64_t Count() const { return count_; }

private:
    Config config_;
    uint64_t count_{0};
    double mean_{0.0};
    double m2_{0.0};
    double sum_{0.0};
    double minimum_{0.0};
    double last_mean_{0.0};
};

}  // namespace agent
}  // namespace dpan
