// File: src/memory/decay_functions.hpp
#pragma once

#include "core/types.hpp"
#include <chrono>
#include <cmath>
#include <memory>

namespace dpan {

/**
 * @brief Abstract interface for decay functions
 *
 * Decay functions model how pattern and association strengths diminish over time
 * without reinforcement. Different decay functions capture different forgetting
 * dynamics inspired by cognitive science research.
 */
class IDecayFunction {
public:
    virtual ~IDecayFunction() = default;

    /**
     * @brief Apply decay to strength based on elapsed time
     *
     * @param initial_strength The original strength value (typically in [0.0, 1.0])
     * @param elapsed_time Time since last reinforcement
     * @return Decayed strength value
     */
    virtual float ApplyDecay(float initial_strength, Timestamp::Duration elapsed_time) const = 0;

    /**
     * @brief Get the amount of strength lost due to decay
     *
     * @param initial_strength The original strength value
     * @param elapsed_time Time since last reinforcement
     * @return Amount of strength lost (always >= 0)
     */
    virtual float GetDecayAmount(float initial_strength, Timestamp::Duration elapsed_time) const {
        float decayed = ApplyDecay(initial_strength, elapsed_time);
        return initial_strength - decayed;
    }

    /**
     * @brief Get a descriptive name for this decay function
     *
     * @return Name of the decay function
     */
    virtual const char* GetName() const = 0;

    /**
     * @brief Clone this decay function
     *
     * @return Unique pointer to a copy of this decay function
     */
    virtual std::unique_ptr<IDecayFunction> Clone() const = 0;
};

/**
 * @brief Exponential decay function based on Ebbinghaus forgetting curve
 *
 * Models forgetting as an exponential process:
 *   s(t) = s_0 × e^(-λt)
 *
 * Where:
 *   s(t) = strength at time t
 *   s_0 = initial strength
 *   λ = decay constant (decay_constant_)
 *   t = time since last reinforcement (in hours)
 *
 * This is the classic forgetting curve observed in memory experiments.
 * Higher decay_constant means faster forgetting.
 */
class ExponentialDecay : public IDecayFunction {
public:
    /**
     * @brief Construct exponential decay with specified decay constant
     *
     * @param decay_constant Decay rate (λ). Default 0.01 gives moderate forgetting.
     *                       Higher values = faster forgetting.
     *                       Typical range: [0.001, 0.1]
     */
    explicit ExponentialDecay(float decay_constant = 0.01f)
        : decay_constant_(decay_constant) {
        if (decay_constant_ < 0.0f) {
            decay_constant_ = 0.0f;
        }
    }

    float ApplyDecay(float initial_strength, Timestamp::Duration elapsed_time) const override {
        if (initial_strength <= 0.0f || decay_constant_ == 0.0f) {
            return initial_strength;
        }

        // Convert elapsed time to hours for meaningful decay rates
        auto hours = std::chrono::duration_cast<std::chrono::duration<float, std::ratio<3600>>>(elapsed_time).count();

        // Apply exponential decay: s(t) = s_0 * e^(-λt)
        float decayed = initial_strength * std::exp(-decay_constant_ * hours);

        // Ensure result is in valid range
        return std::max(0.0f, std::min(decayed, initial_strength));
    }

    const char* GetName() const override {
        return "ExponentialDecay";
    }

    std::unique_ptr<IDecayFunction> Clone() const override {
        return std::make_unique<ExponentialDecay>(decay_constant_);
    }

    /**
     * @brief Get the decay constant (λ)
     */
    float GetDecayConstant() const { return decay_constant_; }

    /**
     * @brief Set the decay constant (λ)
     */
    void SetDecayConstant(float decay_constant) {
        decay_constant_ = std::max(0.0f, decay_constant);
    }

    /**
     * @brief Calculate half-life (time for strength to decay to 50%)
     *
     * Half-life t_1/2 = ln(2) / λ
     *
     * @return Half-life in hours
     */
    float GetHalfLife() const {
        if (decay_constant_ == 0.0f) {
            return std::numeric_limits<float>::infinity();
        }
        return std::log(2.0f) / decay_constant_;
    }

private:
    float decay_constant_;  // λ parameter
};

/**
 * @brief Power-law decay function based on Anderson's ACT-R model
 *
 * Models forgetting as a power-law process:
 *   s(t) = s_0 / (1 + t/τ)^β
 *
 * Where:
 *   s(t) = strength at time t
 *   s_0 = initial strength
 *   t = time since last reinforcement (in hours)
 *   τ = time constant (time_constant_)
 *   β = decay exponent (exponent_)
 *
 * Power-law decay is often more realistic than exponential for long-term memory,
 * as it produces slower decay over long periods. Typical exponent β ≈ 0.5.
 */
class PowerLawDecay : public IDecayFunction {
public:
    /**
     * @brief Construct power-law decay with specified parameters
     *
     * @param time_constant Time scale parameter (τ). Default 1.0.
     *                      Higher values = slower initial decay.
     *                      Typical range: [0.1, 10.0]
     * @param exponent Decay exponent (β). Default 0.5.
     *                 Higher values = faster decay.
     *                 Typical range: [0.3, 1.0]
     */
    explicit PowerLawDecay(float time_constant = 1.0f, float exponent = 0.5f)
        : time_constant_(time_constant), exponent_(exponent) {
        if (time_constant_ <= 0.0f) {
            time_constant_ = 1.0f;
        }
        if (exponent_ < 0.0f) {
            exponent_ = 0.5f;
        }
    }

    float ApplyDecay(float initial_strength, Timestamp::Duration elapsed_time) const override {
        if (initial_strength <= 0.0f) {
            return initial_strength;
        }

        // Convert elapsed time to hours
        auto hours = std::chrono::duration_cast<std::chrono::duration<float, std::ratio<3600>>>(elapsed_time).count();

        // Apply power-law decay: s(t) = s_0 / (1 + t/τ)^β
        float decay_factor = std::pow(1.0f + hours / time_constant_, exponent_);
        float decayed = initial_strength / decay_factor;

        // Ensure result is in valid range
        return std::max(0.0f, std::min(decayed, initial_strength));
    }

    const char* GetName() const override {
        return "PowerLawDecay";
    }

    std::unique_ptr<IDecayFunction> Clone() const override {
        return std::make_unique<PowerLawDecay>(time_constant_, exponent_);
    }

    /**
     * @brief Get the time constant (τ)
     */
    float GetTimeConstant() const { return time_constant_; }

    /**
     * @brief Get the decay exponent (β)
     */
    float GetExponent() const { return exponent_; }

    /**
     * @brief Set the time constant (τ)
     */
    void SetTimeConstant(float time_constant) {
        time_constant_ = std::max(0.001f, time_constant);
    }

    /**
     * @brief Set the decay exponent (β)
     */
    void SetExponent(float exponent) {
        exponent_ = std::max(0.0f, exponent);
    }

private:
    float time_constant_;  // τ parameter
    float exponent_;       // β parameter
};

/**
 * @brief Step decay function with discrete decay intervals
 *
 * Models forgetting as a discrete, step-wise process:
 *   s(t) = s_0 × decay_factor^(floor(t / step_size))
 *
 * Where:
 *   s(t) = strength at time t
 *   s_0 = initial strength
 *   decay_factor = multiplicative decay per step (decay_factor_)
 *   step_size = time interval between decay steps (step_size_)
 *   floor(t / step_size) = number of complete steps elapsed
 *
 * This can model periodic memory consolidation or scheduled forgetting.
 * For example, step_size = 24 hours with decay_factor = 0.9 means
 * 10% strength loss per day.
 */
class StepDecay : public IDecayFunction {
public:
    /**
     * @brief Construct step decay with specified parameters
     *
     * @param decay_factor Multiplicative decay per step. Default 0.9 (10% loss per step).
     *                     Must be in range [0.0, 1.0].
     *                     decay_factor = 1.0 means no decay.
     *                     decay_factor = 0.5 means 50% loss per step.
     * @param step_size Time interval between decay steps. Default 24 hours.
     *                  Typical values: 1 hour, 24 hours, 1 week.
     */
    explicit StepDecay(float decay_factor = 0.9f,
                       Timestamp::Duration step_size = std::chrono::hours(24))
        : decay_factor_(decay_factor), step_size_(step_size) {
        // Clamp decay_factor to valid range
        if (decay_factor_ < 0.0f) decay_factor_ = 0.0f;
        if (decay_factor_ > 1.0f) decay_factor_ = 1.0f;

        // Ensure step_size is positive
        if (step_size_.count() <= 0) {
            step_size_ = std::chrono::hours(24);
        }
    }

    float ApplyDecay(float initial_strength, Timestamp::Duration elapsed_time) const override {
        if (initial_strength <= 0.0f || decay_factor_ == 1.0f) {
            return initial_strength;
        }

        // Calculate number of complete decay steps
        int64_t num_steps = elapsed_time.count() / step_size_.count();

        if (num_steps <= 0) {
            // No complete steps yet, no decay
            return initial_strength;
        }

        // Apply step decay: s(t) = s_0 * decay_factor^num_steps
        float decayed = initial_strength * std::pow(decay_factor_, static_cast<float>(num_steps));

        // Ensure result is in valid range
        return std::max(0.0f, std::min(decayed, initial_strength));
    }

    const char* GetName() const override {
        return "StepDecay";
    }

    std::unique_ptr<IDecayFunction> Clone() const override {
        return std::make_unique<StepDecay>(decay_factor_, step_size_);
    }

    /**
     * @brief Get the decay factor
     */
    float GetDecayFactor() const { return decay_factor_; }

    /**
     * @brief Get the step size
     */
    Timestamp::Duration GetStepSize() const { return step_size_; }

    /**
     * @brief Set the decay factor
     */
    void SetDecayFactor(float decay_factor) {
        decay_factor_ = std::max(0.0f, std::min(1.0f, decay_factor));
    }

    /**
     * @brief Set the step size
     */
    void SetStepSize(Timestamp::Duration step_size) {
        if (step_size.count() > 0) {
            step_size_ = step_size;
        }
    }

    /**
     * @brief Calculate half-life (number of steps for strength to decay to 50%)
     *
     * Half-life = log(0.5) / log(decay_factor)
     *
     * @return Number of steps to reach 50% strength
     */
    float GetHalfLifeSteps() const {
        if (decay_factor_ == 0.0f || decay_factor_ == 1.0f) {
            return std::numeric_limits<float>::infinity();
        }
        return std::log(0.5f) / std::log(decay_factor_);
    }

private:
    float decay_factor_;           // Multiplicative decay per step
    Timestamp::Duration step_size_; // Time interval between steps
};

/**
 * @brief Factory function to create decay functions by name
 *
 * @param name Name of decay function ("exponential", "powerlaw", or "step")
 * @return Unique pointer to decay function, or nullptr if name not recognized
 */
inline std::unique_ptr<IDecayFunction> CreateDecayFunction(const std::string& name) {
    if (name == "exponential") {
        return std::make_unique<ExponentialDecay>();
    } else if (name == "powerlaw") {
        return std::make_unique<PowerLawDecay>();
    } else if (name == "step") {
        return std::make_unique<StepDecay>();
    }
    return nullptr;
}

} // namespace dpan
