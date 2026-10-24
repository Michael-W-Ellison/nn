// File: src/agent/actor_critic.cpp
#include "agent/actor_critic.hpp"

#include "core/binary_io.hpp"
#include "nn/layers.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dpan {
namespace agent {

namespace {

nn::Sequential Mlp(size_t inputs, size_t hidden, size_t outputs, bool tanh_output, uint32_t seed) {
    nn::Sequential model;
    model.Emplace<nn::Dense>(inputs, hidden, nn::Initialization::HE, seed)
        .Emplace<nn::ReLU>()
        .Emplace<nn::Dense>(hidden, hidden, nn::Initialization::HE, seed + 1)
        .Emplace<nn::ReLU>()
        .Emplace<nn::Dense>(hidden, outputs, nn::Initialization::XAVIER, seed + 2);
    if (tanh_output) model.Emplace<nn::Tanh>();
    return model;
}

bool Finite(nn::Sequential& model) {
    for (const auto& parameter : model.Parameters()) {
        const float* v = parameter.value->Data();
        for (size_t i = 0; i < parameter.value->Size(); ++i) {
            if (!std::isfinite(v[i])) return false;
        }
    }
    return true;
}

}  // namespace

ActorCritic::ActorCritic(size_t state_size, size_t action_size, const ActorCriticConfig& config)
    : state_size_(state_size),
      action_size_(action_size),
      config_(config),
      actor_optimizer_(config.actor_learning_rate),
      critic_optimizer_(config.critic_learning_rate) {
    if (state_size == 0 || action_size == 0 || config.hidden == 0) {
        throw std::invalid_argument("ActorCritic: state, action and hidden sizes must be > 0");
    }
    if (config.policy_delay == 0) throw std::invalid_argument("ActorCritic: policy_delay must be > 0");
    if (!(config.saturation_penalty >= 0.0f) || !(config.saturation_threshold >= 0.0f)) {
        throw std::invalid_argument("ActorCritic: saturation settings must be >= 0");
    }
    actor_ = Mlp(state_size, config.hidden, action_size, true, config.seed);
    critic1_ = Mlp(state_size + action_size, config.hidden, 1, false, config.seed + 10);
    critic2_ = Mlp(state_size + action_size, config.hidden, 1, false, config.seed + 20);
    actor_target_ = actor_;
    critic1_target_ = critic1_;
    critic2_target_ = critic2_;
}

nn::Matrix ActorCritic::Join(const nn::Matrix& states, const nn::Matrix& actions) const {
    nn::Matrix joined(states.Rows(), state_size_ + action_size_);
    for (size_t r = 0; r < states.Rows(); ++r) {
        std::copy(states.Data() + r * state_size_, states.Data() + (r + 1) * state_size_,
                  joined.Data() + r * joined.Cols());
        std::copy(actions.Data() + r * action_size_, actions.Data() + (r + 1) * action_size_,
                  joined.Data() + r * joined.Cols() + state_size_);
    }
    return joined;
}

nn::Matrix ActorCritic::Act(const nn::Matrix& states) const { return actor_.Predict(states); }

std::vector<float> ActorCritic::Value(const nn::Matrix& states, const nn::Matrix& actions) const {
    nn::Matrix q = critic1_.Predict(Join(states, actions));
    return std::vector<float>(q.Data(), q.Data() + q.Rows());
}

std::vector<nn::Parameter> ActorCritic::CriticParameters() {
    auto parameters = critic1_.Parameters();
    auto second = critic2_.Parameters();
    parameters.insert(parameters.end(), second.begin(), second.end());
    return parameters;
}

void ActorCritic::SoftUpdate(nn::Sequential& target, nn::Sequential& live, float tau) {
    auto t = target.Parameters();
    auto l = live.Parameters();
    for (size_t p = 0; p < t.size(); ++p) {
        float* tv = t[p].value->Data();
        const float* lv = l[p].value->Data();
        for (size_t i = 0; i < t[p].value->Size(); ++i) tv[i] += tau * (lv[i] - tv[i]);
    }
}

ActorCritic::Losses ActorCritic::Train(const Batch& batch, std::mt19937& rng) {
    const size_t n = batch.states.Rows();
    Losses losses;
    if (n == 0) return losses;

    // Targets: reward + discount x min of the target critics at the target
    // actor's (smoothed) next action
    std::vector<float> targets(batch.rewards);
    std::vector<size_t> bootstrap;
    for (size_t r = 0; r < n; ++r) {
        if (batch.discounts[r] > 0.0f) bootstrap.push_back(r);
    }
    if (!bootstrap.empty()) {
        nn::Matrix next(bootstrap.size(), state_size_);
        for (size_t i = 0; i < bootstrap.size(); ++i) {
            std::copy(batch.next_states.Data() + bootstrap[i] * state_size_,
                      batch.next_states.Data() + (bootstrap[i] + 1) * state_size_, next.Data() + i * state_size_);
        }
        nn::Matrix next_actions = actor_target_.Predict(next);
        std::normal_distribution<float> noise(0.0f, config_.target_noise);
        for (size_t i = 0; i < next_actions.Size(); ++i) {
            float e = std::clamp(noise(rng), -config_.noise_clip, config_.noise_clip);
            next_actions.Data()[i] = std::clamp(next_actions.Data()[i] + e, -1.0f, 1.0f);
        }
        nn::Matrix joined = Join(next, next_actions);
        nn::Matrix q1 = critic1_target_.Predict(joined);
        nn::Matrix q2 = critic2_target_.Predict(joined);
        for (size_t i = 0; i < bootstrap.size(); ++i) {
            size_t r = bootstrap[i];
            targets[r] += batch.discounts[r] * std::min(q1(i, 0), q2(i, 0));
        }
    }

    // Critics: mean squared error toward the targets
    nn::Matrix joined = Join(batch.states, batch.actions);
    critic1_.ZeroGradients();
    critic2_.ZeroGradients();
    double loss = 0.0;
    for (nn::Sequential* critic : {&critic1_, &critic2_}) {
        nn::Matrix q = critic->Forward(joined, true);
        nn::Matrix grad(n, 1);
        for (size_t r = 0; r < n; ++r) {
            float diff = q(r, 0) - targets[r];
            loss += static_cast<double>(diff) * diff;
            grad(r, 0) = 2.0f * diff / static_cast<float>(n);
        }
        critic->Backward(grad);
    }
    critic_optimizer_.Step(CriticParameters());
    losses.critic = static_cast<float>(loss / static_cast<double>(2 * n));
    updates_++;

    // Delayed actor update: maximize Q1(s, actor(s)) less the saturation
    // penalty, then move the targets
    if (updates_ % config_.policy_delay == 0) {
        actor_.ZeroGradients();
        nn::Matrix actions = actor_.Forward(batch.states, true);
        critic1_.Forward(Join(batch.states, actions), true);
        nn::Matrix grad_q(n, 1, -1.0f / static_cast<float>(n));
        nn::Matrix grad_joined = critic1_.Backward(grad_q);
        nn::Matrix grad_actions(n, action_size_);
        for (size_t r = 0; r < n; ++r) {
            for (size_t a = 0; a < action_size_; ++a) {
                float g = grad_joined(r, state_size_ + a);
                // The penalty's gradient on z, divided by the tanh slope the
                // Tanh layer multiplies it by (0 only where tanh rounds to +-1)
                float t = actions(r, a);
                float slope = 1.0f - t * t;
                if (config_.saturation_penalty > 0.0f && slope > 0.0f) {
                    float z = std::atanh(t);
                    float excess = std::fabs(z) - config_.saturation_threshold;
                    if (excess > 0.0f) {
                        g += 2.0f * config_.saturation_penalty * std::copysign(excess, z) /
                             (static_cast<float>(n) * slope);
                    }
                }
                grad_actions(r, a) = g;
            }
        }
        actor_.Backward(grad_actions);
        actor_optimizer_.Step(actor_.Parameters());
        critic1_.ZeroGradients();  // the actor step must not leave critic gradients behind
        losses.actor_updated = true;

        SoftUpdate(actor_target_, actor_, config_.tau);
        SoftUpdate(critic1_target_, critic1_, config_.tau);
        SoftUpdate(critic2_target_, critic2_, config_.tau);
    }
    return losses;
}

size_t ActorCritic::ParameterCount() const {
    return actor_.ParameterCount() + critic1_.ParameterCount() + critic2_.ParameterCount();
}

bool ActorCritic::AllFinite() {
    for (nn::Sequential* model : {&actor_, &actor_target_, &critic1_, &critic1_target_, &critic2_, &critic2_target_}) {
        if (!Finite(*model)) return false;
    }
    return true;
}

void ActorCritic::Serialize(std::ostream& out) const {
    io::WriteSize(out, state_size_);
    io::WriteSize(out, action_size_);
    io::WriteSize(out, config_.hidden);
    io::Write<float>(out, config_.actor_learning_rate);
    io::Write<float>(out, config_.critic_learning_rate);
    io::Write<float>(out, config_.tau);
    io::Write<float>(out, config_.target_noise);
    io::Write<float>(out, config_.noise_clip);
    io::WriteSize(out, config_.policy_delay);
    io::Write<float>(out, config_.saturation_penalty);
    io::Write<float>(out, config_.saturation_threshold);
    io::Write<uint32_t>(out, config_.seed);
    for (const nn::Sequential* model :
         {&actor_, &actor_target_, &critic1_, &critic1_target_, &critic2_, &critic2_target_}) {
        model->Serialize(out);
    }
    actor_optimizer_.SaveState(out);
    critic_optimizer_.SaveState(out);
    io::Write<uint64_t>(out, updates_);
}

std::unique_ptr<ActorCritic> ActorCritic::Deserialize(std::istream& in) {
    std::unique_ptr<ActorCritic> ac(new ActorCritic());
    ac->state_size_ = io::ReadSize(in);
    ac->action_size_ = io::ReadSize(in);
    ac->config_.hidden = io::ReadSize(in);
    ac->config_.actor_learning_rate = io::Read<float>(in);
    ac->config_.critic_learning_rate = io::Read<float>(in);
    ac->config_.tau = io::Read<float>(in);
    ac->config_.target_noise = io::Read<float>(in);
    ac->config_.noise_clip = io::Read<float>(in);
    ac->config_.policy_delay = io::ReadSize(in);
    ac->config_.saturation_penalty = io::Read<float>(in);
    ac->config_.saturation_threshold = io::Read<float>(in);
    ac->config_.seed = io::Read<uint32_t>(in);
    if (ac->config_.policy_delay == 0 || !(ac->config_.saturation_penalty >= 0.0f) ||
        !(ac->config_.saturation_threshold >= 0.0f)) throw std::runtime_error("ActorCritic: malformed data");
    for (nn::Sequential* model : {&ac->actor_, &ac->actor_target_, &ac->critic1_, &ac->critic1_target_,
                                  &ac->critic2_, &ac->critic2_target_}) {
        *model = nn::Sequential::Deserialize(in);
    }
    ac->actor_optimizer_ = nn::Adam(ac->config_.actor_learning_rate);
    ac->critic_optimizer_ = nn::Adam(ac->config_.critic_learning_rate);
    ac->actor_optimizer_.LoadState(in);
    ac->critic_optimizer_.LoadState(in);
    ac->updates_ = io::Read<uint64_t>(in);
    if (ac->actor_.ParameterCount() == 0 || ac->critic1_.ParameterCount() == 0) {
        throw std::runtime_error("ActorCritic: malformed data");
    }
    return ac;
}

}  // namespace agent
}  // namespace dpan
