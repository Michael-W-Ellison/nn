// File: src/agent/agent.cpp
#include "agent/agent.hpp"
#include "agent/actor_critic.hpp"

#include "agent/agent_testing.hpp"
#include "core/binary_io.hpp"
#include "agent/drift_detector.hpp"
#include "agent/network.hpp"
#include "agent/pattern_memory.hpp"
#include "agent/replay_buffer.hpp"
#include "nn/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <cstdio>
#include <fstream>
#include <limits>
#include <locale>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace dpan {
namespace agent {

namespace {

const float kNaN = std::numeric_limits<float>::quiet_NaN();

void CheckConfig(const AgentConfig& c) {
    auto require = [](bool ok, const char* message) {
        if (!ok) throw std::invalid_argument(std::string("AgentConfig: ") + message);
    };
    require(c.learning_rate > 0.0f, "learning_rate must be > 0");
    require(c.min_learning_rate > 0.0f && c.min_learning_rate <= c.learning_rate,
            "min_learning_rate must be in (0, learning_rate]");
    require(c.min_trunk_width > 0 && c.min_trunk_width <= c.max_trunk_width,
            "trunk widths must satisfy 0 < min_trunk_width <= max_trunk_width");
    require(c.gradient_clip_norm >= 0.0f, "gradient_clip_norm must be >= 0");
    require(c.train_every > 0 && c.batch_size > 0, "train_every and batch_size must be > 0");
    require(c.new_sample_share >= 0.0f && c.new_sample_share <= 1.0f, "new_sample_share must be in [0, 1]");
    require(c.replay_capacity > 0, "replay_capacity must be > 0");
    require(c.feedback_timeout > 0, "feedback_timeout must be > 0");
    require(c.plateau_interval > 0 && c.plateau_patience > 0, "plateau_interval and plateau_patience must be > 0");
    require(c.plateau_factor > 0.0f && c.plateau_factor < 1.0f, "plateau_factor must be in (0, 1)");
    require(c.pattern_capacity > 0 && c.pattern_dimension > 0, "pattern_capacity and pattern_dimension must be > 0");
    require(c.neighbors > 0, "neighbors must be > 0");
    require(c.merge_ratio >= 0.0f, "merge_ratio must be >= 0");
    require(c.pattern_label_horizon > 0, "pattern_label_horizon must be > 0");
    require(c.quality_rate > 0.0f && c.quality_rate <= 1.0f, "quality_rate must be in (0, 1]");
    require(c.anomaly_baseline > 0, "anomaly_baseline must be > 0");
    require(c.max_classes > 0, "max_classes must be > 0");
    require(c.memory_budget_bytes > 0, "memory_budget_bytes must be > 0");
    require(c.snapshot_interval > 0 && c.divergence_loss > 0.0f, "snapshot_interval and divergence_loss must be > 0");
    require(c.capacity_growth > 1.0f, "capacity_growth must be > 1");
    require(c.drift_delta >= 0.0 && c.drift_threshold > 0.0, "drift_delta must be >= 0 and drift_threshold > 0");
    require(c.drift_learning_rate_factor >= 1.0f, "drift_learning_rate_factor must be >= 1");
    require(c.drift_new_sample_share >= 0.0f && c.drift_new_sample_share <= 1.0f, "drift_new_sample_share must be in [0, 1]");
    require(c.drift_recent_window > 0 && c.drift_recovery_max_samples > 0,
            "drift_recent_window and drift_recovery_max_samples must be > 0");
    require(c.discount >= 0.0f && c.discount <= 1.0f, "discount must be in [0, 1]");
    require(c.target_sync > 0, "target_sync must be > 0");
    require(c.action_return_steps > 0, "action_return_steps must be > 0");
    require(c.action_train_every > 0, "action_train_every must be > 0");
    require(c.action_learning_rate >= c.min_learning_rate, "action_learning_rate must be >= min_learning_rate");
    require(c.exploration_start >= 0.0f && c.exploration_start <= 1.0f && c.exploration_end >= 0.0f &&
                c.exploration_end <= c.exploration_start,
            "exploration rates must satisfy 0 <= exploration_end <= exploration_start <= 1");
    require(c.continuous_hidden > 0, "continuous_hidden must be > 0");
    require(c.continuous_actor_learning_rate > 0.0f && c.continuous_critic_learning_rate > 0.0f,
            "continuous learning rates must be > 0");
    require(c.continuous_exploration_noise >= 0.0f && c.continuous_target_noise >= 0.0f &&
                c.continuous_noise_clip >= 0.0f,
            "continuous noise settings must be >= 0");
    require(c.continuous_policy_delay > 0, "continuous_policy_delay must be > 0");
    require(c.continuous_updates_per_step > 0, "continuous_updates_per_step must be > 0");
    require(c.continuous_tau > 0.0f && c.continuous_tau <= 1.0f, "continuous_tau must be in (0, 1]");
    ValidateEncoderConfig(c.encoder);
}

std::string FormatNumber(double x) {
    std::ostringstream text;
    text.imbue(std::locale::classic());
    if (std::floor(x) == x && std::fabs(x) < 1e15) {
        text << static_cast<long long>(x);
    } else {
        text.precision(17);
        text << x;
    }
    return text.str();
}

nn::Matrix Row(const std::vector<float>& values) { return nn::Matrix::RowVector(values); }

std::vector<float> Softmax(const nn::Matrix& logits) {
    return nn::Softmax::Compute(logits).Row(0);
}

size_t ArgMax(const std::vector<float>& values) {
    return static_cast<size_t>(std::max_element(values.begin(), values.end()) - values.begin());
}

/// Running estimate that starts as a plain mean and becomes an
/// exponential average with the given rate
struct Quality {
    float value{kNaN};
    uint64_t count{0};

    void Add(float x, float rate) {
        ++count;
        float r = std::max(rate, 1.0f / static_cast<float>(count));
        value = std::isnan(value) ? x : value + r * (x - value);
    }
    bool Measured() const { return count > 0; }
};

/// Percentile of x among the baseline (0.5 if the baseline is empty)
float Percentile(const std::deque<float>& baseline, float x) {
    if (baseline.empty()) return 0.5f;
    double below = 0.0;
    for (float b : baseline) {
        if (b < x) below += 1.0;
        else if (b == x) below += 0.5;
    }
    return static_cast<float>(below / static_cast<double>(baseline.size()));
}

// ============================================================================
// Checkpoint format helpers
// ============================================================================

constexpr char kCheckpointMagic[4] = {'D', 'P', 'A', 'G'};
// 2: image settings in the config; embeddings stored as layer sequences
// 3: CONTINUOUS_ACTION outputs (bounds in the spec, TD3 settings, actor-critics,
//    continuous actions in samples and pending steps)
// 4: continuous_updates_per_step in the config
constexpr uint32_t kCheckpointVersion = 4;

template <typename T>
void WriteVector(std::ostream& out, const std::vector<T>& values) {
    io::WriteSize(out, values.size());
    for (const T& v : values) io::Write<T>(out, v);
}

template <typename T>
std::vector<T> ReadVector(std::istream& in) {
    std::vector<T> values(io::ReadSize(in, 1u << 30));
    for (T& v : values) v = io::Read<T>(in);
    return values;
}

void WriteStrings(std::ostream& out, const std::vector<std::string>& values) {
    io::WriteSize(out, values.size());
    for (const auto& v : values) io::WriteString(out, v);
}

std::vector<std::string> ReadStrings(std::istream& in) {
    std::vector<std::string> values(io::ReadSize(in, 1u << 24));
    for (auto& v : values) v = io::ReadString(in, 1u << 20);
    return values;
}

template <typename Enum>
Enum ReadEnum(std::istream& in, Enum last) {
    auto value = io::Read<uint8_t>(in);
    if (value > static_cast<uint8_t>(last)) throw std::runtime_error("Agent checkpoint: bad enum value");
    return static_cast<Enum>(value);
}

void SaveSpec(std::ostream& out, const EnvironmentSpec& spec) {
    io::WriteSize(out, spec.inputs.size());
    for (const auto& i : spec.inputs) {
        io::WriteString(out, i.name);
        io::Write<uint8_t>(out, static_cast<uint8_t>(i.type));
        io::WriteSize(out, i.size);
        WriteStrings(out, i.categories);
        io::WriteSize(out, i.width);
        io::WriteSize(out, i.height);
        io::WriteSize(out, i.image_channels);
    }
    io::WriteSize(out, spec.feedback.size());
    for (const auto& f : spec.feedback) {
        io::WriteString(out, f.name);
        io::Write<uint8_t>(out, static_cast<uint8_t>(f.type));
        io::WriteString(out, f.output);
    }
    io::WriteSize(out, spec.outputs.size());
    for (const auto& o : spec.outputs) {
        io::WriteString(out, o.name);
        io::Write<uint8_t>(out, static_cast<uint8_t>(o.type));
        WriteStrings(out, o.classes_or_actions);
        io::WriteSize(out, o.size);
        WriteVector(out, o.low);
        WriteVector(out, o.high);
    }
}

EnvironmentSpec LoadSpec(std::istream& in) {
    EnvironmentSpec spec;
    spec.inputs.resize(io::ReadSize(in, 1u << 16));
    for (auto& i : spec.inputs) {
        i.name = io::ReadString(in, 1u << 16);
        i.type = ReadEnum(in, InputType::TIME_SERIES);
        i.size = io::ReadSize(in);
        i.categories = ReadStrings(in);
        i.width = io::ReadSize(in);
        i.height = io::ReadSize(in);
        i.image_channels = io::ReadSize(in);
    }
    spec.feedback.resize(io::ReadSize(in, 1u << 16));
    for (auto& f : spec.feedback) {
        f.name = io::ReadString(in, 1u << 16);
        f.type = ReadEnum(in, FeedbackType::OUTCOME);
        f.output = io::ReadString(in, 1u << 16);
    }
    spec.outputs.resize(io::ReadSize(in, 1u << 16));
    for (auto& o : spec.outputs) {
        o.name = io::ReadString(in, 1u << 16);
        o.type = ReadEnum(in, OutputType::CONTINUOUS_ACTION);
        o.classes_or_actions = ReadStrings(in);
        o.size = io::ReadSize(in);
        o.low = ReadVector<float>(in);
        o.high = ReadVector<float>(in);
    }
    return spec;
}

void SaveConfig(std::ostream& out, const AgentConfig& c) {
    io::Write<uint32_t>(out, c.seed);
    io::Write<float>(out, c.learning_rate);
    io::Write<float>(out, c.min_learning_rate);
    io::WriteSize(out, c.min_trunk_width);
    io::WriteSize(out, c.max_trunk_width);
    io::Write<float>(out, c.gradient_clip_norm);
    io::WriteSize(out, c.train_every);
    io::WriteSize(out, c.batch_size);
    io::Write<float>(out, c.new_sample_share);
    io::WriteSize(out, c.replay_capacity);
    io::WriteSize(out, c.cold_start_samples);
    io::WriteSize(out, c.feedback_timeout);
    io::WriteSize(out, c.plateau_interval);
    io::WriteSize(out, c.plateau_patience);
    io::Write<float>(out, c.plateau_factor);
    io::WriteSize(out, c.action_train_every);
    io::Write<float>(out, c.action_learning_rate);
    io::WriteSize(out, c.action_min_trunk_width);
    io::WriteSize(out, c.action_return_steps);
    io::Write<float>(out, c.discount);
    io::WriteSize(out, c.target_sync);
    io::Write<float>(out, c.exploration_start);
    io::Write<float>(out, c.exploration_end);
    io::WriteSize(out, c.exploration_steps);
    io::Write<uint8_t>(out, c.detect_drift ? 1 : 0);
    io::Write<double>(out, c.drift_delta);
    io::Write<double>(out, c.drift_threshold);
    io::WriteSize(out, c.drift_warmup);
    io::Write<float>(out, c.drift_learning_rate_factor);
    io::Write<float>(out, c.drift_new_sample_share);
    io::WriteSize(out, c.drift_recent_window);
    io::WriteSize(out, c.drift_recovery_max_samples);
    io::WriteSize(out, c.snapshot_interval);
    io::Write<float>(out, c.divergence_loss);
    io::Write<uint8_t>(out, c.grow_capacity ? 1 : 0);
    io::WriteSize(out, c.capacity_min_samples);
    io::Write<float>(out, c.capacity_growth);
    io::WriteSize(out, c.memory_budget_bytes);
    io::WriteSize(out, c.pattern_capacity);
    io::WriteSize(out, c.pattern_dimension);
    io::WriteSize(out, c.neighbors);
    io::Write<float>(out, c.merge_ratio);
    io::WriteSize(out, c.pattern_label_horizon);
    io::Write<float>(out, c.quality_rate);
    io::WriteSize(out, c.anomaly_baseline);
    io::WriteSize(out, c.max_classes);
    io::Write<float>(out, c.encoder.clip_sigma);
    io::Write<uint64_t>(out, c.encoder.statistics_limit);
    io::WriteSize(out, c.encoder.category_buckets);
    io::WriteSize(out, c.encoder.category_embedding_size);
    io::WriteSize(out, c.encoder.text_buckets);
    io::WriteSize(out, c.encoder.text_embedding_size);
    io::WriteSize(out, c.encoder.time_series_window);
    io::WriteSize(out, c.encoder.image_size);
    io::WriteSize(out, c.encoder.max_image_size);
    io::WriteSize(out, c.encoder.image_embedding_size);
    io::WriteSize(out, c.encoder.max_tracked_categories);
    io::WriteSize(out, c.continuous_hidden);
    io::Write<float>(out, c.continuous_actor_learning_rate);
    io::Write<float>(out, c.continuous_critic_learning_rate);
    io::WriteSize(out, c.continuous_random_steps);
    io::Write<float>(out, c.continuous_exploration_noise);
    io::Write<float>(out, c.continuous_target_noise);
    io::Write<float>(out, c.continuous_noise_clip);
    io::WriteSize(out, c.continuous_policy_delay);
    io::Write<float>(out, c.continuous_tau);
    io::WriteSize(out, c.continuous_updates_per_step);
}

AgentConfig LoadConfig(std::istream& in) {
    AgentConfig c;
    c.seed = io::Read<uint32_t>(in);
    c.learning_rate = io::Read<float>(in);
    c.min_learning_rate = io::Read<float>(in);
    c.min_trunk_width = io::ReadSize(in);
    c.max_trunk_width = io::ReadSize(in);
    c.gradient_clip_norm = io::Read<float>(in);
    c.train_every = io::ReadSize(in);
    c.batch_size = io::ReadSize(in);
    c.new_sample_share = io::Read<float>(in);
    c.replay_capacity = io::ReadSize(in);
    c.cold_start_samples = io::ReadSize(in);
    c.feedback_timeout = io::ReadSize(in);
    c.plateau_interval = io::ReadSize(in);
    c.plateau_patience = io::ReadSize(in);
    c.plateau_factor = io::Read<float>(in);
    c.action_train_every = io::ReadSize(in);
    c.action_learning_rate = io::Read<float>(in);
    c.action_min_trunk_width = io::ReadSize(in);
    c.action_return_steps = io::ReadSize(in);
    c.discount = io::Read<float>(in);
    c.target_sync = io::ReadSize(in);
    c.exploration_start = io::Read<float>(in);
    c.exploration_end = io::Read<float>(in);
    c.exploration_steps = io::ReadSize(in);
    c.detect_drift = io::Read<uint8_t>(in) != 0;
    c.drift_delta = io::Read<double>(in);
    c.drift_threshold = io::Read<double>(in);
    c.drift_warmup = io::ReadSize(in);
    c.drift_learning_rate_factor = io::Read<float>(in);
    c.drift_new_sample_share = io::Read<float>(in);
    c.drift_recent_window = io::ReadSize(in);
    c.drift_recovery_max_samples = io::ReadSize(in);
    c.snapshot_interval = io::ReadSize(in);
    c.divergence_loss = io::Read<float>(in);
    c.grow_capacity = io::Read<uint8_t>(in) != 0;
    c.capacity_min_samples = io::ReadSize(in);
    c.capacity_growth = io::Read<float>(in);
    c.memory_budget_bytes = io::ReadSize(in);
    c.pattern_capacity = io::ReadSize(in);
    c.pattern_dimension = io::ReadSize(in);
    c.neighbors = io::ReadSize(in);
    c.merge_ratio = io::Read<float>(in);
    c.pattern_label_horizon = io::ReadSize(in);
    c.quality_rate = io::Read<float>(in);
    c.anomaly_baseline = io::ReadSize(in);
    c.max_classes = io::ReadSize(in);
    c.encoder.clip_sigma = io::Read<float>(in);
    c.encoder.statistics_limit = io::Read<uint64_t>(in);
    c.encoder.category_buckets = io::ReadSize(in);
    c.encoder.category_embedding_size = io::ReadSize(in);
    c.encoder.text_buckets = io::ReadSize(in);
    c.encoder.text_embedding_size = io::ReadSize(in);
    c.encoder.time_series_window = io::ReadSize(in);
    c.encoder.image_size = io::ReadSize(in);
    c.encoder.max_image_size = io::ReadSize(in);
    c.encoder.image_embedding_size = io::ReadSize(in);
    c.encoder.max_tracked_categories = io::ReadSize(in);
    c.continuous_hidden = io::ReadSize(in);
    c.continuous_actor_learning_rate = io::Read<float>(in);
    c.continuous_critic_learning_rate = io::Read<float>(in);
    c.continuous_random_steps = io::ReadSize(in);
    c.continuous_exploration_noise = io::Read<float>(in);
    c.continuous_target_noise = io::Read<float>(in);
    c.continuous_noise_clip = io::Read<float>(in);
    c.continuous_policy_delay = io::ReadSize(in);
    c.continuous_tau = io::Read<float>(in);
    c.continuous_updates_per_step = io::ReadSize(in);
    return c;
}

void SaveQuality(std::ostream& out, const Quality& q) {
    io::Write<float>(out, q.value);
    io::Write<uint64_t>(out, q.count);
}

Quality LoadQuality(std::istream& in) {
    Quality q;
    q.value = io::Read<float>(in);
    q.count = io::Read<uint64_t>(in);
    return q;
}

void SaveOptional(std::ostream& out, const std::optional<StepId>& value) {
    io::Write<uint8_t>(out, value ? 1 : 0);
    io::Write<uint64_t>(out, value.value_or(0));
}

std::optional<StepId> LoadOptional(std::istream& in) {
    bool has = io::Read<uint8_t>(in) != 0;
    uint64_t value = io::Read<uint64_t>(in);
    return has ? std::optional<StepId>(value) : std::nullopt;
}

void SaveIndexMap(std::ostream& out, const std::map<size_t, size_t>& values) {
    io::WriteSize(out, values.size());
    for (const auto& [key, value] : values) {
        io::WriteSize(out, key);
        io::WriteSize(out, value);
    }
}

std::map<size_t, size_t> LoadIndexMap(std::istream& in) {
    std::map<size_t, size_t> values;
    size_t n = io::ReadSize(in, 1u << 20);
    for (size_t i = 0; i < n; ++i) {
        size_t key = io::ReadSize(in);
        values[key] = io::ReadSize(in);
    }
    return values;
}

void SaveRandom(std::ostream& out, const std::mt19937& rng) {
    std::ostringstream text;
    text << rng;
    io::WriteString(out, text.str());
}

void LoadRandom(std::istream& in, std::mt19937& rng) {
    std::istringstream text(io::ReadString(in, 1u << 20));
    text >> rng;
    if (!text) throw std::runtime_error("Agent checkpoint: bad random state");
}

}  // namespace

struct Agent::Impl {
    struct OutputState {
        const OutputSpec* spec{nullptr};
        int head{-1};        // network head
        int class_slot{-1};  // index among CLASS outputs (pattern memory)
        int value_slot{-1};  // index among VALUE outputs
        int continuous_slot{-1};  // index among CONTINUOUS_ACTION outputs (actor_critics)
        bool learn_classes{false};
        bool outcome_only{false};  // CLASS trained by OUTCOME feedback only
        std::vector<std::string> classes;  // class names, or action names for ACTION
        std::vector<PatternID> action_ids;  // ACTION: one pattern ID per action (associations)
        uint64_t acts{0};                   // ACTION: exploring Act calls
        std::map<std::string, size_t> class_index;
        std::vector<RunningStats> target_stats;
        uint64_t samples{0};
        Quality neural;
        Quality pattern_short;  // votes of recently labelled patterns
        Quality pattern_long;   // votes of all patterns (also NEXT_OBSERVATION sequences)

        // Drift (Page-Hinkley on the neural loss) and recovery
        PageHinkley drift;
        bool recovering{false};
        uint64_t recovery_samples{0};
        double drift_baseline{0.0};  // mean loss before the drift
        Quality recovery_loss;
        uint32_t drift_events{0};

        /// Whether the short-term pattern memory is currently the better one
        bool UseShort(bool higher_is_better) const {
            if (recovering) return true;  // old knowledge is suspect after a drift
            if (!pattern_short.Measured() || !pattern_long.Measured()) return false;
            return higher_is_better ? pattern_short.value > pattern_long.value
                                    : pattern_short.value < pattern_long.value;
        }
        const Quality& Pattern(bool higher_is_better) const {
            return UseShort(higher_is_better) ? pattern_short : pattern_long;
        }
    };

    struct Pending {
        StepId step{0};
        bool encoded{false};
        std::vector<float> fused;
        std::vector<float> point;
        std::vector<float> observed;  // raw NUMERIC/TIME_SERIES values
        PatternID pattern;
        SamplePtr sample;
        std::map<size_t, size_t> predicted;  // output -> class returned by Predict
        std::map<size_t, size_t> actions;    // output -> action chosen by Act
        std::map<size_t, std::vector<float>> continuous_actions;  // output -> action chosen by ActContinuous
        bool transition_complete{false};     // next state or terminal known (ACTION)
        float anomaly{0.5f};
        float anomaly_neural_weight{0.0f};
    };

    struct FeedbackState {
        const FeedbackSpec* spec;
        size_t output;
    };

    Impl(EnvironmentSpec s, AgentConfig c)
        : spec(std::move(s)),
          config(c),
          encoder(spec.inputs, config.encoder),
          replay(config.replay_capacity, config.seed + 7),
          schedule(config.plateau_factor, config.plateau_patience, config.min_learning_rate) {
        rng.seed(config.seed + 11);
    }

    // ------------------------------------------------------------------ setup

    void Initialize() {
        for (size_t i = 0; i < spec.outputs.size(); ++i) {
            OutputState state;
            state.spec = &spec.outputs[i];
            PageHinkley::Config drift;
            drift.delta = config.drift_delta;
            drift.threshold = config.drift_threshold;
            drift.warmup = config.drift_warmup;
            state.drift = PageHinkley(drift);
            if (state.spec->type == OutputType::CLASS) {
                state.classes = state.spec->classes_or_actions;
                state.learn_classes = state.classes.empty();
                for (size_t c = 0; c < state.classes.size(); ++c) state.class_index[state.classes[c]] = c;
                state.class_slot = static_cast<int>(class_outputs++);
            } else if (state.spec->type == OutputType::VALUE) {
                state.value_slot = static_cast<int>(value_sizes.size());
                value_sizes.push_back(state.spec->size);
                state.target_stats.resize(state.spec->size);
            } else if (state.spec->type == OutputType::NEXT_OBSERVATION) {
                next_output = static_cast<int>(i);
            } else if (state.spec->type == OutputType::ANOMALY) {
                has_anomaly = true;
            } else if (state.spec->type == OutputType::ACTION) {
                state.classes = state.spec->classes_or_actions;
                for (size_t a = 0; a < state.classes.size(); ++a) {
                    state.class_index[state.classes[a]] = a;
                    state.action_ids.push_back(PatternID::Generate());
                }
                has_actions = true;
            } else if (state.spec->type == OutputType::CONTINUOUS_ACTION) {
                state.continuous_slot = static_cast<int>(continuous_outputs.size());
                continuous_outputs.push_back(i);
            }
            outputs.push_back(std::move(state));
        }
        for (const auto& fb : spec.feedback) {
            for (size_t i = 0; i < spec.outputs.size(); ++i) {
                if (spec.outputs[i].name == fb.output) feedback.emplace(fb.name, FeedbackState{&fb, i});
            }
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (outputs[i].spec->type != OutputType::CLASS) continue;
            bool labelled = std::any_of(spec.feedback.begin(), spec.feedback.end(), [&](const FeedbackSpec& fb) {
                return fb.output == outputs[i].spec->name && fb.type == FeedbackType::LABEL;
            });
            outputs[i].outcome_only = !labelled;
        }
    }

    void Log(LogLevel level, const std::string& message) const {
        if (logger) logger(level, message);
    }

    /// Positions of NUMERIC/TIME_SERIES values and one head per output
    /// (needs the input sizes); used when building and when loading
    std::vector<HeadSpec> PrepareLayout() {
        observed_channels.clear();
        observed_offsets.clear();
        observed_size = 0;
        for (size_t i = 0; i < spec.inputs.size(); ++i) {
            const auto& input = spec.inputs[i];
            if (input.type == InputType::NUMERIC || input.type == InputType::TIME_SERIES) {
                observed_channels.push_back(i);
                observed_offsets.push_back(observed_size);
                observed_size += input.type == InputType::NUMERIC
                                     ? dynamic_cast<const NumericEncoder&>(encoder.Channel(i)).Size()
                                     : dynamic_cast<const TimeSeriesEncoder&>(encoder.Channel(i)).SeriesCount();
            }
        }

        std::vector<HeadSpec> heads;
        for (auto& out : outputs) {
            switch (out.spec->type) {
                case OutputType::CLASS:
                    out.head = static_cast<int>(heads.size());
                    heads.push_back({out.spec->name, HeadKind::CLASSIFIER, out.classes.size()});
                    break;
                case OutputType::VALUE:
                    out.head = static_cast<int>(heads.size());
                    heads.push_back({out.spec->name, HeadKind::REGRESSOR, out.spec->size});
                    break;
                case OutputType::NEXT_OBSERVATION:
                    out.head = static_cast<int>(heads.size());
                    heads.push_back({out.spec->name, HeadKind::REGRESSOR, observed_size});
                    break;
                case OutputType::ACTION:
                    out.head = static_cast<int>(heads.size());
                    heads.push_back({out.spec->name, HeadKind::Q_VALUES, out.classes.size()});
                    break;
                default:
                    break;
            }
        }
        return heads;
    }

    std::unique_ptr<PatternMemory> MakePatternMemory() const {
        PatternMemory::Config memory;
        memory.dimension = config.pattern_dimension;
        memory.memory_budget_bytes = 1;  // set by AllocateMemory()
        memory.capacity = config.pattern_capacity;
        memory.merge_ratio = config.merge_ratio;
        memory.seed = config.seed + 5;
        return std::make_unique<PatternMemory>(encoder.OutputSize(), class_outputs, value_sizes, observed_size,
                                               memory);
    }

    void BuildModels() {
        auto layout = encoder.Layout();
        auto heads = PrepareLayout();
        if (!heads.empty()) {
            network = std::make_unique<AgentNetwork>(layout, heads, config.learning_rate, config.seed,
                                                     config.min_trunk_width, config.max_trunk_width);
            head_loss.assign(heads.size(), kNaN);
            if (has_actions) target_network = std::make_unique<AgentNetwork>(*network);
        }
        if (has_anomaly) {
            autoencoder = std::make_unique<AnomalyAutoencoder>(encoder.OutputSize(), config.learning_rate,
                                                               config.seed + 3);
        }
        actor_critics.clear();
        for (size_t index : continuous_outputs) {
            ActorCriticConfig ac;
            ac.hidden = config.continuous_hidden;
            ac.actor_learning_rate = config.continuous_actor_learning_rate;
            ac.critic_learning_rate = config.continuous_critic_learning_rate;
            ac.tau = config.continuous_tau;
            ac.target_noise = config.continuous_target_noise;
            ac.noise_clip = config.continuous_noise_clip;
            ac.policy_delay = config.continuous_policy_delay;
            ac.seed = config.seed + 100 + static_cast<uint32_t>(actor_critics.size()) * 10;
            actor_critics.push_back(
                std::make_unique<ActorCritic>(encoder.OutputSize(), outputs[index].spec->size, ac));
        }
        patterns = MakePatternMemory();
        AllocateMemory();
        TakeSnapshot();
        Log(LogLevel::INFO, "Inputs ready: fused size " + std::to_string(encoder.OutputSize()) +
                                (network ? ", trunk width " + std::to_string(network->TrunkWidth()) : ""));
    }

    // -------------------------------------------------------------- memory

    static constexpr size_t kMinimumStoreBytes = 1u << 20;

    size_t FixedBytes() const { return ModelBytes() + config.feedback_timeout * PendingEntryBytes(); }

    /// Split the budget: fixed parts first, the rest between the replay
    /// buffer (80%) and the pattern store (20%)
    void AllocateMemory() {
        size_t fixed = FixedBytes();
        size_t remaining = config.memory_budget_bytes > fixed ? config.memory_budget_bytes - fixed : 0;
        if (remaining < kMinimumStoreBytes) {
            Log(LogLevel::WARNING, "memory_budget_bytes (" + std::to_string(config.memory_budget_bytes) +
                                       ") leaves less than 1 MiB after the networks and pending steps (" +
                                       std::to_string(fixed) + " bytes); using 1 MiB");
            remaining = kMinimumStoreBytes;
        }
        replay.SetMemoryBudget(remaining / 10 * 8);
        if (patterns) patterns->SetMemoryBudget(remaining / 10 * 2);
    }

    /// Networks with gradients and Adam moments (4 values per parameter),
    /// target network and the rollback snapshot
    size_t ModelBytes() const {
        size_t parameters = (network ? network->ParameterCount() : 0) + (autoencoder ? autoencoder->ParameterCount() : 0);
        size_t copies = 2 + (target_network ? 1 : 0);  // live + snapshot (+ target)
        size_t bytes = parameters * 4 * sizeof(float) * copies;
        // Actor-critics: live, target and snapshot copies
        for (const auto& ac : actor_critics) bytes += ac->ParameterCount() * 4 * sizeof(float) * 3;
        return bytes;
    }

    /// Upper bound for one pending step, including its sample (which may
    /// have left the replay buffer): dense fused vectors for the state and
    /// the next state, and one target per head
    size_t PendingEntryBytes() const {
        size_t fused = encoder.OutputSize();
        size_t heads = network ? network->HeadCount() : 0;
        size_t entry = sizeof(Pending) + 256 + (fused + config.pattern_dimension + observed_size) * sizeof(float);
        size_t continuous = 0;
        for (size_t index : continuous_outputs) {
            continuous += sizeof(ContinuousTarget) + 64 + 3 * outputs[index].spec->size * sizeof(float);
        }
        size_t sample = sizeof(Sample) + 32 + 2 * fused * (sizeof(uint32_t) + sizeof(float)) +
                        heads * sizeof(HeadTarget) + (observed_size + 16) * sizeof(float) + continuous;
        return entry + sample;
    }

    size_t MemoryBytes() const {
        size_t bytes = ModelBytes() + replay.MemoryBytes() + (patterns ? patterns->MemoryBytes() : 0);
        // Samples outside the replay buffer are kept alive by pending steps
        // and the newest queue; count each once
        std::unordered_set<const Sample*> counted;
        auto add_sample = [&](const SamplePtr& sample) {
            if (sample && !sample->in_replay && counted.insert(sample.get()).second) bytes += sample->MemoryBytes();
        };
        for (const auto& entry : pending) {
            bytes += sizeof(Pending) + 256 + (entry.fused.size() + entry.point.size() + entry.observed.size()) * sizeof(float);
            for (const auto& [index, action] : entry.continuous_actions) bytes += 64 + action.size() * sizeof(float);
            add_sample(entry.sample);
        }
        for (const auto& sample : newest) add_sample(sample);
        return bytes;
    }

    // ----------------------------------------------------- observed values

    std::vector<float> Observed(const Observation& observation) const {
        std::vector<float> values(observed_size, kNaN);
        for (size_t c = 0; c < observed_channels.size(); ++c) {
            auto it = observation.find(spec.inputs[observed_channels[c]].name);
            if (it == observation.end()) continue;
            std::vector<float> raw;
            if (const double* number = std::get_if<double>(&it->second)) {
                raw.assign(1, static_cast<float>(*number));
            } else if (const auto* vector = std::get_if<std::vector<float>>(&it->second)) {
                raw = *vector;
            }
            for (size_t j = 0; j < raw.size() && observed_offsets[c] + j < observed_size; ++j) {
                values[observed_offsets[c] + j] = std::isfinite(raw[j]) ? raw[j] : kNaN;
            }
        }
        return values;
    }

    const RunningStats& ObservedStats(size_t element) const {
        size_t c = static_cast<size_t>(
            std::upper_bound(observed_offsets.begin(), observed_offsets.end(), element) -
            observed_offsets.begin() - 1);
        size_t j = element - observed_offsets[c];
        const Encoder& channel = encoder.Channel(observed_channels[c]);
        if (spec.inputs[observed_channels[c]].type == InputType::NUMERIC) {
            return dynamic_cast<const NumericEncoder&>(channel).Statistics()[j];
        }
        return dynamic_cast<const TimeSeriesEncoder&>(channel).SeriesStatistics(j);
    }

    static float ToZ(float raw, const RunningStats& stats) {
        if (std::isnan(raw)) return kNaN;
        double sd = stats.StdDev();
        return sd > 1e-12 ? static_cast<float>((raw - stats.mean) / sd) : 0.0f;
    }

    static float FromZ(float z, const RunningStats& stats) {
        double sd = stats.StdDev();
        return static_cast<float>(stats.mean + (sd > 1e-12 ? z * sd : 0.0));
    }

    /// Product of two predictions' errors in standard deviations, averaged
    /// over the elements known in both and in `actual` (a == b gives the
    /// mean squared standardized error); NaN if there are none
    static float ErrorProduct(const std::vector<float>& a, const std::vector<float>& b,
                              const std::vector<float>& actual,
                              const std::function<const RunningStats&(size_t)>& stats) {
        double sum = 0.0;
        size_t n = 0;
        for (size_t i = 0; i < actual.size() && i < a.size() && i < b.size(); ++i) {
            if (std::isnan(actual[i]) || std::isnan(a[i]) || std::isnan(b[i])) continue;
            double sd = stats(i).StdDev();
            double scale = sd > 1e-12 ? sd : 1.0;
            sum += (a[i] - actual[i]) / scale * ((b[i] - actual[i]) / scale);
            ++n;
        }
        return n == 0 ? kNaN : static_cast<float>(sum / static_cast<double>(n));
    }

    /// Score regression predictions made before learning from `actual`
    void ScoreRegression(OutputState& out, const std::optional<std::vector<float>>& neural,
                         const std::optional<std::vector<float>>& pattern_short,
                         const std::optional<std::vector<float>>& pattern_long,
                         const std::vector<float>& actual,
                         const std::function<const RunningStats&(size_t)>& stats) {
        auto add = [&](Quality& quality, float value) {
            if (!std::isnan(value)) quality.Add(value, config.quality_rate);
        };
        if (neural) {
            float error = ErrorProduct(*neural, *neural, actual, stats);
            add(out.neural, error);
            ObserveLoss(out, error);
        }
        if (pattern_short) add(out.pattern_short, ErrorProduct(*pattern_short, *pattern_short, actual, stats));
        if (pattern_long) add(out.pattern_long, ErrorProduct(*pattern_long, *pattern_long, actual, stats));
    }

    // ------------------------------------------------------------ pending

    Pending& FindPending(StepId step) const {
        if (step == 0 || step > steps) {
            throw std::out_of_range("Unknown step " + std::to_string(step));
        }
        if (pending.empty() || step < pending.front().step) {
            throw std::out_of_range("Step " + std::to_string(step) + " has expired (feedback_timeout = " +
                                    std::to_string(config.feedback_timeout) + " steps)");
        }
        return pending[step - pending.front().step];
    }

    void MakeTrainable(const SamplePtr& sample) {
        replay.Add(sample);
        newest.push_back(sample);
        while (newest.size() > config.batch_size) newest.pop_front();
        recent.push_back(sample);  // weak: does not keep evicted samples alive
        while (recent.size() > config.drift_recent_window) recent.pop_front();
    }

    // -------------------------------------------------------------- safety

    bool ModelsFinite() {
        for (auto& ac : actor_critics) {
            if (!ac->AllFinite()) return false;
        }
        return (!network || network->AllFinite()) && (!autoencoder || autoencoder->AllFinite());
    }

    void TakeSnapshot() {
        if (network) snapshot_network = std::make_unique<AgentNetwork>(*network);
        if (autoencoder) snapshot_autoencoder = std::make_unique<AnomalyAutoencoder>(*autoencoder);
        snapshot_actor_critics.clear();
        for (const auto& ac : actor_critics) snapshot_actor_critics.push_back(std::make_unique<ActorCritic>(*ac));
        last_snapshot = steps;
    }

    void Rollback() {
        if (snapshot_network) {
            network = std::make_unique<AgentNetwork>(*snapshot_network);
            if (target_network) target_network = std::make_unique<AgentNetwork>(*snapshot_network);
        }
        if (snapshot_autoencoder) autoencoder = std::make_unique<AnomalyAutoencoder>(*snapshot_autoencoder);
        for (size_t i = 0; i < snapshot_actor_critics.size() && i < actor_critics.size(); ++i) {
            actor_critics[i] = std::make_unique<ActorCritic>(*snapshot_actor_critics[i]);
        }
        std::fill(head_loss.begin(), head_loss.end(), kNaN);
        ApplyLearningRate();
        rollbacks++;
        Log(LogLevel::WARNING, "Training produced non-finite or diverging values at step " + std::to_string(steps) +
                                   "; weights restored from step " + std::to_string(last_snapshot));
    }

    // ------------------------------------------------------------ capacity

    /// Largest relative shortfall of the network against the pattern side
    /// (accuracy points, or relative extra error), over outputs with
    /// enough samples; 0 if none trails
    float NetworkGap() const {
        float gap = 0.0f;
        for (const auto& out : outputs) {
            if (out.samples < config.capacity_min_samples || out.head < 0) continue;
            bool higher = out.spec->type == OutputType::CLASS;
            if (out.spec->type != OutputType::CLASS && out.spec->type != OutputType::VALUE &&
                out.spec->type != OutputType::NEXT_OBSERVATION) {
                continue;
            }
            const Quality& pattern = out.Pattern(higher);
            if (!out.neural.Measured() || !pattern.Measured()) continue;
            gap = std::max(gap, higher ? pattern.value - out.neural.value
                                       : (out.neural.value - pattern.value) / std::max(pattern.value, 1e-6f));
        }
        return gap;
    }

    bool ShouldGrow() {
        if (!config.grow_capacity || growth_blocked || !network || network->TrunkWidth() >= config.max_trunk_width) {
            return false;
        }
        float gap = NetworkGap();
        if (widenings > 0 && gap > 0.9f * gap_at_widening) {
            // The last widening did not narrow the gap: capacity is not
            // what the network lacks
            growth_blocked = true;
            Log(LogLevel::INFO, "Trunk widening stopped at " + std::to_string(network->TrunkWidth()) +
                                    " units: the last widening did not narrow the gap to the pattern side");
            return false;
        }
        for (const auto& out : outputs) {
            if (out.samples < config.capacity_min_samples || out.head < 0) continue;
            bool higher = out.spec->type == OutputType::CLASS;
            if (out.spec->type != OutputType::CLASS && out.spec->type != OutputType::VALUE &&
                out.spec->type != OutputType::NEXT_OBSERVATION) {
                continue;
            }
            const Quality& pattern = out.Pattern(higher);
            if (!out.neural.Measured() || !pattern.Measured()) continue;
            // A clear gap, not noise: 2 points of accuracy or 10% more error
            if (higher ? out.neural.value < pattern.value - 0.02f : out.neural.value > 1.1f * pattern.value) {
                return true;
            }
        }
        return false;
    }

    void Grow() {
        size_t old_width = network->TrunkWidth();
        auto width = static_cast<size_t>(std::ceil(static_cast<float>(old_width) * config.capacity_growth));
        width = std::min(config.max_trunk_width, std::max(width, old_width + 1));

        // The wider network must leave room for the replay buffer and patterns
        AgentNetwork wider(*network);
        wider.Widen(width);
        size_t copies = 2 + (target_network ? 1 : 0);
        size_t extra = (wider.ParameterCount() - network->ParameterCount()) * 4 * sizeof(float) * copies;
        if (FixedBytes() + extra + kMinimumStoreBytes > config.memory_budget_bytes) {
            if (!growth_blocked) {
                Log(LogLevel::INFO, "Trunk not widened beyond " + std::to_string(old_width) +
                                        " units: the memory budget leaves no room");
            }
            growth_blocked = true;
            return;
        }
        network = std::make_unique<AgentNetwork>(wider);
        if (target_network) target_network = std::make_unique<AgentNetwork>(*network);
        AllocateMemory();
        ApplyLearningRate();
        TakeSnapshot();
        widenings++;
        gap_at_widening = NetworkGap();
        Log(LogLevel::INFO, "Trunk widened from " + std::to_string(old_width) + " to " + std::to_string(width) +
                                " units at step " + std::to_string(steps));
    }

    // --------------------------------------------------------------- drift

    void ApplyLearningRate() {
        float rate = learning_rate * (recovering_outputs > 0 ? config.drift_learning_rate_factor : 1.0f);
        if (network) network->SetLearningRate(rate);
        if (autoencoder) autoencoder->SetLearningRate(rate);
    }

    /// Feed one prequential loss of an output's network to its drift test
    void ObserveLoss(OutputState& out, float loss) {
        if (!config.detect_drift || out.spec->type == OutputType::ACTION || !std::isfinite(loss)) return;
        if (out.recovering) {
            out.recovery_samples++;
            out.recovery_loss.Add(loss, 0.05f);
            bool back = out.recovery_samples >= config.drift_warmup &&
                        out.recovery_loss.value <= 1.1 * out.drift_baseline + 0.01;
            if (back || out.recovery_samples >= config.drift_recovery_max_samples) EndRecovery(out, back);
        }
        if (out.drift.Add(loss)) StartRecovery(out);
    }

    void StartRecovery(OutputState& out) {
        out.drift_events++;
        drift_events++;
        out.drift_baseline = out.drift.MeanBeforeDrift();
        out.recovery_loss = Quality();
        out.recovery_samples = 0;
        if (!out.recovering) {
            out.recovering = true;
            recovering_outputs++;
        }
        // Re-estimate the blend from scratch
        out.neural = out.pattern_short = out.pattern_long = Quality();
        // Labels and targets from before the drift describe the old concept:
        // drop them from replay (samples observed up to the warm-up length
        // before detection may already follow the new one and are kept)
        if (out.head >= 0 && (out.spec->type == OutputType::CLASS || out.spec->type == OutputType::VALUE)) {
            const auto h = static_cast<size_t>(out.head);
            uint64_t cutoff = steps > config.drift_warmup ? steps - config.drift_warmup : 0;
            replay.Update(
                [&](Sample& sample) {
                    if (sample.step < cutoff && h < sample.targets.size()) sample.targets[h] = HeadTarget();
                },
                [](const Sample& sample) {
                    if (sample.reconstruct || sample.has_next || sample.terminal) return false;
                    return std::all_of(sample.targets.begin(), sample.targets.end(),
                                       [](const HeadTarget& t) { return t.Empty(); });
                });
        }
        // Restart the learning-rate schedule (the loss rose because the data
        // changed, not because training stalled) and raise the rate
        schedule = nn::ReduceOnPlateau(config.plateau_factor, config.plateau_patience, config.min_learning_rate);
        learning_rate = config.learning_rate;
        plateau_sum = 0.0;
        plateau_count = 0;
        ApplyLearningRate();
        Log(LogLevel::WARNING, "Drift detected on output '" + out.spec->name + "' at step " + std::to_string(steps));
    }

    void EndRecovery(OutputState& out, bool recovered) {
        out.recovering = false;
        recovering_outputs--;
        ApplyLearningRate();
        Log(LogLevel::INFO, "Output '" + out.spec->name + (recovered ? "' recovered" : "' recovery period ended") +
                                " after " + std::to_string(out.recovery_samples) + " samples");
    }

    // --------------------------------------------------------- predictions

    /// Neural weight from two running qualities; higher_is_better for accuracy
    float BlendWeight(const OutputState& out, bool neural, bool pattern, bool higher_is_better) const {
        if (!neural) return 0.0f;
        if (!pattern) return 1.0f;
        if (out.samples < config.cold_start_samples) return 0.0f;
        const Quality& pattern_quality = out.Pattern(higher_is_better);
        if (!out.neural.Measured() || !pattern_quality.Measured()) return 0.5f;
        double n, p;
        if (higher_is_better) {
            n = static_cast<double>(out.neural.value) * out.neural.value;
            p = static_cast<double>(pattern_quality.value) * pattern_quality.value;
        } else {
            n = 1.0 / (out.neural.value + 1e-6);
            p = 1.0 / (pattern_quality.value + 1e-6);
        }
        return n + p > 0.0 ? static_cast<float>(n / (n + p)) : 0.5f;
    }

    std::optional<std::vector<float>> NeuralClass(const Pending& entry, const OutputState& out) const {
        // An output trained only by outcomes must predict before it has
        // learned anything, so its outcomes have a prediction to refer to
        if (!network || (out.samples == 0 && !out.outcome_only) || out.classes.empty() ||
            network->Head(static_cast<size_t>(out.head)).size == 0) {
            return std::nullopt;
        }
        auto probabilities = Softmax(network->Infer(Row(entry.fused), out.head));
        probabilities.resize(out.classes.size(), 0.0f);
        return probabilities;
    }

    /// Vote of the short-term (recently labelled) or long-term pattern memory
    std::optional<std::vector<float>> PatternClass(const Pending& entry, const OutputState& out, bool short_term) const {
        if (out.classes.empty()) return std::nullopt;
        return patterns->ClassVote(entry.point, static_cast<size_t>(out.class_slot), out.classes.size(),
                                   config.neighbors, short_term ? config.pattern_label_horizon : 0);
    }

    std::optional<std::vector<float>> PatternValue(const Pending& entry, const OutputState& out, bool short_term) const {
        return patterns->ValueVote(entry.point, static_cast<size_t>(out.value_slot), config.neighbors,
                                   short_term ? config.pattern_label_horizon : 0);
    }

    std::optional<std::vector<float>> NeuralValue(const Pending& entry, const OutputState& out) const {
        if (!network || out.samples == 0) return std::nullopt;
        auto z = network->Infer(Row(entry.fused), out.head).Row(0);
        for (size_t i = 0; i < z.size(); ++i) z[i] = FromZ(z[i], out.target_stats[i]);
        return z;
    }

    std::optional<std::vector<float>> NeuralNext(const Pending& entry, const OutputState& out) const {
        if (!network || out.samples == 0) return std::nullopt;
        auto z = network->Infer(Row(entry.fused), out.head).Row(0);
        for (size_t i = 0; i < z.size(); ++i) z[i] = FromZ(z[i], ObservedStats(i));
        return z;
    }

    /// Records the class returned for a CLASS output (used by OUTCOME feedback)
    // -------------------------------------------------------------- actions

    float Exploration(const OutputState& out) const {
        if (config.exploration_steps == 0) return config.exploration_end;
        float progress = std::min(1.0f, static_cast<float>(out.acts) / static_cast<float>(config.exploration_steps));
        return config.exploration_start - (config.exploration_start - config.exploration_end) * progress;
    }

    /// Q-values once the head is past the cold start
    std::optional<std::vector<float>> NeuralQ(const Pending& entry, const OutputState& out) const {
        if (!network || out.samples < config.cold_start_samples) return std::nullopt;
        return network->Infer(Row(entry.fused), static_cast<size_t>(out.head)).Row(0);
    }

    /// Highest Q-value; ties (and the cold start) decided by association
    /// strength; remaining ties by `rng` if given, else the first action
    size_t GreedyAction(const Pending& entry, const OutputState& out, std::mt19937* tie_rng) const {
        const size_t n = out.classes.size();
        auto q = NeuralQ(entry, out);
        auto strengths = patterns->ActionStrengths(entry.pattern, out.action_ids);
        std::vector<size_t> candidates;
        if (q) {
            float best = *std::max_element(q->begin(), q->end());
            float tolerance = 1e-6f * std::max(1.0f, std::fabs(best));
            for (size_t a = 0; a < n; ++a) {
                if ((*q)[a] >= best - tolerance) candidates.push_back(a);
            }
        } else {
            for (size_t a = 0; a < n; ++a) candidates.push_back(a);
        }
        float strongest = -1.0f;
        for (size_t a : candidates) strongest = std::max(strongest, strengths[a]);
        std::vector<size_t> best;
        for (size_t a : candidates) {
            if (strengths[a] >= strongest) best.push_back(a);
        }
        if (tie_rng && best.size() > 1) {
            return best[std::uniform_int_distribution<size_t>(0, best.size() - 1)(*tie_rng)];
        }
        return best.front();
    }

    std::string ActLocked(StepId step, size_t index, bool explore) {
        Pending& entry = FindPending(step);
        OutputState& out = outputs[index];
        auto existing = entry.actions.find(index);
        if (existing != entry.actions.end()) return out.classes[existing->second];

        float epsilon = Exploration(out);
        if (explore) out.acts++;
        size_t action;
        std::uniform_real_distribution<float> unit(0.0f, 1.0f);
        bool random = !entry.encoded || (explore && unit(rng) < epsilon);
        if (random) {
            action = std::uniform_int_distribution<size_t>(0, out.classes.size() - 1)(rng);
        } else {
            action = GreedyAction(entry, out, &rng);
        }
        entry.actions[index] = action;
        if (entry.encoded) {
            entry.sample->targets[static_cast<size_t>(out.head)].action = static_cast<int>(action);
            entry.sample->targets[static_cast<size_t>(out.head)].exploratory = random;
            last_acted_step = step;
        }
        return out.classes[action];
    }

    // ------------------------------------------------- continuous actions

    /// Bound of dimension i of a CONTINUOUS_ACTION output ([-1, 1] by default)
    static float Low(const OutputSpec& spec, size_t i) { return spec.low.empty() ? -1.0f : spec.low[i]; }
    static float High(const OutputSpec& spec, size_t i) { return spec.high.empty() ? 1.0f : spec.high[i]; }

    /// [-1, 1] -> the output's bounds
    static std::vector<float> ToBounds(const OutputSpec& spec, const float* normalized) {
        std::vector<float> action(spec.size);
        for (size_t i = 0; i < spec.size; ++i) {
            action[i] = Low(spec, i) + (normalized[i] + 1.0f) * 0.5f * (High(spec, i) - Low(spec, i));
        }
        return action;
    }

    /// Exploration noise std now (1 during the uniform random start)
    float ContinuousExploration(const OutputState& out) const {
        return out.acts < config.continuous_random_steps ? 1.0f : config.continuous_exploration_noise;
    }

    std::vector<float> ActContinuousLocked(StepId step, size_t index, bool explore) {
        Pending& entry = FindPending(step);
        OutputState& out = outputs[index];
        auto existing = entry.continuous_actions.find(index);
        if (existing != entry.continuous_actions.end()) return existing->second;

        const size_t n = out.spec->size;
        std::vector<float> normalized(n);
        std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
        bool random = !entry.encoded || (explore && out.acts < config.continuous_random_steps);
        if (explore) out.acts++;
        if (random) {
            for (auto& a : normalized) a = uniform(rng);
        } else {
            const ActorCritic& ac = *actor_critics[static_cast<size_t>(out.continuous_slot)];
            nn::Matrix greedy = ac.Act(Row(entry.fused));
            std::normal_distribution<float> noise(0.0f, config.continuous_exploration_noise);
            for (size_t i = 0; i < n; ++i) {
                float a = greedy(0, i);
                if (explore && config.continuous_exploration_noise > 0.0f) a += noise(rng);
                normalized[i] = std::clamp(a, -1.0f, 1.0f);
            }
        }
        std::vector<float> action = ToBounds(*out.spec, normalized.data());
        entry.continuous_actions[index] = action;
        if (entry.encoded) {
            entry.sample->continuous[static_cast<size_t>(out.continuous_slot)].action = std::move(normalized);
            last_acted_step = step;
        }
        return action;
    }

    static bool HasActions(const Pending& entry) {
        return !entry.actions.empty() || !entry.continuous_actions.empty();
    }

    /// A transition becomes trainable once its next state or its end is known
    void CompleteTransition(Pending& entry) {
        if (!HasActions(entry) || !entry.encoded) return;
        if (!entry.transition_complete) {
            entry.transition_complete = true;
            for (const auto& [index, action] : entry.actions) outputs[index].samples++;
            for (const auto& [index, action] : entry.continuous_actions) outputs[index].samples++;
        }
        MakeTrainable(entry.sample);
    }

    void LearnContinuousReward(Pending& entry, OutputState& out, float reward) {
        entry.sample->continuous[static_cast<size_t>(out.continuous_slot)].reward += reward;
        if (entry.transition_complete) MakeTrainable(entry.sample);
    }

    void LearnReward(Pending& entry, OutputState& out, size_t index, float reward) {
        size_t action = entry.actions.at(index);
        entry.sample->targets[static_cast<size_t>(out.head)].reward += reward;
        patterns->ReinforceAction(entry.pattern, out.action_ids[action], reward);
        if (entry.transition_complete) MakeTrainable(entry.sample);
    }

    Prediction PredictLocked(Pending& entry, size_t index) const {
        const OutputState& out = outputs[index];
        Prediction prediction;
        prediction.output = out.spec->name;
        prediction.type = out.spec->type;
        if (!entry.encoded) return prediction;

        switch (out.spec->type) {
            case OutputType::CLASS: {
                auto neural = NeuralClass(entry, out);
                auto pattern = PatternClass(entry, out, out.UseShort(true));
                if (!neural && !pattern) return prediction;
                float w = BlendWeight(out, neural.has_value(), pattern.has_value(), true);
                std::vector<float> p(out.classes.size(), 0.0f);
                for (size_t c = 0; c < p.size(); ++c) {
                    p[c] = (neural ? w * (*neural)[c] : 0.0f) + (pattern ? (1.0f - w) * (*pattern)[c] : 0.0f);
                }
                size_t best = ArgMax(p);
                prediction.available = true;
                prediction.classes = out.classes;
                prediction.probabilities = p;
                prediction.label = out.classes[best];
                prediction.confidence = p[best];
                prediction.neural_weight = w;
                entry.predicted[index] = best;
                break;
            }
            case OutputType::VALUE:
            case OutputType::NEXT_OBSERVATION: {
                bool next = out.spec->type == OutputType::NEXT_OBSERVATION;
                auto neural = next ? NeuralNext(entry, out) : NeuralValue(entry, out);
                auto pattern = next ? patterns->PredictNextObserved(entry.pattern)
                                    : PatternValue(entry, out, out.UseShort(false));
                if (!neural && !pattern) return prediction;
                float w = BlendWeight(out, neural.has_value(), pattern.has_value(), false);
                size_t size = neural ? neural->size() : pattern->size();
                std::vector<float> values(size, kNaN);
                for (size_t i = 0; i < size; ++i) {
                    float n = neural ? (*neural)[i] : kNaN;
                    float q = pattern ? (*pattern)[i] : kNaN;
                    if (std::isnan(n)) values[i] = q;
                    else if (std::isnan(q)) values[i] = n;
                    else values[i] = w * n + (1.0f - w) * q;
                }
                const Quality& pattern_quality = out.Pattern(false);
                float error = (neural && out.neural.Measured() ? w * out.neural.value : 0.0f) +
                              (pattern && pattern_quality.Measured() ? (1.0f - w) * pattern_quality.value : 0.0f);
                prediction.available = true;
                prediction.values = std::move(values);
                prediction.confidence = 1.0f / (1.0f + error);
                prediction.neural_weight = w;
                break;
            }
            case OutputType::ANOMALY:
                prediction.available = true;
                prediction.score = entry.anomaly;
                prediction.neural_weight = entry.anomaly_neural_weight;
                prediction.confidence = std::min(1.0f, static_cast<float>(pattern_baseline.size()) /
                                                           static_cast<float>(config.anomaly_baseline));
                break;
            case OutputType::CONTINUOUS_ACTION: {
                const ActorCritic& ac = *actor_critics[static_cast<size_t>(out.continuous_slot)];
                nn::Matrix state = Row(entry.fused);
                nn::Matrix greedy = ac.Act(state);
                prediction.available = true;
                prediction.values = ToBounds(*out.spec, greedy.Data());
                prediction.score = ac.Value(state, greedy)[0];
                prediction.confidence = 1.0f - std::min(1.0f, ContinuousExploration(out));
                prediction.neural_weight = ac.Updates() > 0 ? 1.0f : 0.0f;
                break;
            }
            case OutputType::ACTION: {
                size_t greedy = GreedyAction(entry, out, nullptr);
                float epsilon = Exploration(out);
                size_t n = out.classes.size();
                prediction.available = true;
                prediction.classes = out.classes;
                prediction.probabilities.assign(n, epsilon / static_cast<float>(n));
                prediction.probabilities[greedy] += 1.0f - epsilon;
                prediction.label = out.classes[greedy];
                prediction.confidence = prediction.probabilities[greedy];
                if (auto q = NeuralQ(entry, out)) {
                    prediction.values = *q;
                    prediction.neural_weight = 1.0f;
                }
                break;
            }
        }
        return prediction;
    }

    // ------------------------------------------------------------ feedback

    size_t ClassIndex(OutputState& out, const Value& value) {
        std::string name;
        if (const auto* text = std::get_if<std::string>(&value)) {
            name = *text;
        } else if (const double* number = std::get_if<double>(&value)) {
            if (!out.learn_classes) {
                if (*number < 0 || *number >= out.classes.size() || std::floor(*number) != *number) {
                    throw std::invalid_argument("Label " + FormatNumber(*number) + " for output '" +
                                                out.spec->name + "' is not a class index in [0, " +
                                                std::to_string(out.classes.size()) + ")");
                }
                return static_cast<size_t>(*number);
            }
            name = FormatNumber(*number);
        } else {
            throw std::invalid_argument("Label for output '" + out.spec->name +
                                        "' must be a class name or index, got " + ValueTypeName(value));
        }
        auto it = out.class_index.find(name);
        if (it != out.class_index.end()) return it->second;
        if (!out.learn_classes) {
            throw std::invalid_argument("Output '" + out.spec->name + "' has no class '" + name + "'");
        }
        if (out.classes.size() >= config.max_classes) {
            throw std::invalid_argument("Output '" + out.spec->name + "' already has max_classes (" +
                                        std::to_string(config.max_classes) + ") classes");
        }
        size_t index = out.classes.size();
        out.classes.push_back(name);
        out.class_index.emplace(name, index);
        if (network) network->GrowClasses(out.head, out.classes.size());
        Log(LogLevel::INFO, "Output '" + out.spec->name + "': new class '" + name + "'");
        return index;
    }

    void LearnLabel(Pending& entry, OutputState& out, size_t cls) {
        // Score both sides on this sample before learning from it
        if (auto neural = NeuralClass(entry, out)) {
            float correct = ArgMax(*neural) == cls ? 1.0f : 0.0f;
            out.neural.Add(correct, config.quality_rate);
            ObserveLoss(out, 1.0f - correct);
        }
        if (!out.classes.empty()) {
            auto votes = patterns->ClassVotes(entry.point, static_cast<size_t>(out.class_slot), out.classes.size(),
                                              config.neighbors, config.pattern_label_horizon);
            if (votes.short_term) out.pattern_short.Add(ArgMax(*votes.short_term) == cls ? 1.0f : 0.0f, config.quality_rate);
            if (votes.long_term) out.pattern_long.Add(ArgMax(*votes.long_term) == cls ? 1.0f : 0.0f, config.quality_rate);
        }

        HeadTarget& target = entry.sample->targets[out.head];
        target.label = static_cast<int>(cls);
        target.negative = -1;
        patterns->AddLabel(entry.pattern, out.class_slot, cls);
        if (entry.sample->balance_class < 0) {
            int old = entry.sample->balance_class;
            entry.sample->balance_class = static_cast<int>(cls);
            replay.Reclassify(entry.sample, old);
        }
        out.samples++;
        MakeTrainable(entry.sample);
    }

    void FeedbackLocked(StepId step, const std::string& name, const Value& value) {
        auto fb = feedback.find(name);
        if (fb == feedback.end()) {
            throw std::invalid_argument("Unknown feedback '" + name + "'");
        }
        OutputState& out = outputs[fb->second.output];
        Pending& entry = FindPending(step);
        FeedbackType type = fb->second.spec->type;

        // Check the value before anything is learned
        std::vector<float> target_values;
        bool correct = false;
        float reward = 0.0f;
        if (type == FeedbackType::TARGET) {
            if (const double* number = std::get_if<double>(&value)) {
                target_values.assign(1, static_cast<float>(*number));
            } else if (const auto* vector = std::get_if<std::vector<float>>(&value)) {
                target_values = *vector;
            } else {
                throw std::invalid_argument("Feedback '" + name + "' (TARGET) expects a number or vector, got " +
                                            ValueTypeName(value));
            }
            if (target_values.size() != out.spec->size) {
                throw std::invalid_argument("Feedback '" + name + "' expects " + std::to_string(out.spec->size) +
                                            " values, got " + std::to_string(target_values.size()));
            }
        } else if (type == FeedbackType::OUTCOME) {
            const double* number = std::get_if<double>(&value);
            if (!number) {
                throw std::invalid_argument("Feedback '" + name + "' (OUTCOME) expects a number (nonzero = correct), got " +
                                            ValueTypeName(value));
            }
            correct = *number != 0.0;
        } else if (type == FeedbackType::REWARD) {
            const double* number = std::get_if<double>(&value);
            if (!number || !std::isfinite(*number)) {
                throw std::invalid_argument("Feedback '" + name + "' (REWARD) expects a finite number, got " +
                                            (number ? FormatNumber(*number) : std::string(ValueTypeName(value))));
            }
            reward = static_cast<float>(*number);
        }
        if (out.spec->type == OutputType::ACTION && entry.actions.count(fb->second.output) == 0) {
            throw std::logic_error("Feedback '" + name + "' for step " + std::to_string(step) +
                                   ": no action of output '" + out.spec->name + "' was chosen for it (call Act first)");
        }
        if (out.spec->type == OutputType::CONTINUOUS_ACTION && entry.continuous_actions.count(fb->second.output) == 0) {
            throw std::logic_error("Feedback '" + name + "' for step " + std::to_string(step) +
                                   ": no action of output '" + out.spec->name +
                                   "' was chosen for it (call ActContinuous first)");
        }

        if (!entry.encoded) {
            ignored_feedback++;
            return;
        }
        feedback_count++;

        if (out.spec->type == OutputType::ACTION) {
            LearnReward(entry, out, fb->second.output, type == FeedbackType::OUTCOME ? (correct ? 1.0f : -1.0f) : reward);
            return;
        }
        if (out.spec->type == OutputType::CONTINUOUS_ACTION) {
            LearnContinuousReward(entry, out, type == FeedbackType::OUTCOME ? (correct ? 1.0f : -1.0f) : reward);
            return;
        }

        switch (type) {
            case FeedbackType::LABEL:
                LearnLabel(entry, out, ClassIndex(out, value));
                break;
            case FeedbackType::OUTCOME: {
                size_t index = fb->second.output;
                auto it = entry.predicted.find(index);
                std::optional<size_t> predicted;
                if (it != entry.predicted.end()) {
                    predicted = it->second;
                } else {
                    Prediction p = PredictLocked(entry, index);
                    if (p.available) predicted = ArgMax(p.probabilities);
                }
                if (!predicted) {
                    ignored_feedback++;
                    feedback_count--;
                    return;
                }
                if (correct) {
                    LearnLabel(entry, out, *predicted);
                } else {
                    HeadTarget& target = entry.sample->targets[out.head];
                    if (target.label < 0) target.negative = static_cast<int>(*predicted);
                    patterns->AddNegative(entry.pattern, out.class_slot, *predicted);
                    out.samples++;
                    MakeTrainable(entry.sample);
                }
                break;
            }
            case FeedbackType::TARGET: {
                for (size_t i = 0; i < target_values.size(); ++i) {
                    if (std::isfinite(target_values[i])) out.target_stats[i].Add(target_values[i]);
                    else target_values[i] = kNaN;
                }
                auto stats = [&](size_t i) -> const RunningStats& { return out.target_stats[i]; };
                auto votes = patterns->ValueVotes(entry.point, static_cast<size_t>(out.value_slot), config.neighbors,
                                                  config.pattern_label_horizon);
                ScoreRegression(out, NeuralValue(entry, out), votes.short_term, votes.long_term, target_values, stats);
                entry.sample->targets[out.head].values = target_values;
                patterns->AddValue(entry.pattern, out.value_slot, target_values);
                out.samples++;
                MakeTrainable(entry.sample);
                break;
            }
            case FeedbackType::REWARD:
                break;
        }
    }

    // ------------------------------------------------------------ observe

    StepId ObserveLocked(const Observation& observation) {
        auto fused = encoder.Encode(observation);
        StepId step = ++steps;
        if (!patterns && encoder.Ready()) BuildModels();

        Pending entry;
        entry.step = step;
        if (fused && patterns) {
            entry.encoded = true;
            entry.fused = std::move(*fused);
            entry.observed = Observed(observation);
            entry.point = patterns->Project(entry.fused);

            auto placement = patterns->Insert(entry.point, step, entry.observed);
            entry.pattern = placement.id;
            // Distance to the nearest pattern before this state was stored
            if (has_anomaly) ScoreAnomaly(entry, placement.nearest_distance);

            entry.sample = std::make_shared<Sample>();
            entry.sample->step = step;
            entry.sample->SetFused(entry.fused);
            entry.sample->targets.resize(network ? network->HeadCount() : 0);
            entry.sample->continuous.resize(actor_critics.size());

            if (has_anomaly) {
                entry.sample->reconstruct = true;
                anomaly_samples++;
                MakeTrainable(entry.sample);
            }

            if (previous_step) {
                Pending* previous = TryFindPending(*previous_step);
                if (previous && previous->encoded) {
                    patterns->RecordTransition(previous->pattern, entry.pattern);
                    if (next_output >= 0) LearnNext(*previous, entry.observed);
                    if (HasActions(*previous)) {
                        previous->sample->SetNext(*entry.sample);
                        previous->sample->next_sample = entry.sample;
                        CompleteTransition(*previous);
                    }
                }
            }
            previous_step = step;
        } else {
            previous_step.reset();
        }

        pending.push_back(std::move(entry));
        while (pending.size() > config.feedback_timeout) pending.pop_front();

        if (++since_training >= config.train_every) {
            since_training = 0;
            Train();
        }
        if (patterns && steps - last_snapshot >= config.snapshot_interval && ModelsFinite()) TakeSnapshot();
        return step;
    }

    Pending* TryFindPending(StepId step) {
        if (pending.empty() || step < pending.front().step || step > pending.back().step) return nullptr;
        return &pending[step - pending.front().step];
    }

    void LearnNext(Pending& previous, const std::vector<float>& observed) {
        OutputState& out = outputs[static_cast<size_t>(next_output)];
        auto stats = [&](size_t i) -> const RunningStats& { return ObservedStats(i); };
        ScoreRegression(out, NeuralNext(previous, out), std::nullopt,
                        patterns->PredictNextObserved(previous.pattern), observed, stats);
        previous.sample->targets[out.head].values = observed;
        out.samples++;
        MakeTrainable(previous.sample);
    }

    void ScoreAnomaly(Pending& entry, float distance) {
        float pattern_percentile = std::isfinite(distance) ? Percentile(pattern_baseline, distance) : 0.5f;
        float neural_percentile = 0.5f;
        float weight = 0.0f;
        float error = autoencoder->Errors(Row(entry.fused))[0];
        if (anomaly_samples >= config.cold_start_samples) {
            neural_percentile = Percentile(neural_baseline, error);
            weight = 0.5f;
        }
        entry.anomaly = weight * neural_percentile + (1.0f - weight) * pattern_percentile;
        entry.anomaly_neural_weight = weight;

        if (std::isfinite(distance)) pattern_baseline.push_back(distance);
        neural_baseline.push_back(error);
        if (anomaly_samples >= config.cold_start_samples) {
            for (auto& out : outputs) {
                if (out.spec->type == OutputType::ANOMALY) ObserveLoss(out, error);
            }
        }
        while (pattern_baseline.size() > config.anomaly_baseline) pattern_baseline.pop_front();
        while (neural_baseline.size() > config.anomaly_baseline) neural_baseline.pop_front();
    }

    // -------------------------------------------------------------- train

    /// Multi-step return of one transition: rewards of up to
    /// action_return_steps consecutive steps of the episode, discounted.
    /// Stops early at a terminal step, where the chain is unknown, and
    /// before a random (exploring) action, so the return estimates the
    /// greedy policy (as in Watkins's Q(lambda))
    struct Return {
        float rewards{0.0f};
        float discount{1.0f};             // discount of the bootstrap value
        const Sample* bootstrap{nullptr};  // sample whose next state is bootstrapped (nullptr = terminal)
    };

    Return MultiStepReturn(const Sample& first, size_t h) const {
        Return result;
        const Sample* current = &first;
        std::shared_ptr<Sample> hold;  // keeps the current link alive
        for (size_t k = 0;; ++k) {
            result.rewards += result.discount * current->targets[h].reward;
            result.discount *= config.discount;
            if (current->terminal) {
                result.bootstrap = nullptr;
                return result;
            }
            auto next = current->next_sample.lock();
            bool usable = next && next->targets[h].action >= 0 && !next->targets[h].exploratory &&
                          (next->has_next || next->terminal);
            if (k + 1 >= config.action_return_steps || !usable) {
                result.bootstrap = current;
                return result;
            }
            hold = std::move(next);
            current = hold.get();
        }
    }

    /// Double DQN targets for one ACTION head: the multi-step return plus
    /// its discount x the target network's value of the action the online
    /// network picks in the state after the last step (no future part
    /// after a terminal step)
    void QTargets(const OutputState& out, const std::vector<SamplePtr>& batch,
                  std::vector<std::vector<HeadTarget>>& owned,
                  std::vector<std::vector<const HeadTarget*>>& targets) const {
        const auto h = static_cast<size_t>(out.head);
        const size_t width = encoder.OutputSize();
        const size_t n = out.classes.size();
        std::vector<size_t> rows, bootstrap;
        std::map<size_t, Return> returns;
        for (size_t r = 0; r < batch.size(); ++r) {
            const Sample& sample = *batch[r];
            if (sample.targets[h].action < 0 || !(sample.has_next || sample.terminal)) continue;
            rows.push_back(r);
            returns[r] = MultiStepReturn(sample, h);
            if (returns[r].bootstrap) bootstrap.push_back(r);
        }
        if (rows.empty()) return;

        std::map<size_t, float> future;
        if (!bootstrap.empty()) {
            nn::Matrix next(bootstrap.size(), width);
            for (size_t i = 0; i < bootstrap.size(); ++i) {
                returns[bootstrap[i]].bootstrap->FillNext(next.Data() + i * width, width);
            }
            nn::Matrix online = network->Infer(next, h);
            nn::Matrix target = target_network->Infer(next, h);
            for (size_t i = 0; i < bootstrap.size(); ++i) {
                size_t best = 0;
                for (size_t a = 1; a < n; ++a) {
                    if (online(i, a) > online(i, best)) best = a;
                }
                future[bootstrap[i]] = target(i, best);
            }
        }
        for (size_t r : rows) {
            const HeadTarget& stored = batch[r]->targets[h];
            HeadTarget& t = owned[h][r];
            t = HeadTarget();
            t.values.assign(n, kNaN);
            const Return& g = returns[r];
            auto it = future.find(r);
            t.values[static_cast<size_t>(stored.action)] =
                g.rewards + (it != future.end() ? g.discount * it->second : 0.0f);
            targets[h][r] = &t;
        }
    }

    /// One TD3 update for a CONTINUOUS_ACTION output from the batch's
    /// complete transitions; returns the critic loss (NaN if none)
    float TrainContinuous(size_t slot, const std::vector<SamplePtr>& batch) {
        const size_t width = encoder.OutputSize();
        const size_t n = outputs[continuous_outputs[slot]].spec->size;
        std::vector<const Sample*> rows;
        for (const auto& sample : batch) {
            if (!sample->continuous[slot].action.empty() && (sample->has_next || sample->terminal)) {
                rows.push_back(sample.get());
            }
        }
        if (rows.empty()) return kNaN;
        ActorCritic::Batch b{nn::Matrix(rows.size(), width), nn::Matrix(rows.size(), n),
                             std::vector<float>(rows.size()), nn::Matrix(rows.size(), width),
                             std::vector<float>(rows.size())};
        for (size_t r = 0; r < rows.size(); ++r) {
            const Sample& sample = *rows[r];
            sample.Fill(b.states.Data() + r * width, width);
            std::copy(sample.continuous[slot].action.begin(), sample.continuous[slot].action.end(),
                      b.actions.Data() + r * n);
            b.rewards[r] = sample.continuous[slot].reward;
            if (sample.terminal) {
                b.discounts[r] = 0.0f;
            } else {
                sample.FillNext(b.next_states.Data() + r * width, width);
                b.discounts[r] = config.discount;
            }
        }
        return actor_critics[slot]->Train(b, rng).critic;
    }

    void Train() {
        if (replay.Size() == 0 || (!network && !autoencoder && actor_critics.empty())) return;

        // New samples: not yet trained ones, and while recovering from a
        // drift also random recent ones, up to the share; the rest replay
        float share = recovering_outputs > 0 ? config.drift_new_sample_share : config.new_sample_share;
        auto wanted = static_cast<size_t>(std::lround(config.batch_size * share));
        size_t from_newest = std::min(newest.size(), wanted);
        std::vector<SamplePtr> batch(newest.end() - static_cast<std::ptrdiff_t>(from_newest), newest.end());
        newest.clear();
        if (recovering_outputs > 0 && !recent.empty()) {
            std::uniform_int_distribution<size_t> pick(0, recent.size() - 1);
            for (size_t tries = 0; batch.size() < wanted && tries < 4 * config.batch_size; ++tries) {
                if (auto sample = recent[pick(rng)].lock()) batch.push_back(std::move(sample));
            }
        }
        auto drawn = replay.Draw(config.batch_size - batch.size());
        batch.insert(batch.end(), drawn.begin(), drawn.end());
        if (batch.empty()) return;

        const size_t width = encoder.OutputSize();
        float total_loss = 0.0f;
        bool monitored = false;  // a loss that measures progress was added
        bool diverged = false;
        auto bad = [&](float loss) { return std::isinf(loss) || loss > config.divergence_loss; };

        if (network) {
            nn::Matrix inputs(batch.size(), width);
            const size_t heads = network->HeadCount();
            std::vector<std::vector<HeadTarget>> owned(heads, std::vector<HeadTarget>(batch.size()));
            std::vector<std::vector<const HeadTarget*>> targets(heads, std::vector<const HeadTarget*>(batch.size()));
            for (size_t r = 0; r < batch.size(); ++r) {
                batch[r]->Fill(inputs.Data() + r * width, width);
            }
            for (const auto& out : outputs) {
                if (out.head < 0 || out.spec->type == OutputType::ACTION) continue;
                auto h = static_cast<size_t>(out.head);
                for (size_t r = 0; r < batch.size(); ++r) {
                    const HeadTarget& stored = batch[r]->targets[h];
                    if (stored.Empty()) continue;
                    HeadTarget& t = owned[h][r];
                    t.label = stored.label;
                    t.negative = stored.negative;
                    if (!stored.values.empty()) {
                        t.values.resize(stored.values.size());
                        for (size_t i = 0; i < stored.values.size(); ++i) {
                            t.values[i] = out.spec->type == OutputType::VALUE ? ToZ(stored.values[i], out.target_stats[i])
                                                                               : ToZ(stored.values[i], ObservedStats(i));
                        }
                    }
                    targets[h][r] = &t;
                }
            }
            for (const auto& out : outputs) {
                if (out.spec->type == OutputType::ACTION) QTargets(out, batch, owned, targets);
            }
            std::vector<float> scale(heads, 1.0f);
            for (size_t h = 0; h < heads; ++h) {
                if (!std::isnan(head_loss[h])) scale[h] = 1.0f / std::max(head_loss[h], 1e-3f);
            }
            auto losses = network->TrainBatch(inputs, targets, scale, config.gradient_clip_norm);
            for (size_t h = 0; h < heads; ++h) diverged = diverged || bad(losses[h]);
            for (size_t h = 0; h < heads; ++h) {
                if (std::isnan(losses[h])) continue;
                head_loss[h] = std::isnan(head_loss[h]) ? losses[h] : 0.95f * head_loss[h] + 0.05f * losses[h];
                // Q-learning targets move with the network, so their loss is
                // no measure of progress; it does not drive the plateau check
                if (network->Head(h).kind != HeadKind::Q_VALUES) {
                    total_loss += losses[h];
                    monitored = true;
                }
            }
        }

        if (autoencoder) {
            std::vector<const Sample*> rows;
            for (const auto& sample : batch) {
                if (sample->reconstruct) rows.push_back(sample.get());
            }
            if (!rows.empty()) {
                nn::Matrix inputs(rows.size(), width);
                for (size_t r = 0; r < rows.size(); ++r) rows[r]->Fill(inputs.Data() + r * width, width);
                float loss = autoencoder->TrainBatch(inputs, config.gradient_clip_norm);
                diverged = diverged || bad(loss) || std::isnan(loss);
                total_loss += loss;
                monitored = true;
            }
        }

        for (size_t slot = 0; slot < actor_critics.size(); ++slot) {
            // Q-learning targets move with the critics, so the loss is not
            // compared with divergence_loss (rewards set its scale); only
            // non-finite values roll back
            // The first update uses this batch; each further one draws its own
            for (size_t update = 0; update < config.continuous_updates_per_step; ++update) {
                float loss = TrainContinuous(slot, update == 0 ? batch : replay.Draw(config.batch_size));
                diverged = diverged || std::isinf(loss);
            }
        }

        training_steps++;
        if (diverged || !ModelsFinite()) {
            Rollback();
            return;
        }
        if (target_network && training_steps % config.target_sync == 0) {
            target_network = std::make_unique<AgentNetwork>(*network);
        }
        if (!monitored) return;
        plateau_sum += total_loss;
        if (++plateau_count >= config.plateau_interval) {
            float mean = static_cast<float>(plateau_sum / static_cast<double>(plateau_count));
            plateau_checks++;
            float rate = schedule.NextRate(plateau_checks, config.learning_rate, learning_rate, mean);
            if (rate < learning_rate && ShouldGrow()) {
                // A plateau while the network trails the pattern side: add
                // capacity instead of lowering the rate
                Grow();
                schedule = nn::ReduceOnPlateau(config.plateau_factor, config.plateau_patience, config.min_learning_rate);
            } else if (rate != learning_rate) {
                learning_rate = rate;
                ApplyLearningRate();
                Log(LogLevel::INFO, "Learning rate reduced to " + FormatNumber(rate));
            }
            plateau_sum = 0.0;
            plateau_count = 0;
        }
    }

    // ---------------------------------------------------------- checkpoint

    void SaveState(std::ostream& out) const {
        encoder.SaveState(out);

        // Outputs
        for (const auto& out_state : outputs) {
            WriteStrings(out, out_state.classes);
            io::WriteSize(out, out_state.action_ids.size());
            for (PatternID id : out_state.action_ids) io::Write<uint64_t>(out, id.value());
            io::Write<uint64_t>(out, out_state.acts);
            io::WriteSize(out, out_state.target_stats.size());
            for (const auto& stats : out_state.target_stats) stats.Save(out);
            io::Write<uint64_t>(out, out_state.samples);
            SaveQuality(out, out_state.neural);
            SaveQuality(out, out_state.pattern_short);
            SaveQuality(out, out_state.pattern_long);
            out_state.drift.Save(out);
            io::Write<uint8_t>(out, out_state.recovering ? 1 : 0);
            io::Write<uint64_t>(out, out_state.recovery_samples);
            io::Write<double>(out, out_state.drift_baseline);
            SaveQuality(out, out_state.recovery_loss);
            io::Write<uint32_t>(out, out_state.drift_events);
        }

        // Models
        io::Write<uint8_t>(out, patterns ? 1 : 0);
        if (patterns) {
            for (const AgentNetwork* n : {network.get(), target_network.get(), snapshot_network.get()}) {
                io::Write<uint8_t>(out, n ? 1 : 0);
                if (n) n->Serialize(out);
            }
            for (const AnomalyAutoencoder* a : {autoencoder.get(), snapshot_autoencoder.get()}) {
                io::Write<uint8_t>(out, a ? 1 : 0);
                if (a) a->Serialize(out);
            }
            for (const auto* list : {&actor_critics, &snapshot_actor_critics}) {
                io::WriteSize(out, list->size());
                for (const auto& ac : *list) ac->Serialize(out);
            }
            patterns->SaveState(out);
        }

        // Samples, each once; everything else refers to them by number
        std::map<const Sample*, uint64_t> ids;
        std::vector<const Sample*> ordered;
        auto add = [&](const Sample* sample) {
            if (sample && ids.emplace(sample, ordered.size() + 1).second) ordered.push_back(sample);
        };
        replay.ForEach([&](const SamplePtr& sample) { add(sample.get()); });
        for (const auto& entry : pending) add(entry.sample.get());
        for (const auto& sample : newest) add(sample.get());
        auto id_of = [&](const Sample* sample) -> uint64_t {
            auto it = ids.find(sample);
            return it == ids.end() ? 0 : it->second;
        };
        io::WriteSize(out, ordered.size());
        for (const Sample* sample : ordered) sample->Save(out);
        for (const Sample* sample : ordered) io::Write<uint64_t>(out, id_of(sample->next_sample.lock().get()));
        replay.SaveState(out, id_of);
        io::WriteSize(out, newest.size());
        for (const auto& sample : newest) io::Write<uint64_t>(out, id_of(sample.get()));
        io::WriteSize(out, recent.size());
        for (const auto& sample : recent) io::Write<uint64_t>(out, id_of(sample.lock().get()));

        // Pending steps
        io::WriteSize(out, pending.size());
        for (const auto& entry : pending) {
            io::Write<uint64_t>(out, entry.step);
            io::Write<uint8_t>(out, entry.encoded ? 1 : 0);
            WriteVector(out, entry.fused);
            WriteVector(out, entry.point);
            WriteVector(out, entry.observed);
            io::Write<uint64_t>(out, entry.pattern.value());
            io::Write<uint64_t>(out, id_of(entry.sample.get()));
            SaveIndexMap(out, entry.predicted);
            SaveIndexMap(out, entry.actions);
            io::WriteSize(out, entry.continuous_actions.size());
            for (const auto& [index, action] : entry.continuous_actions) {
                io::WriteSize(out, index);
                WriteVector(out, action);
            }
            io::Write<uint8_t>(out, entry.transition_complete ? 1 : 0);
            io::Write<float>(out, entry.anomaly);
            io::Write<float>(out, entry.anomaly_neural_weight);
        }

        // Counters, schedules, random state
        WriteVector(out, head_loss);
        schedule.SaveState(out);
        io::Write<float>(out, learning_rate);
        io::Write<double>(out, plateau_sum);
        io::WriteSize(out, plateau_count);
        io::WriteSize(out, plateau_checks);
        SaveOptional(out, previous_step);
        SaveOptional(out, last_acted_step);
        WriteVector(out, std::vector<float>(pattern_baseline.begin(), pattern_baseline.end()));
        WriteVector(out, std::vector<float>(neural_baseline.begin(), neural_baseline.end()));
        for (uint64_t v : {anomaly_samples, steps, feedback_count, ignored_feedback, training_steps, episodes,
                           drift_events, last_snapshot, rollbacks, widenings}) {
            io::Write<uint64_t>(out, v);
        }
        io::WriteSize(out, since_training);
        io::WriteSize(out, recovering_outputs);
        io::Write<uint8_t>(out, growth_blocked ? 1 : 0);
        io::Write<float>(out, gap_at_widening);
        io::WriteSize(out, metrics_every);
        SaveRandom(out, rng);
    }

    void LoadState(std::istream& in) {
        encoder.LoadState(in);

        for (auto& out_state : outputs) {
            out_state.classes = ReadStrings(in);
            out_state.class_index.clear();
            for (size_t c = 0; c < out_state.classes.size(); ++c) out_state.class_index[out_state.classes[c]] = c;
            out_state.action_ids.resize(io::ReadSize(in, 1u << 20));
            for (auto& id : out_state.action_ids) id = PatternID(io::Read<uint64_t>(in));
            out_state.acts = io::Read<uint64_t>(in);
            out_state.target_stats.resize(io::ReadSize(in, 1u << 20));
            for (auto& stats : out_state.target_stats) stats.Load(in);
            out_state.samples = io::Read<uint64_t>(in);
            out_state.neural = LoadQuality(in);
            out_state.pattern_short = LoadQuality(in);
            out_state.pattern_long = LoadQuality(in);
            out_state.drift.Load(in);
            out_state.recovering = io::Read<uint8_t>(in) != 0;
            out_state.recovery_samples = io::Read<uint64_t>(in);
            out_state.drift_baseline = io::Read<double>(in);
            out_state.recovery_loss = LoadQuality(in);
            out_state.drift_events = io::Read<uint32_t>(in);
        }

        PatternID::ValueType max_id = 0;
        for (const auto& out_state : outputs) {
            for (PatternID id : out_state.action_ids) max_id = std::max(max_id, id.value());
        }
        if (io::Read<uint8_t>(in)) {
            if (!encoder.Ready()) throw std::runtime_error("Agent checkpoint: models without known input sizes");
            auto heads = PrepareLayout();
            for (auto* n : {&network, &target_network, &snapshot_network}) {
                *n = io::Read<uint8_t>(in) ? AgentNetwork::Deserialize(in) : nullptr;
            }
            for (auto* a : {&autoencoder, &snapshot_autoencoder}) {
                *a = io::Read<uint8_t>(in) ? AnomalyAutoencoder::Deserialize(in) : nullptr;
            }
            for (auto* list : {&actor_critics, &snapshot_actor_critics}) {
                list->resize(io::ReadSize(in, 1u << 16));
                for (auto& ac : *list) ac = ActorCritic::Deserialize(in);
            }
            if (actor_critics.size() != continuous_outputs.size() ||
                snapshot_actor_critics.size() != continuous_outputs.size()) {
                throw std::runtime_error("Agent checkpoint: actor-critics do not match the spec");
            }
            for (size_t slot = 0; slot < actor_critics.size(); ++slot) {
                if (actor_critics[slot]->StateSize() != encoder.OutputSize() ||
                    actor_critics[slot]->ActionSize() != outputs[continuous_outputs[slot]].spec->size) {
                    throw std::runtime_error("Agent checkpoint: actor-critic sizes do not match the spec");
                }
            }
            if (network && network->HeadCount() != heads.size()) {
                throw std::runtime_error("Agent checkpoint: network heads do not match the spec");
            }
            patterns = MakePatternMemory();
            patterns->LoadState(in);
            max_id = std::max(max_id, patterns->MaxPatternId());
        }
        // New pattern IDs must not collide with restored ones
        PatternID::ReserveThrough(max_id);

        std::vector<SamplePtr> samples(io::ReadSize(in, 1u << 30));
        for (auto& sample : samples) sample = Sample::Load(in);
        auto lookup = [&](uint64_t id) -> SamplePtr {
            if (id == 0) return nullptr;
            if (id > samples.size()) throw std::runtime_error("Agent checkpoint: bad sample reference");
            return samples[id - 1];
        };
        for (auto& sample : samples) {
            if (auto next = lookup(io::Read<uint64_t>(in))) sample->next_sample = next;
        }
        replay.LoadState(in, [&](uint64_t id) {
            auto sample = lookup(id);
            if (!sample) throw std::runtime_error("Agent checkpoint: bad replay entry");
            return sample;
        });
        newest.clear();
        size_t count = io::ReadSize(in, 1u << 30);
        for (size_t i = 0; i < count; ++i) newest.push_back(lookup(io::Read<uint64_t>(in)));
        recent.clear();
        count = io::ReadSize(in, 1u << 30);
        for (size_t i = 0; i < count; ++i) recent.push_back(lookup(io::Read<uint64_t>(in)));

        pending.clear();
        count = io::ReadSize(in, 1u << 30);
        for (size_t i = 0; i < count; ++i) {
            Pending entry;
            entry.step = io::Read<uint64_t>(in);
            entry.encoded = io::Read<uint8_t>(in) != 0;
            entry.fused = ReadVector<float>(in);
            entry.point = ReadVector<float>(in);
            entry.observed = ReadVector<float>(in);
            entry.pattern = PatternID(io::Read<uint64_t>(in));
            entry.sample = lookup(io::Read<uint64_t>(in));
            entry.predicted = LoadIndexMap(in);
            entry.actions = LoadIndexMap(in);
            size_t continuous_count = io::ReadSize(in, 1u << 16);
            for (size_t c = 0; c < continuous_count; ++c) {
                size_t index = io::ReadSize(in, 1u << 16);
                entry.continuous_actions[index] = ReadVector<float>(in);
            }
            entry.transition_complete = io::Read<uint8_t>(in) != 0;
            entry.anomaly = io::Read<float>(in);
            entry.anomaly_neural_weight = io::Read<float>(in);
            pending.push_back(std::move(entry));
        }

        head_loss = ReadVector<float>(in);
        schedule.LoadState(in);
        learning_rate = io::Read<float>(in);
        plateau_sum = io::Read<double>(in);
        plateau_count = io::ReadSize(in);
        plateau_checks = io::ReadSize(in);
        previous_step = LoadOptional(in);
        last_acted_step = LoadOptional(in);
        auto baseline = ReadVector<float>(in);
        pattern_baseline.assign(baseline.begin(), baseline.end());
        baseline = ReadVector<float>(in);
        neural_baseline.assign(baseline.begin(), baseline.end());
        for (uint64_t* v : {&anomaly_samples, &steps, &feedback_count, &ignored_feedback, &training_steps, &episodes,
                            &drift_events, &last_snapshot, &rollbacks, &widenings}) {
            *v = io::Read<uint64_t>(in);
        }
        since_training = io::ReadSize(in);
        recovering_outputs = io::ReadSize(in);
        growth_blocked = io::Read<uint8_t>(in) != 0;
        gap_at_widening = io::Read<float>(in);
        metrics_every = io::ReadSize(in);
        LoadRandom(in, rng);
    }

    AgentStats StatsLocked() const {
        const Impl& m = *this;
        AgentStats stats;
        stats.steps = m.steps;
        stats.episodes = m.episodes;
        stats.feedback = m.feedback_count;
        stats.ignored_feedback = m.ignored_feedback;
        stats.training_steps = m.training_steps;
        stats.pending_steps = m.pending.size();
        stats.replay_size = m.replay.Size();
        stats.patterns = m.patterns ? m.patterns->Size() : 0;
        stats.associations = m.patterns ? m.patterns->Associations().GetAssociationCount() : 0;
        stats.patterns_pruned = m.patterns ? m.patterns->Pruned() : 0;
        stats.drift_events = m.drift_events;
        stats.rollbacks = m.rollbacks;
        stats.widenings = m.widenings;
        stats.recovering = m.recovering_outputs > 0;
        stats.memory_bytes = m.MemoryBytes();
        stats.memory_budget_bytes = m.config.memory_budget_bytes;
        stats.network_ready = m.patterns != nullptr;
        stats.fused_size = m.encoder.OutputSize();
        stats.trunk_width = m.network ? m.network->TrunkWidth() : 0;
        stats.learning_rate = m.learning_rate;
        for (const auto& out : m.outputs) {
            OutputStats o;
            o.name = out.spec->name;
            o.type = out.spec->type;
            o.samples = out.spec->type == OutputType::ANOMALY ? m.anomaly_samples : out.samples;
            o.classes = out.classes.size();
            o.neural_quality = out.neural.value;
            o.drift_events = out.drift_events;
            o.recovering = out.recovering;
            o.pattern_quality = out.Pattern(out.spec->type == OutputType::CLASS).value;
            if (out.spec->type == OutputType::ANOMALY) {
                o.neural_weight = m.anomaly_samples >= m.config.cold_start_samples ? 0.5f : 0.0f;
            } else if (out.spec->type == OutputType::ACTION) {
                o.neural_weight = out.samples >= m.config.cold_start_samples ? 1.0f : 0.0f;
                o.exploration = m.Exploration(out);
            } else if (out.spec->type == OutputType::CONTINUOUS_ACTION) {
                bool trained = !m.actor_critics.empty() &&
                               m.actor_critics[static_cast<size_t>(out.continuous_slot)]->Updates() > 0;
                o.neural_weight = trained ? 1.0f : 0.0f;
                o.exploration = m.ContinuousExploration(out);
            } else if (m.network && out.head >= 0) {
                o.neural_weight = m.BlendWeight(out, out.samples > 0,
                                                out.Pattern(out.spec->type == OutputType::CLASS).Measured(),
                                                out.spec->type == OutputType::CLASS);
            }
            stats.outputs.push_back(o);
        }
        return stats;
    }

    // -------------------------------------------------------------- state

    EnvironmentSpec spec;
    AgentConfig config;
    ObservationEncoder encoder;
    std::vector<OutputState> outputs;
    std::map<std::string, FeedbackState> feedback;
    size_t class_outputs{0};
    std::vector<size_t> value_sizes;
    int next_output{-1};
    bool has_anomaly{false};
    std::vector<size_t> continuous_outputs;  // output index of each CONTINUOUS_ACTION output
    std::vector<std::unique_ptr<ActorCritic>> actor_critics;           // one per CONTINUOUS_ACTION output
    std::vector<std::unique_ptr<ActorCritic>> snapshot_actor_critics;  // last healthy copies

    std::vector<size_t> observed_channels;
    std::vector<size_t> observed_offsets;
    size_t observed_size{0};

    std::unique_ptr<AgentNetwork> network;
    std::unique_ptr<AgentNetwork> target_network;  // ACTION heads (Double DQN)
    std::unique_ptr<AgentNetwork> snapshot_network;          // last healthy weights
    std::unique_ptr<AnomalyAutoencoder> snapshot_autoencoder;
    uint64_t last_snapshot{0};
    uint64_t rollbacks{0};
    uint64_t widenings{0};
    bool growth_blocked{false};
    float gap_at_widening{0.0f};
    std::function<void(const AgentStats&)> metrics_hook;
    size_t metrics_every{1000};
    bool has_actions{false};
    std::mt19937 rng;
    std::optional<StepId> last_acted_step;  // in the current episode
    uint64_t episodes{0};
    std::unique_ptr<AnomalyAutoencoder> autoencoder;
    std::unique_ptr<PatternMemory> patterns;
    ReplayBuffer replay;
    std::deque<SamplePtr> newest;
    std::deque<std::weak_ptr<Sample>> recent;  // last drift_recent_window trainable samples
    size_t recovering_outputs{0};
    uint64_t drift_events{0};
    std::vector<float> head_loss;

    nn::ReduceOnPlateau schedule;
    float learning_rate{0.0f};
    double plateau_sum{0.0};
    size_t plateau_count{0};
    size_t plateau_checks{0};

    mutable std::deque<Pending> pending;
    std::optional<StepId> previous_step;
    std::deque<float> pattern_baseline;
    std::deque<float> neural_baseline;
    uint64_t anomaly_samples{0};

    uint64_t steps{0};
    uint64_t feedback_count{0};
    uint64_t ignored_feedback{0};
    uint64_t training_steps{0};
    size_t since_training{0};

    Logger logger;
    mutable std::mutex mutex;
};

// ============================================================================
// Agent
// ============================================================================

Agent::Agent(EnvironmentSpec spec, AgentConfig config) {
    Validate(spec);
    bool has_numeric = std::any_of(spec.inputs.begin(), spec.inputs.end(), [](const InputSpec& input) {
        return input.type == InputType::NUMERIC || input.type == InputType::TIME_SERIES;
    });
    for (const auto& output : spec.outputs) {
        if (output.type == OutputType::NEXT_OBSERVATION && !has_numeric) {
            throw std::invalid_argument("Output '" + output.name +
                                        "' (NEXT_OBSERVATION) needs a NUMERIC or TIME_SERIES input");
        }
    }
    for (const auto& output : spec.outputs) {
        bool labelled = std::any_of(spec.feedback.begin(), spec.feedback.end(), [&](const FeedbackSpec& fb) {
            return fb.output == output.name && fb.type == FeedbackType::LABEL;
        });
        if (output.type == OutputType::CLASS && !labelled && output.classes_or_actions.empty()) {
            throw std::invalid_argument("Output '" + output.name +
                                        "' (CLASS) is trained by OUTCOME feedback only and must list its classes");
        }
    }
    CheckConfig(config);
    bool has_actions = std::any_of(spec.outputs.begin(), spec.outputs.end(), [](const OutputSpec& output) {
        return output.type == OutputType::ACTION || output.type == OutputType::CONTINUOUS_ACTION;
    });
    if (has_actions) {
        config.train_every = config.action_train_every;
        config.learning_rate = config.action_learning_rate;
        config.min_trunk_width = std::min(config.max_trunk_width,
                                          std::max(config.min_trunk_width, config.action_min_trunk_width));
    }

    impl_ = std::make_unique<Impl>(std::move(spec), config);
    impl_->learning_rate = config.learning_rate;
    impl_->Initialize();
}

Agent::~Agent() = default;
Agent::Agent(Agent&&) noexcept = default;
Agent& Agent::operator=(Agent&&) noexcept = default;

StepId Agent::Observe(const Observation& observation) {
    std::function<void(const AgentStats&)> hook;
    AgentStats stats;
    StepId step;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        step = impl_->ObserveLocked(observation);
        if (impl_->metrics_hook && step % impl_->metrics_every == 0) {
            hook = impl_->metrics_hook;
            stats = impl_->StatsLocked();
        }
    }
    if (hook) hook(stats);
    return step;
}

void Agent::SetMetricsHook(std::function<void(const AgentStats&)> hook, size_t every_steps) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->metrics_hook = std::move(hook);
    impl_->metrics_every = std::max<size_t>(1, every_steps);
}

bool AgentTestAccess::InjectNaN(Agent& agent) {
    std::lock_guard<std::mutex> lock(agent.impl_->mutex);
    Agent::Impl& m = *agent.impl_;
    std::vector<nn::Parameter> parameters;
    if (m.network) parameters = m.network->AllParameters();
    else if (m.autoencoder) parameters = m.autoencoder->Parameters();
    else if (!m.actor_critics.empty()) parameters = m.actor_critics.front()->ActorParameters();
    if (parameters.empty()) return false;
    parameters.front().value->Data()[0] = std::numeric_limits<float>::quiet_NaN();
    return true;
}

Prediction Agent::Predict(StepId step, const std::string& output) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (size_t i = 0; i < impl_->outputs.size(); ++i) {
        if (impl_->outputs[i].spec->name == output) {
            return impl_->PredictLocked(impl_->FindPending(step), i);
        }
    }
    throw std::invalid_argument("Unknown output '" + output + "'");
}

void Agent::Feedback(StepId step, const std::string& feedback, const Value& value) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->FeedbackLocked(step, feedback, value);
}

std::string Agent::Act(StepId step, const std::string& action_output, bool explore) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (size_t i = 0; i < impl_->outputs.size(); ++i) {
        if (impl_->outputs[i].spec->name != action_output) continue;
        if (impl_->outputs[i].spec->type == OutputType::CONTINUOUS_ACTION) {
            throw std::invalid_argument("Output '" + action_output +
                                        "' is a CONTINUOUS_ACTION output; use ActContinuous");
        }
        if (impl_->outputs[i].spec->type != OutputType::ACTION) {
            throw std::invalid_argument("Output '" + action_output + "' is not an ACTION output");
        }
        return impl_->ActLocked(step, i, explore);
    }
    throw std::invalid_argument("Unknown output '" + action_output + "'");
}

std::vector<float> Agent::ActContinuous(StepId step, const std::string& action_output, bool explore) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (size_t i = 0; i < impl_->outputs.size(); ++i) {
        if (impl_->outputs[i].spec->name != action_output) continue;
        if (impl_->outputs[i].spec->type != OutputType::CONTINUOUS_ACTION) {
            throw std::invalid_argument("Output '" + action_output + "' is not a CONTINUOUS_ACTION output");
        }
        return impl_->ActContinuousLocked(step, i, explore);
    }
    throw std::invalid_argument("Unknown output '" + action_output + "'");
}

void Agent::EndEpisode(bool terminal) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    Impl& m = *impl_;
    if (terminal && m.last_acted_step) {
        // The last action of the episode has no future value
        if (auto* last = m.TryFindPending(*m.last_acted_step); last && last->encoded) {
            last->sample->terminal = true;
            m.CompleteTransition(*last);
        }
    }
    m.encoder.ResetEpisode();
    m.previous_step.reset();
    m.last_acted_step.reset();
    m.episodes++;
}

AgentStats Agent::Stats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->StatsLocked();
}

// ============================================================================
// Checkpoints
// ============================================================================

void Agent::Save(std::ostream& out) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    out.write(kCheckpointMagic, sizeof(kCheckpointMagic));
    io::Write<uint32_t>(out, kCheckpointVersion);
    SaveSpec(out, impl_->spec);
    SaveConfig(out, impl_->config);
    impl_->SaveState(out);
    if (!out) throw std::runtime_error("Agent::Save: write failed");
}

bool Agent::Save(const std::string& path) const {
    // Write to a temporary file, then replace: a failed save leaves any
    // earlier checkpoint intact
    std::string temporary = path + ".tmp";
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file) return false;
        try {
            Save(file);
        } catch (const std::exception&) {
            std::remove(temporary.c_str());
            return false;
        }
        file.flush();
        if (!file) {
            std::remove(temporary.c_str());
            return false;
        }
    }
    return std::rename(temporary.c_str(), path.c_str()) == 0;
}

Agent Agent::Load(std::istream& in) {
    char magic[sizeof(kCheckpointMagic)];
    in.read(magic, sizeof(magic));
    if (!in || !std::equal(magic, magic + sizeof(magic), kCheckpointMagic)) {
        throw std::runtime_error("Agent::Load: not an agent checkpoint");
    }
    uint32_t version = io::Read<uint32_t>(in);
    if (version != kCheckpointVersion) {
        throw std::runtime_error("Agent::Load: checkpoint version " + std::to_string(version) +
                                 " is not supported (expected " + std::to_string(kCheckpointVersion) + ")");
    }
    EnvironmentSpec spec = LoadSpec(in);
    AgentConfig config = LoadConfig(in);
    Agent agent(std::move(spec), config);
    agent.impl_->LoadState(in);
    return agent;
}

Agent Agent::Load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Agent::Load: cannot open " + path);
    return Load(file);
}

const EnvironmentSpec& Agent::Spec() const { return impl_->spec; }
const AgentConfig& Agent::Config() const { return impl_->config; }

void Agent::SetClock(std::shared_ptr<const Clock> clock) { dpan::SetClock(std::move(clock)); }

void Agent::SetLogger(Logger logger) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->logger = std::move(logger);
}

}  // namespace agent
}  // namespace dpan
