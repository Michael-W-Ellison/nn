// File: src/agent/agent.cpp
#include "agent/agent.hpp"

#include "agent/network.hpp"
#include "agent/pattern_memory.hpp"
#include "agent/replay_buffer.hpp"
#include "nn/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <limits>
#include <locale>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>

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

}  // namespace

struct Agent::Impl {
    struct OutputState {
        const OutputSpec* spec{nullptr};
        int head{-1};        // network head
        int class_slot{-1};  // index among CLASS outputs (pattern memory)
        int value_slot{-1};  // index among VALUE outputs
        bool learn_classes{false};
        bool outcome_only{false};  // CLASS trained by OUTCOME feedback only
        std::vector<std::string> classes;
        std::map<std::string, size_t> class_index;
        std::vector<RunningStats> target_stats;
        uint64_t samples{0};
        Quality neural;
        Quality pattern_short;  // votes of recently labelled patterns
        Quality pattern_long;   // votes of all patterns (also NEXT_OBSERVATION sequences)

        /// Whether the short-term pattern memory is currently the better one
        bool UseShort(bool higher_is_better) const {
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
          schedule(config.plateau_factor, config.plateau_patience, config.min_learning_rate) {}

    // ------------------------------------------------------------------ setup

    void Initialize() {
        for (size_t i = 0; i < spec.outputs.size(); ++i) {
            OutputState state;
            state.spec = &spec.outputs[i];
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

    void BuildModels() {
        auto layout = encoder.Layout();
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
                default:
                    break;
            }
        }
        if (!heads.empty()) {
            network = std::make_unique<AgentNetwork>(layout, heads, config.learning_rate, config.seed,
                                                     config.min_trunk_width, config.max_trunk_width);
            head_loss.assign(heads.size(), kNaN);
        }
        if (has_anomaly) {
            autoencoder = std::make_unique<AnomalyAutoencoder>(encoder.OutputSize(), config.learning_rate,
                                                               config.seed + 3);
        }
        PatternMemory::Config memory;
        memory.dimension = config.pattern_dimension;
        memory.capacity = config.pattern_capacity;
        memory.merge_ratio = config.merge_ratio;
        memory.seed = config.seed + 5;
        patterns = std::make_unique<PatternMemory>(encoder.OutputSize(), class_outputs, value_sizes,
                                                   observed_size, memory);
        Log(LogLevel::INFO, "Inputs ready: fused size " + std::to_string(encoder.OutputSize()) +
                                (network ? ", trunk width " + std::to_string(network->TrunkWidth()) : ""));
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
        if (neural) add(out.neural, ErrorProduct(*neural, *neural, actual, stats));
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
            case OutputType::ACTION:
                break;
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
        if (auto neural = NeuralClass(entry, out)) out.neural.Add(ArgMax(*neural) == cls ? 1.0f : 0.0f, config.quality_rate);
        if (auto pattern = PatternClass(entry, out, true)) out.pattern_short.Add(ArgMax(*pattern) == cls ? 1.0f : 0.0f, config.quality_rate);
        if (auto pattern = PatternClass(entry, out, false)) out.pattern_long.Add(ArgMax(*pattern) == cls ? 1.0f : 0.0f, config.quality_rate);

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
        }

        if (!entry.encoded) {
            ignored_feedback++;
            return;
        }
        feedback_count++;

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
                ScoreRegression(out, NeuralValue(entry, out), PatternValue(entry, out, true),
                                PatternValue(entry, out, false), target_values, stats);
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

            if (has_anomaly) ScoreAnomaly(entry);

            auto placement = patterns->Insert(entry.point, step, entry.observed);
            entry.pattern = placement.id;

            entry.sample = std::make_shared<Sample>();
            entry.sample->step = step;
            entry.sample->SetFused(entry.fused);
            entry.sample->targets.resize(network ? network->HeadCount() : 0);

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

    void ScoreAnomaly(Pending& entry) {
        float distance = patterns->NearestDistance(entry.point);
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
        while (pattern_baseline.size() > config.anomaly_baseline) pattern_baseline.pop_front();
        while (neural_baseline.size() > config.anomaly_baseline) neural_baseline.pop_front();
    }

    // -------------------------------------------------------------- train

    void Train() {
        if (replay.Size() == 0 || (!network && !autoencoder)) return;

        size_t from_newest = std::min(newest.size(), static_cast<size_t>(std::lround(
                                                         config.batch_size * config.new_sample_share)));
        std::vector<SamplePtr> batch(newest.end() - static_cast<std::ptrdiff_t>(from_newest), newest.end());
        newest.clear();
        auto drawn = replay.Draw(config.batch_size - batch.size());
        batch.insert(batch.end(), drawn.begin(), drawn.end());
        if (batch.empty()) return;

        const size_t width = encoder.OutputSize();
        float total_loss = 0.0f;

        if (network) {
            nn::Matrix inputs(batch.size(), width);
            const size_t heads = network->HeadCount();
            std::vector<std::vector<HeadTarget>> owned(heads, std::vector<HeadTarget>(batch.size()));
            std::vector<std::vector<const HeadTarget*>> targets(heads, std::vector<const HeadTarget*>(batch.size()));
            for (size_t r = 0; r < batch.size(); ++r) {
                batch[r]->Fill(inputs.Data() + r * width, width);
            }
            for (const auto& out : outputs) {
                if (out.head < 0) continue;
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
            std::vector<float> scale(heads, 1.0f);
            for (size_t h = 0; h < heads; ++h) {
                if (!std::isnan(head_loss[h])) scale[h] = 1.0f / std::max(head_loss[h], 1e-3f);
            }
            auto losses = network->TrainBatch(inputs, targets, scale, config.gradient_clip_norm);
            for (size_t h = 0; h < heads; ++h) {
                if (std::isnan(losses[h])) continue;
                head_loss[h] = std::isnan(head_loss[h]) ? losses[h] : 0.95f * head_loss[h] + 0.05f * losses[h];
                total_loss += losses[h];
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
                total_loss += autoencoder->TrainBatch(inputs, config.gradient_clip_norm);
            }
        }

        training_steps++;
        plateau_sum += total_loss;
        if (++plateau_count >= config.plateau_interval) {
            float mean = static_cast<float>(plateau_sum / static_cast<double>(plateau_count));
            plateau_checks++;
            float rate = schedule.NextRate(plateau_checks, config.learning_rate, learning_rate, mean);
            if (rate != learning_rate) {
                learning_rate = rate;
                if (network) network->SetLearningRate(rate);
                if (autoencoder) autoencoder->SetLearningRate(rate);
                Log(LogLevel::INFO, "Learning rate reduced to " + FormatNumber(rate));
            }
            plateau_sum = 0.0;
            plateau_count = 0;
        }
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

    std::vector<size_t> observed_channels;
    std::vector<size_t> observed_offsets;
    size_t observed_size{0};

    std::unique_ptr<AgentNetwork> network;
    std::unique_ptr<AnomalyAutoencoder> autoencoder;
    std::unique_ptr<PatternMemory> patterns;
    ReplayBuffer replay;
    std::deque<SamplePtr> newest;
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
    for (const auto& output : spec.outputs) {
        if (output.type == OutputType::ACTION) {
            throw std::invalid_argument("Output '" + output.name +
                                        "': ACTION outputs are not supported yet (planned for Phase 3)");
        }
    }
    for (const auto& fb : spec.feedback) {
        if (fb.type == FeedbackType::REWARD) {
            throw std::invalid_argument("Feedback '" + fb.name +
                                        "': REWARD feedback is not supported yet (planned for Phase 3)");
        }
    }
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

    impl_ = std::make_unique<Impl>(std::move(spec), config);
    impl_->learning_rate = config.learning_rate;
    impl_->Initialize();
}

Agent::~Agent() = default;
Agent::Agent(Agent&&) noexcept = default;
Agent& Agent::operator=(Agent&&) noexcept = default;

StepId Agent::Observe(const Observation& observation) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->ObserveLocked(observation);
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

void Agent::EndEpisode() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->encoder.ResetEpisode();
    impl_->previous_step.reset();
}

AgentStats Agent::Stats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const Impl& m = *impl_;
    AgentStats stats;
    stats.steps = m.steps;
    stats.feedback = m.feedback_count;
    stats.ignored_feedback = m.ignored_feedback;
    stats.training_steps = m.training_steps;
    stats.pending_steps = m.pending.size();
    stats.replay_size = m.replay.Size();
    stats.patterns = m.patterns ? m.patterns->Size() : 0;
    stats.associations = m.patterns ? m.patterns->Associations().GetAssociationCount() : 0;
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
        o.pattern_quality = out.Pattern(out.spec->type == OutputType::CLASS).value;
        if (out.spec->type == OutputType::ANOMALY) {
            o.neural_weight = m.anomaly_samples >= m.config.cold_start_samples ? 0.5f : 0.0f;
        } else if (m.network && out.head >= 0) {
            o.neural_weight = m.BlendWeight(out, out.samples > 0,
                                            out.Pattern(out.spec->type == OutputType::CLASS).Measured(),
                                            out.spec->type == OutputType::CLASS);
        }
        stats.outputs.push_back(o);
    }
    return stats;
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
