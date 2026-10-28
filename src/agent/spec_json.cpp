// File: src/agent/spec_json.cpp
#include "agent/spec_json.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace dpan {
namespace agent {
namespace {

[[noreturn]] void Fail(const std::string& where, const std::string& message) {
    throw std::invalid_argument(where.empty() ? message : where + ": " + message);
}

std::string Upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

std::string Lower(const char* s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

const Json::Object& RequireObject(const Json& json, const std::string& where) {
    if (!json.IsObject()) Fail(where, std::string("expected an object, got ") + ToString(json.GetType()));
    return json.AsObject();
}

void CheckMembers(const Json& json, const std::string& where, std::initializer_list<const char*> allowed) {
    for (const auto& member : RequireObject(json, where)) {
        bool known = std::any_of(allowed.begin(), allowed.end(), [&](const char* a) { return member.first == a; });
        if (!known) Fail(where, "unknown member \"" + member.first + "\"");
    }
}

std::string GetString(const Json& object, const char* key, const std::string& where, bool required) {
    const Json* value = object.Find(key);
    if (!value) {
        if (required) Fail(where, std::string("missing \"") + key + "\"");
        return "";
    }
    if (!value->IsString()) Fail(where, std::string("\"") + key + "\" must be a string");
    return value->AsString();
}

uint64_t ToUnsigned(const Json& value, const std::string& where, uint64_t max) {
    if (!value.IsNumber()) Fail(where, std::string("expected a number, got ") + ToString(value.GetType()));
    double v = value.AsNumber();
    if (!(v >= 0) || v != std::floor(v) || v > static_cast<double>(max)) {
        Fail(where, "expected a whole number from 0 to " + std::to_string(max));
    }
    return static_cast<uint64_t>(v);
}

size_t GetSize(const Json& object, const char* key, const std::string& where, size_t fallback) {
    const Json* value = object.Find(key);
    if (!value) return fallback;
    return static_cast<size_t>(ToUnsigned(*value, where + ": \"" + key + "\"", std::numeric_limits<size_t>::max()));
}

std::vector<std::string> GetStrings(const Json& object, const char* key, const std::string& where) {
    std::vector<std::string> out;
    const Json* value = object.Find(key);
    if (!value) return out;
    if (!value->IsArray()) Fail(where, std::string("\"") + key + "\" must be an array of strings");
    for (const Json& item : value->AsArray()) {
        if (!item.IsString()) Fail(where, std::string("\"") + key + "\" must be an array of strings");
        out.push_back(item.AsString());
    }
    return out;
}

std::vector<float> GetNumbers(const Json& object, const char* key, const std::string& where) {
    std::vector<float> out;
    const Json* value = object.Find(key);
    if (!value) return out;
    if (!value->IsArray()) Fail(where, std::string("\"") + key + "\" must be an array of numbers");
    for (const Json& item : value->AsArray()) {
        if (!item.IsNumber()) Fail(where, std::string("\"") + key + "\" must be an array of numbers");
        out.push_back(static_cast<float>(item.AsNumber()));
    }
    return out;
}

template <typename Enum, size_t N>
Enum ParseEnum(const std::string& name, const Enum (&values)[N], const std::string& where) {
    std::string wanted = Upper(name);
    std::string options;
    for (Enum value : values) {
        if (wanted == ToString(value)) return value;
        options += (options.empty() ? "" : ", ") + Lower(ToString(value));
    }
    Fail(where, "unknown type \"" + name + "\" (expected one of: " + options + ")");
}

constexpr InputType kInputTypes[] = {InputType::NUMERIC, InputType::CATEGORICAL, InputType::TEXT, InputType::IMAGE,
                                     InputType::TIME_SERIES};
constexpr FeedbackType kFeedbackTypes[] = {FeedbackType::LABEL, FeedbackType::TARGET, FeedbackType::REWARD,
                                           FeedbackType::OUTCOME};
constexpr OutputType kOutputTypes[] = {OutputType::CLASS, OutputType::VALUE, OutputType::ACTION,
                                       OutputType::NEXT_OBSERVATION, OutputType::ANOMALY,
                                       OutputType::CONTINUOUS_ACTION};

const Json::Array& GetList(const Json& root, const char* key) {
    static const Json::Array kEmpty;
    const Json* list = root.Find(key);
    if (!list) return kEmpty;
    if (!list->IsArray()) Fail("", std::string("\"") + key + "\" must be an array");
    return list->AsArray();
}

Json Strings(const std::vector<std::string>& values) {
    Json out = Json::MakeArray();
    for (const auto& v : values) out.Push(v);
    return out;
}

Json Numbers(const std::vector<float>& values) {
    Json out = Json::MakeArray();
    for (float v : values) out.Push(v);
    return out;
}

// ---------------------------------------------------------------------------
// Config fields

// size_t and uint64_t are the same type on some platforms (Linux) and not
// on others (macOS, Windows)
template <typename Owner>
using Field = std::variant<bool Owner::*, float Owner::*, double Owner::*, uint32_t Owner::*, size_t Owner::*,
                           std::conditional_t<std::is_same_v<size_t, uint64_t>, std::monostate, uint64_t Owner::*>,
                           std::string Owner::*>;

template <typename Owner>
struct FieldEntry {
    const char* name;
    Field<Owner> field;
};

#define DPAN_FIELD(owner, name) FieldEntry<owner>{#name, &owner::name}

const std::vector<FieldEntry<EncoderConfig>>& EncoderFields() {
    static const std::vector<FieldEntry<EncoderConfig>> fields = {
        DPAN_FIELD(EncoderConfig, clip_sigma),
        DPAN_FIELD(EncoderConfig, statistics_limit),
        DPAN_FIELD(EncoderConfig, category_buckets),
        DPAN_FIELD(EncoderConfig, category_embedding_size),
        DPAN_FIELD(EncoderConfig, text_buckets),
        DPAN_FIELD(EncoderConfig, text_embedding_size),
        DPAN_FIELD(EncoderConfig, text_sequence_length),
        DPAN_FIELD(EncoderConfig, text_vocabulary),
        DPAN_FIELD(EncoderConfig, text_word_vectors),
        DPAN_FIELD(EncoderConfig, text_word_vectors_limit),
        DPAN_FIELD(EncoderConfig, text_word_vector_norm),
        DPAN_FIELD(EncoderConfig, train_word_vectors),
        DPAN_FIELD(EncoderConfig, text_bidirectional),
        DPAN_FIELD(EncoderConfig, text_word_dropout),
        DPAN_FIELD(EncoderConfig, text_dropout),
        DPAN_FIELD(EncoderConfig, text_word_dimension),
        DPAN_FIELD(EncoderConfig, text_sequence_units),
        DPAN_FIELD(EncoderConfig, time_series_window),
        DPAN_FIELD(EncoderConfig, image_size),
        DPAN_FIELD(EncoderConfig, max_image_size),
        DPAN_FIELD(EncoderConfig, image_embedding_size),
        DPAN_FIELD(EncoderConfig, max_tracked_categories),
    };
    return fields;
}

const std::vector<FieldEntry<AgentConfig>>& AgentFields() {
    static const std::vector<FieldEntry<AgentConfig>> fields = {
        DPAN_FIELD(AgentConfig, seed),
        DPAN_FIELD(AgentConfig, learning_rate),
        DPAN_FIELD(AgentConfig, min_learning_rate),
        DPAN_FIELD(AgentConfig, min_trunk_width),
        DPAN_FIELD(AgentConfig, max_trunk_width),
        DPAN_FIELD(AgentConfig, gradient_clip_norm),
        DPAN_FIELD(AgentConfig, train_every),
        DPAN_FIELD(AgentConfig, batch_size),
        DPAN_FIELD(AgentConfig, new_sample_share),
        DPAN_FIELD(AgentConfig, replay_capacity),
        DPAN_FIELD(AgentConfig, cold_start_samples),
        DPAN_FIELD(AgentConfig, feedback_timeout),
        DPAN_FIELD(AgentConfig, plateau_interval),
        DPAN_FIELD(AgentConfig, plateau_patience),
        DPAN_FIELD(AgentConfig, plateau_factor),
        DPAN_FIELD(AgentConfig, action_train_every),
        DPAN_FIELD(AgentConfig, action_learning_rate),
        DPAN_FIELD(AgentConfig, action_min_trunk_width),
        DPAN_FIELD(AgentConfig, action_return_steps),
        DPAN_FIELD(AgentConfig, discount),
        DPAN_FIELD(AgentConfig, target_sync),
        DPAN_FIELD(AgentConfig, exploration_start),
        DPAN_FIELD(AgentConfig, exploration_end),
        DPAN_FIELD(AgentConfig, exploration_steps),
        DPAN_FIELD(AgentConfig, continuous_hidden),
        DPAN_FIELD(AgentConfig, continuous_actor_learning_rate),
        DPAN_FIELD(AgentConfig, continuous_critic_learning_rate),
        DPAN_FIELD(AgentConfig, continuous_random_steps),
        DPAN_FIELD(AgentConfig, continuous_exploration_noise),
        DPAN_FIELD(AgentConfig, continuous_target_noise),
        DPAN_FIELD(AgentConfig, continuous_noise_clip),
        DPAN_FIELD(AgentConfig, continuous_policy_delay),
        DPAN_FIELD(AgentConfig, continuous_tau),
        DPAN_FIELD(AgentConfig, continuous_updates_per_step),
        DPAN_FIELD(AgentConfig, continuous_saturation_penalty),
        DPAN_FIELD(AgentConfig, continuous_saturation_threshold),
        DPAN_FIELD(AgentConfig, detect_drift),
        DPAN_FIELD(AgentConfig, drift_delta),
        DPAN_FIELD(AgentConfig, drift_threshold),
        DPAN_FIELD(AgentConfig, drift_warmup),
        DPAN_FIELD(AgentConfig, drift_learning_rate_factor),
        DPAN_FIELD(AgentConfig, drift_new_sample_share),
        DPAN_FIELD(AgentConfig, drift_recent_window),
        DPAN_FIELD(AgentConfig, drift_recovery_max_samples),
        DPAN_FIELD(AgentConfig, snapshot_interval),
        DPAN_FIELD(AgentConfig, divergence_loss),
        DPAN_FIELD(AgentConfig, grow_capacity),
        DPAN_FIELD(AgentConfig, capacity_min_samples),
        DPAN_FIELD(AgentConfig, capacity_growth),
        DPAN_FIELD(AgentConfig, memory_budget_bytes),
        DPAN_FIELD(AgentConfig, pattern_capacity),
        DPAN_FIELD(AgentConfig, pattern_dimension),
        DPAN_FIELD(AgentConfig, neighbors),
        DPAN_FIELD(AgentConfig, merge_ratio),
        DPAN_FIELD(AgentConfig, pattern_label_horizon),
        DPAN_FIELD(AgentConfig, quality_rate),
        DPAN_FIELD(AgentConfig, class_blend_sharpness),
        DPAN_FIELD(AgentConfig, class_blend_rate),
        DPAN_FIELD(AgentConfig, anomaly_baseline),
        DPAN_FIELD(AgentConfig, max_classes),
        DPAN_FIELD(AgentConfig, holdout_every),
        DPAN_FIELD(AgentConfig, holdout_capacity),
        DPAN_FIELD(AgentConfig, holdout_check_interval),
        DPAN_FIELD(AgentConfig, holdout_min_samples),
        DPAN_FIELD(AgentConfig, max_training_slowdown),
        DPAN_FIELD(AgentConfig, memory_units),
        DPAN_FIELD(AgentConfig, memory_window),
        DPAN_FIELD(AgentConfig, memory_batch),
        DPAN_FIELD(AgentConfig, memory_history),
        DPAN_FIELD(AgentConfig, memory_burn_in),
        DPAN_FIELD(AgentConfig, action_memory_window),
        DPAN_FIELD(AgentConfig, action_memory_batch),
    };
    return fields;
}

#undef DPAN_FIELD

template <typename Owner>
void ReadField(Owner& owner, const Field<Owner>& field, const Json& value, const std::string& where) {
    std::visit(
        [&](auto member) {
            if constexpr (!std::is_same_v<decltype(member), std::monostate>) {
            using T = std::remove_reference_t<decltype(owner.*member)>;
            if constexpr (std::is_same_v<T, bool>) {
                if (!value.IsBool()) Fail(where, std::string("expected a boolean, got ") + ToString(value.GetType()));
                owner.*member = value.AsBool();
            } else if constexpr (std::is_same_v<T, std::string>) {
                if (!value.IsString()) Fail(where, std::string("expected a string, got ") + ToString(value.GetType()));
                owner.*member = value.AsString();
            } else if constexpr (std::is_floating_point_v<T>) {
                if (!value.IsNumber()) Fail(where, std::string("expected a number, got ") + ToString(value.GetType()));
                double v = value.AsNumber();
                if (!std::isfinite(static_cast<T>(v))) Fail(where, "number out of range");
                owner.*member = static_cast<T>(v);
            } else {
                owner.*member = static_cast<T>(ToUnsigned(value, where, std::numeric_limits<T>::max()));
            }
            }
        },
        field);
}

template <typename Owner>
Json WriteField(const Owner& owner, const Field<Owner>& field) {
    return std::visit(
        [&](auto member) {
            if constexpr (std::is_same_v<decltype(member), std::monostate>) {
                return Json();
            } else {
                return Json(owner.*member);
            }
        },
        field);
}

template <typename Owner>
void ReadFields(Owner& owner, const std::vector<FieldEntry<Owner>>& fields, const Json::Object::value_type& member,
                const std::string& prefix) {
    for (const auto& entry : fields) {
        if (member.first == entry.name) {
            ReadField(owner, entry.field, member.second, prefix + member.first);
            return;
        }
    }
    Fail(prefix.empty() ? "config" : "config: " + prefix.substr(0, prefix.size() - 1),
         "unknown field \"" + member.first + "\"");
}

}  // namespace

// ---------------------------------------------------------------------------

EnvironmentSpec SpecFromJson(const std::string& text) {
    Json root = Json::Parse(text);
    CheckMembers(root, "spec", {"inputs", "feedback", "outputs"});
    EnvironmentSpec spec;

    const auto& inputs = GetList(root, "inputs");
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::string where = "inputs[" + std::to_string(i) + "]";
        const Json& item = inputs[i];
        CheckMembers(item, where, {"name", "type", "size", "categories", "width", "height", "channels"});
        InputSpec input;
        input.name = GetString(item, "name", where, true);
        input.type = ParseEnum(GetString(item, "type", where, true), kInputTypes, where);
        input.size = GetSize(item, "size", where, 0);
        input.categories = GetStrings(item, "categories", where);
        input.width = GetSize(item, "width", where, 0);
        input.height = GetSize(item, "height", where, 0);
        input.image_channels = GetSize(item, "channels", where, 1);
        spec.inputs.push_back(std::move(input));
    }

    const auto& feedback = GetList(root, "feedback");
    for (size_t i = 0; i < feedback.size(); ++i) {
        std::string where = "feedback[" + std::to_string(i) + "]";
        const Json& item = feedback[i];
        CheckMembers(item, where, {"name", "type", "output"});
        FeedbackSpec f;
        f.name = GetString(item, "name", where, true);
        f.type = ParseEnum(GetString(item, "type", where, true), kFeedbackTypes, where);
        f.output = GetString(item, "output", where, true);
        spec.feedback.push_back(std::move(f));
    }

    const auto& outputs = GetList(root, "outputs");
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::string where = "outputs[" + std::to_string(i) + "]";
        const Json& item = outputs[i];
        CheckMembers(item, where, {"name", "type", "classes", "actions", "size", "low", "high"});
        OutputSpec output;
        output.name = GetString(item, "name", where, true);
        output.type = ParseEnum(GetString(item, "type", where, true), kOutputTypes, where);
        if (item.Find("classes") && item.Find("actions")) Fail(where, "give \"classes\" or \"actions\", not both");
        output.classes_or_actions = GetStrings(item, item.Find("actions") ? "actions" : "classes", where);
        output.size = GetSize(item, "size", where, 1);
        output.low = GetNumbers(item, "low", where);
        output.high = GetNumbers(item, "high", where);
        spec.outputs.push_back(std::move(output));
    }
    return spec;
}

std::string SpecToJson(const EnvironmentSpec& spec) {
    Json root = Json::MakeObject();
    Json& inputs = root.Set("inputs", Json::MakeArray());
    for (const auto& input : spec.inputs) {
        Json item = Json::MakeObject();
        item.Set("name", input.name);
        item.Set("type", Lower(ToString(input.type)));
        if (input.type == InputType::NUMERIC || input.type == InputType::TIME_SERIES) item.Set("size", input.size);
        if (input.type == InputType::CATEGORICAL) item.Set("categories", Strings(input.categories));
        if (input.type == InputType::IMAGE) {
            item.Set("width", input.width);
            item.Set("height", input.height);
            item.Set("channels", input.image_channels);
        }
        inputs.Push(std::move(item));
    }
    Json& feedback = root.Set("feedback", Json::MakeArray());
    for (const auto& f : spec.feedback) {
        Json item = Json::MakeObject();
        item.Set("name", f.name);
        item.Set("type", Lower(ToString(f.type)));
        item.Set("output", f.output);
        feedback.Push(std::move(item));
    }
    Json& outputs = root.Set("outputs", Json::MakeArray());
    for (const auto& output : spec.outputs) {
        Json item = Json::MakeObject();
        item.Set("name", output.name);
        item.Set("type", Lower(ToString(output.type)));
        if (output.type == OutputType::ACTION) item.Set("actions", Strings(output.classes_or_actions));
        if (output.type == OutputType::CLASS) item.Set("classes", Strings(output.classes_or_actions));
        if (output.type == OutputType::VALUE) item.Set("size", output.size);
        if (output.type == OutputType::CONTINUOUS_ACTION) {
            item.Set("size", output.size);
            item.Set("low", Numbers(output.low));
            item.Set("high", Numbers(output.high));
        }
        outputs.Push(std::move(item));
    }
    return root.Dump();
}

AgentConfig ConfigFromJson(const std::string& text, const AgentConfig& base) {
    Json root = Json::Parse(text);
    RequireObject(root, "config");
    AgentConfig config = base;
    for (const auto& member : root.AsObject()) {
        if (member.first == "encoder") {
            RequireObject(member.second, "config: encoder");
            for (const auto& field : member.second.AsObject()) {
                ReadFields(config.encoder, EncoderFields(), field, "encoder.");
            }
        } else {
            ReadFields(config, AgentFields(), member, "");
        }
    }
    return config;
}

std::string ConfigToJson(const AgentConfig& config) {
    Json root = Json::MakeObject();
    for (const auto& entry : AgentFields()) root.Set(entry.name, WriteField(config, entry.field));
    Json& encoder = root.Set("encoder", Json::MakeObject());
    for (const auto& entry : EncoderFields()) encoder.Set(entry.name, WriteField(config.encoder, entry.field));
    return root.Dump();
}

Value ValueFromJson(const Json& json) {
    switch (json.GetType()) {
        case Json::Type::NUMBER: return json.AsNumber();
        case Json::Type::STRING: return json.AsString();
        case Json::Type::ARRAY: {
            std::vector<float> values;
            values.reserve(json.AsArray().size());
            for (const Json& item : json.AsArray()) {
                if (item.IsNull()) values.push_back(std::numeric_limits<float>::quiet_NaN());
                else if (item.IsNumber()) values.push_back(static_cast<float>(item.AsNumber()));
                else Fail("", "array values must be numbers or null");
            }
            return values;
        }
        case Json::Type::OBJECT: {
            CheckMembers(json, "image", {"width", "height", "channels", "pixels"});
            Image image;
            image.width = GetSize(json, "width", "image", 0);
            image.height = GetSize(json, "height", "image", 0);
            image.channels = GetSize(json, "channels", "image", 1);
            const Json* pixels = json.Find("pixels");
            if (!pixels || !pixels->IsArray()) Fail("image", "\"pixels\" must be an array of numbers");
            image.pixels.reserve(pixels->AsArray().size());
            for (const Json& p : pixels->AsArray()) {
                if (!p.IsNumber()) Fail("image", "\"pixels\" must be an array of numbers");
                image.pixels.push_back(static_cast<float>(p.AsNumber()));
            }
            return image;
        }
        default:
            Fail("", std::string("a value must be a number, array, string or image object, got ") +
                         ToString(json.GetType()));
    }
}

Observation ObservationFromJson(const std::string& text) {
    Json root = Json::Parse(text);
    Observation observation;
    RequireObject(root, "observation");
    for (const auto& member : root.AsObject()) {
        if (member.second.IsNull()) continue;  // missing channel
        try {
            observation[member.first] = ValueFromJson(member.second);
        } catch (const std::invalid_argument& e) {
            Fail("observation \"" + member.first + "\"", e.what());
        }
    }
    return observation;
}

std::string PredictionToJson(const Prediction& p) {
    Json root = Json::MakeObject();
    root.Set("output", p.output);
    root.Set("type", Lower(ToString(p.type)));
    root.Set("available", p.available);
    root.Set("label", p.label);
    root.Set("classes", Strings(p.classes));
    root.Set("probabilities", Numbers(p.probabilities));
    root.Set("values", Numbers(p.values));
    root.Set("score", p.score);
    root.Set("confidence", p.confidence);
    root.Set("neural_weight", p.neural_weight);
    return root.Dump();
}

std::string StatsToJson(const AgentStats& s) {
    Json root = Json::MakeObject();
    root.Set("steps", s.steps);
    root.Set("episodes", s.episodes);
    root.Set("feedback", s.feedback);
    root.Set("ignored_feedback", s.ignored_feedback);
    root.Set("training_steps", s.training_steps);
    root.Set("pending_steps", s.pending_steps);
    root.Set("replay_size", s.replay_size);
    root.Set("held_out_samples", s.held_out_samples);
    root.Set("training_slowdown", s.training_slowdown);
    root.Set("patterns", s.patterns);
    root.Set("patterns_pruned", s.patterns_pruned);
    root.Set("drift_events", s.drift_events);
    root.Set("recovering", s.recovering);
    root.Set("rollbacks", s.rollbacks);
    root.Set("widenings", s.widenings);
    root.Set("associations", s.associations);
    root.Set("memory_bytes", s.memory_bytes);
    root.Set("memory_budget_bytes", s.memory_budget_bytes);
    root.Set("network_ready", s.network_ready);
    root.Set("fused_size", s.fused_size);
    root.Set("trunk_width", s.trunk_width);
    root.Set("learning_rate", s.learning_rate);
    Json& outputs = root.Set("outputs", Json::MakeArray());
    for (const auto& o : s.outputs) {
        Json item = Json::MakeObject();
        item.Set("name", o.name);
        item.Set("type", Lower(ToString(o.type)));
        item.Set("samples", o.samples);
        item.Set("classes", o.classes);
        item.Set("neural_quality", o.neural_quality);
        item.Set("pattern_quality", o.pattern_quality);
        item.Set("neural_log_loss", o.neural_log_loss);
        item.Set("pattern_log_loss", o.pattern_log_loss);
        item.Set("neural_weight", o.neural_weight);
        item.Set("exploration", o.exploration);
        item.Set("drift_events", o.drift_events);
        item.Set("recovering", o.recovering);
        outputs.Push(std::move(item));
    }
    return root.Dump();
}

}  // namespace agent
}  // namespace dpan
