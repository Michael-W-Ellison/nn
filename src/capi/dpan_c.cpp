// File: src/capi/dpan_c.cpp
#include "capi/dpan_c.h"

#include "agent/agent.hpp"
#include "agent/spec_json.hpp"
#include "core/clock.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>

using dpan::agent::Agent;
using dpan::agent::Observation;
using dpan::agent::StepId;

struct dpan_agent {
    explicit dpan_agent(Agent a) : agent(std::move(a)) {}
    Agent agent;
};

struct dpan_observation {
    Observation values;
};

namespace {

thread_local std::string last_error;

dpan_status Report(dpan_status status, const char* message) {
    last_error = message;
    return status;
}

/// Runs `body`, turning exceptions into a status and the error message
template <typename Body>
dpan_status Guard(Body&& body) {
    try {
        body();
        return DPAN_OK;
    } catch (const std::bad_alloc&) {
        return Report(DPAN_OUT_OF_MEMORY, "out of memory");
    } catch (const std::invalid_argument& e) {
        return Report(DPAN_INVALID_ARGUMENT, e.what());
    } catch (const std::out_of_range& e) {
        return Report(DPAN_OUT_OF_RANGE, e.what());
    } catch (const std::logic_error& e) {
        return Report(DPAN_LOGIC_ERROR, e.what());
    } catch (const std::runtime_error& e) {
        return Report(DPAN_RUNTIME_ERROR, e.what());
    } catch (const std::exception& e) {
        return Report(DPAN_ERROR, e.what());
    } catch (...) {
        return Report(DPAN_ERROR, "unknown error");
    }
}

void Require(const void* pointer, const char* name) {
    if (!pointer) throw std::invalid_argument(std::string(name) + " is NULL");
}

char* Copy(const std::string& text) {
    char* out = static_cast<char*>(std::malloc(text.size() + 1));
    if (!out) throw std::bad_alloc();
    std::memcpy(out, text.c_str(), text.size() + 1);
    return out;
}

std::mutex clock_mutex;
std::shared_ptr<dpan::ManualClock> manual_clock;

}  // namespace

extern "C" {

const char* dpan_version(void) { return DPAN_VERSION_STRING; }

const char* dpan_last_error(void) { return last_error.c_str(); }

void dpan_string_free(char* text) { std::free(text); }

dpan_status dpan_agent_create(const char* spec_json, const char* config_json, dpan_agent** out) {
    return Guard([&] {
        Require(spec_json, "spec_json");
        Require(out, "out");
        auto spec = dpan::agent::SpecFromJson(spec_json);
        dpan::agent::AgentConfig config;
        if (config_json && *config_json) config = dpan::agent::ConfigFromJson(config_json);
        *out = new dpan_agent(Agent(std::move(spec), config));
    });
}

dpan_status dpan_agent_load(const char* path, dpan_agent** out) {
    return Guard([&] {
        Require(path, "path");
        Require(out, "out");
        *out = new dpan_agent(Agent::Load(std::string(path)));
    });
}

dpan_status dpan_agent_save(const dpan_agent* agent, const char* path) {
    return Guard([&] {
        Require(agent, "agent");
        Require(path, "path");
        if (!agent->agent.Save(std::string(path))) {
            throw std::runtime_error(std::string("cannot write checkpoint '") + path + "'");
        }
    });
}

void dpan_agent_free(dpan_agent* agent) { delete agent; }

dpan_status dpan_agent_spec_json(const dpan_agent* agent, char** out) {
    return Guard([&] {
        Require(agent, "agent");
        Require(out, "out");
        *out = Copy(dpan::agent::SpecToJson(agent->agent.Spec()));
    });
}

dpan_status dpan_agent_config_json(const dpan_agent* agent, char** out) {
    return Guard([&] {
        Require(agent, "agent");
        Require(out, "out");
        *out = Copy(dpan::agent::ConfigToJson(agent->agent.Config()));
    });
}

dpan_status dpan_agent_observe(dpan_agent* agent, const dpan_observation* observation, dpan_step* out) {
    return Guard([&] {
        Require(agent, "agent");
        Require(observation, "observation");
        Require(out, "out");
        *out = agent->agent.Observe(observation->values);
    });
}

dpan_status dpan_agent_observe_json(dpan_agent* agent, const char* observation_json, dpan_step* out) {
    return Guard([&] {
        Require(agent, "agent");
        Require(observation_json, "observation_json");
        Require(out, "out");
        *out = agent->agent.Observe(dpan::agent::ObservationFromJson(observation_json));
    });
}

dpan_status dpan_agent_predict_json(const dpan_agent* agent, dpan_step step, const char* output, char** out) {
    return Guard([&] {
        Require(agent, "agent");
        Require(output, "output");
        Require(out, "out");
        *out = Copy(dpan::agent::PredictionToJson(agent->agent.Predict(step, output)));
    });
}

dpan_status dpan_agent_act(dpan_agent* agent, dpan_step step, const char* output, int explore, const char** action,
                           size_t* index) {
    return Guard([&] {
        Require(agent, "agent");
        Require(output, "output");
        Require(action, "action");
        std::string chosen = agent->agent.Act(step, output, explore != 0);
        // Point into the spec the agent keeps, so the name outlives this call
        for (const auto& spec : agent->agent.Spec().outputs) {
            if (spec.name != output) continue;
            for (size_t i = 0; i < spec.classes_or_actions.size(); ++i) {
                if (spec.classes_or_actions[i] == chosen) {
                    *action = spec.classes_or_actions[i].c_str();
                    if (index) *index = i;
                    return;
                }
            }
        }
        throw std::logic_error("action '" + chosen + "' is not in the spec");
    });
}

dpan_status dpan_agent_act_continuous(dpan_agent* agent, dpan_step step, const char* output, int explore,
                                      float* values, size_t capacity, size_t* count) {
    return Guard([&] {
        Require(agent, "agent");
        Require(output, "output");
        for (const auto& spec : agent->agent.Spec().outputs) {
            if (spec.name != output || spec.type != dpan::agent::OutputType::CONTINUOUS_ACTION) continue;
            if (!values || capacity < spec.size) {
                throw std::invalid_argument("output '" + spec.name + "' needs room for " + std::to_string(spec.size) +
                                            " values");
            }
        }
        // Unknown or non-continuous outputs are reported by ActContinuous
        std::vector<float> action = agent->agent.ActContinuous(step, output, explore != 0);
        std::copy(action.begin(), action.end(), values);
        if (count) *count = action.size();
    });
}

dpan_status dpan_agent_feedback_number(dpan_agent* agent, dpan_step step, const char* feedback, double value) {
    return Guard([&] {
        Require(agent, "agent");
        Require(feedback, "feedback");
        agent->agent.Feedback(step, feedback, value);
    });
}

dpan_status dpan_agent_feedback_vector(dpan_agent* agent, dpan_step step, const char* feedback, const float* values,
                                       size_t count) {
    return Guard([&] {
        Require(agent, "agent");
        Require(feedback, "feedback");
        if (count > 0) Require(values, "values");
        agent->agent.Feedback(step, feedback, std::vector<float>(values, values + count));
    });
}

dpan_status dpan_agent_feedback_text(dpan_agent* agent, dpan_step step, const char* feedback, const char* text) {
    return Guard([&] {
        Require(agent, "agent");
        Require(feedback, "feedback");
        Require(text, "text");
        agent->agent.Feedback(step, feedback, std::string(text));
    });
}

dpan_status dpan_agent_feedback_json(dpan_agent* agent, dpan_step step, const char* feedback,
                                     const char* value_json) {
    return Guard([&] {
        Require(agent, "agent");
        Require(feedback, "feedback");
        Require(value_json, "value_json");
        agent->agent.Feedback(step, feedback,
                              dpan::agent::ValueFromJson(dpan::agent::Json::Parse(value_json)));
    });
}

dpan_status dpan_agent_end_episode(dpan_agent* agent, int terminal) {
    return Guard([&] {
        Require(agent, "agent");
        agent->agent.EndEpisode(terminal != 0);
    });
}

dpan_status dpan_agent_stats_json(const dpan_agent* agent, char** out) {
    return Guard([&] {
        Require(agent, "agent");
        Require(out, "out");
        *out = Copy(dpan::agent::StatsToJson(agent->agent.Stats()));
    });
}

dpan_observation* dpan_observation_create(void) { return new (std::nothrow) dpan_observation(); }

void dpan_observation_free(dpan_observation* observation) { delete observation; }

void dpan_observation_clear(dpan_observation* observation) {
    if (observation) observation->values.clear();
}

dpan_status dpan_observation_set_number(dpan_observation* observation, const char* channel, double value) {
    return Guard([&] {
        Require(observation, "observation");
        Require(channel, "channel");
        observation->values[channel] = value;
    });
}

dpan_status dpan_observation_set_vector(dpan_observation* observation, const char* channel, const float* values,
                                        size_t count) {
    return Guard([&] {
        Require(observation, "observation");
        Require(channel, "channel");
        if (count > 0) Require(values, "values");
        observation->values[channel] = std::vector<float>(values, values + count);
    });
}

dpan_status dpan_observation_set_text(dpan_observation* observation, const char* channel, const char* text) {
    return Guard([&] {
        Require(observation, "observation");
        Require(channel, "channel");
        Require(text, "text");
        observation->values[channel] = std::string(text);
    });
}

dpan_status dpan_observation_set_image(dpan_observation* observation, const char* channel, size_t width,
                                       size_t height, size_t channels, const float* pixels) {
    return Guard([&] {
        Require(observation, "observation");
        Require(channel, "channel");
        const size_t max = std::numeric_limits<size_t>::max();
        if ((height != 0 && width > max / height) || (channels != 0 && width * height > max / channels)) {
            throw std::invalid_argument("image size overflows");
        }
        size_t count = width * height * channels;
        if (count > 0) Require(pixels, "pixels");
        dpan::agent::Image image;
        image.width = width;
        image.height = height;
        image.channels = channels;
        image.pixels.assign(pixels, pixels + count);
        observation->values[channel] = std::move(image);
    });
}

void dpan_clock_use_manual(int64_t start_microseconds) {
    std::lock_guard<std::mutex> lock(clock_mutex);
    manual_clock = std::make_shared<dpan::ManualClock>(dpan::Timestamp::FromMicros(start_microseconds));
    dpan::SetClock(manual_clock);
}

dpan_status dpan_clock_advance(int64_t microseconds) {
    return Guard([&] {
        std::lock_guard<std::mutex> lock(clock_mutex);
        if (!manual_clock) throw std::logic_error("no manual clock in use (call dpan_clock_use_manual first)");
        manual_clock->Advance(std::chrono::microseconds(microseconds));
    });
}

void dpan_clock_use_system(void) {
    std::lock_guard<std::mutex> lock(clock_mutex);
    manual_clock.reset();
    dpan::SetClock(nullptr);
}

}  // extern "C"
