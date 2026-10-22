// File: src/agent/spec_json.hpp
//
// JSON forms of the agent's interface types, used by the C API and the
// Python bindings (and usable from C++, e.g. to read a spec from a file).
//
// Spec:
//   {"inputs":   [{"name": "x", "type": "numeric", "size": 4},
//                 {"name": "region", "type": "categorical", "categories": ["north", "south"]},
//                 {"name": "note", "type": "text"},
//                 {"name": "load", "type": "time_series", "size": 1},
//                 {"name": "camera", "type": "image", "width": 32, "height": 32, "channels": 1}],
//    "feedback": [{"name": "label", "type": "label", "output": "kind"}],
//    "outputs":  [{"name": "kind", "type": "class", "classes": ["a", "b"]},
//                 {"name": "move", "type": "action", "actions": ["left", "right"]},
//                 {"name": "y", "type": "value", "size": 1},
//                 {"name": "next", "type": "next_observation"},
//                 {"name": "unusual", "type": "anomaly"}]}
// Type names are case-insensitive. Members other than "name" and "type" are
// optional (defaults as in environment.hpp). Unknown members are errors.
//
// Config: an object with any AgentConfig field by its C++ name, e.g.
//   {"seed": 7, "memory_budget_bytes": 67108864, "encoder": {"text_buckets": 512}}
// Fields left out keep their defaults; unknown fields are errors.
//
// Values (observations and feedback): a number, an array of numbers (null =
// missing element, NaN), a string, or an image object
//   {"width": 2, "height": 1, "channels": 1, "pixels": [0, 255]}.
// An observation is an object of values by input name.
#pragma once

#include "agent/agent.hpp"
#include "agent/json.hpp"

#include <string>

namespace dpan {
namespace agent {

/// @throws std::invalid_argument for malformed JSON or unknown members or
///         type names (the spec itself is validated by the Agent constructor)
EnvironmentSpec SpecFromJson(const std::string& json);
std::string SpecToJson(const EnvironmentSpec& spec);

/// Fields present in `json` override those of `base`
/// @throws std::invalid_argument for malformed JSON, unknown fields or
///         values of the wrong type or out of the field's range
AgentConfig ConfigFromJson(const std::string& json, const AgentConfig& base = {});
std::string ConfigToJson(const AgentConfig& config);

/// @throws std::invalid_argument for anything that is not a value
Value ValueFromJson(const Json& json);
Observation ObservationFromJson(const std::string& json);

std::string PredictionToJson(const Prediction& prediction);
std::string StatsToJson(const AgentStats& stats);

}  // namespace agent
}  // namespace dpan
