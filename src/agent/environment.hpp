// File: src/agent/environment.hpp
//
// Description of the environment an agent runs in: its input channels, the
// feedback it receives and the outputs it produces (docs/AGENT_PLAN.md,
// section 1). The host declares each channel's type; sizes, value ranges and
// vocabularies left unspecified are inferred from the data.
#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace dpan {
namespace agent {

enum class InputType { NUMERIC, CATEGORICAL, TEXT, IMAGE, TIME_SERIES };
enum class FeedbackType { LABEL, TARGET, REWARD, OUTCOME };
enum class OutputType { CLASS, VALUE, ACTION, NEXT_OBSERVATION, ANOMALY, CONTINUOUS_ACTION };

const char* ToString(InputType type);
const char* ToString(FeedbackType type);
const char* ToString(OutputType type);

struct InputSpec {
    std::string name;
    InputType type{InputType::NUMERIC};

    /// NUMERIC: vector length. TIME_SERIES: number of series.
    /// 0 = infer from the first value. Must be 0 for other types.
    size_t size{0};

    /// CATEGORICAL only: the known categories (one-hot encoded).
    /// Empty = open vocabulary (hashed).
    std::vector<std::string> categories;

    /// IMAGE only: expected size (both 0 = infer from the first image)
    size_t width{0};
    size_t height{0};

    /// IMAGE only: 1 (gray), 3 (RGB) or 4 (RGBA). Images with another
    /// channel count are converted.
    size_t image_channels{1};
};

struct FeedbackSpec {
    std::string name;
    FeedbackType type{FeedbackType::LABEL};

    /// Name of the output this feedback trains
    std::string output;
};

struct OutputSpec {
    std::string name;
    OutputType type{OutputType::CLASS};

    /// CLASS: class names (empty = learned from LABEL feedback).
    /// ACTION: action names (required). Empty for other types.
    std::vector<std::string> classes_or_actions;

    /// VALUE: number of values predicted. CONTINUOUS_ACTION: number of
    /// action values (dimensions).
    size_t size{1};

    /// CONTINUOUS_ACTION only: bounds of each action value (low < high).
    /// Both empty = [-1, 1] in every dimension; otherwise `size` values each.
    std::vector<float> low;
    std::vector<float> high;
};

struct EnvironmentSpec {
    std::vector<InputSpec> inputs;
    std::vector<FeedbackSpec> feedback;
    std::vector<OutputSpec> outputs;
};

/// Image in row-major order with interleaved channels:
/// pixels[(y * width + x) * channels + c]. Values in [0, 1] or [0, 255].
struct Image {
    size_t width{0};
    size_t height{0};
    size_t channels{1};
    std::vector<float> pixels;
};

/// One channel's value in an observation or a feedback message:
///   NUMERIC      number or vector of numbers (NaN = missing element)
///   CATEGORICAL  string, or for a known category set its index as a number
///   TEXT         string
///   TIME_SERIES  number (one series) or vector (one value per series)
///   IMAGE        Image
using Value = std::variant<double, std::vector<float>, std::string, Image>;

/// Values by input channel name. Channels may be left out (missing).
using Observation = std::map<std::string, Value>;

/// Name of the alternative a Value holds ("number", "vector", "text", "image")
const char* ValueTypeName(const Value& value);

/// All problems with a spec, one message each (empty = valid)
std::vector<std::string> ValidationErrors(const EnvironmentSpec& spec);

/// Problems with the input channels only
std::vector<std::string> InputValidationErrors(const std::vector<InputSpec>& inputs);

/// @throws std::invalid_argument listing every problem
void Validate(const EnvironmentSpec& spec);

}  // namespace agent
}  // namespace dpan
