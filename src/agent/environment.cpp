// File: src/agent/environment.cpp
#include "agent/environment.hpp"

#include <set>
#include <stdexcept>

namespace dpan {
namespace agent {

const char* ToString(InputType type) {
    switch (type) {
        case InputType::NUMERIC: return "NUMERIC";
        case InputType::CATEGORICAL: return "CATEGORICAL";
        case InputType::TEXT: return "TEXT";
        case InputType::IMAGE: return "IMAGE";
        case InputType::TIME_SERIES: return "TIME_SERIES";
    }
    return "UNKNOWN";
}

const char* ToString(FeedbackType type) {
    switch (type) {
        case FeedbackType::LABEL: return "LABEL";
        case FeedbackType::TARGET: return "TARGET";
        case FeedbackType::REWARD: return "REWARD";
        case FeedbackType::OUTCOME: return "OUTCOME";
    }
    return "UNKNOWN";
}

const char* ToString(OutputType type) {
    switch (type) {
        case OutputType::CLASS: return "CLASS";
        case OutputType::VALUE: return "VALUE";
        case OutputType::ACTION: return "ACTION";
        case OutputType::NEXT_OBSERVATION: return "NEXT_OBSERVATION";
        case OutputType::ANOMALY: return "ANOMALY";
    }
    return "UNKNOWN";
}

const char* ValueTypeName(const Value& value) {
    switch (value.index()) {
        case 0: return "number";
        case 1: return "vector";
        case 2: return "text";
        default: return "image";
    }
}

namespace {

std::string Quote(const std::string& name) { return "'" + name + "'"; }

/// Reports empty and duplicate names in one list
template <typename Spec>
void CheckNames(const std::vector<Spec>& specs, const std::string& kind,
                std::vector<std::string>& errors) {
    std::set<std::string> seen;
    for (size_t i = 0; i < specs.size(); ++i) {
        const std::string& name = specs[i].name;
        if (name.empty()) {
            errors.push_back(kind + " #" + std::to_string(i) + " has no name");
        } else if (!seen.insert(name).second) {
            errors.push_back("duplicate " + kind + " name " + Quote(name));
        }
    }
}

/// Reports empty and duplicate entries in a list of category/class names
void CheckNameList(const std::vector<std::string>& names, const std::string& owner,
                   const std::string& what, std::vector<std::string>& errors) {
    std::set<std::string> seen;
    for (const auto& name : names) {
        if (name.empty()) {
            errors.push_back(owner + " has an empty " + what + " name");
        } else if (!seen.insert(name).second) {
            errors.push_back(owner + " lists " + what + " " + Quote(name) + " twice");
        }
    }
}

bool Accepts(OutputType output, FeedbackType feedback) {
    switch (feedback) {
        case FeedbackType::LABEL: return output == OutputType::CLASS;
        case FeedbackType::TARGET: return output == OutputType::VALUE;
        case FeedbackType::REWARD: return output == OutputType::ACTION;
        case FeedbackType::OUTCOME:
            return output == OutputType::CLASS || output == OutputType::ACTION;
    }
    return false;
}

}  // namespace

std::vector<std::string> InputValidationErrors(const std::vector<InputSpec>& inputs) {
    std::vector<std::string> errors;
    if (inputs.empty()) {
        errors.push_back("no input channels");
    }
    CheckNames(inputs, "input", errors);

    for (const auto& input : inputs) {
        std::string owner = "input " + Quote(input.name) + " (" + ToString(input.type) + ")";
        bool numeric_like = input.type == InputType::NUMERIC || input.type == InputType::TIME_SERIES;
        if (!numeric_like && input.size != 0) {
            errors.push_back(owner + ": size must be 0 (only NUMERIC and TIME_SERIES have a size)");
        }
        if (input.type == InputType::CATEGORICAL) {
            CheckNameList(input.categories, owner, "category", errors);
        } else if (!input.categories.empty()) {
            errors.push_back(owner + ": categories are only allowed for CATEGORICAL inputs");
        }
        if (input.type == InputType::IMAGE) {
            if ((input.width == 0) != (input.height == 0)) {
                errors.push_back(owner + ": set both width and height, or neither");
            }
            size_t c = input.image_channels;
            if (c != 1 && c != 3 && c != 4) {
                errors.push_back(owner + ": image_channels must be 1, 3 or 4");
            }
        } else if (input.width != 0 || input.height != 0) {
            errors.push_back(owner + ": width and height are only allowed for IMAGE inputs");
        }
    }
    return errors;
}

std::vector<std::string> ValidationErrors(const EnvironmentSpec& spec) {
    std::vector<std::string> errors = InputValidationErrors(spec.inputs);

    if (spec.outputs.empty()) {
        errors.push_back("no outputs");
    }
    CheckNames(spec.outputs, "output", errors);
    CheckNames(spec.feedback, "feedback", errors);

    std::map<std::string, const OutputSpec*> outputs;
    for (const auto& output : spec.outputs) {
        outputs.emplace(output.name, &output);
        std::string owner = "output " + Quote(output.name) + " (" + ToString(output.type) + ")";
        switch (output.type) {
            case OutputType::CLASS:
                CheckNameList(output.classes_or_actions, owner, "class", errors);
                break;
            case OutputType::ACTION:
                if (output.classes_or_actions.empty()) {
                    errors.push_back(owner + ": list its actions in classes_or_actions");
                }
                CheckNameList(output.classes_or_actions, owner, "action", errors);
                break;
            default:
                if (!output.classes_or_actions.empty()) {
                    errors.push_back(owner + ": classes_or_actions is only allowed for CLASS "
                                             "and ACTION outputs");
                }
                break;
        }
        if (output.type == OutputType::VALUE && output.size == 0) {
            errors.push_back(owner + ": size must be at least 1");
        }
    }

    std::set<std::string> trained;
    for (const auto& feedback : spec.feedback) {
        std::string owner = "feedback " + Quote(feedback.name) + " (" + ToString(feedback.type) + ")";
        auto it = outputs.find(feedback.output);
        if (it == outputs.end()) {
            errors.push_back(owner + ": unknown output " + Quote(feedback.output));
            continue;
        }
        if (!Accepts(it->second->type, feedback.type)) {
            errors.push_back(owner + " cannot train output " + Quote(feedback.output) + " (" +
                             ToString(it->second->type) + ")");
            continue;
        }
        trained.insert(feedback.output);
    }

    for (const auto& output : spec.outputs) {
        bool self_supervised = output.type == OutputType::NEXT_OBSERVATION ||
                               output.type == OutputType::ANOMALY;
        if (!self_supervised && !output.name.empty() && trained.count(output.name) == 0) {
            const char* needed = output.type == OutputType::CLASS    ? "LABEL or OUTCOME"
                                 : output.type == OutputType::VALUE  ? "TARGET"
                                                                     : "REWARD or OUTCOME";
            errors.push_back("output " + Quote(output.name) + " (" + ToString(output.type) +
                             ") has no feedback channel; add " + needed + " feedback for it");
        }
    }
    return errors;
}

void Validate(const EnvironmentSpec& spec) {
    auto errors = ValidationErrors(spec);
    if (errors.empty()) {
        return;
    }
    std::string message = "Invalid environment spec:";
    for (const auto& error : errors) {
        message += "\n  - " + error;
    }
    throw std::invalid_argument(message);
}

}  // namespace agent
}  // namespace dpan
