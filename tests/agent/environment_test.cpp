// File: tests/agent/environment_test.cpp
#include "agent/environment.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>

namespace dpan {
namespace agent {
namespace {

EnvironmentSpec ValidSpec() {
    EnvironmentSpec spec;
    spec.inputs = {
        {"temperature", InputType::NUMERIC},
        {"weather", InputType::CATEGORICAL, 0, {"sun", "rain"}},
        {"notes", InputType::TEXT},
        {"camera", InputType::IMAGE, 0, {}, 32, 32, 3},
        {"load", InputType::TIME_SERIES, 2},
    };
    spec.outputs = {
        {"kind", OutputType::CLASS, {"a", "b"}},
        {"price", OutputType::VALUE, {}, 2},
        {"move", OutputType::ACTION, {"left", "right"}},
        {"next", OutputType::NEXT_OBSERVATION},
        {"odd", OutputType::ANOMALY},
    };
    spec.feedback = {
        {"label", FeedbackType::LABEL, "kind"},
        {"target", FeedbackType::TARGET, "price"},
        {"reward", FeedbackType::REWARD, "move"},
        {"correct", FeedbackType::OUTCOME, "kind"},
    };
    return spec;
}

/// True if some error message contains `text`
bool HasError(const EnvironmentSpec& spec, const std::string& text) {
    auto errors = ValidationErrors(spec);
    return std::any_of(errors.begin(), errors.end(),
                       [&](const std::string& e) { return e.find(text) != std::string::npos; });
}

TEST(EnvironmentSpecTest, ValidSpecHasNoErrors) {
    EXPECT_TRUE(ValidationErrors(ValidSpec()).empty());
    EXPECT_NO_THROW(Validate(ValidSpec()));
}

TEST(EnvironmentSpecTest, MinimalSelfSupervisedSpecIsValid) {
    EnvironmentSpec spec;
    spec.inputs = {{"x", InputType::NUMERIC}};
    spec.outputs = {{"odd", OutputType::ANOMALY}};
    EXPECT_TRUE(ValidationErrors(spec).empty());
}

TEST(EnvironmentSpecTest, ClassesMayBeLearnedFromLabels) {
    auto spec = ValidSpec();
    spec.outputs[0].classes_or_actions.clear();
    EXPECT_TRUE(ValidationErrors(spec).empty());
}

TEST(EnvironmentSpecTest, RejectsMissingInputsAndOutputs) {
    EnvironmentSpec spec;
    EXPECT_TRUE(HasError(spec, "no input channels"));
    EXPECT_TRUE(HasError(spec, "no outputs"));
}

TEST(EnvironmentSpecTest, RejectsEmptyAndDuplicateNames) {
    auto spec = ValidSpec();
    spec.inputs.push_back({"", InputType::NUMERIC});
    spec.inputs.push_back({"notes", InputType::TEXT});
    spec.outputs.push_back({"odd", OutputType::ANOMALY});
    spec.feedback.push_back({"label", FeedbackType::LABEL, "kind"});
    EXPECT_TRUE(HasError(spec, "input #5 has no name"));
    EXPECT_TRUE(HasError(spec, "duplicate input name 'notes'"));
    EXPECT_TRUE(HasError(spec, "duplicate output name 'odd'"));
    EXPECT_TRUE(HasError(spec, "duplicate feedback name 'label'"));
}

TEST(EnvironmentSpecTest, RejectsFieldsThatDoNotApplyToTheInputType) {
    auto spec = ValidSpec();
    spec.inputs[0].categories = {"a"};
    spec.inputs[2].size = 4;
    spec.inputs[0].width = 8;
    EXPECT_TRUE(HasError(spec, "input 'temperature' (NUMERIC): categories are only allowed"));
    EXPECT_TRUE(HasError(spec, "input 'notes' (TEXT): size must be 0"));
    EXPECT_TRUE(HasError(spec, "input 'temperature' (NUMERIC): width and height are only allowed"));
}

TEST(EnvironmentSpecTest, RejectsBadCategoryLists) {
    auto spec = ValidSpec();
    spec.inputs[1].categories = {"sun", "", "sun"};
    EXPECT_TRUE(HasError(spec, "empty category name"));
    EXPECT_TRUE(HasError(spec, "lists category 'sun' twice"));
}

TEST(EnvironmentSpecTest, RejectsBadImageShape) {
    auto spec = ValidSpec();
    spec.inputs[3].height = 0;
    spec.inputs[3].image_channels = 2;
    EXPECT_TRUE(HasError(spec, "set both width and height"));
    EXPECT_TRUE(HasError(spec, "image_channels must be 1, 3 or 4"));
}

TEST(EnvironmentSpecTest, RejectsBadOutputs) {
    auto spec = ValidSpec();
    spec.outputs[1].size = 0;
    spec.outputs[2].classes_or_actions.clear();
    spec.outputs[3].classes_or_actions = {"x"};
    EXPECT_TRUE(HasError(spec, "output 'price' (VALUE): size must be at least 1"));
    EXPECT_TRUE(HasError(spec, "output 'move' (ACTION): list its actions"));
    EXPECT_TRUE(HasError(spec, "output 'next' (NEXT_OBSERVATION): classes_or_actions is only"));
}

TEST(EnvironmentSpecTest, RejectsFeedbackForUnknownOrIncompatibleOutputs) {
    auto spec = ValidSpec();
    spec.feedback.push_back({"stray", FeedbackType::LABEL, "nothing"});
    spec.feedback.push_back({"wrong", FeedbackType::TARGET, "kind"});
    spec.feedback.push_back({"self", FeedbackType::LABEL, "next"});
    EXPECT_TRUE(HasError(spec, "feedback 'stray' (LABEL): unknown output 'nothing'"));
    EXPECT_TRUE(HasError(spec, "feedback 'wrong' (TARGET) cannot train output 'kind' (CLASS)"));
    EXPECT_TRUE(HasError(spec, "feedback 'self' (LABEL) cannot train output 'next'"));
}

TEST(EnvironmentSpecTest, RequiresFeedbackForSupervisedOutputs) {
    auto spec = ValidSpec();
    spec.feedback.clear();
    EXPECT_TRUE(HasError(spec, "output 'kind' (CLASS) has no feedback channel; add LABEL or OUTCOME"));
    EXPECT_TRUE(HasError(spec, "output 'price' (VALUE) has no feedback channel; add TARGET"));
    EXPECT_TRUE(HasError(spec, "output 'move' (ACTION) has no feedback channel; add REWARD or OUTCOME"));
    EXPECT_FALSE(HasError(spec, "'next'"));
    EXPECT_FALSE(HasError(spec, "'odd'"));
}

TEST(EnvironmentSpecTest, OutcomeCanTrainActions) {
    auto spec = ValidSpec();
    spec.feedback[2] = {"worked", FeedbackType::OUTCOME, "move"};
    EXPECT_TRUE(ValidationErrors(spec).empty());
}

TEST(EnvironmentSpecTest, ValidateListsEveryProblem) {
    auto spec = ValidSpec();
    spec.inputs[2].size = 4;
    spec.feedback.clear();
    try {
        Validate(spec);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        std::string message = e.what();
        EXPECT_NE(std::string::npos, message.find("Invalid environment spec:"));
        EXPECT_NE(std::string::npos, message.find("'notes'"));
        EXPECT_NE(std::string::npos, message.find("'kind'"));
        EXPECT_NE(std::string::npos, message.find("'price'"));
        EXPECT_NE(std::string::npos, message.find("'move'"));
    }
}

TEST(ValueTest, HoldsEachChannelKind) {
    Value number = 2.5;
    Value vector = std::vector<float>{1.0f, 2.0f};
    Value text = std::string("hello");
    Value image = Image{2, 1, 1, {0.0f, 1.0f}};
    EXPECT_STREQ("number", ValueTypeName(number));
    EXPECT_STREQ("vector", ValueTypeName(vector));
    EXPECT_STREQ("text", ValueTypeName(text));
    EXPECT_STREQ("image", ValueTypeName(image));
}

}  // namespace
}  // namespace agent
}  // namespace dpan
