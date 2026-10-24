// File: tests/agent/json_test.cpp
#include "agent/json.hpp"
#include "agent/spec_json.hpp"

#include <gtest/gtest.h>

#include <locale>
#include <cmath>
#include <limits>

namespace dpan {
namespace agent {
namespace {

// ============================================================================
// Json
// ============================================================================

TEST(JsonTest, ParsesAllTypes) {
    Json j = Json::Parse(R"( {"a": [1, -2.5e3, true, false, null], "b": {"c": "x\"\\\/\n\u00e9\ud83d\ude00"}} )");
    const auto& a = j.Find("a")->AsArray();
    ASSERT_EQ(5u, a.size());
    EXPECT_EQ(1.0, a[0].AsNumber());
    EXPECT_EQ(-2500.0, a[1].AsNumber());
    EXPECT_TRUE(a[2].AsBool());
    EXPECT_FALSE(a[3].AsBool());
    EXPECT_TRUE(a[4].IsNull());
    EXPECT_EQ("x\"\\/\n\xC3\xA9\xF0\x9F\x98\x80", j.Find("b")->Find("c")->AsString());
    EXPECT_EQ(nullptr, j.Find("missing"));
}

TEST(JsonTest, RoundTrips) {
    const std::string text = R"({"n":0.10000000000000001,"i":-42,"s":"tab\tquote\"","e":[],"o":{},"z":null,"t":true})";
    EXPECT_EQ(text, Json::Parse(text).Dump());
    float f = 0.1f;
    EXPECT_EQ(f, static_cast<float>(Json::Parse(Json(f).Dump()).AsNumber()));
    EXPECT_EQ("null", Json(std::numeric_limits<double>::quiet_NaN()).Dump());
    EXPECT_EQ("\"\\u0001\"", Json(std::string(1, '\x01')).Dump());
}

TEST(JsonTest, RejectsMalformedTextWithPosition) {
    for (const char* bad : {"", "{", "[1,]", "{\"a\" 1}", "01", "1.", "-", "tru", "\"abc", "\"\\x\"", "[1] 2",
                            "{\"a\":1,\"a\":2}", "\"\\ud800\"", "\"a\nb\""}) {
        EXPECT_THROW(Json::Parse(bad), std::invalid_argument) << bad;
    }
    try {
        Json::Parse("{\n  \"a\": nope}");
        FAIL();
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string::npos, std::string(e.what()).find("line 2, column 8")) << e.what();
    }
    EXPECT_THROW(Json::Parse(std::string(1000, '[') + std::string(1000, ']')), std::invalid_argument);
}

TEST(JsonTest, WrongTypeAccessThrows) {
    Json j = Json::Parse("[1]");
    EXPECT_THROW(j.AsObject(), std::invalid_argument);
    EXPECT_THROW(j.AsArray()[0].AsString(), std::invalid_argument);
}

/// Decimal comma and thousands grouping, as in many host locales
struct CommaPunct : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
    char do_thousands_sep() const override { return '.'; }
    std::string do_grouping() const override { return "\3"; }
};

TEST(JsonTest, NumbersIgnoreTheHostLocale) {
    std::locale previous = std::locale::global(std::locale(std::locale::classic(), new CommaPunct));
    double parsed = Json::Parse("1234.5").AsNumber();
    std::string written = Json(1234.5).Dump() + " " + Json(1234567.0).Dump();
    std::locale::global(previous);
    EXPECT_EQ(1234.5, parsed);
    EXPECT_EQ("1234.5 1234567", written);
}

// ============================================================================
// Spec, config, values
// ============================================================================

const char* kSpec = R"({
  "inputs": [
    {"name": "x", "type": "numeric", "size": 4},
    {"name": "region", "type": "CATEGORICAL", "categories": ["north", "south"]},
    {"name": "note", "type": "text"},
    {"name": "load", "type": "time_series", "size": 1},
    {"name": "camera", "type": "image", "width": 8, "height": 6, "channels": 3}
  ],
  "feedback": [
    {"name": "label", "type": "label", "output": "kind"},
    {"name": "reward", "type": "reward", "output": "move"},
    {"name": "target", "type": "target", "output": "y"}
  ],
  "outputs": [
    {"name": "kind", "type": "class", "classes": ["a", "b"]},
    {"name": "move", "type": "action", "actions": ["left", "right"]},
    {"name": "y", "type": "value", "size": 2},
    {"name": "next", "type": "next_observation"},
    {"name": "unusual", "type": "anomaly"}
  ]
})";

TEST(SpecJsonTest, ReadsEveryChannelType) {
    EnvironmentSpec spec = SpecFromJson(kSpec);
    ASSERT_EQ(5u, spec.inputs.size());
    EXPECT_EQ(InputType::NUMERIC, spec.inputs[0].type);
    EXPECT_EQ(4u, spec.inputs[0].size);
    EXPECT_EQ(InputType::CATEGORICAL, spec.inputs[1].type);
    EXPECT_EQ((std::vector<std::string>{"north", "south"}), spec.inputs[1].categories);
    EXPECT_EQ(InputType::TEXT, spec.inputs[2].type);
    EXPECT_EQ(InputType::TIME_SERIES, spec.inputs[3].type);
    EXPECT_EQ(InputType::IMAGE, spec.inputs[4].type);
    EXPECT_EQ(8u, spec.inputs[4].width);
    EXPECT_EQ(6u, spec.inputs[4].height);
    EXPECT_EQ(3u, spec.inputs[4].image_channels);
    ASSERT_EQ(3u, spec.feedback.size());
    EXPECT_EQ(FeedbackType::REWARD, spec.feedback[1].type);
    EXPECT_EQ("move", spec.feedback[1].output);
    ASSERT_EQ(5u, spec.outputs.size());
    EXPECT_EQ((std::vector<std::string>{"left", "right"}), spec.outputs[1].classes_or_actions);
    EXPECT_EQ(2u, spec.outputs[2].size);
    EXPECT_EQ(OutputType::ANOMALY, spec.outputs[4].type);
    EXPECT_NO_THROW(Validate(spec));
    // Written and read again: the same spec
    EXPECT_EQ(SpecToJson(spec), SpecToJson(SpecFromJson(SpecToJson(spec))));
}

TEST(SpecJsonTest, ReportsWhereTheProblemIs) {
    auto message = [](const std::string& json) {
        try {
            SpecFromJson(json);
        } catch (const std::invalid_argument& e) {
            return std::string(e.what());
        }
        return std::string("no error");
    };
    EXPECT_EQ("inputs[0]: unknown member \"sizes\"", message(R"({"inputs":[{"name":"x","type":"numeric","sizes":3}]})"));
    EXPECT_EQ("inputs[0]: unknown type \"picture\" (expected one of: numeric, categorical, text, image, time_series)",
              message(R"({"inputs":[{"name":"x","type":"picture"}]})"));
    EXPECT_EQ("outputs[0]: missing \"name\"", message(R"({"outputs":[{"type":"class"}]})"));
    EXPECT_EQ("inputs[0]: \"size\": expected a whole number from 0 to 18446744073709551615",
              message(R"({"inputs":[{"name":"x","type":"numeric","size":-1}]})"));
    EXPECT_EQ("spec: unknown member \"input\"", message(R"({"input":[]})"));
    EXPECT_NE(std::string::npos, message("{").find("JSON:"));
}

TEST(SpecJsonTest, ConfigFieldsByName) {
    AgentConfig config = ConfigFromJson(
        R"({"seed": 7, "learning_rate": 0.01, "detect_drift": false, "memory_budget_bytes": 1048576,
            "drift_threshold": 12.5, "encoder": {"text_buckets": 512, "statistics_limit": 0}})");
    EXPECT_EQ(7u, config.seed);
    EXPECT_FLOAT_EQ(0.01f, config.learning_rate);
    EXPECT_FALSE(config.detect_drift);
    EXPECT_EQ(1048576u, config.memory_budget_bytes);
    EXPECT_EQ(12.5, config.drift_threshold);
    EXPECT_EQ(512u, config.encoder.text_buckets);
    EXPECT_EQ(0u, config.encoder.statistics_limit);
    EXPECT_EQ(AgentConfig{}.batch_size, config.batch_size);  // untouched

    // Every field is written and read back
    AgentConfig round = ConfigFromJson(ConfigToJson(config));
    EXPECT_EQ(ConfigToJson(config), ConfigToJson(round));

    EXPECT_THROW(ConfigFromJson(R"({"sed": 1})"), std::invalid_argument);
    EXPECT_THROW(ConfigFromJson(R"({"seed": 1.5})"), std::invalid_argument);
    EXPECT_THROW(ConfigFromJson(R"({"seed": 5000000000})"), std::invalid_argument);
    EXPECT_THROW(ConfigFromJson(R"({"detect_drift": 1})"), std::invalid_argument);
    EXPECT_THROW(ConfigFromJson(R"({"learning_rate": "fast"})"), std::invalid_argument);
    EXPECT_THROW(ConfigFromJson(R"({"learning_rate": 1e300})"), std::invalid_argument);
    EXPECT_THROW(ConfigFromJson(R"({"encoder": {"bucket": 3}})"), std::invalid_argument);
    EXPECT_THROW(ConfigFromJson("[]"), std::invalid_argument);
}

TEST(SpecJsonTest, ObservationValues) {
    Observation o = ObservationFromJson(
        R"({"x": [1, null, 3], "t": 2.5, "s": "rain", "missing": null,
            "img": {"width": 2, "height": 1, "pixels": [0, 255]}})");
    ASSERT_EQ(4u, o.size());
    const auto& x = std::get<std::vector<float>>(o["x"]);
    EXPECT_EQ(1.0f, x[0]);
    EXPECT_TRUE(std::isnan(x[1]));
    EXPECT_EQ(2.5, std::get<double>(o["t"]));
    EXPECT_EQ("rain", std::get<std::string>(o["s"]));
    const auto& img = std::get<Image>(o["img"]);
    EXPECT_EQ(2u, img.width);
    EXPECT_EQ(1u, img.channels);
    EXPECT_EQ((std::vector<float>{0, 255}), img.pixels);
    EXPECT_THROW(ObservationFromJson(R"({"x": true})"), std::invalid_argument);
    EXPECT_THROW(ObservationFromJson(R"({"x": ["a"]})"), std::invalid_argument);
    EXPECT_THROW(ObservationFromJson(R"({"x": {"width": 1}})"), std::invalid_argument);
}

TEST(SpecJsonTest, PredictionsAndStatsAreJson) {
    EnvironmentSpec spec = SpecFromJson(R"({"inputs":[{"name":"x","type":"numeric"}],
        "feedback":[{"name":"label","type":"label","output":"kind"}],
        "outputs":[{"name":"kind","type":"class"}]})");
    Agent agent(spec);
    for (int i = 0; i < 30; ++i) {
        StepId step = agent.Observe({{"x", std::vector<float>{float(i % 2), 1.0f}}});
        agent.Feedback(step, "label", std::string(i % 2 ? "odd" : "even"));
    }
    StepId step = agent.Observe({{"x", std::vector<float>{1.0f, 1.0f}}});
    Json p = Json::Parse(PredictionToJson(agent.Predict(step, "kind")));
    EXPECT_EQ("kind", p.Find("output")->AsString());
    EXPECT_EQ("class", p.Find("type")->AsString());
    EXPECT_TRUE(p.Find("available")->AsBool());
    EXPECT_EQ(2u, p.Find("classes")->AsArray().size());
    EXPECT_EQ(2u, p.Find("probabilities")->AsArray().size());

    Json s = Json::Parse(StatsToJson(agent.Stats()));
    EXPECT_EQ(31.0, s.Find("steps")->AsNumber());
    EXPECT_EQ(30.0, s.Find("feedback")->AsNumber());
    ASSERT_EQ(1u, s.Find("outputs")->AsArray().size());
    EXPECT_EQ("kind", s.Find("outputs")->AsArray()[0].Find("name")->AsString());
}

}  // namespace
}  // namespace agent
}  // namespace dpan
