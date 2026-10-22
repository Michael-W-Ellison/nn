// File: tests/capi/c_api_test.cpp
//
// The C API (dpan_c.h) through the shared library dpan_c.
#include "dpan_c.h"

#include "../test_temp_path.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {

const char* kClassifier = R"({
  "inputs": [{"name": "x", "type": "numeric", "size": 2}, {"name": "note", "type": "text"}],
  "feedback": [{"name": "label", "type": "label", "output": "kind"}],
  "outputs": [{"name": "kind", "type": "class", "classes": ["a", "b"]},
              {"name": "unusual", "type": "anomaly"}]
})";

const char* kGrid = R"({
  "inputs": [{"name": "position", "type": "numeric", "size": 2}],
  "feedback": [{"name": "reward", "type": "reward", "output": "move"}],
  "outputs": [{"name": "move", "type": "action", "actions": ["up", "down", "left", "right"]}]
})";

/// Owns a string returned by the library
std::string Take(char* text) {
    std::string out = text ? text : "";
    dpan_string_free(text);
    return out;
}

class CApiTest : public ::testing::Test {
protected:
    void SetUp() override { dpan_clock_use_manual(1'700'000'000'000'000); }
    void TearDown() override { dpan_clock_use_system(); }
};

TEST_F(CApiTest, ReportsErrorsAsStatusAndMessage) {
    dpan_agent* agent = nullptr;
    EXPECT_EQ(DPAN_INVALID_ARGUMENT, dpan_agent_create("{", nullptr, &agent));
    EXPECT_NE(std::string::npos, std::string(dpan_last_error()).find("JSON"));
    EXPECT_EQ(nullptr, agent);  // left unchanged
    EXPECT_EQ(DPAN_INVALID_ARGUMENT, dpan_agent_create(R"({"outputs": [{"name": "y", "type": "value"}]})", nullptr,
                                                       &agent));
    EXPECT_NE(std::string::npos, std::string(dpan_last_error()).find("TARGET"));
    EXPECT_EQ(DPAN_INVALID_ARGUMENT, dpan_agent_create(kClassifier, R"({"sed": 1})", &agent));
    EXPECT_EQ(DPAN_INVALID_ARGUMENT, dpan_agent_create(nullptr, nullptr, &agent));
    EXPECT_STREQ("spec_json is NULL", dpan_last_error());
    EXPECT_EQ(DPAN_RUNTIME_ERROR, dpan_agent_load("/nonexistent/agent.dpag", &agent));

    ASSERT_EQ(DPAN_OK, dpan_agent_create(kGrid, "", &agent));
    dpan_step step = 0;
    EXPECT_EQ(DPAN_OUT_OF_RANGE, dpan_agent_feedback_number(agent, 999, "reward", 1.0));
    ASSERT_EQ(DPAN_OK, dpan_agent_observe_json(agent, R"({"position": [0, 0]})", &step));
    EXPECT_EQ(DPAN_LOGIC_ERROR, dpan_agent_feedback_number(agent, step, "reward", 1.0));  // before act
    EXPECT_EQ(DPAN_INVALID_ARGUMENT, dpan_agent_feedback_number(agent, step, "bonus", 1.0));
    EXPECT_EQ(DPAN_INVALID_ARGUMENT, dpan_agent_observe_json(agent, R"({"position": "here"})", &step));
    EXPECT_EQ(DPAN_INVALID_ARGUMENT, dpan_agent_save(agent, nullptr));
    EXPECT_EQ(DPAN_RUNTIME_ERROR, dpan_agent_save(agent, "/nonexistent/dir/agent.dpag"));
    dpan_agent_free(agent);
    dpan_agent_free(nullptr);
    dpan_string_free(nullptr);

    dpan_clock_use_system();
    EXPECT_EQ(DPAN_LOGIC_ERROR, dpan_clock_advance(1));
}

TEST_F(CApiTest, LastErrorIsPerThread) {
    dpan_agent* agent = nullptr;
    ASSERT_EQ(DPAN_INVALID_ARGUMENT, dpan_agent_create("[", nullptr, &agent));
    std::string other;
    std::thread([&] { other = dpan_last_error(); }).join();
    EXPECT_EQ("", other);
    EXPECT_NE("", std::string(dpan_last_error()));
}

TEST_F(CApiTest, BuilderAndJsonObservationsAreEquivalent) {
    dpan_agent *built = nullptr, *parsed = nullptr;
    ASSERT_EQ(DPAN_OK, dpan_agent_create(kClassifier, R"({"seed": 5})", &built));
    ASSERT_EQ(DPAN_OK, dpan_agent_create(kClassifier, R"({"seed": 5})", &parsed));
    dpan_observation* obs = dpan_observation_create();
    std::string last_built, last_parsed;
    for (int i = 0; i < 400; ++i) {
        float x[2] = {static_cast<float>(i % 2) + 0.01f * static_cast<float>(i % 7), 0.5f};
        const char* note = i % 2 ? "warm day" : "cold night";
        dpan_observation_clear(obs);
        ASSERT_EQ(DPAN_OK, dpan_observation_set_vector(obs, "x", x, 2));
        ASSERT_EQ(DPAN_OK, dpan_observation_set_text(obs, "note", note));
        dpan_step a = 0, b = 0;
        ASSERT_EQ(DPAN_OK, dpan_agent_observe(built, obs, &a));
        char json[128];
        std::snprintf(json, sizeof(json), R"({"x": [%.9g, 0.5], "note": "%s"})", x[0], note);
        ASSERT_EQ(DPAN_OK, dpan_agent_observe_json(parsed, json, &b));
        char *pa = nullptr, *pb = nullptr;
        ASSERT_EQ(DPAN_OK, dpan_agent_predict_json(built, a, "kind", &pa));
        ASSERT_EQ(DPAN_OK, dpan_agent_predict_json(parsed, b, "kind", &pb));
        last_built = Take(pa);
        last_parsed = Take(pb);
        ASSERT_EQ(last_built, last_parsed) << i;
        if (i % 2) {
            ASSERT_EQ(DPAN_OK, dpan_agent_feedback_text(built, a, "label", "b"));
            ASSERT_EQ(DPAN_OK, dpan_agent_feedback_json(parsed, b, "label", "\"b\""));
        } else {
            ASSERT_EQ(DPAN_OK, dpan_agent_feedback_number(built, a, "label", 0));
            ASSERT_EQ(DPAN_OK, dpan_agent_feedback_json(parsed, b, "label", "0"));
        }
        ASSERT_EQ(DPAN_OK, dpan_clock_advance(1'000'000));
    }
    EXPECT_NE(std::string::npos, last_built.find(R"("available":true)")) << last_built;
    EXPECT_NE(std::string::npos, last_built.find(R"("label":"b")")) << last_built;
    char *sa = nullptr, *sb = nullptr;
    ASSERT_EQ(DPAN_OK, dpan_agent_stats_json(built, &sa));
    ASSERT_EQ(DPAN_OK, dpan_agent_stats_json(parsed, &sb));
    EXPECT_EQ(Take(sa), Take(sb));
    dpan_observation_free(obs);
    dpan_agent_free(built);
    dpan_agent_free(parsed);
}

TEST_F(CApiTest, ActSaveAndLoad) {
    dpan_agent* agent = nullptr;
    ASSERT_EQ(DPAN_OK, dpan_agent_create(kGrid, nullptr, &agent));
    dpan_step step = 0;
    for (int i = 0; i < 300; ++i) {
        char json[64];
        std::snprintf(json, sizeof(json), R"({"position": [%d, %d]})", i % 5, (i / 5) % 5);
        ASSERT_EQ(DPAN_OK, dpan_agent_observe_json(agent, json, &step));
        const char* move = nullptr;
        size_t index = 99;
        ASSERT_EQ(DPAN_OK, dpan_agent_act(agent, step, "move", 1, &move, &index));
        ASSERT_LT(index, 4u);
        static const char* names[] = {"up", "down", "left", "right"};
        ASSERT_STREQ(names[index], move);
        ASSERT_EQ(DPAN_OK, dpan_agent_feedback_number(agent, step, "reward", index == 3 ? 1.0 : -0.1));
        if (i % 10 == 9) {
            ASSERT_EQ(DPAN_OK, dpan_agent_end_episode(agent, 1));
        }
    }
    const std::string path = dpan::test::UniqueTempPath("c_api_agent");
    ASSERT_EQ(DPAN_OK, dpan_agent_save(agent, path.c_str()));
    dpan_agent* loaded = nullptr;
    ASSERT_EQ(DPAN_OK, dpan_agent_load(path.c_str(), &loaded));
    std::remove(path.c_str());

    char *spec_a = nullptr, *spec_b = nullptr, *config_a = nullptr, *config_b = nullptr;
    ASSERT_EQ(DPAN_OK, dpan_agent_spec_json(agent, &spec_a));
    ASSERT_EQ(DPAN_OK, dpan_agent_spec_json(loaded, &spec_b));
    ASSERT_EQ(DPAN_OK, dpan_agent_config_json(agent, &config_a));
    ASSERT_EQ(DPAN_OK, dpan_agent_config_json(loaded, &config_b));
    EXPECT_EQ(Take(spec_a), Take(spec_b));
    std::string config = Take(config_a);
    EXPECT_EQ(config, Take(config_b));
    EXPECT_NE(std::string::npos, config.find(R"("action_learning_rate":)"));

    // The same steps give the same actions and Q-values
    for (int i = 0; i < 50; ++i) {
        char json[64];
        std::snprintf(json, sizeof(json), R"({"position": [%d, %d]})", i % 5, (i * 3) % 5);
        dpan_step a = 0, b = 0;
        ASSERT_EQ(DPAN_OK, dpan_agent_observe_json(agent, json, &a));
        ASSERT_EQ(DPAN_OK, dpan_agent_observe_json(loaded, json, &b));
        const char *ma = nullptr, *mb = nullptr;
        ASSERT_EQ(DPAN_OK, dpan_agent_act(agent, a, "move", 1, &ma, nullptr));
        ASSERT_EQ(DPAN_OK, dpan_agent_act(loaded, b, "move", 1, &mb, nullptr));
        ASSERT_STREQ(ma, mb) << i;
        char *pa = nullptr, *pb = nullptr;
        ASSERT_EQ(DPAN_OK, dpan_agent_predict_json(agent, a, "move", &pa));
        ASSERT_EQ(DPAN_OK, dpan_agent_predict_json(loaded, b, "move", &pb));
        ASSERT_EQ(Take(pa), Take(pb)) << i;
        ASSERT_EQ(DPAN_OK, dpan_agent_feedback_number(agent, a, "reward", 0.5));
        ASSERT_EQ(DPAN_OK, dpan_agent_feedback_number(loaded, b, "reward", 0.5));
    }
    dpan_agent_free(agent);
    dpan_agent_free(loaded);
}

TEST_F(CApiTest, ImagesAndVectorFeedback) {
    const char* spec = R"({
      "inputs": [{"name": "camera", "type": "image", "width": 4, "height": 4}],
      "feedback": [{"name": "target", "type": "target", "output": "y"}],
      "outputs": [{"name": "y", "type": "value", "size": 2}]})";
    dpan_agent* agent = nullptr;
    ASSERT_EQ(DPAN_OK, dpan_agent_create(spec, nullptr, &agent));
    dpan_observation* obs = dpan_observation_create();
    std::vector<float> pixels(16);
    dpan_step step = 0;
    for (int i = 0; i < 50; ++i) {
        for (size_t p = 0; p < pixels.size(); ++p) pixels[p] = static_cast<float>((p * 17 + i) % 256);
        ASSERT_EQ(DPAN_OK, dpan_observation_set_image(obs, "camera", 4, 4, 1, pixels.data()));
        ASSERT_EQ(DPAN_OK, dpan_agent_observe(agent, obs, &step));
        float target[2] = {static_cast<float>(i % 3), 1.0f};
        ASSERT_EQ(DPAN_OK, dpan_agent_feedback_vector(agent, step, "target", target, 2));
    }
    char* prediction = nullptr;
    ASSERT_EQ(DPAN_OK, dpan_agent_predict_json(agent, step, "y", &prediction));
    EXPECT_NE(std::string::npos, Take(prediction).find(R"("type":"value")"));
    EXPECT_EQ(DPAN_INVALID_ARGUMENT, dpan_observation_set_image(obs, "camera", 4, 4, 1, nullptr));
    EXPECT_EQ(DPAN_INVALID_ARGUMENT,
              dpan_observation_set_image(obs, "camera", SIZE_MAX / 2, 4, 1, pixels.data()));
    EXPECT_EQ(DPAN_INVALID_ARGUMENT, dpan_agent_feedback_vector(agent, step, "target", nullptr, 2));
    dpan_observation_free(obs);
    dpan_agent_free(agent);
}

TEST_F(CApiTest, Version) { EXPECT_STRNE("", dpan_version()); }

}  // namespace
