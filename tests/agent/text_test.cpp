// File: tests/agent/text_test.cpp
//
// Phase 10 (docs/AGENT_PLAN.md): sentence meaning. Criteria were fixed
// before the first run. T1 and T3 need files outside the repository and run
// only when they are given:
//   DPAN_SST2_DIR     directory with train.tsv and test.tsv ("sentence\tlabel")
//   DPAN_WORD_VECTORS GloVe text file (glove.6B.100d.txt, uncompressed)
// With DPAN_TEXT_BASELINE set, every test runs without the word sequence
// (text_sequence_length 0: the hashed bag of words only) and only prints
// its results. DPAN_TEXT_CONFIG (JSON, as in AgentConfig) changes settings.

#include "agent/agent.hpp"
#include "agent/spec_json.hpp"
#include "core/clock.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace dpan {
namespace agent {
namespace {

using Example = std::pair<std::string, std::string>;  // sentence, label

class TextTest : public ::testing::TestWithParam<uint32_t> {
protected:
    void SetUp() override {
        clock_ = std::make_shared<ManualClock>(Timestamp::FromMicros(1'700'000'000'000'000));
        scope_ = std::make_unique<ScopedClock>(clock_);
    }
    void TearDown() override { scope_.reset(); }

    static bool Baseline() { return std::getenv("DPAN_TEXT_BASELINE") != nullptr; }
    static const char* Vectors() { return std::getenv("DPAN_WORD_VECTORS"); }

    static AgentConfig Config(uint32_t seed, bool vectors) {
        AgentConfig config;
        config.seed = seed;
        if (Baseline()) config.encoder.text_sequence_length = 0;
        if (vectors && !Baseline()) config.encoder.text_word_vectors = Vectors();
        if (const char* extra = std::getenv("DPAN_TEXT_CONFIG")) config = ConfigFromJson(extra, config);
        return config;
    }
    static void Check(bool pass, const std::string& result) {
        std::cout << (Baseline() ? "[bag of words] " : "") << result << (pass ? " (pass)" : " (fail)") << "\n";
        if (!Baseline()) EXPECT_TRUE(pass) << result;
    }

    /// Streams `train` once (label after each prediction), then the test
    /// accuracy without labels
    double Accuracy(const AgentConfig& config, const std::vector<Example>& train, const std::vector<Example>& test) {
        EnvironmentSpec spec;
        spec.inputs = {{"sentence", InputType::TEXT}};
        spec.outputs = {{"meaning", OutputType::CLASS}};
        spec.feedback = {{"label", FeedbackType::LABEL, "meaning"}};
        Agent agent(spec, config);
        for (const auto& [text, label] : train) {
            StepId step = agent.Observe({{"sentence", text}});
            agent.Feedback(step, "label", label);
            clock_->Advance(std::chrono::seconds(1));
        }
        size_t correct = 0;
        for (const auto& [text, label] : test) {
            Prediction p = agent.Predict(agent.Observe({{"sentence", text}}), "meaning");
            correct += p.available && p.label == label;
            clock_->Advance(std::chrono::seconds(1));
        }
        return static_cast<double>(correct) / static_cast<double>(test.size());
    }

    static std::string Describe(const std::string& name, uint32_t seed, double accuracy) {
        std::ostringstream s;
        s << name << " seed " << seed << ": test accuracy " << accuracy;
        return s.str();
    }

    std::shared_ptr<ManualClock> clock_;
    std::unique_ptr<ScopedClock> scope_;
};

const std::vector<std::string> kNouns = {"film", "meal", "book", "trip", "show", "song", "game", "room", "car", "plan"};
const std::vector<std::string> kAnimals = {"cat", "dog", "fox", "owl", "rat", "bear"};

/// "the <noun> was [not] <adjective>": positive iff the adjective is
/// positive and there is no "not", or negative with "not"
std::vector<Example> Negation(size_t count, std::mt19937& rng, const std::vector<std::string>& positive,
                              const std::vector<std::string>& negative, bool with_not) {
    std::uniform_int_distribution<size_t> noun(0, kNouns.size() - 1), adjective(0, positive.size() - 1);
    std::bernoulli_distribution coin(0.5);
    std::vector<Example> examples;
    for (size_t i = 0; i < count; ++i) {
        bool good = coin(rng), negated = with_not && coin(rng);
        std::string text = "the " + kNouns[noun(rng)] + " was " + (negated ? "not " : "") +
                           (good ? positive : negative)[adjective(rng)];
        examples.emplace_back(text, good != negated ? "positive" : "negative");
    }
    return examples;
}

/// "the A chased the B" / "the B was chased by the A": label A
std::vector<Example> Chases(size_t count, std::mt19937& rng) {
    std::uniform_int_distribution<size_t> animal(0, kAnimals.size() - 1);
    std::bernoulli_distribution passive(0.5);
    std::vector<Example> examples;
    for (size_t i = 0; i < count; ++i) {
        size_t a = animal(rng), b = animal(rng);
        while (b == a) b = animal(rng);
        std::string text = passive(rng) ? "the " + kAnimals[b] + " was chased by the " + kAnimals[a]
                                        : "the " + kAnimals[a] + " chased the " + kAnimals[b];
        examples.emplace_back(text, kAnimals[a]);
    }
    return examples;
}

std::vector<Example> ReadTsv(const std::string& path) {
    std::ifstream in(path);
    std::vector<Example> rows;
    std::string line;
    while (std::getline(in, line)) {
        size_t tab = line.rfind('\t');
        if (tab != std::string::npos) rows.emplace_back(line.substr(0, tab), line.substr(tab + 1));
    }
    return rows;
}

// ---------------------------------------------------------------------- T1
TEST_P(TextTest, T1_Sst2) {
    const char* dir = std::getenv("DPAN_SST2_DIR");
    if (!dir) GTEST_SKIP() << "DPAN_SST2_DIR not set";
    auto train = ReadTsv(std::string(dir) + "/train.tsv");
    auto test = ReadTsv(std::string(dir) + "/test.tsv");
    ASSERT_FALSE(train.empty());
    ASSERT_FALSE(test.empty());
    std::mt19937 rng(GetParam());
    std::shuffle(train.begin(), train.end(), rng);
    double plain = Accuracy(Config(GetParam(), false), train, test);
    Check(plain >= 0.75, Describe("T1 SST-2 without word vectors", GetParam(), plain));
    if (Vectors()) {
        double glove = Accuracy(Config(GetParam(), true), train, test);
        Check(glove >= 0.818, Describe("T1 SST-2 with word vectors", GetParam(), glove));
    }
}

// ---------------------------------------------------------------------- T2
TEST_P(TextTest, T2a_Negation) {
    const std::vector<std::string> positive = {"good", "great", "fine", "nice"};
    const std::vector<std::string> negative = {"bad", "awful", "poor", "dull"};
    std::mt19937 rng(GetParam() + 100);
    auto train = Negation(3000, rng, positive, negative, true);
    auto test = Negation(1000, rng, positive, negative, true);
    double accuracy = Accuracy(Config(GetParam(), false), train, test);
    Check(accuracy >= 0.95, Describe("T2a negation", GetParam(), accuracy));
}

TEST_P(TextTest, T2b_WordOrder) {
    std::mt19937 rng(GetParam() + 200);
    auto train = Chases(3000, rng);
    auto test = Chases(1000, rng);
    double accuracy = Accuracy(Config(GetParam(), false), train, test);
    Check(accuracy >= 0.95, Describe("T2b word order", GetParam(), accuracy));
}

// ---------------------------------------------------------------------- T3
TEST_P(TextTest, T3_UnseenSynonyms) {
    if (!Vectors()) GTEST_SKIP() << "DPAN_WORD_VECTORS not set";
    std::mt19937 rng(GetParam() + 300);
    auto train = Negation(3000, rng, {"good", "great", "excellent", "wonderful"},
                          {"bad", "terrible", "awful", "horrible"}, false);
    auto test = Negation(1000, rng, {"fantastic", "superb", "pleasant", "lovely"},
                         {"poor", "dreadful", "nasty", "lousy"}, false);
    double accuracy = Accuracy(Config(GetParam(), true), train, test);
    Check(accuracy >= 0.85, Describe("T3 unseen synonyms", GetParam(), accuracy));
}

INSTANTIATE_TEST_SUITE_P(Seeds, TextTest, ::testing::Values(1u, 2u, 3u));

}  // namespace
}  // namespace agent
}  // namespace dpan
