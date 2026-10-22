// File: examples/agent/agent_example.cpp
//
// A drop-in agent that learns to classify readings from labels that arrive
// with each observation, and flags unusual readings.

#include "agent/agent.hpp"

#include <iostream>
#include <random>

using namespace dpan::agent;

int main() {
    EnvironmentSpec spec;
    spec.inputs = {{"reading", InputType::NUMERIC},         // size inferred
                   {"source", InputType::CATEGORICAL}};     // open vocabulary
    spec.outputs = {{"state", OutputType::CLASS},            // classes learned from labels
                    {"unusual", OutputType::ANOMALY}};
    spec.feedback = {{"label", FeedbackType::LABEL, "state"}};
    Agent agent(spec);

    std::mt19937 rng(1);
    std::normal_distribution<float> noise(0.0f, 0.3f);
    int correct = 0;
    for (int i = 0; i < 3000; ++i) {
        bool hot = i % 3 == 0;
        std::vector<float> reading = {(hot ? 2.0f : -1.0f) + noise(rng), noise(rng)};
        StepId step = agent.Observe({{"reading", reading}, {"source", std::string(i % 2 ? "a" : "b")}});

        Prediction state = agent.Predict(step, "state");
        if (i >= 2000 && state.label == (hot ? "hot" : "normal")) ++correct;
        agent.Feedback(step, "label", std::string(hot ? "hot" : "normal"));
    }
    std::cout << "accuracy on the last 1000 readings: " << correct / 1000.0 << "\n";

    StepId odd = agent.Observe({{"reading", std::vector<float>{9.0f, -9.0f}}, {"source", std::string("a")}});
    std::cout << "anomaly percentile of an odd reading: " << agent.Predict(odd, "unusual").score << "\n";

    AgentStats stats = agent.Stats();
    std::cout << "patterns " << stats.patterns << ", training steps " << stats.training_steps << "\n";
    return 0;
}
