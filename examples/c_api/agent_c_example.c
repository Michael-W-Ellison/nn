/* File: examples/c_api/agent_c_example.c
 *
 * The drop-in agent from C (src/capi/dpan_c.h):
 *   1. classification: numeric and categorical inputs, labels as feedback
 *   2. a 5x5 gridworld learned from rewards, saved and loaded again
 *
 * Build: link with the shared library dpan_c (CMake target dpan_c, or
 * dpan::c_api from an installed DPAN). Exits with 1 if a call fails or the
 * agent does not learn.
 */
#include "dpan_c.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call)                                                             \
    do {                                                                        \
        dpan_status status_ = (call);                                           \
        if (status_ != DPAN_OK) {                                               \
            fprintf(stderr, "%s failed (%d): %s\n", #call, (int)status_,         \
                    dpan_last_error());                                         \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

/* Small deterministic generator (the example's data, not the agent's) */
static unsigned long long rng_state = 12345;
static double uniform(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(rng_state >> 11) / 9007199254740992.0;
}

/* Value of "label" in a prediction's JSON (enough for this example) */
static void predicted_label(const char* json, char* label, size_t size) {
    const char* key = strstr(json, "\"label\":\"");
    size_t n = 0;
    if (key) {
        key += 9;
        while (key[n] && key[n] != '"' && n + 1 < size) ++n;
        memcpy(label, key, n);
    }
    label[n] = '\0';
}

static int classification(void) {
    static const char* spec =
        "{\"inputs\": [{\"name\": \"x\", \"type\": \"numeric\", \"size\": 2},"
        "              {\"name\": \"weather\", \"type\": \"categorical\"}],"
        " \"feedback\": [{\"name\": \"label\", \"type\": \"label\", \"output\": \"kind\"}],"
        " \"outputs\": [{\"name\": \"kind\", \"type\": \"class\"}]}";
    static const char* names[] = {"low", "high"};
    static const char* weather[] = {"rain", "sun"};
    dpan_agent* agent = NULL;
    dpan_observation* obs = dpan_observation_create();
    int correct = 0, i;
    CHECK(dpan_agent_create(spec, "{\"seed\": 3}", &agent));

    for (i = 0; i < 2000; ++i) {
        int kind = uniform() < 0.5 ? 0 : 1;
        float x[2];
        char* prediction = NULL;
        char label[64];
        dpan_step step;
        x[0] = (float)(kind * 2.0 - 1.0 + 0.5 * (uniform() - 0.5));
        x[1] = (float)(uniform() - 0.5);
        dpan_observation_clear(obs);
        CHECK(dpan_observation_set_vector(obs, "x", x, 2));
        CHECK(dpan_observation_set_text(obs, "weather", weather[uniform() < 0.8 ? kind : 1 - kind]));
        CHECK(dpan_agent_observe(agent, obs, &step));
        CHECK(dpan_agent_predict_json(agent, step, "kind", &prediction));
        predicted_label(prediction, label, sizeof(label));
        if (i >= 1000 && strcmp(label, names[kind]) == 0) ++correct;
        dpan_string_free(prediction);
        CHECK(dpan_agent_feedback_text(agent, step, "label", names[kind]));
        CHECK(dpan_clock_advance(1000000));
    }
    printf("classification: %d of the last 1000 predictions correct\n", correct);
    dpan_observation_free(obs);
    dpan_agent_free(agent);
    return correct >= 950;
}

/* Runs episodes; returns how many took a shortest path */
static int run_gridworld(dpan_agent* agent, int episodes, int explore) {
    int optimal = 0, e, t;
    for (e = 0; e < episodes; ++e) {
        int x, y, shortest;
        do {
            x = (int)(uniform() * 5);
            y = (int)(uniform() * 5);
        } while (x == 4 && y == 4);
        shortest = (4 - x) + (4 - y);
        for (t = 0; t < 50; ++t) {
            char observation[64];
            const char* move = NULL;
            dpan_step step;
            int goal;
            snprintf(observation, sizeof(observation), "{\"position\": [%d, %d]}", x, y);
            CHECK(dpan_agent_observe_json(agent, observation, &step));
            CHECK(dpan_agent_act(agent, step, "move", explore, &move, NULL));
            if (strcmp(move, "up") == 0 && y < 4) ++y;
            else if (strcmp(move, "down") == 0 && y > 0) --y;
            else if (strcmp(move, "left") == 0 && x > 0) --x;
            else if (strcmp(move, "right") == 0 && x < 4) ++x;
            goal = x == 4 && y == 4;
            CHECK(dpan_agent_feedback_number(agent, step, "reward", goal ? 1.0 : -0.04));
            CHECK(dpan_clock_advance(1000000));
            if (goal) {
                CHECK(dpan_agent_end_episode(agent, 1));
                if (t + 1 == shortest) ++optimal;
                break;
            }
        }
        if (t == 50) CHECK(dpan_agent_end_episode(agent, 0));
    }
    return optimal;
}

static int gridworld(const char* checkpoint) {
    static const char* spec =
        "{\"inputs\": [{\"name\": \"position\", \"type\": \"numeric\", \"size\": 2}],"
        " \"feedback\": [{\"name\": \"reward\", \"type\": \"reward\", \"output\": \"move\"}],"
        " \"outputs\": [{\"name\": \"move\", \"type\": \"action\","
        "              \"actions\": [\"up\", \"down\", \"left\", \"right\"]}]}";
    dpan_agent* agent = NULL;
    dpan_agent* loaded = NULL;
    char* stats = NULL;
    int optimal;
    CHECK(dpan_agent_create(spec, NULL, &agent));
    run_gridworld(agent, 500, 1);
    CHECK(dpan_agent_save(agent, checkpoint));
    dpan_agent_free(agent);

    CHECK(dpan_agent_load(checkpoint, &loaded));
    optimal = run_gridworld(loaded, 20, 0);
    CHECK(dpan_agent_stats_json(loaded, &stats));
    printf("gridworld: %d of 20 greedy episodes on a shortest path after loading\n", optimal);
    printf("stats: %.120s...\n", stats);
    dpan_string_free(stats);
    dpan_agent_free(loaded);
    remove(checkpoint);
    return optimal >= 18;
}

int main(int argc, char** argv) {
    const char* checkpoint = argc > 1 ? argv[1] : "agent_c_example.dpag";
    dpan_agent* agent = NULL;
    int ok;
    printf("dpan %s\n", dpan_version());

    /* Errors come back as a status and a message */
    if (dpan_agent_create("{\"inputs\": [{\"name\": \"x\", \"type\": \"vector\"}]}", NULL, &agent) != DPAN_OK) {
        printf("expected error: %s\n", dpan_last_error());
    }

    dpan_clock_use_manual(0); /* reproducible run */
    ok = classification();
    ok = gridworld(checkpoint) && ok;
    dpan_clock_use_system();
    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
