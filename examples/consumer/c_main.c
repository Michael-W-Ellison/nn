/* Uses the agent through the C API of an installed or embedded DPAN */
#include "dpan_c.h"

#include <stdio.h>

int main(void) {
    const char* spec =
        "{\"inputs\": [{\"name\": \"x\", \"type\": \"numeric\", \"size\": 1}],"
        " \"outputs\": [{\"name\": \"unusual\", \"type\": \"anomaly\"}]}";
    dpan_agent* agent = NULL;
    dpan_step step = 0;
    char* prediction = NULL;
    int i;
    if (dpan_agent_create(spec, NULL, &agent) != DPAN_OK) {
        fprintf(stderr, "%s\n", dpan_last_error());
        return 1;
    }
    for (i = 0; i < 100; ++i) {
        char observation[48];
        snprintf(observation, sizeof(observation), "{\"x\": %d}", i % 10);
        if (dpan_agent_observe_json(agent, observation, &step) != DPAN_OK) return 1;
    }
    if (dpan_agent_predict_json(agent, step, "unusual", &prediction) != DPAN_OK) return 1;
    printf("dpan %s: %s\n", dpan_version(), prediction);
    dpan_string_free(prediction);
    dpan_agent_free(agent);
    return 0;
}
