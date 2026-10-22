/* File: src/capi/dpan_c.h
 *
 * C interface to the drop-in agent (docs/AGENT_PLAN.md), built as the shared
 * library dpan_c. Every C++ type stays behind an opaque handle; the
 * environment spec and configuration are JSON text (format in
 * agent/spec_json.hpp), predictions and statistics come back as JSON text.
 *
 *   dpan_agent* agent = NULL;
 *   if (dpan_agent_create(spec_json, NULL, &agent) != DPAN_OK) {
 *       fprintf(stderr, "%s\n", dpan_last_error());
 *   }
 *   dpan_observation* obs = dpan_observation_create();
 *   dpan_observation_set_vector(obs, "x", values, 4);
 *   dpan_step step;
 *   dpan_agent_observe(agent, obs, &step);
 *   char* prediction = NULL;
 *   dpan_agent_predict_json(agent, step, "kind", &prediction);
 *   ...
 *   dpan_string_free(prediction);
 *   dpan_agent_feedback_text(agent, step, "label", "b");
 *   dpan_observation_free(obs);
 *   dpan_agent_free(agent);
 *
 * Errors: functions return a dpan_status; on anything but DPAN_OK,
 * dpan_last_error() holds the message (per thread) and output arguments are
 * left unchanged. Functions on one agent may be called from several threads.
 * Strings returned through char** are allocated by the library and must be
 * released with dpan_string_free.
 */
#ifndef DPAN_C_H
#define DPAN_C_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(DPAN_C_BUILDING)
#    define DPAN_C_API __declspec(dllexport)
#  else
#    define DPAN_C_API __declspec(dllimport)
#  endif
#else
#  define DPAN_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dpan_agent dpan_agent;
typedef struct dpan_observation dpan_observation;
typedef uint64_t dpan_step;

typedef enum dpan_status {
    DPAN_OK = 0,
    DPAN_INVALID_ARGUMENT = 1, /* bad JSON, spec, config, channel, value or a NULL argument */
    DPAN_OUT_OF_RANGE = 2,     /* unknown or expired step */
    DPAN_LOGIC_ERROR = 3,      /* call not allowed in this state (e.g. reward before act) */
    DPAN_RUNTIME_ERROR = 4,    /* file could not be read or written, malformed checkpoint */
    DPAN_OUT_OF_MEMORY = 5,
    DPAN_ERROR = 6             /* anything else */
} dpan_status;

/* Library version, e.g. "0.1.0" */
DPAN_C_API const char* dpan_version(void);

/* Message of the last failed call on this thread ("" if none). Valid until
 * the next call on this thread. */
DPAN_C_API const char* dpan_last_error(void);

/* Releases a string returned by this library (NULL is ignored) */
DPAN_C_API void dpan_string_free(char* text);

/* ---------------------------------------------------------------------------
 * Agent
 */

/* config_json: NULL or "" for the defaults, otherwise an object of the
 * AgentConfig fields to change */
DPAN_C_API dpan_status dpan_agent_create(const char* spec_json, const char* config_json, dpan_agent** out);
DPAN_C_API dpan_status dpan_agent_load(const char* path, dpan_agent** out);
DPAN_C_API dpan_status dpan_agent_save(const dpan_agent* agent, const char* path);
DPAN_C_API void dpan_agent_free(dpan_agent* agent);

/* The spec and the configuration in effect, as JSON */
DPAN_C_API dpan_status dpan_agent_spec_json(const dpan_agent* agent, char** out);
DPAN_C_API dpan_status dpan_agent_config_json(const dpan_agent* agent, char** out);

/* Learn from an observation and open a step for predictions and feedback */
DPAN_C_API dpan_status dpan_agent_observe(dpan_agent* agent, const dpan_observation* observation, dpan_step* out);
/* The same with the observation as a JSON object of values by input name */
DPAN_C_API dpan_status dpan_agent_observe_json(dpan_agent* agent, const char* observation_json, dpan_step* out);

/* Prediction of an output as a JSON object: output, type, available, label,
 * classes, probabilities, values, score, confidence, neural_weight */
DPAN_C_API dpan_status dpan_agent_predict_json(const dpan_agent* agent, dpan_step step, const char* output,
                                               char** out);

/* Chooses an action of an ACTION output (explore: nonzero = may pick a
 * random action). *action points to the action's name, owned by the agent
 * and valid until it is freed; *index (if not NULL) is its position in the
 * spec. */
DPAN_C_API dpan_status dpan_agent_act(dpan_agent* agent, dpan_step step, const char* output, int explore,
                                      const char** action, size_t* index);

/* Feedback for a step: a number (label index, target, reward, outcome), a
 * vector (target of several values), a text (class name) or a JSON value */
DPAN_C_API dpan_status dpan_agent_feedback_number(dpan_agent* agent, dpan_step step, const char* feedback,
                                                  double value);
DPAN_C_API dpan_status dpan_agent_feedback_vector(dpan_agent* agent, dpan_step step, const char* feedback,
                                                  const float* values, size_t count);
DPAN_C_API dpan_status dpan_agent_feedback_text(dpan_agent* agent, dpan_step step, const char* feedback,
                                                const char* text);
DPAN_C_API dpan_status dpan_agent_feedback_json(dpan_agent* agent, dpan_step step, const char* feedback,
                                                const char* value_json);

/* terminal: nonzero = the episode ended in a terminal state */
DPAN_C_API dpan_status dpan_agent_end_episode(dpan_agent* agent, int terminal);

DPAN_C_API dpan_status dpan_agent_stats_json(const dpan_agent* agent, char** out);

/* ---------------------------------------------------------------------------
 * Observation builder: values by input channel name. Setting a channel again
 * replaces its value. Channels left out are missing.
 */

DPAN_C_API dpan_observation* dpan_observation_create(void);
DPAN_C_API void dpan_observation_free(dpan_observation* observation);
DPAN_C_API void dpan_observation_clear(dpan_observation* observation);
/* NUMERIC scalar, TIME_SERIES value, or a known category's index */
DPAN_C_API dpan_status dpan_observation_set_number(dpan_observation* observation, const char* channel, double value);
/* NUMERIC vector or one value per series (NaN = missing element) */
DPAN_C_API dpan_status dpan_observation_set_vector(dpan_observation* observation, const char* channel,
                                                   const float* values, size_t count);
/* CATEGORICAL or TEXT */
DPAN_C_API dpan_status dpan_observation_set_text(dpan_observation* observation, const char* channel,
                                                 const char* text);
/* IMAGE: pixels[(y * width + x) * channels + c], values in [0, 1] or [0, 255] */
DPAN_C_API dpan_status dpan_observation_set_image(dpan_observation* observation, const char* channel, size_t width,
                                                  size_t height, size_t channels, const float* pixels);

/* ---------------------------------------------------------------------------
 * Process clock (shared by every agent in the process). By default the
 * system clock; a manual clock makes runs reproducible.
 */

DPAN_C_API void dpan_clock_use_manual(int64_t start_microseconds);
/* Fails with DPAN_LOGIC_ERROR unless a manual clock is in use */
DPAN_C_API dpan_status dpan_clock_advance(int64_t microseconds);
DPAN_C_API void dpan_clock_use_system(void);

#ifdef __cplusplus
}
#endif

#endif /* DPAN_C_H */
