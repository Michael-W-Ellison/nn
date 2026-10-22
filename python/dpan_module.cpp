// File: python/dpan_module.cpp
//
// Python bindings for the drop-in agent (package dpan, extension dpan._core).
//
//   import dpan
//   agent = dpan.Agent({"inputs": [{"name": "x", "type": "numeric", "size": 4}],
//                       "feedback": [{"name": "label", "type": "label", "output": "kind"}],
//                       "outputs": [{"name": "kind", "type": "class"}]})
//   step = agent.observe({"x": numpy.array([0.1, 0.2, 0.3, 0.4])})
//   print(agent.predict(step, "kind").label)
//   agent.feedback(step, "label", "b")
//
// The spec and config are dicts (or JSON text) in the format of
// agent/spec_json.hpp. Observation values: int/float, str, a 1-D sequence or
// array (vector), a 2-D (height x width) or 3-D (height x width x channels)
// array (image), or None (missing).
#include "agent/agent.hpp"
#include "agent/spec_json.hpp"
#include "core/clock.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <mutex>
#include <string>

namespace py = pybind11;
using namespace pybind11::literals;
using dpan::agent::Agent;
using dpan::agent::Prediction;
using dpan::agent::StepId;
using dpan::agent::Value;

namespace {

std::string ToJsonText(const py::handle& object) {
    if (py::isinstance<py::str>(object)) return object.cast<std::string>();
    return py::module_::import("json").attr("dumps")(object).cast<std::string>();
}

py::object FromJsonText(const std::string& text) { return py::module_::import("json").attr("loads")(text); }

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

Value ToValue(const py::handle& value, const std::string& name) {
    if (py::isinstance<py::str>(value)) return value.cast<std::string>();
    if (py::isinstance<py::bool_>(value) || py::isinstance<py::int_>(value) || py::isinstance<py::float_>(value)) {
        return value.cast<double>();
    }
    FloatArray array = FloatArray::ensure(value);
    if (!array) {
        throw py::type_error("'" + name + "': expected a number, str, sequence or array of numbers, got " +
                             std::string(py::str(py::type::of(value).attr("__name__"))));
    }
    const float* data = array.data();
    switch (array.ndim()) {
        case 0: return static_cast<double>(*data);  // NumPy scalar
        case 1: return std::vector<float>(data, data + array.shape(0));
        case 2:
        case 3: {
            dpan::agent::Image image;
            image.height = static_cast<size_t>(array.shape(0));
            image.width = static_cast<size_t>(array.shape(1));
            image.channels = array.ndim() == 3 ? static_cast<size_t>(array.shape(2)) : 1;
            image.pixels.assign(data, data + array.size());
            return image;
        }
        default:
            throw py::value_error("'" + name + "': arrays must have 1 (vector), 2 or 3 (image) dimensions, got " +
                                  std::to_string(array.ndim()));
    }
}

dpan::agent::Observation ToObservation(const py::dict& values) {
    dpan::agent::Observation observation;
    for (const auto& item : values) {
        if (!py::isinstance<py::str>(item.first)) throw py::type_error("observation keys must be channel names (str)");
        std::string name = item.first.cast<std::string>();
        if (item.second.is_none()) continue;  // missing channel
        observation[name] = ToValue(item.second, name);
    }
    return observation;
}

std::mutex clock_mutex;
std::shared_ptr<dpan::ManualClock> manual_clock;

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "DPAN drop-in agent: learns from whatever inputs, feedback and outputs its environment declares";
    m.attr("__version__") = DPAN_VERSION_STRING;

    py::class_<Prediction>(m, "Prediction", "Prediction of one output for one step")
        .def_readonly("output", &Prediction::output)
        .def_property_readonly("type",
                               [](const Prediction& p) {
                                   std::string name = dpan::agent::ToString(p.type);
                                   for (auto& c : name) c = static_cast<char>(std::tolower(c));
                                   return name;
                               })
        .def_readonly("available", &Prediction::available,
                      "False if nothing has been learned for this output yet")
        .def_readonly("label", &Prediction::label, "CLASS: most likely class. ACTION: greedy action")
        .def_readonly("classes", &Prediction::classes)
        .def_readonly("probabilities", &Prediction::probabilities)
        .def_readonly("values", &Prediction::values,
                      "VALUE: predictions. ACTION: Q-values. NEXT_OBSERVATION: next numeric input values")
        .def_readonly("score", &Prediction::score, "ANOMALY: percentile among recent observations (0-1)")
        .def_readonly("confidence", &Prediction::confidence)
        .def_readonly("neural_weight", &Prediction::neural_weight)
        .def("to_dict", [](const Prediction& p) { return FromJsonText(dpan::agent::PredictionToJson(p)); })
        .def("__repr__", [](const Prediction& p) { return "Prediction(" + dpan::agent::PredictionToJson(p) + ")"; });

    py::class_<Agent>(m, "Agent", "Drop-in learner configured by an environment spec")
        .def(py::init([](const py::object& spec, const py::object& config) {
                 auto parsed_spec = dpan::agent::SpecFromJson(ToJsonText(spec));
                 dpan::agent::AgentConfig parsed_config;
                 if (!config.is_none()) parsed_config = dpan::agent::ConfigFromJson(ToJsonText(config));
                 return std::make_unique<Agent>(std::move(parsed_spec), parsed_config);
             }),
             "spec"_a, "config"_a = py::none(),
             "spec: dict or JSON text (inputs, feedback, outputs). config: dict of AgentConfig fields to change.")
        .def(
            "observe",
            [](Agent& agent, const py::dict& observation) {
                auto values = ToObservation(observation);
                py::gil_scoped_release release;
                return agent.Observe(values);
            },
            "observation"_a, "Learn from an observation (dict of values by input name); returns the step id")
        .def(
            "predict",
            [](const Agent& agent, StepId step, const std::string& output) {
                py::gil_scoped_release release;
                return agent.Predict(step, output);
            },
            "step"_a, "output"_a)
        .def(
            "act",
            [](Agent& agent, StepId step, const std::string& output, bool explore) {
                py::gil_scoped_release release;
                return agent.Act(step, output, explore);
            },
            "step"_a, "output"_a, "explore"_a = true, "Choose an action of an ACTION output; returns its name")
        .def(
            "feedback",
            [](Agent& agent, StepId step, const std::string& feedback, const py::object& value) {
                Value converted = ToValue(value, feedback);
                py::gil_scoped_release release;
                agent.Feedback(step, feedback, converted);
            },
            "step"_a, "feedback"_a, "value"_a,
            "Label (class name or index), target (number or sequence), reward or outcome (number)")
        .def(
            "end_episode",
            [](Agent& agent, bool terminal) {
                py::gil_scoped_release release;
                agent.EndEpisode(terminal);
            },
            "terminal"_a = true)
        .def("stats", [](const Agent& agent) { return FromJsonText(dpan::agent::StatsToJson(agent.Stats())); })
        .def(
            "save",
            [](const Agent& agent, const std::string& path) {
                bool ok;
                {
                    py::gil_scoped_release release;
                    ok = agent.Save(path);
                }
                if (!ok) {
                    PyErr_SetString(PyExc_OSError, ("cannot write checkpoint '" + path + "'").c_str());
                    throw py::error_already_set();
                }
            },
            "path"_a)
        .def_static(
            "load",
            [](const std::string& path) {
                py::gil_scoped_release release;
                return std::make_unique<Agent>(Agent::Load(path));
            },
            "path"_a)
        .def_property_readonly("spec", [](const Agent& a) { return FromJsonText(dpan::agent::SpecToJson(a.Spec())); })
        .def_property_readonly("config",
                               [](const Agent& a) { return FromJsonText(dpan::agent::ConfigToJson(a.Config())); });

    m.def(
        "use_manual_clock",
        [](int64_t start_microseconds) {
            std::lock_guard<std::mutex> lock(clock_mutex);
            manual_clock = std::make_shared<dpan::ManualClock>(dpan::Timestamp::FromMicros(start_microseconds));
            dpan::SetClock(manual_clock);
        },
        "start_microseconds"_a = 0,
        "Use a clock that moves only with advance_clock (process-wide; makes runs reproducible)");
    m.def(
        "advance_clock",
        [](double seconds) {
            std::lock_guard<std::mutex> lock(clock_mutex);
            if (!manual_clock) throw std::logic_error("no manual clock in use (call use_manual_clock first)");
            manual_clock->Advance(std::chrono::microseconds(static_cast<int64_t>(seconds * 1e6)));
        },
        "seconds"_a = 1.0);
    m.def("use_system_clock", [] {
        std::lock_guard<std::mutex> lock(clock_mutex);
        manual_clock.reset();
        dpan::SetClock(nullptr);
    });
}
