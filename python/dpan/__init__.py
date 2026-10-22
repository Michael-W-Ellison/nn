"""DPAN drop-in agent.

Describe the environment once, then stream observations and send feedback:

    import dpan
    agent = dpan.Agent({
        "inputs": [{"name": "x", "type": "numeric", "size": 4}],
        "feedback": [{"name": "label", "type": "label", "output": "kind"}],
        "outputs": [{"name": "kind", "type": "class"}],
    })
    step = agent.observe({"x": [0.1, 0.2, 0.3, 0.4]})
    prediction = agent.predict(step, "kind")
    agent.feedback(step, "label", "b")

See docs/AGENT_PLAN.md for the spec format and src/agent/spec_json.hpp for
the JSON (dict) form used here.
"""

from ._core import (
    Agent,
    Prediction,
    __version__,
    advance_clock,
    use_manual_clock,
    use_system_clock,
)

__all__ = [
    "Agent",
    "Prediction",
    "__version__",
    "advance_clock",
    "use_manual_clock",
    "use_system_clock",
]
