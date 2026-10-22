"""The Python bindings: conversions, errors, persistence."""

import json
import os
import tempfile
import unittest

import numpy as np

import dpan

CLASSIFIER = {
    "inputs": [{"name": "x", "type": "numeric", "size": 2},
               {"name": "weather", "type": "categorical"},
               {"name": "note", "type": "text"}],
    "feedback": [{"name": "label", "type": "label", "output": "kind"}],
    "outputs": [{"name": "kind", "type": "class"}, {"name": "unusual", "type": "anomaly"}],
}

GRID = {
    "inputs": [{"name": "position", "type": "numeric", "size": 2}],
    "feedback": [{"name": "reward", "type": "reward", "output": "move"}],
    "outputs": [{"name": "move", "type": "action", "actions": ["up", "down", "left", "right"]}],
}


class ApiTest(unittest.TestCase):
    def setUp(self):
        dpan.use_manual_clock(1_700_000_000_000_000)

    def tearDown(self):
        dpan.use_system_clock()

    def test_version_and_exports(self):
        self.assertTrue(dpan.__version__)
        for name in ("Agent", "Prediction", "use_manual_clock", "advance_clock", "use_system_clock"):
            self.assertTrue(hasattr(dpan, name))

    def test_spec_as_dict_or_json_text(self):
        a = dpan.Agent(CLASSIFIER)
        b = dpan.Agent(json.dumps(CLASSIFIER))
        self.assertEqual(a.spec, b.spec)
        self.assertEqual("categorical", a.spec["inputs"][1]["type"])
        self.assertEqual(42, a.config["seed"])
        c = dpan.Agent(CLASSIFIER, {"seed": 9, "encoder": {"text_buckets": 256}})
        self.assertEqual(9, c.config["seed"])
        self.assertEqual(256, c.config["encoder"]["text_buckets"])

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "unknown type"):
            dpan.Agent({"inputs": [{"name": "x", "type": "vector"}]})
        with self.assertRaisesRegex(ValueError, "TARGET"):
            dpan.Agent({"outputs": [{"name": "y", "type": "value"}]})
        with self.assertRaisesRegex(ValueError, "unknown field"):
            dpan.Agent(CLASSIFIER, {"sed": 1})
        with self.assertRaisesRegex(ValueError, "JSON"):
            dpan.Agent("{")
        agent = dpan.Agent(GRID)
        with self.assertRaises(TypeError):
            agent.observe({"position": object()})
        with self.assertRaises(TypeError):
            agent.observe({1: [0, 0]})
        with self.assertRaisesRegex(ValueError, "dimensions"):
            agent.observe({"position": np.zeros((1, 1, 1, 1))})
        with self.assertRaises(ValueError):
            agent.observe({"unknown": 1.0})
        with self.assertRaises(IndexError):
            agent.predict(12345, "move")
        step = agent.observe({"position": [0, 0]})
        with self.assertRaises(RuntimeError):  # reward before act (std::logic_error)
            agent.feedback(step, "reward", 1.0)
        with self.assertRaises(OSError):
            agent.save("/nonexistent/dir/agent.dpag")
        with self.assertRaises(RuntimeError):
            dpan.Agent.load("/nonexistent/agent.dpag")
        dpan.use_system_clock()
        with self.assertRaisesRegex(RuntimeError, "manual clock"):
            dpan.advance_clock(1)

    def test_values_from_python_and_numpy(self):
        agent = dpan.Agent(CLASSIFIER)
        inputs = [
            {"x": [0.5, 1], "weather": "rain", "note": "cold"},
            {"x": (1, 2), "weather": "sun", "note": None},          # None: missing
            {"x": np.array([1.5, 2.5], dtype=np.float64)},
            {"x": np.array([1, 2], dtype=np.int32), "weather": "fog"},
            {"x": [np.float32(1.0), np.nan], "note": "warm"},        # NaN: missing element
        ]
        for i in range(60):
            step = agent.observe(inputs[i % len(inputs)])
            agent.feedback(step, "label", ["a", "b"][i % 2] if i % 3 else i % 2)
            dpan.advance_clock(1)
        stats = agent.stats()
        self.assertEqual(60, stats["steps"])
        self.assertEqual(60, stats["feedback"])
        self.assertEqual(0, stats["ignored_feedback"])
        p = agent.predict(step, "kind")
        self.assertIsInstance(p, dpan.Prediction)
        self.assertEqual("class", p.type)
        self.assertTrue(p.available)
        self.assertEqual(sorted(p.classes), ["0", "1", "a", "b"])
        self.assertAlmostEqual(1.0, sum(p.probabilities), places=4)
        self.assertEqual(p.to_dict()["label"], p.label)
        self.assertIn("Prediction(", repr(p))
        anomaly = agent.predict(step, "unusual")
        self.assertTrue(0.0 <= anomaly.score <= 1.0)

    def test_images_and_vector_targets(self):
        agent = dpan.Agent({
            "inputs": [{"name": "gray", "type": "image"}, {"name": "color", "type": "image", "channels": 3}],
            "feedback": [{"name": "target", "type": "target", "output": "y"}],
            "outputs": [{"name": "y", "type": "value", "size": 2}],
        })
        rng = np.random.default_rng(0)
        for i in range(40):
            gray = rng.integers(0, 256, (8, 6)).astype(np.uint8)      # height x width
            color = rng.random((8, 6, 3), dtype=np.float32)             # height x width x channels
            step = agent.observe({"gray": gray, "color": color})
            agent.feedback(step, "target", np.array([i % 3, 1.0]))
        self.assertEqual(2, len(agent.predict(step, "y").values))
        stats = agent.stats()
        self.assertTrue(stats["network_ready"] or stats["steps"] == 40)

    def test_actions_save_and_load(self):
        agent = dpan.Agent(GRID)
        for i in range(300):
            step = agent.observe({"position": [i % 5, (i // 5) % 5]})
            move = agent.act(step, "move")
            self.assertIn(move, ["up", "down", "left", "right"])
            self.assertEqual(move, agent.act(step, "move"))  # the same step: the same action
            agent.feedback(step, "reward", 1.0 if move == "right" else -0.1)
            if i % 10 == 9:
                agent.end_episode()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "agent.dpag")
            agent.save(path)
            loaded = dpan.Agent.load(path)
        self.assertEqual(agent.spec, loaded.spec)
        self.assertEqual(agent.config, loaded.config)
        for i in range(50):
            observation = {"position": [i % 5, (i * 3) % 5]}
            a, b = agent.observe(observation), loaded.observe(observation)
            self.assertEqual(agent.act(a, "move"), loaded.act(b, "move"))
            self.assertEqual(agent.predict(a, "move").values, loaded.predict(b, "move").values)
            agent.feedback(a, "reward", 0.5)
            loaded.feedback(b, "reward", 0.5)
        self.assertEqual(agent.stats(), loaded.stats())


if __name__ == "__main__":
    unittest.main()
