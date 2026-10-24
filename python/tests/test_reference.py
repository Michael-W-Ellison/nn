"""Reference environments A-F and I (docs/AGENT_PLAN.md, section 6) through
the Python bindings.

The environments and pass criteria are those of
tests/agent/reference_environments_test.cpp. The data comes from NumPy's
generator, not from the C++ test's std::mt19937, so the samples (and the
results) differ from the C++ tests. Every agent uses the default config and
a manual clock advanced one second per observation.
"""

import math
import unittest

import numpy as np

import dpan


class ReferenceTest(unittest.TestCase):
    def setUp(self):
        dpan.use_manual_clock(1_700_000_000_000_000)

    def tearDown(self):
        dpan.use_system_clock()

    # ------------------------------------------------------------------
    # A: stream classification with drift. 3 classes, a 4-dimensional
    # numeric input around one center per class and a categorical region
    # that matches the class 60% of the time. Labels arrive right after
    # each prediction. At sample 6000 the labels rotate. Pass: accuracy
    # >= 0.9 on samples 2000-6000, on the 200 samples ending 1000 after the
    # drift, and from 1000 after the drift to the end (9000).
    def test_a_stream_classification_with_drift(self):
        regions = ["north", "south", "east", "west", "center"]
        agent = dpan.Agent({
            "inputs": [{"name": "x", "type": "numeric", "size": 4},
                       {"name": "region", "type": "categorical", "categories": regions}],
            "feedback": [{"name": "label", "type": "label", "output": "kind"}],
            "outputs": [{"name": "kind", "type": "class"}],
        })
        centers = np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0]], dtype=np.float32)
        rng = np.random.default_rng(11)
        total, drift = 9000, 6000
        correct = np.zeros(total)
        for i in range(total):
            cluster = int(rng.integers(3))
            x = centers[cluster] + rng.normal(0.0, 0.6, 4).astype(np.float32)
            region = regions[cluster] if rng.random() < 0.6 else regions[int(rng.integers(5))]
            label = cluster if i < drift else (cluster + 1) % 3
            step = agent.observe({"x": x, "region": region})
            p = agent.predict(step, "kind")
            correct[i] = p.available and p.label == str(label)
            agent.feedback(step, "label", label)
            dpan.advance_clock(1)
        before = correct[2000:drift].mean()
        recovered = correct[drift + 800:drift + 1000].mean()
        after = correct[drift + 1000:].mean()
        print(f"A: accuracy before drift {before:.4f}, 800-1000 after drift {recovered:.4f}, after {after:.4f}")
        self.assertGreaterEqual(before, 0.90)
        self.assertGreaterEqual(recovered, 0.90)
        self.assertGreaterEqual(after, 0.90)

    # ------------------------------------------------------------------
    # B: stream regression. y = sin(1.5 x0) + 0.5 x1^2 - x2 + 0.3 x0 x3 +
    # noise(0.3), x uniform in [-2, 2]^4; 8000 samples streamed with
    # immediate targets, then 1000 test samples. The offline model has the
    # agent's trunk shape (4 -> 32 -> 32 -> 1, ReLU) and is trained with
    # Adam (1e-3, batch 32) for up to 300 epochs, stopping when the loss on
    # the last 10% has not improved for 20 epochs. Pass: agent test error
    # <= 1.2 x offline test error.
    def test_b_stream_regression(self):
        agent = dpan.Agent({
            "inputs": [{"name": "x", "type": "numeric", "size": 4}],
            "feedback": [{"name": "target", "type": "target", "output": "y"}],
            "outputs": [{"name": "y", "type": "value"}],
        })
        rng = np.random.default_rng(21)

        def make(n):
            x = rng.uniform(-2.0, 2.0, (n, 4)).astype(np.float32)
            y = (np.sin(1.5 * x[:, 0]) + 0.5 * x[:, 1] ** 2 - x[:, 2] + 0.3 * x[:, 0] * x[:, 3]
                 + rng.normal(0.0, 0.3, n)).astype(np.float32)
            return x, y

        train_x, train_y = make(8000)
        test_x, test_y = make(1000)
        for x, y in zip(train_x, train_y):
            step = agent.observe({"x": x})
            agent.feedback(step, "target", float(y))
            dpan.advance_clock(1)
        predictions = []
        for x in test_x:
            p = agent.predict(agent.observe({"x": x}), "y")
            self.assertTrue(p.available)
            predictions.append(p.values[0])
        agent_error = float(np.mean((np.array(predictions) - test_y) ** 2))
        offline_error = float(np.mean((_fit_offline_mlp(train_x, train_y, test_x) - test_y) ** 2))
        print(f"B: agent test MSE {agent_error:.5f}, offline {offline_error:.5f} "
              f"(ratio {agent_error / offline_error:.4f})")
        self.assertLessEqual(agent_error, 1.2 * offline_error)

    # ------------------------------------------------------------------
    # C: time-series forecasting. s(t) = sin(2 pi t / 25) + 0.5 sin(2 pi t
    # / 7 + 1) + noise(0.1); NEXT_OBSERVATION output, 6000 steps. Pass:
    # mean squared error over steps 3000-6000 <= 0.8 x that of predicting
    # the last value.
    def test_c_time_series_forecasting(self):
        agent = dpan.Agent({
            "inputs": [{"name": "load", "type": "time_series", "size": 1}],
            "outputs": [{"name": "next", "type": "next_observation"}],
        })
        rng = np.random.default_rng(31)
        total = 6000
        t = np.arange(total + 1)
        s = (np.sin(2 * math.pi * t / 25) + 0.5 * np.sin(2 * math.pi * t / 7 + 1)
             + rng.normal(0.0, 0.1, total + 1)).astype(np.float32)
        agent_error = last_value_error = 0.0
        counted = 0
        for i in range(total):
            step = agent.observe({"load": float(s[i])})
            p = agent.predict(step, "next")
            if i >= 3000:
                self.assertTrue(p.available)
                agent_error += (p.values[0] - s[i + 1]) ** 2
                last_value_error += (s[i] - s[i + 1]) ** 2
                counted += 1
            dpan.advance_clock(1)
        agent_error /= counted
        last_value_error /= counted
        print(f"C: agent MSE {agent_error:.5f}, last value {last_value_error:.5f} "
              f"(ratio {agent_error / last_value_error:.4f})")
        self.assertLessEqual(agent_error, 0.8 * last_value_error)

    # ------------------------------------------------------------------
    # D: anomaly detection. Metrics m1 = z, m2 = 2z + noise, m3 independent,
    # and a log message from 8 templates. 2% anomalies, a third each:
    # broken correlation, an extreme value (m3 = +-6), a message of unusual
    # words. No feedback. Pass: area under the ROC curve of the ANOMALY
    # score >= 0.9 over observations 1000-10000.
    def test_d_anomaly_detection(self):
        agent = dpan.Agent({
            "inputs": [{"name": "metrics", "type": "numeric", "size": 3},
                       {"name": "message", "type": "text"}],
            "outputs": [{"name": "unusual", "type": "anomaly"}],
        })
        templates = ["user {} logged in", "user {} logged out", "disk usage at {} percent",
                     "request served in {} ms", "cache hit for key {}", "backup job {} finished",
                     "connection from host {} accepted", "scheduled task {} started"]
        unusual = ["kernel", "panic", "segmentation", "fault", "overheat", "corrupted", "breach",
                   "unauthorized"]
        rng = np.random.default_rng(41)
        scores, labels = [], []
        for i in range(10000):
            z = rng.normal()
            metrics = [z, 2 * z + 0.3 * rng.normal(), rng.normal()]
            message = templates[int(rng.integers(8))].format(int(rng.integers(1, 1000)))
            anomaly = rng.random() < 0.02
            if anomaly:
                kind = rng.random()
                if kind < 1 / 3:
                    magnitude = 1.5 + rng.random()
                    metrics[0] = magnitude if rng.random() < 0.5 else -magnitude
                    metrics[1] = -2 * metrics[0]
                elif kind < 2 / 3:
                    metrics[2] = 6.0 if rng.random() < 0.5 else -6.0
                else:
                    message = " ".join(unusual[int(rng.integers(8))] for _ in range(3))
            step = agent.observe({"metrics": metrics, "message": message})
            p = agent.predict(step, "unusual")
            if i >= 1000:
                scores.append(p.score)
                labels.append(anomaly)
            dpan.advance_clock(1)
        auc = _area_under_roc(np.array(scores), np.array(labels))
        print(f"D: area under ROC {auc:.4f} ({int(np.sum(labels))} anomalies)")
        self.assertGreaterEqual(auc, 0.9)

    # ------------------------------------------------------------------
    # E: 5x5 gridworld. Goal at (4, 4); episodes start in a random other
    # cell; moves into the border leave the position unchanged. Reward -0.04
    # per step, +1 at the goal (terminal); cut off after 50 steps. 500
    # training episodes with exploration, then 100 greedy episodes. Pass:
    # >= 95 of them take a shortest path.
    def test_e_gridworld(self):
        agent = dpan.Agent({
            "inputs": [{"name": "position", "type": "numeric", "size": 2}],
            "feedback": [{"name": "reward", "type": "reward", "output": "move"}],
            "outputs": [{"name": "move", "type": "action", "actions": ["up", "down", "left", "right"]}],
        })
        rng = np.random.default_rng(51)

        def episode(explore):
            while True:
                x, y = int(rng.integers(5)), int(rng.integers(5))
                if (x, y) != (4, 4):
                    break
            shortest = (4 - x) + (4 - y)
            for t in range(50):
                step = agent.observe({"position": [x, y]})
                move = agent.act(step, "move", explore)
                if move == "up":
                    y = min(4, y + 1)
                elif move == "down":
                    y = max(0, y - 1)
                elif move == "left":
                    x = max(0, x - 1)
                else:
                    x = min(4, x + 1)
                goal = x == 4 and y == 4
                agent.feedback(step, "reward", 1.0 if goal else -0.04)
                dpan.advance_clock(1)
                if goal:
                    agent.end_episode(True)
                    return t + 1 == shortest
            agent.end_episode(False)
            return False

        for _ in range(500):
            episode(True)
        optimal = sum(episode(False) for _ in range(100))
        stats = agent.stats()
        print(f"E: shortest path in {optimal} of 100 greedy episodes ({stats['steps']} steps)")
        self.assertGreaterEqual(optimal, 95)

    # ------------------------------------------------------------------
    # F: cart-pole with the CartPole-v0 constants (gravity 9.8, cart mass
    # 1.0, pole mass 0.1, half length 0.5, force 10, Euler steps of 0.02 s).
    # Failure beyond +-12 degrees or +-2.4; cut off after 200 steps; start
    # state uniform in +-0.05. Reward +1 per step. Pass: the average reward
    # of 100 consecutive (exploring) episodes reaches 195 within 1000.
    def test_f_cart_pole(self):
        agent = dpan.Agent({
            "inputs": [{"name": "state", "type": "numeric", "size": 4}],
            "feedback": [{"name": "reward", "type": "reward", "output": "push"}],
            "outputs": [{"name": "push", "type": "action", "actions": ["left", "right"]}],
        })
        rng = np.random.default_rng(61)
        rewards = []
        solved_at = -1
        best = 0.0
        for episode in range(1000):
            state = [float(v) for v in rng.uniform(-0.05, 0.05, 4)]
            total = 0
            failed = False
            while total < 200 and not failed:
                step = agent.observe({"state": state})
                state, failed = _cart_pole_step(state, agent.act(step, "push") == "right")
                agent.feedback(step, "reward", 1.0)
                total += 1
                dpan.advance_clock(1)
            agent.end_episode(failed)  # cut off at 200 steps: not terminal
            rewards.append(total)
            if len(rewards) >= 100:
                average = sum(rewards[-100:]) / 100.0
                best = max(best, average)
                if average >= 195.0:
                    solved_at = episode + 1
                    break
        result = (f"average reward >= 195 after {solved_at} episodes" if solved_at > 0
                  else "not solved in 1000 episodes")
        print(f"F: {result} (best 100-episode average {best:.2f}, {agent.stats()['steps']} steps)")
        self.assertGreater(solved_at, 0)


    # ------------------------------------------------------------------
    # I: pendulum swing-up (Pendulum-v1): angle theta (0 = upright), speed
    # limited to +-8, torque in [-2, 2], steps of 0.05 s. Input (cos theta,
    # sin theta, speed); reward -(theta^2 + 0.1 speed^2 + 0.001 torque^2).
    # Episodes start at a uniform angle and a speed in [-1, 1] and are cut
    # off after 200 steps. 200 training episodes, then 20 greedy episodes.
    # Pass: their mean return >= -200.
    def test_i_pendulum_swing_up(self):
        agent = dpan.Agent({
            "inputs": [{"name": "state", "type": "numeric", "size": 3}],
            "feedback": [{"name": "reward", "type": "reward", "output": "torque"}],
            "outputs": [{"name": "torque", "type": "continuous_action", "size": 1,
                         "low": [-2], "high": [2]}],
        })
        rng = np.random.default_rng(81)

        def episode(explore):
            theta, speed = rng.uniform(-math.pi, math.pi), rng.uniform(-1.0, 1.0)
            total = 0.0
            for _ in range(200):
                step = agent.observe({"state": [math.cos(theta), math.sin(theta), speed]})
                torque = min(2.0, max(-2.0, agent.act_continuous(step, "torque", explore)[0]))
                angle = math.remainder(theta, 2 * math.pi)
                reward = -(angle ** 2 + 0.1 * speed ** 2 + 0.001 * torque ** 2)
                speed = min(8.0, max(-8.0, speed + (15.0 * math.sin(theta) + 3.0 * torque) * 0.05))
                theta += speed * 0.05
                agent.feedback(step, "reward", reward)
                total += reward
                dpan.advance_clock(1)
            agent.end_episode(False)  # time limit: not terminal
            return total

        for _ in range(200):
            episode(True)
        greedy = sum(episode(False) for _ in range(20)) / 20
        print(f"I: mean return of 20 greedy episodes {greedy:.1f} ({agent.stats()['steps']} steps)")
        self.assertGreaterEqual(greedy, -200.0)


def _cart_pole_step(state, push_right):
    gravity, cart_mass, pole_mass, half_length, tau = 9.8, 1.0, 0.1, 0.5, 0.02
    total_mass = cart_mass + pole_mass
    pole_moment = pole_mass * half_length
    force = 10.0 if push_right else -10.0
    x, x_dot, theta, theta_dot = state
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    temp = (force + pole_moment * theta_dot * theta_dot * sin_t) / total_mass
    theta_acc = (gravity * sin_t - cos_t * temp) / (
        half_length * (4.0 / 3.0 - pole_mass * cos_t * cos_t / total_mass))
    x_acc = temp - pole_moment * theta_acc * cos_t / total_mass
    x += tau * x_dot
    x_dot += tau * x_acc
    theta += tau * theta_dot
    theta_dot += tau * theta_acc
    # Single precision, as the C++ simulator stores it
    state = [float(np.float32(v)) for v in (x, x_dot, theta, theta_dot)]
    failed = abs(state[0]) > 2.4 or abs(state[2]) > 12.0 * 2.0 * math.pi / 360.0
    return state, failed


def _area_under_roc(scores, labels):
    """Rank-sum (Mann-Whitney) with average ranks for ties"""
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2.0
        i = j
    positives = labels.sum()
    negatives = len(labels) - positives
    return (ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives)


def _fit_offline_mlp(x, y, test_x, epochs=300, batch=32, patience=20, rate=1e-3):
    """4 -> 32 -> 32 -> 1 ReLU network trained with Adam in NumPy (float64)"""
    rng = np.random.default_rng(42)
    split = int(len(x) * 0.9)
    train_x, train_y = x[:split].astype(np.float64), y[:split, None].astype(np.float64)
    valid_x, valid_y = x[split:].astype(np.float64), y[split:, None].astype(np.float64)
    sizes = [4, 32, 32, 1]
    weights = [rng.normal(0.0, math.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(2)]
    weights.append(rng.uniform(-1, 1, (32, 1)) * math.sqrt(6.0 / (32 + 1)))
    biases = [np.zeros((1, n)) for n in sizes[1:]]
    params = weights + biases
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def forward(inputs):
        h1 = np.maximum(inputs @ weights[0] + biases[0], 0.0)
        h2 = np.maximum(h1 @ weights[1] + biases[1], 0.0)
        return h1, h2, h2 @ weights[2] + biases[2]

    best, stale = math.inf, 0
    for _ in range(epochs):
        order = rng.permutation(split)
        for start in range(0, split, batch):
            idx = order[start:start + batch]
            xb, yb = train_x[idx], train_y[idx]
            h1, h2, out = forward(xb)
            g_out = 2.0 * (out - yb) / len(idx)
            g_w3, g_b3 = h2.T @ g_out, g_out.sum(0, keepdims=True)
            g_h2 = (g_out @ weights[2].T) * (h2 > 0)
            g_w2, g_b2 = h1.T @ g_h2, g_h2.sum(0, keepdims=True)
            g_h1 = (g_h2 @ weights[1].T) * (h1 > 0)
            g_w1, g_b1 = xb.T @ g_h1, g_h1.sum(0, keepdims=True)
            grads = [g_w1, g_w2, g_w3, g_b1, g_b2, g_b3]
            t += 1
            for k, (p, g) in enumerate(zip(params, grads)):
                m[k] = beta1 * m[k] + (1 - beta1) * g
                v[k] = beta2 * v[k] + (1 - beta2) * g * g
                p -= rate * (m[k] / (1 - beta1 ** t)) / (np.sqrt(v[k] / (1 - beta2 ** t)) + eps)
        loss = float(np.mean((forward(valid_x)[2] - valid_y) ** 2))
        if loss < best:
            best, stale = loss, 0
        else:
            stale += 1
            if stale >= patience:
                break
    return forward(test_x.astype(np.float64))[2][:, 0]


if __name__ == "__main__":
    unittest.main()
