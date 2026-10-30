"""Games (docs/AGENT_PLAN.md, Phase 12): the agent learns to play from
rewards alone with its ACTION outputs and the default config.

Run when DPAN_GAMES is set; C1-C4 also need the gymnasium package
(gymnasium[box2d] for LunarLander). DPAN_GAMES_SEEDS (default "1,2,3")
selects the seeds; DPAN_GAMES_CONFIG (JSON) changes agent settings. Each
test prints its results and the random (and rule-based) baselines.
"""

import json
import os
import unittest

import numpy as np

import dpan

try:
    import gymnasium
except ImportError:  # C1-C4 are skipped
    gymnasium = None

RUN = os.environ.get("DPAN_GAMES") is not None
SEEDS = [int(s) for s in os.environ.get("DPAN_GAMES_SEEDS", "1,2,3").split(",")]
CONFIG = json.loads(os.environ.get("DPAN_GAMES_CONFIG", "{}"))


def _agent(inputs, actions, seed):
    return dpan.Agent({
        "inputs": [{"name": "state", "type": "numeric", "size": inputs}],
        "feedback": [{"name": "reward", "type": "reward", "output": "move"}],
        "outputs": [{"name": "move", "type": "action", "actions": [str(a) for a in range(actions)]}],
    }, {**CONFIG, "seed": seed})


# ----------------------------------------------------------------------
# Classic control (C1-C4)

def _play_control(name, training_steps, seed):
    """Train for `training_steps` exploring steps; return the mean return of
    100 greedy episodes and of 100 random-policy episodes"""
    env = gymnasium.make(name)
    agent = _agent(env.observation_space.shape[0], env.action_space.n, seed)
    steps, episode = 0, 0
    while steps < training_steps:
        obs, _ = env.reset(seed=seed * 1_000_000 + episode)
        episode += 1
        while True:
            step = agent.observe({"state": [float(v) for v in obs]})
            obs, reward, terminated, truncated, _ = env.step(int(agent.act(step, "move", True)))
            agent.feedback(step, "reward", float(reward))
            dpan.advance_clock(1)
            steps += 1
            if terminated or truncated:
                agent.end_episode(terminated)
                break
    returns = []
    for k in range(100):
        obs, _ = env.reset(seed=900_000_000 + seed * 1000 + k)
        total = 0.0
        while True:
            step = agent.observe({"state": [float(v) for v in obs]})
            obs, reward, terminated, truncated, _ = env.step(int(agent.act(step, "move", False)))
            total += reward
            dpan.advance_clock(1)
            if terminated or truncated:
                agent.end_episode(terminated)
                break
        returns.append(total)
    rng = np.random.default_rng(seed)
    random_returns = []
    for k in range(100):
        env.reset(seed=900_000_000 + seed * 1000 + k)
        total = 0.0
        while True:
            _, reward, terminated, truncated, _ = env.step(int(rng.integers(env.action_space.n)))
            total += reward
            if terminated or truncated:
                break
        random_returns.append(total)
    return float(np.mean(returns)), float(np.mean(random_returns)), episode, env.spec.reward_threshold


@unittest.skipUnless(RUN, "set DPAN_GAMES to run the game tests")
@unittest.skipIf(gymnasium is None, "the gymnasium package is not installed")
class ClassicControlTest(unittest.TestCase):
    def setUp(self):
        dpan.use_manual_clock(1_700_000_000_000_000)

    def tearDown(self):
        dpan.use_system_clock()

    def _check(self, label, name, training_steps):
        failed = []  # every seed runs before the test fails
        for seed in SEEDS:
            mean, random_mean, episodes, threshold = _play_control(name, training_steps, seed)
            print(f"{label} {name} seed {seed}: mean return {mean:.1f} over 100 greedy episodes "
                  f"(threshold {threshold}; random play {random_mean:.1f}; {episodes} training episodes)", flush=True)
            if mean < threshold:
                failed.append(seed)
        self.assertEqual([], failed, f"{label}: seeds below the threshold")

    def test_c1_cart_pole(self):
        self._check("C1", "CartPole-v1", 300_000)

    def test_c2_acrobot(self):
        self._check("C2", "Acrobot-v1", 200_000)

    def test_c3_mountain_car(self):
        self._check("C3", "MountainCar-v0", 500_000)

    def test_c4_lunar_lander(self):
        try:
            gymnasium.make("LunarLander-v3")
        except Exception as error:  # gymnasium[box2d] missing
            self.skipTest(f"LunarLander-v3 is not available: {error}")
        self._check("C4", "LunarLander-v3", 600_000)


# ----------------------------------------------------------------------
# Board games (B1-B2). Boards are flat lists: 0 empty, 1 and -1 the players.

class TicTacToe:
    cells, actions = 9, 9
    LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

    @staticmethod
    def new():
        return [0] * 9

    @staticmethod
    def legal(board):
        return [i for i in range(9) if board[i] == 0]

    @staticmethod
    def play(board, move, player):
        """The board after `player` takes `move`, or None if it is not legal"""
        if board[move] != 0:
            return None
        board = list(board)
        board[move] = player
        return board

    @staticmethod
    def winner(board):
        for a, b, c in TicTacToe.LINES:
            if board[a] != 0 and board[a] == board[b] == board[c]:
                return board[a]
        return 0

    @staticmethod
    def full(board):
        return all(v != 0 for v in board)


class ConnectFour:
    rows, columns = 6, 7
    cells, actions = 42, 7

    @staticmethod
    def new():
        return [0] * 42

    @staticmethod
    def legal(board):
        return [c for c in range(7) if board[c] == 0]  # row 0 is the top

    @staticmethod
    def play(board, column, player):
        if board[column] != 0:
            return None
        board = list(board)
        for row in range(5, -1, -1):
            if board[row * 7 + column] == 0:
                board[row * 7 + column] = player
                return board

    @staticmethod
    def winner(board):
        for row in range(6):
            for col in range(7):
                p = board[row * 7 + col]
                if p == 0:
                    continue
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    r, c = row + 3 * dr, col + 3 * dc
                    if 0 <= r < 6 and 0 <= c < 7 and all(board[(row + k * dr) * 7 + col + k * dc] == p for k in range(4)):
                        return p
        return 0

    @staticmethod
    def full(board):
        return all(v != 0 for v in board[:7])


def random_player(game, board, player, rng):
    moves = game.legal(board)
    return moves[int(rng.integers(len(moves)))]


_minimax_cache = {}


def minimax_player(game, board, player, rng):
    """Perfect tic-tac-toe: a random move among those with the best value"""
    def value(b, to_move):
        key = (tuple(b), to_move)
        if key not in _minimax_cache:
            w = game.winner(b)
            if w != 0:
                _minimax_cache[key] = w
            elif game.full(b):
                _minimax_cache[key] = 0
            else:
                results = [value(game.play(b, m, to_move), -to_move) for m in game.legal(b)]
                _minimax_cache[key] = max(results) if to_move == 1 else min(results)
        return _minimax_cache[key]
    moves = game.legal(board)
    scores = [value(game.play(board, m, player), -player) * player for m in moves]
    best = max(scores)
    choices = [m for m, s in zip(moves, scores) if s == best]
    return choices[int(rng.integers(len(choices)))]


def rule_player(game, board, player, rng):
    """Connect Four: win if possible, else block the opponent's win, else random"""
    moves = game.legal(board)
    for who in (player, -player):
        for m in moves:
            if game.winner(game.play(board, m, who)) == who:
                return m
    return moves[int(rng.integers(len(moves)))]


def play_game(game, agent, opponent, agent_first, rng, explore):
    """One game; returns +1 (agent wins), -1 (agent loses) or 0. The agent
    is player 1 and sees the board with its own stones as +1."""
    board = game.new()
    player = 1 if agent_first else -1  # whose turn
    pending = None  # the agent's last step, waiting for its reward
    while True:
        if player == 1:
            step = agent.observe({"state": [float(v) for v in board]})
            move = int(agent.act(step, "move", explore))
            dpan.advance_clock(1)
            after = game.play(board, move, 1)
            if after is None:  # an illegal move loses the game
                agent.feedback(step, "reward", -1.0)
                agent.end_episode(True)
                return -1
            board = after
            if game.winner(board) == 1:
                agent.feedback(step, "reward", 1.0)
                agent.end_episode(True)
                return 1
            if game.full(board):
                agent.feedback(step, "reward", 0.0)
                agent.end_episode(True)
                return 0
            pending = step
        else:
            board = game.play(board, opponent(game, board, -1, rng), -1)
            if game.winner(board) == -1 or game.full(board):
                result = -1 if game.winner(board) == -1 else 0
                if pending is not None:
                    agent.feedback(pending, "reward", float(result))
                agent.end_episode(True)
                return result
            if pending is not None:
                agent.feedback(pending, "reward", 0.0)
                pending = None
        player = -player


def evaluate(game, agent, opponent, games, rng):
    results = [play_game(game, agent, opponent, k % 2 == 0, rng, False) for k in range(games)]
    return results.count(1) / games, results.count(-1) / games


def baseline(game, first, second, games, rng):
    """Wins and losses of `first` (alternating who starts) against `second`"""
    wins = losses = 0
    for k in range(games):
        board, player = game.new(), (1 if k % 2 == 0 else -1)
        while True:
            move = (first if player == 1 else second)(game, board, player, rng)
            board = game.play(board, move, player)
            w = game.winner(board)
            if w != 0 or game.full(board):
                wins += w == 1
                losses += w == -1
                break
            player = -player
    return wins / games, losses / games


@unittest.skipUnless(RUN, "set DPAN_GAMES to run the game tests")
class BoardGameTest(unittest.TestCase):
    def setUp(self):
        dpan.use_manual_clock(1_700_000_000_000_000)

    def tearDown(self):
        dpan.use_system_clock()

    def _train(self, game, opponents, games, seed):
        agent = _agent(game.cells, game.actions, seed)
        rng = np.random.default_rng(seed)
        for k in range(games):
            play_game(game, agent, opponents[k % 2], (k // 2) % 2 == 0, rng, True)
        return agent, rng

    def test_b1_tic_tac_toe(self):
        game = TicTacToe
        rng = np.random.default_rng(0)
        print("B1 baselines: random vs random wins %.3f losses %.3f; random vs minimax wins %.3f losses %.3f" % (
            *baseline(game, random_player, random_player, 1000, rng),
            *baseline(game, random_player, minimax_player, 200, rng)), flush=True)
        failed = []
        for seed in SEEDS:
            agent, rng = self._train(game, (random_player, minimax_player), 30_000, seed)
            wins, losses = evaluate(game, agent, random_player, 1000, rng)
            m_wins, m_losses = evaluate(game, agent, minimax_player, 200, rng)
            print(f"B1 tic-tac-toe seed {seed}: against random wins {wins:.3f} losses {losses:.3f}; "
                  f"against minimax wins {m_wins:.3f} losses {m_losses:.3f}", flush=True)
            if not (losses <= 0.01 and wins >= 0.80 and m_losses <= 0.05):
                failed.append(seed)
        self.assertEqual([], failed, "B1: seeds that do not pass")

    def test_b2_connect_four(self):
        game = ConnectFour
        rng = np.random.default_rng(0)
        print("B2 baselines: random vs random wins %.3f losses %.3f; random vs rule-based wins %.3f losses %.3f; "
              "rule-based vs random wins %.3f losses %.3f" % (
                  *baseline(game, random_player, random_player, 1000, rng),
                  *baseline(game, random_player, rule_player, 1000, rng),
                  *baseline(game, rule_player, random_player, 1000, rng)), flush=True)
        failed = []
        for seed in SEEDS:
            agent, rng = self._train(game, (random_player, rule_player), 50_000, seed)
            wins, losses = evaluate(game, agent, random_player, 1000, rng)
            r_wins, r_losses = evaluate(game, agent, rule_player, 1000, rng)
            print(f"B2 Connect Four seed {seed}: against random wins {wins:.3f} losses {losses:.3f}; "
                  f"against rule-based wins {r_wins:.3f} losses {r_losses:.3f}", flush=True)
            if not (wins >= 0.95 and r_wins >= 0.50):
                failed.append(seed)
        self.assertEqual([], failed, "B2: seeds that do not pass")


if __name__ == "__main__":
    unittest.main()
