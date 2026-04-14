"""Tests for Phantom Tic-Tac-Toe environment."""

import jax
import jax.numpy as jnp
import pytest

from jaxpot.env.phantom_ttt import PhantomTTT
from jaxpot.env.phantom_ttt.game import Game, GameState


# ── Game logic tests ──────────────────────────────────────────────────────


class TestGame:
    def test_init(self):
        game = Game()
        state = game.init()
        assert jnp.all(state.board == 0)
        assert jnp.all(state.x_view == 0)
        assert jnp.all(state.o_view == 0)
        assert state.color == 0
        assert state.winner == -1

    def test_successful_move(self):
        game = Game()
        state = game.init()
        # X places at cell 4 (center)
        state = game.step(state, jnp.int32(4))
        assert state.board[4] == 1  # X mark
        assert state.x_view[4] == 1  # X sees own mark
        assert state.o_view[4] == 0  # O doesn't see it
        assert state.color == 1  # turn switched to O
        assert state.move_succeeded

    def test_failed_move_classical(self):
        game = Game(abrupt=False)
        state = game.init()
        # X places at cell 0
        state = game.step(state, jnp.int32(0))
        assert state.color == 1  # O's turn
        # O places at cell 1
        state = game.step(state, jnp.int32(1))
        assert state.color == 0  # X's turn
        # X tries to place at cell 1 (occupied by O)
        state = game.step(state, jnp.int32(1))
        assert state.board[1] == 2  # still O's mark
        assert state.x_view[1] == 2  # X now sees O's mark
        assert state.color == 0  # classical: X keeps turn
        assert not state.move_succeeded

    def test_failed_move_abrupt(self):
        game = Game(abrupt=True)
        state = game.init()
        # X places at cell 0
        state = game.step(state, jnp.int32(0))
        assert state.color == 1  # O's turn
        # O places at cell 1
        state = game.step(state, jnp.int32(1))
        assert state.color == 0  # X's turn
        # X tries to place at cell 1 (occupied by O)
        state = game.step(state, jnp.int32(1))
        assert state.board[1] == 2  # still O's mark
        assert state.x_view[1] == 2  # X now sees O's mark
        assert state.color == 1  # abrupt: turn switches
        assert not state.move_succeeded

    def test_x_wins(self):
        game = Game()
        state = game.init()
        # X: 0, 1, 2 (top row); O: 3, 4
        for x_move, o_move in [(0, 3), (1, 4)]:
            state = game.step(state, jnp.int32(x_move))
            state = game.step(state, jnp.int32(o_move))
        state = game.step(state, jnp.int32(2))  # X wins
        assert state.winner == 0
        assert game.is_terminal(state)
        rewards = game.rewards(state)
        assert rewards[0] == 1.0
        assert rewards[1] == -1.0

    def test_o_wins(self):
        game = Game()
        state = game.init()
        # X: 0, 1, 6; O: 3, 4, 5 (middle row)
        for x_move, o_move in [(0, 3), (1, 4), (6, 5)]:
            state = game.step(state, jnp.int32(x_move))
            state = game.step(state, jnp.int32(o_move))
        assert state.winner == 1
        assert game.is_terminal(state)

    def test_draw(self):
        game = Game()
        state = game.init()
        # X O X    (0,1,2)
        # X X O    (3,4,5)
        # O X O    (6,7,8)
        moves = [0, 1, 2, 4, 5, 3, 7, 8, 6]
        for move in moves:
            state = game.step(state, jnp.int32(move))
        assert state.winner == 2  # draw
        rewards = game.rewards(state)
        assert rewards[0] == 0.0
        assert rewards[1] == 0.0

    def test_legal_actions(self):
        game = Game()
        state = game.init()
        # All cells legal initially
        mask = game.legal_action_mask(state)
        assert jnp.all(mask)

        # After X places at 4, X's view shows 4 occupied
        state = game.step(state, jnp.int32(4))
        # O's turn: O's view is all empty, so all 9 legal
        mask = game.legal_action_mask(state)
        assert jnp.all(mask)

    def test_legal_actions_after_failed_move(self):
        """After a failed move, the revealed cell should be illegal."""
        game = Game(abrupt=False)
        state = game.init()
        # X places at 0
        state = game.step(state, jnp.int32(0))
        # O places at 1
        state = game.step(state, jnp.int32(1))
        # X tries cell 1 (fails, reveals O)
        state = game.step(state, jnp.int32(1))
        # X's view now shows cell 0 (own) and cell 1 (opponent) as occupied
        mask = game.legal_action_mask(state)
        assert not mask[0]  # X's own mark
        assert not mask[1]  # revealed opponent mark
        assert mask[2]  # still empty in view

    def test_jit_compatible(self):
        game = Game()

        @jax.jit
        def play_game():
            state = game.init()
            state = game.step(state, jnp.int32(0))
            state = game.step(state, jnp.int32(4))
            state = game.step(state, jnp.int32(1))
            return state

        state = play_game()
        assert state.board[0] == 1
        assert state.board[4] == 2
        assert state.board[1] == 1

    def test_vmap_compatible(self):
        game = Game()

        @jax.jit
        @jax.vmap
        def batch_step(actions):
            state = game.init()
            return game.step(state, actions)

        actions = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        states = batch_step(actions)
        # Each state should have exactly one X mark at the given cell
        assert jnp.all(jnp.sum(states.board == 1, axis=1) == 1)


# ── PGX Environment tests ────────────────────────────────────────────────


class TestPhantomTTTEnv:
    @pytest.fixture
    def env(self):
        return PhantomTTT()

    @pytest.fixture
    def env_abrupt(self):
        return PhantomTTT(abrupt=True)

    def test_init(self, env):
        key = jax.random.key(42)
        state = jax.jit(env.init)(key)
        assert state.legal_action_mask.shape == (9,)
        assert jnp.all(state.legal_action_mask)
        assert not state.terminated
        assert state.observation.shape == (3, 3, 3)

    def test_step_basic(self, env):
        key = jax.random.key(42)
        init_fn = jax.jit(env.init)
        step_fn = jax.jit(env.step)

        state = init_fn(key)
        p0 = state.current_player

        state = step_fn(state, jnp.int32(4), key)
        # Player should have switched (successful move)
        assert state.current_player == 1 - p0
        assert not state.terminated

    def test_failed_move_classical_env(self, env):
        """In classical mode, current_player stays the same on failed move."""
        key = jax.random.key(0)
        init_fn = jax.jit(env.init)
        step_fn = jax.jit(env.step)

        state = init_fn(key)
        p0 = state.current_player

        # p0 places at cell 0
        state = step_fn(state, jnp.int32(0), key)
        assert state.current_player == 1 - p0

        # p1 places at cell 1
        state = step_fn(state, jnp.int32(1), key)
        assert state.current_player == p0

        # p0 tries cell 1 (occupied by p1) - fails, stays p0
        state = step_fn(state, jnp.int32(1), key)
        assert state.current_player == p0

    def test_failed_move_abrupt_env(self, env_abrupt):
        """In abrupt mode, current_player switches even on failed move."""
        key = jax.random.key(0)
        init_fn = jax.jit(env_abrupt.init)
        step_fn = jax.jit(env_abrupt.step)

        state = init_fn(key)
        p0 = state.current_player

        # p0 places at cell 0
        state = step_fn(state, jnp.int32(0), key)
        assert state.current_player == 1 - p0

        # p1 places at cell 1
        state = step_fn(state, jnp.int32(1), key)
        assert state.current_player == p0

        # p0 tries cell 1 (occupied) - abrupt: switches to p1
        state = step_fn(state, jnp.int32(1), key)
        assert state.current_player == 1 - p0

    def test_observation_perspective(self, env):
        """Each player should only see their own marks and revealed cells."""
        key = jax.random.key(42)
        init_fn = jax.jit(env.init)
        step_fn = jax.jit(env.step)
        observe_fn = jax.jit(env.observe)

        state = init_fn(key)
        p0 = state.current_player

        # p0 places at cell 4
        state = step_fn(state, jnp.int32(4), key)

        # p1's observation should show all empty (doesn't know about p0's move)
        obs_p1 = observe_fn(state, state.current_player)
        # Channel 0 (empty) should be all 1s for p1
        assert jnp.sum(obs_p1[:, :, 0]) == 9.0

        # p0's observation should show cell 4 as "mine"
        obs_p0 = observe_fn(state, 1 - state.current_player)
        assert obs_p0[1, 1, 1] == 1.0  # cell 4 = (1,1), channel 1 = mine

    def test_terminal_rewards(self, env):
        """Test that rewards are correctly assigned at game end."""
        key = jax.random.key(0)
        init_fn = jax.jit(env.init)
        step_fn = jax.jit(env.step)

        state = init_fn(key)
        p0 = state.current_player

        # Play a quick game where first player wins with top row
        # p0: 0, 1, 2; p1: 3, 4
        state = step_fn(state, jnp.int32(0), key)
        state = step_fn(state, jnp.int32(3), key)
        state = step_fn(state, jnp.int32(1), key)
        state = step_fn(state, jnp.int32(4), key)
        state = step_fn(state, jnp.int32(2), key)

        assert state.terminated
        # p0 should get +1
        assert state.rewards[p0] == 1.0
        assert state.rewards[1 - p0] == -1.0

    def test_vectorized(self, env):
        """Test vmap works for batched environments."""
        init_fn = jax.jit(jax.vmap(env.init))
        step_fn = jax.jit(jax.vmap(env.step))

        keys = jax.random.split(jax.random.key(42), 16)
        states = init_fn(keys)

        actions = jnp.full(16, 4, dtype=jnp.int32)
        states = step_fn(states, actions, keys)
        assert states.legal_action_mask.shape == (16, 9)

    def test_flat_observation(self):
        """Test flat observation variant."""
        env = PhantomTTT(observation_cls="flat")
        key = jax.random.key(42)
        state = jax.jit(env.init)(key)
        obs = jax.jit(env.observe)(state, state.current_player)
        assert obs.shape == (27,)
        # Initially all cells empty: every 3rd value (starting at 0) should be 1
        assert jnp.sum(obs) == 9.0  # 9 empty cells

    def test_max_steps_truncation(self):
        """Test that the game truncates after max_steps."""
        env = PhantomTTT(max_steps=3)
        key = jax.random.key(0)
        init_fn = jax.jit(env.init)
        step_fn = jax.jit(env.step)

        state = init_fn(key)
        # Take 4 steps (step_count starts at 0 and increments before _step)
        for i in range(4):
            if not state.terminated:
                state = step_fn(state, jnp.int32(i), key)

        # Should be terminated via truncation
        assert state.terminated or state.truncated
