"""Tests for Dark Hex environment."""

import jax
import jax.numpy as jnp
import pytest

from jaxpot.env.dark_hex import DarkHex
from jaxpot.env.dark_hex.game import Game, GameState, _build_neighbor_table


class TestNeighborTable:
    def test_3x3_center_has_6_neighbors(self):
        neighbors = _build_neighbor_table(3, 3)
        # Cell 4 (center of 3x3) should have all 6 neighbors
        center_neighbors = neighbors[4]
        sentinel = 9
        valid = center_neighbors[center_neighbors != sentinel]
        assert len(valid) == 6

    def test_3x3_corner_has_correct_neighbors(self):
        neighbors = _build_neighbor_table(3, 3)
        sentinel = 9
        # Cell 0 (top-left corner): neighbors are NE=invalid, E=1, S=3, SW=invalid, W=invalid, N=invalid
        # Hex neighbors: N(-1,0)=invalid, NE(-1,1)=invalid, E(0,1)=1, S(1,0)=3, SW(1,-1)=invalid, W(0,-1)=invalid
        valid = neighbors[0][neighbors[0] != sentinel]
        assert set(valid.tolist()) == {1, 3}

    def test_3x3_cell1_neighbors(self):
        neighbors = _build_neighbor_table(3, 3)
        sentinel = 9
        # Cell 1 at (0,1): N=invalid, NE=invalid, E=2, S=4, SW=3, W=0
        valid = neighbors[1][neighbors[1] != sentinel]
        assert set(valid.tolist()) == {0, 2, 3, 4}


class TestGameLogic:
    def test_init(self):
        game = Game(3, 3)
        state = game.init()
        assert state.color == 0
        assert jnp.all(state.board == 0)
        assert state.winner == -1

    def test_successful_move(self):
        game = Game(3, 3)
        state = game.init()
        state = game.step(state, jnp.int32(4))  # Black plays center
        assert state.board[4] == 1  # Black stone
        assert state.black_view[4] == 1
        assert state.white_view[4] == 0  # White can't see it
        assert state.color == 1  # White's turn
        assert state.move_succeeded

    def test_collision_classical(self):
        game = Game(3, 3, abrupt=False)
        state = game.init()
        # Black plays cell 4
        state = game.step(state, jnp.int32(4))
        assert state.color == 1  # White's turn
        # White tries cell 4 (collision)
        state = game.step(state, jnp.int32(4))
        assert state.board[4] == 1  # Still Black's stone
        assert state.white_view[4] == 1  # White now sees Black's stone
        assert state.color == 1  # Classical: White retries (same turn)
        assert not state.move_succeeded

    def test_collision_abrupt(self):
        game = Game(3, 3, abrupt=True)
        state = game.init()
        # Black plays cell 4
        state = game.step(state, jnp.int32(4))
        assert state.color == 1
        # White tries cell 4 (collision)
        state = game.step(state, jnp.int32(4))
        assert state.board[4] == 1  # Still Black's stone
        assert state.white_view[4] == 1  # White sees it
        assert state.color == 0  # Abrupt: turn passes back to Black
        assert not state.move_succeeded

    def test_legal_actions(self):
        game = Game(3, 3)
        state = game.init()
        mask = game.legal_action_mask(state)
        assert jnp.all(mask)  # All 9 cells legal at start

        # Black plays cell 0
        state = game.step(state, jnp.int32(0))
        # White's legal mask: all cells appear empty (can't see Black's stone)
        mask = game.legal_action_mask(state)
        assert jnp.all(mask)  # White sees all empty

    def test_legal_actions_after_collision(self):
        game = Game(3, 3)
        state = game.init()
        state = game.step(state, jnp.int32(0))  # Black plays 0
        state = game.step(state, jnp.int32(0))  # White collides
        mask = game.legal_action_mask(state)
        assert not mask[0]  # White now knows cell 0 is occupied
        assert mask[1]  # Other cells still legal

    def test_black_wins_vertical(self):
        """Black connects North (row 0) to South (row 2) on a 3x3 board."""
        game = Game(3, 3)
        state = game.init()
        # Black: 0, 3, 6 (column 0, all rows) - but need to alternate with White
        state = game.step(state, jnp.int32(0))  # Black plays (0,0)
        state = game.step(state, jnp.int32(1))  # White plays (0,1)
        state = game.step(state, jnp.int32(3))  # Black plays (1,0)
        state = game.step(state, jnp.int32(4))  # White plays (1,1)
        state = game.step(state, jnp.int32(6))  # Black plays (2,0) - connects N to S

        assert state.winner == 0  # Black wins
        assert game.is_terminal(state)

    def test_white_wins_horizontal(self):
        """White connects West (col 0) to East (col 2) on a 3x3 board."""
        game = Game(3, 3)
        state = game.init()
        # Need to play so White gets a row connected left to right
        state = game.step(state, jnp.int32(6))  # Black plays (2,0)
        state = game.step(state, jnp.int32(0))  # White plays (0,0)
        state = game.step(state, jnp.int32(7))  # Black plays (2,1)
        state = game.step(state, jnp.int32(1))  # White plays (0,1)
        state = game.step(state, jnp.int32(3))  # Black plays (1,0)
        state = game.step(state, jnp.int32(2))  # White plays (0,2) - connects W to E

        assert state.winner == 1  # White wins
        assert game.is_terminal(state)

    def test_rewards(self):
        game = Game(3, 3)
        state = GameState(
            color=jnp.int32(0),
            board=jnp.zeros(9, dtype=jnp.int32),
            black_view=jnp.zeros(9, dtype=jnp.int32),
            white_view=jnp.zeros(9, dtype=jnp.int32),
            winner=jnp.int32(0),
            move_succeeded=jnp.bool_(True),
        )
        rewards = game.rewards(state)
        assert rewards[0] == 1.0  # Black wins
        assert rewards[1] == -1.0

    def test_configurable_board_size(self):
        game = Game(5, 5)
        state = game.init()
        assert state.board.shape == (25,)
        assert game.num_actions == 25
        mask = game.legal_action_mask(state)
        assert mask.shape == (25,)


class TestDarkHexEnv:
    def test_init_and_step(self):
        env = DarkHex(num_rows=3, num_cols=3)
        key = jax.random.PRNGKey(0)
        state = env.init(key)
        assert state.legal_action_mask.shape == (9,)
        assert not state.terminated

        state = env.step(state, jnp.int32(4), key)
        assert not state.terminated

    def test_observe(self):
        env = DarkHex(num_rows=3, num_cols=3)
        key = jax.random.PRNGKey(0)
        state = env.init(key)
        obs = env.observe(state, jnp.int32(0))
        assert obs.shape == (3, 3, 3)

    def test_observe_flat(self):
        env = DarkHex(num_rows=3, num_cols=3, observation_cls="flat")
        key = jax.random.PRNGKey(0)
        state = env.init(key)
        obs = env.observe(state, jnp.int32(0))
        assert obs.shape == (27,)

    def test_vmap_compatible(self):
        env = DarkHex(num_rows=3, num_cols=3)
        init_fn = jax.vmap(env.init)
        step_fn = jax.vmap(env.step)

        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        states = init_fn(keys)
        assert states.legal_action_mask.shape == (4, 9)

        actions = jnp.array([0, 1, 2, 3])
        new_keys = jax.random.split(jax.random.PRNGKey(1), 4)
        states = step_fn(states, actions, new_keys)
        assert states.legal_action_mask.shape == (4, 9)

    def test_jit_compatible(self):
        env = DarkHex(num_rows=3, num_cols=3)
        init_fn = jax.jit(env.init)
        step_fn = jax.jit(env.step)

        key = jax.random.PRNGKey(0)
        state = init_fn(key)
        state = step_fn(state, jnp.int32(4), key)
        assert not state.terminated

    def test_classical_variant(self):
        env = DarkHex(num_rows=3, num_cols=3, abrupt=False)
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        # Play a few steps to verify it works
        for i in range(5):
            state = env.step(state, jnp.int32(i), key)

    def test_abrupt_variant(self):
        env = DarkHex(num_rows=3, num_cols=3, abrupt=True)
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        for i in range(5):
            state = env.step(state, jnp.int32(i), key)

    def test_5x5_board(self):
        env = DarkHex(num_rows=5, num_cols=5)
        key = jax.random.PRNGKey(0)
        state = env.init(key)
        assert state.legal_action_mask.shape == (25,)
        obs = env.observe(state, jnp.int32(0))
        assert obs.shape == (5, 5, 3)

    def test_truncation(self):
        env = DarkHex(num_rows=3, num_cols=3, max_steps=3, abrupt=True)
        key = jax.random.PRNGKey(0)
        state = env.init(key)
        # Step 4 times (max_steps=3, so step 4 should truncate)
        for i in range(4):
            state = env.step(state, jnp.int32(i % 9), key)
        assert state.truncated or state.terminated
