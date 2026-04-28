"""Tests for Liar's Dice environment."""

import jax
import jax.numpy as jnp
import pytest

from jaxpot.env.liars_dice import LiarsDice
from jaxpot.env.liars_dice.game import (
    DICE_SIDES,
    Game,
    _bid_face,
    _bid_quantity,
    _count_matching,
)


class TestGameLogic:
    """Test pure game logic."""

    def test_action_encoding(self):
        """Bid (1, 1) = action 0, (1, 6) = action 5, (2, 1) = action 6."""
        assert _bid_quantity(jnp.int32(0)) == 1
        assert _bid_face(jnp.int32(0)) == 1
        assert _bid_quantity(jnp.int32(5)) == 1
        assert _bid_face(jnp.int32(5)) == 6
        assert _bid_quantity(jnp.int32(6)) == 2
        assert _bid_face(jnp.int32(6)) == 1

    def test_count_matching_no_wilds(self):
        """Count matching dice (face != 6, no wilds involved)."""
        dice = jnp.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        assert _count_matching(dice, jnp.int32(1)) == 2  # two 1s
        assert _count_matching(dice, jnp.int32(3)) == 2  # two 3s

    def test_count_matching_with_wilds(self):
        """Face 6 is wild and counts toward any non-6 face."""
        dice = jnp.array([[1, 6, 3, 4, 5], [6, 2, 3, 4, 5]])
        # Counting 1s: one actual 1 + two 6s (wild) = 3
        assert _count_matching(dice, jnp.int32(1)) == 3
        # Counting 6s: only actual 6s, no double-count
        assert _count_matching(dice, jnp.int32(6)) == 2

    def test_init_and_dice_range(self):
        """All dice values should be in [1, 6]."""
        game = Game(num_dice=5)
        key = jax.random.PRNGKey(42)
        state = game.init(key)
        assert state.dice.shape == (2, 5)
        assert jnp.all(state.dice >= 1)
        assert jnp.all(state.dice <= 6)

    def test_legal_actions_initial(self):
        """Initially all bids are legal, liar is not."""
        game = Game(num_dice=2)
        state = game.init(jax.random.PRNGKey(0))
        mask = game.legal_action_mask(state)
        # 4 total dice * 6 sides = 24 bids + 1 liar = 25 actions
        assert mask.shape == (25,)
        # All bids legal
        assert jnp.all(mask[:24])
        # Liar not legal (no bid yet)
        assert not mask[24]

    def test_legal_actions_after_bid(self):
        """After a bid, only higher bids and liar are legal."""
        game = Game(num_dice=2)
        state = game.init(jax.random.PRNGKey(0))
        # Bid action 5 = (1, 6)
        state = game.step(state, jnp.int32(5))
        mask = game.legal_action_mask(state)
        # Actions 0-5 should be illegal
        assert not jnp.any(mask[:6])
        # Actions 6-23 should be legal
        assert jnp.all(mask[6:24])
        # Liar should be legal
        assert mask[24]

    def test_liar_call_bidder_wins(self):
        """If the bid is correct, bidder wins."""
        game = Game(num_dice=2)
        # Set up known dice: all 1s
        state = game.init(jax.random.PRNGKey(0))
        state = state._replace(dice=jnp.ones((2, 2), dtype=jnp.int32))
        # Bid: (1, 1) - at least one 1 among 4 dice (true, we have 4)
        state = game.step(state, jnp.int32(0))  # player 0 bids (1,1)
        # Player 1 calls liar
        state = game.step(state, jnp.int32(game.liar_action))
        assert game.is_terminal(state)
        # Bidder (player 0) should win
        assert state.winner == 0

    def test_liar_call_challenger_wins(self):
        """If the bid is wrong, challenger wins."""
        game = Game(num_dice=2)
        state = game.init(jax.random.PRNGKey(0))
        # Set dice: all 1s
        state = state._replace(dice=jnp.ones((2, 2), dtype=jnp.int32))
        # Bid: (4, 3) = action (4-1)*6 + (3-1) = 20 — four 3s (false, we have zero 3s)
        state = game.step(state, jnp.int32(20))
        state = game.step(state, jnp.int32(game.liar_action))
        assert game.is_terminal(state)
        # Challenger (player 1) should win
        assert state.winner == 1

    def test_rewards(self):
        """Winner gets +1, loser gets -1."""
        game = Game(num_dice=2)
        state = game.init(jax.random.PRNGKey(0))
        state = state._replace(
            dice=jnp.ones((2, 2), dtype=jnp.int32),
            winner=jnp.int32(0),
        )
        rewards = game.rewards(state)
        assert rewards[0] == 1.0
        assert rewards[1] == -1.0

    def test_num_actions(self):
        """Action space size scales with num_dice."""
        assert Game(num_dice=1).num_actions == 2 * 1 * 6 + 1  # 13
        assert Game(num_dice=5).num_actions == 2 * 5 * 6 + 1  # 61
        assert Game(num_dice=3).num_actions == 2 * 3 * 6 + 1  # 37


class TestEnv:
    """Test PGX-compatible environment wrapper."""

    def test_init_and_step(self):
        """Basic init/step cycle."""
        env = LiarsDice(num_dice=2)
        key = jax.random.PRNGKey(0)
        state = env.init(key)
        assert not state.terminated
        assert state.legal_action_mask.shape == (25,)

        # Make a valid bid
        action = jnp.int32(0)
        state = env.step(state, action)
        assert not state.terminated

    def test_observe(self):
        """Observation has correct shape."""
        env = LiarsDice(num_dice=2)
        key = jax.random.PRNGKey(0)
        state = env.init(key)
        obs = env.observe(state, state.current_player)
        # dice: 2*6=12, bids: 4*6=24 → total 36
        assert obs.shape == (36,)

    def test_vmap(self):
        """Environment works with vmap."""
        env = LiarsDice(num_dice=2)
        keys = jax.random.split(jax.random.PRNGKey(0), 8)
        states = jax.vmap(env.init)(keys)
        assert states.terminated.shape == (8,)

        actions = jnp.zeros(8, dtype=jnp.int32)  # bid (1,1)
        states = jax.vmap(env.step)(states, actions)
        assert states.terminated.shape == (8,)

    def test_full_game(self):
        """Play a complete game: bid then call liar."""
        env = LiarsDice(num_dice=2)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Player bids (1, 1)
        state = env.step(state, jnp.int32(0))
        assert not state.terminated

        # Other player calls liar
        liar_action = 2 * 2 * 6  # = 24
        state = env.step(state, jnp.int32(liar_action))
        assert state.terminated
        # One player should have +1, other -1
        assert jnp.abs(state.rewards).sum() == 2.0

    def test_configurable_dice(self):
        """Different dice counts produce correct action spaces."""
        for n in [1, 3, 5]:
            env = LiarsDice(num_dice=n)
            expected_actions = 2 * n * 6 + 1
            key = jax.random.PRNGKey(0)
            state = env.init(key)
            assert state.legal_action_mask.shape == (expected_actions,)

    def test_compact_observation(self):
        """Compact observation class works."""
        env = LiarsDice(num_dice=2, observation_cls="compact")
        key = jax.random.PRNGKey(0)
        state = env.init(key)
        obs = env.observe(state, state.current_player)
        # dice: 2*6=12, quantity: 1, face: 6, remaining: 1 → 20
        assert obs.shape == (20,)
