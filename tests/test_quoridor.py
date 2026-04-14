"""Tests for the Quoridor environment."""

import jax
import jax.numpy as jnp
import pytest

from jaxpot.env.quoridor import Quoridor, QuoridorSpatialObservation
from jaxpot.env.quoridor.observation import QuoridorSpatialScalarObservation
from jaxpot.env.quoridor.game import (
    BOARD_SIZE,
    NUM_ACTIONS,
    NUM_WALLS_PER_PLAYER,
    WALL_SIZE,
    Game,
    GameState,
    _move_legal_mask,
    _pos_to_rc,
    _rc_to_pos,
)


@pytest.fixture
def env():
    return Quoridor()


@pytest.fixture
def game():
    return Game()


@pytest.fixture
def init_state(env):
    return env.init(jax.random.PRNGKey(0))


class TestGameState:
    def test_initial_positions(self):
        gs = GameState()
        assert int(gs.pawn_pos[0]) == 4  # (0, 4)
        assert int(gs.pawn_pos[1]) == 76  # (8, 4)
        assert int(gs.color) == 0
        assert int(gs.winner) == -1

    def test_initial_walls(self):
        gs = GameState()
        assert int(gs.walls_remaining[0]) == NUM_WALLS_PER_PLAYER
        assert int(gs.walls_remaining[1]) == NUM_WALLS_PER_PLAYER
        assert not gs.h_walls.any()
        assert not gs.v_walls.any()


class TestHelpers:
    def test_pos_to_rc(self):
        r, c = _pos_to_rc(jnp.int32(0))
        assert int(r) == 0 and int(c) == 0

        r, c = _pos_to_rc(jnp.int32(4))
        assert int(r) == 0 and int(c) == 4

        r, c = _pos_to_rc(jnp.int32(76))
        assert int(r) == 8 and int(c) == 4

        r, c = _pos_to_rc(jnp.int32(80))
        assert int(r) == 8 and int(c) == 8

    def test_rc_to_pos(self):
        assert int(_rc_to_pos(jnp.int32(0), jnp.int32(4))) == 4
        assert int(_rc_to_pos(jnp.int32(8), jnp.int32(4))) == 76


class TestMovement:
    def test_cardinal_moves_from_start(self, game):
        gs = game.init()
        mask = _move_legal_mask(gs)
        # P0 at (0, 4): can go S, E, W but not N
        assert not bool(mask[0])  # N - out of bounds
        assert bool(mask[1])  # S
        assert bool(mask[2])  # E
        assert bool(mask[3])  # W

    def test_move_south(self, game):
        gs = game.init()
        gs2 = game.step(gs, jnp.int32(1))  # Move S
        r, c = _pos_to_rc(gs2.pawn_pos[0])
        assert int(r) == 1 and int(c) == 4

    def test_move_east(self, game):
        gs = game.init()
        gs2 = game.step(gs, jnp.int32(2))  # Move E
        r, c = _pos_to_rc(gs2.pawn_pos[0])
        assert int(r) == 0 and int(c) == 5

    def test_move_west(self, game):
        gs = game.init()
        gs2 = game.step(gs, jnp.int32(3))  # Move W
        r, c = _pos_to_rc(gs2.pawn_pos[0])
        assert int(r) == 0 and int(c) == 3

    def test_corner_moves(self, game):
        # Place P0 at (0,0) - only S and E are legal
        gs = GameState(pawn_pos=jnp.array([0, 80], dtype=jnp.int32))
        mask = _move_legal_mask(gs)
        assert not bool(mask[0])  # N
        assert bool(mask[1])  # S
        assert bool(mask[2])  # E
        assert not bool(mask[3])  # W

    def test_occupied_blocks_cardinal(self, game):
        # P0 at (4,4), P1 at (5,4) - P0 can't move S (occupied)
        gs = GameState(
            pawn_pos=jnp.array([_rc_to_pos(jnp.int32(4), jnp.int32(4)), _rc_to_pos(jnp.int32(5), jnp.int32(4))], dtype=jnp.int32)
        )
        mask = _move_legal_mask(gs)
        assert not bool(mask[1])  # S blocked by opponent


class TestJumps:
    def test_straight_jump_over_opponent(self, game):
        # P0 at (4,4), P1 at (5,4) - P0 can jump S over P1 to (6,4)
        gs = GameState(
            pawn_pos=jnp.array([_rc_to_pos(jnp.int32(4), jnp.int32(4)), _rc_to_pos(jnp.int32(5), jnp.int32(4))], dtype=jnp.int32)
        )
        mask = _move_legal_mask(gs)
        assert bool(mask[4 + 1])  # Jump S (action 5)

        gs2 = game.step(gs, jnp.int32(5))  # Jump S
        r, c = _pos_to_rc(gs2.pawn_pos[0])
        assert int(r) == 6 and int(c) == 4

    def test_jump_blocked_by_wall(self, game):
        # P0 at (4,4), P1 at (5,4), wall between (5,4) and (6,4) blocks jump
        gs = GameState(
            pawn_pos=jnp.array([_rc_to_pos(jnp.int32(4), jnp.int32(4)), _rc_to_pos(jnp.int32(5), jnp.int32(4))], dtype=jnp.int32),
            h_walls=jnp.zeros((WALL_SIZE, WALL_SIZE), dtype=jnp.bool_).at[5, 4].set(True),
        )
        mask = _move_legal_mask(gs)
        assert not bool(mask[4 + 1])  # Jump S blocked by wall

    def test_jump_blocked_by_edge(self, game):
        # P0 at (7,4), P1 at (8,4) - jump would go to (9,4) which is off-board
        gs = GameState(
            pawn_pos=jnp.array([_rc_to_pos(jnp.int32(7), jnp.int32(4)), _rc_to_pos(jnp.int32(8), jnp.int32(4))], dtype=jnp.int32)
        )
        mask = _move_legal_mask(gs)
        assert not bool(mask[4 + 1])  # Jump S blocked by edge

    def test_diagonal_jump_when_straight_blocked(self, game):
        # P0 at (7,4), P1 at (8,4) - straight jump S blocked by edge
        # Diagonal SE/SW should be legal
        gs = GameState(
            pawn_pos=jnp.array([_rc_to_pos(jnp.int32(7), jnp.int32(4)), _rc_to_pos(jnp.int32(8), jnp.int32(4))], dtype=jnp.int32)
        )
        mask = _move_legal_mask(gs)
        assert bool(mask[8 + 2])  # Diagonal SE (approach S, slide E)
        assert bool(mask[8 + 3])  # Diagonal SW (approach S, slide W)

    def test_diagonal_jump_lands_correctly(self, game):
        # P0 at (7,4), P1 at (8,4) - diagonal SE should land at (8,5)
        gs = GameState(
            pawn_pos=jnp.array([_rc_to_pos(jnp.int32(7), jnp.int32(4)), _rc_to_pos(jnp.int32(8), jnp.int32(4))], dtype=jnp.int32)
        )
        gs2 = game.step(gs, jnp.int32(10))  # Diagonal SE
        r, c = _pos_to_rc(gs2.pawn_pos[0])
        assert int(r) == 8 and int(c) == 5


class TestWalls:
    def test_horizontal_wall_placement(self, game):
        gs = game.init()
        action = 12 + 3 * 8 + 3  # h_wall at (3, 3)
        gs2 = game.step(gs, jnp.int32(action))
        assert bool(gs2.h_walls[3, 3])
        assert int(gs2.walls_remaining[0]) == NUM_WALLS_PER_PLAYER - 1

    def test_vertical_wall_placement(self, game):
        gs = game.init()
        action = 76 + 2 * 8 + 5  # v_wall at (2, 5)
        gs2 = game.step(gs, jnp.int32(action))
        assert bool(gs2.v_walls[2, 5])
        assert int(gs2.walls_remaining[0]) == NUM_WALLS_PER_PLAYER - 1

    def test_wall_blocks_movement(self, game):
        # Place horizontal wall between rows 0/1 at cols 4/5
        # This should block P0 from moving South from (0,4)
        gs = GameState(h_walls=jnp.zeros((WALL_SIZE, WALL_SIZE), dtype=jnp.bool_).at[0, 4].set(True))
        mask = _move_legal_mask(gs)
        assert not bool(mask[1])  # S should be blocked

    def test_no_overlapping_horizontal_walls(self, game):
        gs = game.init()
        # Place h_wall at (3, 3)
        gs2 = game.step(gs, jnp.int32(12 + 3 * 8 + 3))
        # Adjacent h_walls should be illegal
        mask = game.legal_action_mask(gs2)
        assert not bool(mask[12 + 3 * 8 + 2])  # (3, 2) overlaps
        assert not bool(mask[12 + 3 * 8 + 4])  # (3, 4) overlaps
        assert not bool(mask[12 + 3 * 8 + 3])  # same position

    def test_no_crossing_walls(self, game):
        gs = game.init()
        gs2 = game.step(gs, jnp.int32(12 + 3 * 8 + 3))  # h_wall at (3, 3)
        mask = game.legal_action_mask(gs2)
        assert not bool(mask[76 + 3 * 8 + 3])  # v_wall at (3, 3) crosses

    def test_no_walls_when_depleted(self, game):
        gs = GameState(walls_remaining=jnp.array([0, 10], dtype=jnp.int32))
        mask = game.legal_action_mask(gs)
        # All wall actions should be illegal for P0
        assert not mask[12:].any()


class TestWinCondition:
    def test_player0_wins(self, game):
        # P0 one step from goal row 8
        gs = GameState(
            pawn_pos=jnp.array([_rc_to_pos(jnp.int32(7), jnp.int32(4)), _rc_to_pos(jnp.int32(0), jnp.int32(0))], dtype=jnp.int32)
        )
        gs2 = game.step(gs, jnp.int32(1))  # Move S to row 8
        assert int(gs2.winner) == 0
        assert bool(game.is_terminal(gs2))
        rewards = game.rewards(gs2)
        assert float(rewards[0]) == 1.0
        assert float(rewards[1]) == -1.0

    def test_player1_wins(self, game):
        # P1 one step from goal row 0
        gs = GameState(
            color=jnp.int32(1),
            pawn_pos=jnp.array([_rc_to_pos(jnp.int32(8), jnp.int32(8)), _rc_to_pos(jnp.int32(1), jnp.int32(4))], dtype=jnp.int32),
        )
        gs2 = game.step(gs, jnp.int32(0))  # Move N to row 0
        assert int(gs2.winner) == 1
        assert bool(game.is_terminal(gs2))
        rewards = game.rewards(gs2)
        assert float(rewards[0]) == -1.0
        assert float(rewards[1]) == 1.0

    def test_not_terminal_initially(self, game):
        gs = game.init()
        assert not bool(game.is_terminal(gs))


class TestEnvWrapper:
    def test_init(self, env):
        state = env.init(jax.random.PRNGKey(0))
        assert state.observation.shape == (BOARD_SIZE, BOARD_SIZE, 4)
        assert state.legal_action_mask.shape == (NUM_ACTIONS,)
        assert not bool(state.terminated)

    def test_step(self, env, init_state):
        state = env.step(init_state, jnp.int32(1))  # Move S
        assert int(state.current_player) == 1 - int(init_state.current_player)
        assert not bool(state.terminated)

    def test_illegal_action_penalty(self, env, init_state):
        # Action 0 (N) is illegal for P0 at row 0
        assert not bool(init_state.legal_action_mask[0])
        state = env.step(init_state, jnp.int32(0))
        # PGX penalizes illegal actions by giving -1 reward to the player
        assert bool(state.terminated)

    def test_observation_perspective(self, env, init_state):
        # Observe from player 0's perspective
        obs0 = env.observe(init_state, jnp.int32(init_state.current_player))
        assert obs0.shape == (BOARD_SIZE, BOARD_SIZE, 4)
        # Channel 0 should have my pawn
        assert float(obs0[:, :, 0].sum()) == 1.0
        # Channel 1 should have opponent pawn
        assert float(obs0[:, :, 1].sum()) == 1.0

    def test_env_properties(self, env):
        assert env.id == "quoridor"
        assert env.version == "v1"
        assert env.num_players == 2


class TestObservation:
    def test_observation_shape(self):
        gs = GameState()
        obs = QuoridorSpatialObservation.from_state(gs, color=jnp.int32(0))
        assert obs.shape == (BOARD_SIZE, BOARD_SIZE, 4)

    def test_pawn_channels(self):
        gs = GameState()
        obs = QuoridorSpatialObservation.from_state(gs, color=jnp.int32(0))
        # Channel 0: my pawn (P0 at (0,4))
        assert float(obs[0, 4, 0]) == 1.0
        assert float(obs[:, :, 0].sum()) == 1.0
        # Channel 1: opponent pawn (P1 at (8,4))
        assert float(obs[8, 4, 1]) == 1.0
        assert float(obs[:, :, 1].sum()) == 1.0

    def test_wall_channels(self):
        gs = GameState(h_walls=jnp.zeros((WALL_SIZE, WALL_SIZE), dtype=jnp.bool_).at[3, 3].set(True))
        obs = QuoridorSpatialObservation.from_state(gs, color=jnp.int32(0))
        assert float(obs[3, 3, 2]) == 1.0  # h_wall channel
        assert float(obs[:, :, 2].sum()) == 1.0

    def test_perspective_swap(self):
        gs = GameState()
        obs0 = QuoridorSpatialObservation.from_state(gs, color=jnp.int32(0))
        obs1 = QuoridorSpatialObservation.from_state(gs, color=jnp.int32(1))
        # P0's "my pawn" should be P1's "opponent pawn"
        assert jnp.allclose(obs0[:, :, 0], obs1[:, :, 1])
        assert jnp.allclose(obs0[:, :, 1], obs1[:, :, 0])


class TestSVG:
    def test_svg_generation(self, env, init_state):
        svg = init_state.to_svg()
        assert isinstance(svg, str)
        assert svg.startswith("<svg")
        assert "</svg>" in svg

    def test_svg_with_walls(self, env, init_state):
        state = env.step(init_state, jnp.int32(12 + 3 * 8 + 3))  # Place wall
        svg = state.to_svg()
        assert isinstance(svg, str)
        assert ">9<" in svg  # wall count for the player who placed


class TestJAXCompatibility:
    def test_jit_init(self, env):
        init_fn = jax.jit(env.init)
        state = init_fn(jax.random.PRNGKey(42))
        assert state.observation.shape == (BOARD_SIZE, BOARD_SIZE, 4)

    def test_jit_step(self, env, init_state):
        step_fn = jax.jit(env.step)
        state = step_fn(init_state, jnp.int32(1))
        assert not bool(state.terminated)

    def test_vmap_init(self, env):
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        init_fn = jax.vmap(env.init)
        states = init_fn(keys)
        assert states.observation.shape == (4, BOARD_SIZE, BOARD_SIZE, 4)
        assert states.legal_action_mask.shape == (4, NUM_ACTIONS)

    def test_vmap_step(self, env):
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        states = jax.vmap(env.init)(keys)
        actions = jnp.array([1, 1, 2, 3], dtype=jnp.int32)  # Various legal moves
        step_fn = jax.vmap(env.step)
        new_states = step_fn(states, actions)
        assert new_states.observation.shape == (4, BOARD_SIZE, BOARD_SIZE, 4)


class TestPerspectiveNormalization:
    """Both colors should see the board from the same perspective."""

    def test_symmetric_start_same_observation(self, env):
        """At start, both colors see identical observations (my pawn at row 0, opp at row 8)."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 100)
        states = jax.vmap(env.init)(keys)

        # Find a game where color 0 goes first and one where color 1 goes first
        # (current_player determines which external player maps to internal color 0)
        # All should have identical observations because both see the canonical view
        # Check first game's observation: my pawn at (0,4), opp at (8,4)
        obs = states.observation[0]
        assert float(obs[0, 4, 0]) == 1.0  # my pawn at (0, 4)
        assert float(obs[8, 4, 1]) == 1.0  # opponent at (8, 4)

    def test_both_colors_see_same_initial_board(self, env):
        """All games have identical initial observations regardless of current_player."""
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 20)
        states = jax.vmap(env.init)(keys)

        # Every game should see the exact same observation
        first_obs = states.observation[0]
        for i in range(1, 20):
            assert jnp.allclose(first_obs, states.observation[i])

    def test_action_s_moves_toward_opponent_for_both(self, env):
        """Action S (1) should move toward opponent's base for both colors."""
        key = jax.random.PRNGKey(0)
        state = env.init(key)

        # Step 1: color 0 plays S (action 1) — absolute S, moves pawn from row 0 to row 1
        state2 = env.step(state, jnp.int32(1))
        # Internal pawn should have moved to row 1
        r0, c0 = state2._x.pawn_pos[0] // 9, state2._x.pawn_pos[0] % 9
        assert int(r0) == 1  # color 0 moved south (toward row 8)

        # Step 2: color 1 plays S (action 1) — should be flipped to absolute N
        state3 = env.step(state2, jnp.int32(1))
        r1, c1 = state3._x.pawn_pos[1] // 9, state3._x.pawn_pos[1] % 9
        assert int(r1) == 7  # color 1 moved toward row 0 (their goal)

    def test_wall_placement_flipped_for_color1(self, env):
        """Wall at (0,0) in canonical coords should be at (7,7) in absolute for color 1."""
        key = jax.random.PRNGKey(0)
        state = env.init(key)

        # Color 0 places h-wall at canonical (0,0) = absolute (0,0) = action 12
        state2 = env.step(state, jnp.int32(12))
        assert bool(state2._x.h_walls[0, 0])

        # Color 1 places h-wall at canonical (0,0) = absolute (7,7) = action 12
        state3 = env.step(state2, jnp.int32(12))
        assert bool(state3._x.h_walls[7, 7])

    def test_legal_mask_flipped_for_color1(self, env):
        """Legal mask should be in canonical coords for both colors."""
        key = jax.random.PRNGKey(0)
        state = env.init(key)

        # At init (color 0): N is illegal (row 0, can't go up)
        assert not bool(state.legal_action_mask[0])  # N illegal
        assert bool(state.legal_action_mask[1])       # S legal

        # After one move, color 1 acts. In canonical view, color 1 also sees
        # themselves at row 0, so N should also be illegal
        state2 = env.step(state, jnp.int32(1))  # color 0 moves S
        assert not bool(state2.legal_action_mask[0])  # N illegal for color 1 canonical
        assert bool(state2.legal_action_mask[1])       # S legal for color 1 canonical

    def test_flip_action_roundtrip(self):
        """Flipping an action twice returns the original."""
        from jaxpot.env.quoridor.env import _flip_action
        for a in range(NUM_ACTIONS):
            action = jnp.int32(a)
            assert int(_flip_action(_flip_action(action))) == a

    def test_flip_mask_roundtrip(self):
        """Flipping a mask twice returns the original."""
        from jaxpot.env.quoridor.env import _flip_mask
        mask = jax.random.bernoulli(jax.random.PRNGKey(0), shape=(NUM_ACTIONS,))
        roundtrip = _flip_mask(_flip_mask(mask))
        assert jnp.array_equal(mask, roundtrip)


class TestObservationClasses:
    def test_spatial_shape(self):
        obs = QuoridorSpatialObservation()
        assert obs.shape == (BOARD_SIZE, BOARD_SIZE, 4)

    def test_spatial_scalar_shape(self):
        obs = QuoridorSpatialScalarObservation()
        # 9 * 9 * 4 spatial flat + 2 wall scalars = 326
        assert obs.shape == (BOARD_SIZE * BOARD_SIZE * 4 + 2,)

    def test_spatial_scalar_first_part_matches_spatial(self):
        """First 324 elements of the flat scalar obs equal the spatial obs flattened."""
        gs = GameState()
        color = jnp.int32(0)
        spatial = QuoridorSpatialObservation.from_state(gs, color=color)
        flat = QuoridorSpatialScalarObservation.from_state(gs, color=color)
        spatial_size = BOARD_SIZE * BOARD_SIZE * 4
        assert jnp.allclose(spatial.reshape(-1), flat[:spatial_size])

    def test_spatial_scalar_wall_counts(self):
        """Trailing scalar features encode my/opp walls remaining (normalized)."""
        gs = GameState()
        spatial_size = BOARD_SIZE * BOARD_SIZE * 4
        # Both players start with 10 walls -> 1.0 after normalization
        flat0 = QuoridorSpatialScalarObservation.from_state(gs, color=jnp.int32(0))
        assert float(flat0[spatial_size]) == 1.0
        assert float(flat0[spatial_size + 1]) == 1.0

    def test_env_with_spatial_scalar_observation(self):
        env = Quoridor(observation_cls="spatial_scalar")
        state = env.init(jax.random.PRNGKey(0))
        spatial_size = BOARD_SIZE * BOARD_SIZE * 4
        assert state.observation.shape == (spatial_size + 2,)

        state = env.step(state, jnp.int32(1))
        assert state.observation.shape == (spatial_size + 2,)

    def test_env_spatial_scalar_vmap(self):
        env = Quoridor(observation_cls="spatial_scalar")
        spatial_size = BOARD_SIZE * BOARD_SIZE * 4
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        states = jax.vmap(env.init)(keys)
        assert states.observation.shape == (4, spatial_size + 2)

        actions = jnp.array([1, 1, 2, 3], dtype=jnp.int32)
        new_states = jax.vmap(env.step)(states, actions)
        assert new_states.observation.shape == (4, spatial_size + 2)
