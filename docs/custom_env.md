# Tutorial: Write a custom environment from scratch

If your game is not in `pgx`, you can write your own. Any class that subclasses `pgx.core.Env` and implements four methods works with Jaxpot end-to-end. This walkthrough builds a tiny **Subtraction Nim** environment in a single file and trains PPO on it.

> **The game.** Two players. Start with `15` stones. On your turn, take `1`, `2`, or `3` stones. Whoever takes the **last** stone wins. (Optimal play: leave the opponent with a multiple of 4.)

## 1. The interface you must implement

Subclass `pgx.core.Env` and implement:


| Member                                | Purpose                                                                                  |
| ------------------------------------- | ---------------------------------------------------------------------------------------- |
| `_init(key) -> State`                 | Build the initial state.                                                                 |
| `_step(state, action, key) -> State`  | Apply an action; set `rewards`, `terminated`, `current_player`, and `legal_action_mask`. |
| `_observe(state, player_id) -> Array` | Build the observation vector for the agent that is about to move.                        |
| `id`, `version`, `num_players`        | Metadata properties.                                                                     |


You also need a `State` dataclass that subclasses `pgx.core.State`. The base `State` already has `current_player`, `observation`, `rewards`, `terminated`, `truncated`, `legal_action_mask`, and `_step_count` — you just give them default values for your game and add any extra fields (here: `_stones`).

A few rules `pgx.core.Env` enforces for you (so you don't have to handle them):

- Illegal actions auto-terminate the episode with a `-1` reward to the offender.
- After `terminated == True`, further `step` calls return zero rewards.
- All code must be JAX-traceable: use `jnp` and `jax.lax.cond` / `jnp.where` instead of Python `if`.

## 2. The env (one file)

Create `src/jaxpot/env/nim/__init__.py` (empty) and `src/jaxpot/env/nim/env.py`:

```python
"""Subtraction Nim — minimal custom Jaxpot/pgx environment.

Two players. Start with 15 stones; on your turn take 1, 2, or 3.
Whoever takes the last stone wins.
"""

import jax
import jax.numpy as jnp
import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

NUM_STONES = 15
MAX_TAKE = 3  # action 0 -> take 1, action 1 -> take 2, action 2 -> take 3


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros(NUM_STONES + 1, dtype=jnp.float32)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    legal_action_mask: Array = jnp.ones(MAX_TAKE, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _stones: Array = jnp.int32(NUM_STONES)

    @property
    def env_id(self) -> core.EnvId:
        return "nim"


class Nim(core.Env):
    def _init(self, key: PRNGKey) -> State:
        # Randomize who moves first so PPO trains both seats.
        first = jax.random.bernoulli(key).astype(jnp.int32)
        return State(  # type: ignore
            current_player=first,
            _stones=jnp.int32(NUM_STONES),
            legal_action_mask=jnp.ones(MAX_TAKE, dtype=jnp.bool_),
        )

    def _step(self, state: State, action: Array, key) -> State:
        take = action + 1                       # 0/1/2 -> 1/2/3 stones
        new_stones = state._stones - take
        terminated = new_stones <= 0

        # Player who just moved took the last stone -> they win.
        winner = state.current_player
        rewards = jnp.where(
            terminated,
            jnp.where(jnp.arange(2) == winner, 1.0, -1.0).astype(jnp.float32),
            jnp.zeros(2, dtype=jnp.float32),
        )

        next_player = 1 - state.current_player
        legal = jnp.arange(MAX_TAKE) < jnp.minimum(new_stones, MAX_TAKE)
        # pgx overwrites this to all-True at terminal states; this is just a safe default.
        legal = jnp.where(terminated, jnp.ones(MAX_TAKE, dtype=jnp.bool_), legal)

        return state.replace(  # type: ignore
            current_player=next_player,
            _stones=jnp.maximum(new_stones, 0),
            rewards=rewards,
            terminated=terminated,
            legal_action_mask=legal,
        )

    def _observe(self, state: State, player_id: Array) -> Array:
        # One-hot encoding of remaining stones (0..NUM_STONES).
        return jax.nn.one_hot(state._stones, NUM_STONES + 1, dtype=jnp.float32)

    @property
    def id(self) -> core.EnvId:
        return "nim"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2
```

That's the entire game — under 80 lines.

> **Sanity-check it before training.** Drop this at the bottom of the file and run `python -m jaxpot.env.nim.env`:
>
> ```python
> if __name__ == "__main__":
>     env = Nim()
>     print(env.observation_shape, env.num_actions)  # (16,) 3
>     state = env.init(jax.random.PRNGKey(0))
>     while not bool(state.terminated):
>         state = env.step(state, jnp.int32(0))      # always take 1
>     print("rewards", state.rewards)                 # [1. -1.] or [-1. 1.]
> ```

## 3. Wire it into the configs

Three small YAML files, mirroring the tic-tac-toe tutorial above.

`**config/env/nim/default.yaml**` — point Hydra at your class:

```yaml
_target_: jaxpot.env.nim.env.Nim
```

`**config/model/nim_mlp.yaml**` — `observation_shape` is `(16,)` and `num_actions` is `3`:

```yaml
_target_: jaxpot.models.architectures.mlp.MLPModel
action_dim: 3
input_dim: 16
hidden_dims: [64, 64]
```

`**config/experiment/nim/fast.yaml**`:

```yaml
# @package _global_

defaults:
  - override /logger: none
  - override /model: nim_mlp
  - override /trainer: ppo
  - override /env: nim/default
  - override /eval: random
  - _self_

tags: ["nim"]
experiment_name: "nim_selfplay"

trainer:
  num_epochs: 2
  batch_size: 1024
  auxiliary_losses: []
  clip_eps: 0.2
  entropy_decay_iterations: 1_000

seed: 42
lr: 3e-4
lr_schedule: "constant"
multi_gpu: false
use_target_selfplay: false
selfplay_num_envs: 1024
random_num_envs: 512
league_num_envs: 0
archive_num_envs: 0
random_warmup_iters: 0
league_add_every: 0
base_unit: 64
num_steps: 16          # max game length is ~15 moves
total_iters: 1_000
grad_accum_steps: 1
gamma: 1.0             # short, terminal-reward game
gae_lambda: 0.95

max_grad_norm: 1.0
save_every: 200
keep_last_k: 3
best_checkpoint_top_k: 3
resume_from: null
```

## 4. Train

```bash
python train_selfplay.py experiment=nim/fast
```

The `eval_vs_random` win-rate should climb toward 100% within a few hundred iterations — Nim is solved, so a converged PPO agent should beat a random opponent essentially every game.

## Adapting this template to your own game

The Nim file is the smallest realistic template. To turn it into a different game:

1. **State fields.** Add whatever your game needs to `State` (board tensor, hand, pot, etc.) with `jnp` defaults.
2. `**_init`.** Set the starting state and `legal_action_mask`. Use `key` if you need randomness (e.g. dealing cards).
3. `**_step`.** Apply the action, compute the next `legal_action_mask`, set `terminated` and `rewards` (zero-sum: `[+1, -1]`, `[-1, +1]`, or `[0, 0]` on a draw). Flip `current_player` for alternating-move games.
4. `**_observe`.** Return a flat or shaped float array — whatever your model's `input_dim` / `observation_shape` expects. For imperfect-information games, only include what `player_id` is allowed to see.
5. **Update the model config** so `input_dim` / `action_dim` match your env's `observation_shape` / `num_actions`.

Everything else — vectorized rollouts, GAE, PPO updates, checkpoints, evaluators — is reused as-is by the framework.

