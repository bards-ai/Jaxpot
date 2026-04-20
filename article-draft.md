# Train Self-Play RL Agents Fast 🏎️ With Jaxpot

In this post you will:

1. Train self-play agent with Jaxpot in Colab.
2. Inspect the training, with W&B.
3. Reuse this training stack in more demanding imperfect-information game: Dark Hex.

---

## Jaxpot

It is built around three practical ideas:

- **PPO and AlphaZero-style training.** PPO gives you a strong policy-gradient baseline for self-play. AlphaZero-style components are useful when you want search and value-guided planning.
- **JAX.**  Vectorized rollouts and training, compiled, and run efficiently on accelerators. Pushing the training speed to the hardware limit.
- **Hydra configs.** Experiments are composed from small config files for the game, model, trainer, evaluator, and logger. Changing the game or training setup requires just changing the config.

---

## What Jaxpot Gives You

At a high level, it provides:

- JAX-native environments through `pgx` or custom `pgx.core.Env` implementations.
- PPO training for policy/value agents.
- AlphaZero-style components for search-based agents.
- Self-play rollouts compiled with JAX.
- League and archive play, so agents can train against older opponents.
- Evaluators against random agents, baselines, archived policies, and small-game Nash exploitability.
- Hydra configs for experiments, models, environments, trainers, and loggers.
- TensorBoard and Weights & Biases logging integrated.

---

## Why It Is A Good Fit For Imperfect-Information Games

In perfect information games like Chess, Go, and standard Hex, both players see the full board. The game state is public.

Imperfect-information games split the state into public and private parts. Poker hides cards. Liar's Dice hides dice. Dark Hex hides opponent stones unless you collide with them.

That changes everything. The optimal strategy is not a single fixed policy. A poker bot that always plays the highest win-rate line becomes predictable and easy to exploit.

Self-play already helps: the opponent keeps changing, so the policy cannot overfit to a fixed strategy. Entropy scheduling keeps the policy exploring early in training, which matters when the game demands mixed strategies.

On top of that, Jaxpot supports League play. The agent trains against frozen snapshots of itself from earlier in the run, and opponents it struggles against get higher sampling weight. That prevents the policy from "forgetting" how to beat older versions of itself. When the league fills up, surplus opponents move to an archive that can reactivate if the agent starts losing to them again.


---

## Quick Start: Tic-Tac-Toe In Colab

The fastest way to use the library is the companion Colab notebook:

```text
tic_tac_toe_colab_quickstart.ipynb
```

TODO: In the published repo this notebook should be linked near the top of the README, ideally with an "Open in Colab" badge.

Now switch to the Colab and run your first experiments with Jaxpot!

---

## Reading The Training Output

When the run starts, Jaxpot prints a TUI dashboard.

![TUI dashboard](public/jaxpot tui.png)

The dashboard is useful while the run is active, but curves are better for understanding training. You should see something like this after the run:

![policy loss](public/policy loss.png)

![win rate](public/win rate.png)

Jaxpot supports Weights & Biases. For this article, there is a public project where you can see your training:

```text
https://wandb.ai/team-bards-ai/Jaxpot%20Public
```

The curves worth checking first are:

- policy loss: is the policy still changing?
- value loss: is the value head learning to predict outcomes?
- entropy: is the policy still exploring or collapsing too early?
- KL: are PPO updates staying in a reasonable range?
- evaluation win rate: is the policy improving against the configured opponent?
- samples/sec: is the run using your hardware effectively?

---

## From Toy Game To Real Game: Dark Hex

Tic-Tac-Toe proves the pipeline works. Now let's switch to a game that is still small enough to understand, but much more interesting.

Dark Hex is an imperfect-information version of Hex.

In normal Hex, both players see the whole board. In Dark Hex, each player sees:

- empty cells according to their own view
- their own stones
- opponent stones only when revealed through failed placement attempts

The true board exists, but the agent does not get to see it.

This is a much better demonstration of why environment design matters. The agent is not just choosing a move from a board. It is acting under uncertainty.

Jaxpot already includes the Dark Hex environment:

```text
src/jaxpot/env/dark_hex/
```

And the environment configs:

```text
config/env/dark_hex/classical.yaml
config/env/dark_hex/abrupt.yaml
```

The classical version works like this:

- if you choose an empty cell, you place your stone and the turn passes
- if you choose an occupied cell, the cell is revealed and you try again

The abrupt version is harsher:

- if you choose an occupied cell, the cell is revealed and you lose the turn

That one rule changes the information economics of the game. A failed move is no longer just information; it is information plus tempo loss.

---

## Creating A Dark Hex Experiment

To train on Dark Hex, we do not need a new training loop. We need an experiment config.

Create:

```text
config/experiment/dark_hex/fast.yaml
```

With:

```yaml
# @package _global_

defaults:
  - override /logger: tensorboard
  - override /model: mlp
  - override /trainer: ppo
  - override /env: dark_hex/classical
  - override /eval: random
  - _self_

tags: ["dark_hex", "imperfect_information", "quickstart"]
experiment_name: "dark_hex_fast"

trainer:
  num_epochs: 2
  batch_size: 1024
  auxiliary_losses: []
  clip_eps: 0.2
  entropy_coeff: 0.01
  entropy_coeff_start: 0.05
  entropy_decay_iterations: 500

model:
  hidden_dims: [128, 128]

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
num_steps: 32
total_iters: 500
grad_accum_steps: 1
gamma: 0.99
gae_lambda: 0.95

max_grad_norm: 1.0
save_every: 100
keep_last_k: 3
best_checkpoint_top_k: 3
resume_from: null

eval:
  - _target_: jaxpot.evaluator.random.RandomEvaluator
    eval_every: 50
    num_envs: 1024
    num_steps: ${num_steps}
    deterministic: false
    name: eval_vs_random
  - _target_: jaxpot.evaluator.random.RandomEvaluator
    eval_every: 50
    num_envs: 1024
    num_steps: ${num_steps}
    deterministic: true
    name: eval_vs_random_deterministic
```

Then run:

```bash
uv run python train_selfplay.py experiment=dark_hex/fast
```

For a smoke test, reduce the iteration count:

```bash
uv run python train_selfplay.py experiment=dark_hex/fast total_iters=10
```

For the abrupt variant, swap the env:

```bash
uv run python train_selfplay.py experiment=dark_hex/fast env=dark_hex/abrupt
```

That is the moment where the framework starts to pay off. We changed the game dynamics with a config override.

No new rollout code.

No new trainer.

No new logger.

Just a different experiment.

---

## What You Need To Add Your Own Game

There are two paths.

The easy path is to use an existing `pgx` environment. That is what the Tic-Tac-Toe example does.

The custom path is to implement a `pgx.core.Env`. That is what Dark Hex does.

A custom Jaxpot-compatible environment needs:

- `_init(key)`: create the initial state
- `_step(state, action, key)`: apply one action
- `_observe(state, player_id)`: return what this player is allowed to see
- `legal_action_mask`: tell the agent which moves are available
- `rewards`: return zero-sum rewards at terminal states
- `terminated` / `truncated`: stop episodes cleanly

For imperfect-information games, `_observe` is where the design lives.

You usually do not want to expose the true state. You want to expose the information state: what the acting player knows at that moment.

In Dark Hex, the true board contains both players' stones. But the observation has only three channels:

```text
empty cells in my view
my stones
opponent stones revealed by failed attempts
```

That is the whole point. The hidden state exists, but the policy cannot cheat.

Once the environment exists, the rest is configuration:

```text
config/env/my_game/default.yaml
config/model/my_model.yaml
config/experiment/my_game/fast.yaml
```

If the game is small, start with an MLP. If the observation is spatial and larger, try a convolutional or ResNet-style model. If the game depends heavily on memory, reach for recurrent models.

And start small. The first run should answer one question:

```text
Does the full training loop execute?
```

Only after that should you scale environment counts, rollout length, model size, league play, and total iterations.

---

## What I Like About This Workflow

The best thing about Jaxpot is not that it hides RL complexity.

It does not.

You still need to understand the game, rewards, and compute.

The best thing is that it puts the complexity in the right places.

Game rules live in the environment.

Architecture lives in the model config.

Training behavior lives in the experiment config.

Metrics go to TensorBoard or W&B.

Checkpoints go under the run directory.

That separation is what makes the system usable for experimentation. A researcher can compare game variants. A software engineer can reproduce a run. A CTO can look at a project and see whether the team is building reusable infrastructure or a museum of one-off notebooks.

For me, that is the real test of an RL framework.

Not "can it solve the toy problem?"

But:

```text
Can I run the toy problem, understand the result, and then move to the real problem without starting over?
```

Jaxpot is built for that second question.