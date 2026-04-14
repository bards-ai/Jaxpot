# Modules

## The Main Loop

The core workflow of Jaxpot is based on an Reinforcement Learning (RL) loop.

1. **Self-Play Data Collection**: Agents interact with the environment to generate samples (rollouts).
2. **Target Gathering**: Raw rollout data is processed to targets.
3. **Neural Network Training**: The collected data is used to update the policy and value networks.
4. **Evaluation**: The updated agent is evaluated against baselines or previous versions.
5. **Checkpointing & Logging**: Models and optimizer states are saved periodically, and metrics are sent to trackers like WandB.

## Configuration

Configuration is managed via [Hydra](https://hydra.cc/). The configuration files (e.g., `config/experiment/go_9x9/go_9_alphazero_config.yaml`) define the runs, specifying the environment, model architecture, trainer and evaluator settings.

## Environments (`src/jaxpot/env/`)

The project depends on `[pgx.core.Env](https://github.com/sotetsuk/pgx)` to provide JAX-native environments. You can use their environments or implement your own.

## Evaluators (`src/jaxpot/evaluator/`)

Evaluators can be configured via the config files and should implement the `BaseEvaluator` interface. The following evaluators are implemented:

- `BaselineModelEvaluator`: Evaluates agents against pre-trained PGX baselines (e.g., Go, Connect4).
- `RandomEvaluator`: Evaluates the agent against a random opponent.
- `ArchivedLeagueEvaluator`: Evaluates against archived league opponents.
- `NashExploitabilityEvaluator`: Computes Nash exploitability for small games.

## Loggers (`src/jaxpot/loggers/`)

The loggers can be easily configurated via config.

- Implementations include `WandbLogger` and `TensorBoardLogger`.
- `MultiLogger` allows broadcasting metrics to multiple backends simultaneously.

# Core Modules & Low-Level Mechanisms

## Rollouts & Agents (`src/jaxpot/rollout/`, `src/jaxpot/agents/`)

- **Compiled Unrolling**: Data collection is fully compiled and executed on-device.
- **Rollout Controllers**: The `RolloutController` abstraction (e.g., `SelfPlayController`, `AgentOpponentRolloutController`) decouples the environment transition logic from agent action selection.
- **Agents**:
  - `BaseRolloutActor`: The base interface for rollout actors.
  - `MCTSActor`: Uses `mctx.gumbel_muzero_policy` for AlphaZero-style tree search.
  - `PPOAgent`: Standard actor-critic agent for PPO.
- **Buffers**: Data collection and training use distinct buffer abstractions. The `RolloutBuffer` preserves the temporal and per-environment structure needed for advantage and target calculations, while the `TrainingDataBuffer` flattens and shuffles this data for efficient neural network batch updates.

## Advantage Gatherers (`src/jaxpot/rollout/advantage_gatherers.py`)

Once a rollout is complete, the raw rewards and values need to be converted into training targets. The `AdvantageGatherer` protocol handles this.

- `**GAEAdvantageGatherer`**: Implements Generalized Advantage Estimation (GAE). It is primarily used for PPO to compute advantages and value targets using a combination of rewards and value estimates.
- `**PerSeatReturnGatherer**`: Computes per-seat terminal returns via a backward scan. This is equivalent to PGX AlphaZero value targets and is used for AlphaZero training in zero-sum alternating-move games.

## Models (`src/jaxpot/models/`)

Neural networks are built using Flax NNX.

- `**PolicyValueModel**`: The base class for all policy/value networks, returning a `ModelOutput` containing `policy_logits` and `value`.
- **Implementations**: Includes a variety of architectures tailored to different games, such as `AZNetModel` (AlphaZero-style ResNet), `ResNetModel`, `MLPModel`, `LSTMModel`, and `TransformerLSTMModel`.

## Trainers & On-Device Training (`src/jaxpot/rl/`)

- **The `Trainer` Base Class**: abstract class, it handles multi-GPU data sharding and metric accumulation.
- **Implementations**:
  - `PPOTrainer`: Implements Proximal Policy Optimization with a clipped surrogate objective and value loss.
  - `AlphaZeroTrainer`: Implements AlphaZero training (value MSE + policy cross-entropy against MCTS targets).

## Losses (`src/jaxpot/rl/losses/`)

- **Core Losses**: Standard RL losses including `value_loss_fn`, `policy_loss_fn`, and `alphazero_loss_fn`.
- **Auxiliary Losses**: The `AuxiliaryLoss` abstract class allows injecting additional training signals to speed up representation learning. Implementations include:
  - `OpponentCardsCrossEntropyLoss`: Predicting opponent hole cards.
  - `EquityMSELoss`: Predicting hand equity.
  - `GameProgressMSELoss`: Predicting how far along the game is.