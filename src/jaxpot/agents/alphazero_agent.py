"""
AlphaZeroAgent: stateful training wrapper for AlphaZero-style training.

Implements BaseTrainingAgent so train_selfplay.py can instantiate it via
Hydra config without any algorithm-specific branching.

rollout_actor() returns an MCTSActor (the search-backed PyTree actor).
"""

from jaxpot.agents.base_training_agent import BaseTrainingAgent
from jaxpot.agents.mcts_actor import MCTSActor
from jaxpot.alphazero.mcts import MCTSConfig
from jaxpot.models.base import PolicyValueModel
from jaxpot.rl.alphazero_trainer import AlphaZeroTrainer


class AlphaZeroAgent(BaseTrainingAgent):
    """
    AlphaZero Agent that wraps a model and AlphaZeroTrainer.

    This is a stateful training wrapper — NOT a JAX PyTree. For use in
    JIT-compiled rollout functions, call `rollout_actor()` to get a
    PyTree-compatible MCTSActor.

    Parameters
    ----------
    model : BaseModel
        Policy/value network.
    trainer : AlphaZeroTrainer
        Trainer instance for performing updates.
    mcts_config : MCTSConfig
        MCTS search hyperparameters used during data collection.
    """

    def __init__(
        self,
        model: PolicyValueModel,
        trainer: AlphaZeroTrainer,
        mcts_config: MCTSConfig,
    ):
        super().__init__(model, trainer, MCTSActor(model=model, mcts_config=mcts_config))
