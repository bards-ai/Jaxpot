"""
Agents module for poker rollouts.

Provides agent abstractions for action selection during rollouts.

Agent Types:
- PolicyActor, RandomActor, MCTSActor: JAX PyTree nodes for use in JIT-compiled rollouts
- PPOAgent, AlphaZeroAgent: Stateful training wrappers (not PyTrees) implementing
  BaseTrainingAgent. Call .rollout_actor() to get the PyTree actor for JIT rollouts.
"""

from jaxpot.agents.alphazero_agent import AlphaZeroAgent
from jaxpot.agents.base_rollout_actor import AgentOutput, BaseRolloutActor
from jaxpot.agents.base_training_agent import BaseTrainingAgent
from jaxpot.agents.mcts_actor import MCTSActor
from jaxpot.agents.policy_actor import PolicyActor
from jaxpot.agents.ppo_agent import PPOAgent, PPOUpdateMetrics
from jaxpot.agents.random_actor import RandomActor

__all__ = [
    "AgentOutput",
    "AlphaZeroAgent",
    "BaseRolloutActor",
    "BaseTrainingAgent",
    "MCTSActor",
    "PolicyActor",
    "PPOAgent",
    "PPOUpdateMetrics",
    "RandomActor",
]
