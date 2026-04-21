"""
AlphaZero-style training module backed by mctx.
"""

from jaxpot.alphazero.mcts import MCTSConfig, make_root_and_recurrent_fn

__all__ = [
    "MCTSConfig",
    "make_root_and_recurrent_fn",
]
