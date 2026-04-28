"""Evaluator module for running evaluation during training."""

from jaxpot.evaluator.base import BaseEvaluator
from jaxpot.evaluator.baseline_model import BaselineModelEvaluator
from jaxpot.evaluator.evaluate import (
    evaluate_vs_opponent,
    evaluate_vs_opponent_jited,
    evaluate_vs_random,
    evaluate_vs_random_jited,
)
from jaxpot.evaluator.league import ArchivedLeagueEvaluator
from jaxpot.evaluator.random import RandomEvaluator

__all__ = [
    "BaseEvaluator",
    "BaselineModelEvaluator",
    "ArchivedLeagueEvaluator",
    "RandomEvaluator",
    "evaluate_vs_opponent",
    "evaluate_vs_opponent_jited",
    "evaluate_vs_random",
    "evaluate_vs_random_jited",
]
