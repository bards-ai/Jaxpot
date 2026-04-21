import math
from typing import Any

import numpy as np
from scipy.optimize import minimize

from jaxpot.evaluator.base import EvaluationOutput
from jaxpot.loggers.logger import Logger


def _elo_expected_score(delta: float, scale: float = 400.0) -> float:
    """Expected score for one side: 1 / (1 + 10^(-delta/scale))."""
    return 1.0 / (1.0 + 10.0 ** (-delta / scale))


def _clamp_probability(probability: float) -> float:
    return max(min(probability, 1.0 - 1e-10), 1e-10)


def _effective_score(win_rate: float, draw_rate: float) -> float:
    return float(win_rate) + 0.5 * float(draw_rate)


def bayesian_elo(
    wins_p0: float,
    losses_p0: float,
    draws_p0: float,
    wins_p1: float,
    losses_p1: float,
    draws_p1: float,
    baseline_elo: float = 1000.0,
    prior_scale: float = 400.0,
    scale: float = 400.0,
) -> tuple[float, float]:
    """
    Bayesian Elo rating for agent vs baseline from game counts per seat.

    Uses the model from https://www.remi-coulom.fr/Bayesian-Elo/:
    - f(Delta) = 1 / (1 + 10^(-Delta/400)); P(agent win) = f(elo_agent - elo_baseline).
    - elo_baseline is fixed at 0 (offset applied at the end), so P(agent win) = f(e).
    - This is the same for both seats; seat assignment affects who is P0/P1 in the env
      but does not change the Elo likelihood.
    - Draws are counted as 0.5 win + 0.5 loss for the likelihood.
    - A Gaussian prior on elo_agent (mean 0, std prior_scale) regularizes so that
      high rating differences require more evidence (10-0 is not the same as 1-0).

    Parameters
    ----------
    wins_p0 : float
        Agent wins when agent was player 0.
    losses_p0 : float
        Agent losses when agent was player 0.
    draws_p0 : float
        Draws when agent was player 0.
    wins_p1 : float
        Agent wins when agent was player 1.
    losses_p1 : float
        Agent losses when agent was player 1.
    draws_p1 : float
        Draws when agent was player 1.
    baseline_elo : float, default=1000.0
        Elo of the baseline. The returned Elo is reported relative to this value.
    prior_scale : float, default=400.0
        Standard deviation of the Gaussian prior on the agent Elo.
    scale : float, default=400.0
        Elo scale factor.

    Returns
    -------
    tuple[float, float]
        Agent Elo and approximate posterior standard deviation.
    """
    total_effective_wins = (
        float(wins_p0)
        + float(wins_p1)
        + 0.5 * (float(draws_p0) + float(draws_p1))
    )
    total_effective_losses = (
        float(losses_p0)
        + float(losses_p1)
        + 0.5 * (float(draws_p0) + float(draws_p1))
    )

    def neg_log_posterior(elo_agent: np.ndarray) -> float:
        e = float(elo_agent.flat[0])
        p_agent = _clamp_probability(_elo_expected_score(e, scale))
        log_lik = (
            total_effective_wins * math.log(p_agent)
            + total_effective_losses * math.log(1.0 - p_agent)
        )
        log_prior = -0.5 * (e / prior_scale) ** 2
        return -(log_lik + log_prior)

    res = minimize(
        neg_log_posterior,
        x0=np.array([0.0]),
        method="L-BFGS-B",
        bounds=[(-2000.0, 2000.0)],
    )
    agent_elo_rel = float(res.x.flat[0])
    agent_elo = baseline_elo + agent_elo_rel

    p_agent = _clamp_probability(_elo_expected_score(agent_elo_rel, scale))
    logistic_scale = math.log(10.0) / scale
    curvature = (1.0 / prior_scale**2) + (
        (total_effective_wins + total_effective_losses)
        * logistic_scale**2
        * p_agent
        * (1.0 - p_agent)
    )
    elo_std = math.sqrt(1.0 / max(curvature, 1e-12))

    return agent_elo, elo_std


def calculate_elo(
    win_rate: float,
    draw_rate: float = 0.0,
    baseline_elo: float = 1000.0,
) -> float:
    """
    Calculate Elo rating from win and draw rates relative to a baseline.

    Parameters
    ----------
    win_rate : float
        Fraction of games won.
    draw_rate : float, default=0.0
        Fraction of games drawn.
    baseline_elo : float, default=1000.0
        Elo rating of the opponent or baseline.

    Returns
    -------
    float
        Elo implied by the expected score ``win_rate + 0.5 * draw_rate``.
    """
    expected_score = _clamp_probability(_effective_score(win_rate, draw_rate))
    return baseline_elo + 400.0 * math.log10(expected_score / (1.0 - expected_score))


def log_evaluation_output(
    evaluation_output: EvaluationOutput,
    log_payload: dict[str, Any],
    experiment_tracker: Logger,
    step: int,
) -> None:
    log_payload.update(evaluation_output.metrics)
    for histogram in evaluation_output.histograms:
        experiment_tracker.log_histogram(
            histogram.name, histogram.counts_array, histogram.bins, step
        )
