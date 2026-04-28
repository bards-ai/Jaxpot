import importlib.util
import math
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest


def _load_utils_module():
    jaxpot_pkg = types.ModuleType("jaxpot")
    jaxpot_pkg.__path__ = []
    evaluator_pkg = types.ModuleType("jaxpot.evaluator")
    evaluator_pkg.__path__ = []
    loggers_pkg = types.ModuleType("jaxpot.loggers")
    loggers_pkg.__path__ = []

    @dataclass
    class EvaluationOutput:
        metrics: dict = field(default_factory=dict)
        histograms: list = field(default_factory=list)

    class Logger:
        pass

    evaluator_base = types.ModuleType("jaxpot.evaluator.base")
    evaluator_base.EvaluationOutput = EvaluationOutput
    logger_module = types.ModuleType("jaxpot.loggers.logger")
    logger_module.Logger = Logger

    sys.modules["jaxpot"] = jaxpot_pkg
    sys.modules["jaxpot.evaluator"] = evaluator_pkg
    sys.modules["jaxpot.evaluator.base"] = evaluator_base
    sys.modules["jaxpot.loggers"] = loggers_pkg
    sys.modules["jaxpot.loggers.logger"] = logger_module

    utils_path = Path(__file__).resolve().parents[1] / "src" / "jaxpot" / "evaluator" / "utils.py"
    spec = importlib.util.spec_from_file_location("test_jaxpot_evaluator_utils", utils_path)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {utils_path}")
    spec.loader.exec_module(module)
    return module


UTILS = _load_utils_module()


def test_calculate_elo_treats_draw_as_half_win():
    assert UTILS.calculate_elo(win_rate=0.0, draw_rate=1.0) == pytest.approx(1000.0)
    assert UTILS.calculate_elo(win_rate=0.25, draw_rate=0.5) == pytest.approx(1000.0)


def test_bayesian_elo_depends_only_on_aggregate_record():
    split_by_seat = UTILS.bayesian_elo(3, 1, 2, 5, 4, 1)
    combined = UTILS.bayesian_elo(8, 5, 3, 0, 0, 0)

    assert split_by_seat[0] == pytest.approx(combined[0])
    assert split_by_seat[1] == pytest.approx(combined[1])


def test_bayesian_elo_std_is_finite_and_shrinks_with_more_games():
    elo_small, std_small = UTILS.bayesian_elo(1, 0, 0, 1, 0, 0)
    elo_large, std_large = UTILS.bayesian_elo(10, 0, 0, 10, 0, 0)

    assert elo_large > elo_small
    assert math.isfinite(std_small)
    assert math.isfinite(std_large)
    assert 0.0 < std_large < std_small
