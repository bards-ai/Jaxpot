import jax.numpy as jnp
import numpy as np
import pytest

from jaxpot.rollout.collect_samples import _allocate_envs_per_opponent


def test_allocates_in_base_unit_chunks_and_matches_total():
    counts = _allocate_envs_per_opponent(
        jnp.array([0.6, 0.4], dtype=jnp.float32),
        num_envs=512 * 5,
    )

    assert counts.sum() == 512 * 5
    assert np.all(counts % 256 == 0)
    assert counts.tolist() == [512 * 3, 512 * 2]


def test_rejects_num_envs_not_divisible_by_base_unit():
    with pytest.raises(ValueError):
        _allocate_envs_per_opponent(
            jnp.array([0.5, 0.5], dtype=jnp.float32),
            num_envs=1000,
        )


def test_rejects_non_positive_weight_sum():
    with pytest.raises(ValueError):
        _allocate_envs_per_opponent(
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            num_envs=512,
        )


def test_heavier_weight_receives_more_units():
    counts = _allocate_envs_per_opponent(
        jnp.array([0.65, 0.25, 0.1], dtype=jnp.float32),
        num_envs=512 * 4,
    )

    assert counts.sum() == 512 * 4
    assert np.all(counts % 256 == 0)
    assert counts[0] > counts[1] >= counts[2]
