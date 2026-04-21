from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import jax

# TypeVar for observation output - can be Self (PyTreeNode) or jax.Array
ObsT = TypeVar("ObsT", bound="jax.Array | Observation")


class Observation(ABC, Generic[ObsT]):
    """Abstract base class for observations.

    Observations can be either:
    - A PyTreeNode subclass (struct.PyTreeNode) where from_state returns Self
    - A simple wrapper where from_state returns jax.Array

    Subclasses should specify the type parameter:
    - `class MyObs(Observation["MyObs"], struct.PyTreeNode): ...` for PyTreeNode
    - `class ArrayObs(Observation[jax.Array]): ...` for Array output
    """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Shape of the flattened observation."""
        ...

    @classmethod
    @abstractmethod
    def from_state(cls, state, **kwargs) -> ObsT:
        """Create observation from environment state.

        Parameters
        ----------
        state
            The current environment/game state.
        **kwargs
            Additional game-specific arguments (e.g. color, step_count).

        Returns
        -------
        ObsT
            Either Self (for PyTreeNode observations) or jax.Array.
        """
        ...


class ArrayObservation(Observation[jax.Array]):
    """Base class for observations that return a flat jax.Array.

    Subclasses must implement `shape` and `from_state` to return a jax.Array.

    Example
    -------
    ```python
    class SimpleObs(ArrayObservation):
        @property
        def shape(self) -> tuple[int, ...]:
            return (10,)

        @classmethod
        def from_state(cls, state, **kwargs) -> jax.Array:
            return jnp.concatenate([state.stacks, state.pot_commits])
    ```
    """

    pass
