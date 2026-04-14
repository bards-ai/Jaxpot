"""Base web adapter interface for game environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pgx.core as core


class WebAdapter(ABC):
    """Abstract interface that each game must implement for web play.

    Bridges the gap between the PGX env internals and the JSON/HTML
    expected by the universal web server and frontend.
    """

    env: core.Env

    @abstractmethod
    def state_to_json(self, state: core.State, human_player: int) -> dict[str, Any]:
        """Serialize game state to a JSON-friendly dict for the frontend.

        Must include at minimum:
        - ``"legal_actions"``: list of action dicts with ``"id"`` key
        - ``"terminated"``: bool
        - ``"result"``: ``"win"`` | ``"lose"`` | ``"draw"`` | None
        - ``"current_player"``: int
        """
        ...

    @abstractmethod
    def action_from_input(self, user_input: dict[str, Any], state: core.State) -> int:
        """Convert frontend user input to an action index.

        Parameters
        ----------
        user_input : dict
            Arbitrary dict from the frontend (e.g. ``{"action_id": 3}``).
        state : core.State
            Current game state (may be needed for context).

        Returns
        -------
        int
            Action index for ``env.step``.
        """
        ...

    @abstractmethod
    def action_to_display(self, action: int, state: core.State) -> str | None:
        """Convert an action index to a human-readable string.

        Used to show the bot's last move. Return None if not applicable.
        """
        ...

    @abstractmethod
    def get_frontend_path(self) -> Path:
        """Return the path to the game-specific frontend HTML file."""
        ...

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return game metadata for the frontend shell.

        Should include at minimum:
        - ``"name"``: str — display name (e.g. "Quoridor")
        - ``"num_players"``: int
        """
        ...
