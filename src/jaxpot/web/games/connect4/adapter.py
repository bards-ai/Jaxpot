"""Connect4 web adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pgx

from jaxpot.web.adapter import WebAdapter
from jaxpot.web.server import load_model_from_checkpoint, register_game


@register_game("connect4")
class Connect4WebAdapter(WebAdapter):

    def __init__(self, checkpoint: str | None = None, **model_kwargs):
        self.env = pgx.make("connect_four")

        if checkpoint:
            from jaxpot.models.architectures.resnet import ResNetModel

            params = {
                "num_filters": model_kwargs.get("num_filters", 128),
                "num_blocks": model_kwargs.get("num_blocks", 6),
            }
            self.model = load_model_from_checkpoint(
                checkpoint,
                ResNetModel,
                {"action_dim": 7, "obs_shape": (6, 7, 2), **params},
            )
            print("Loaded trained Connect4 model")
        else:
            from jaxpot.models.games.connect4.baseline import Connect4Baseline

            self.model = Connect4Baseline()
            print("Using Connect4 heuristic baseline")

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "CONNECT 4", "num_players": 2}

    def get_frontend_path(self) -> Path:
        return Path(__file__).parent / "frontend.html"

    def action_to_display(self, action: int, state) -> str | None:
        return f"column {action}"

    def action_from_input(self, user_input: dict[str, Any], state) -> int:
        return int(user_input["action_id"])

    def state_to_json(self, state, human_player: int) -> dict[str, Any]:
        # PGX connect_four observation: (6, 7, 2)
        # channel 0 = current player's pieces, channel 1 = opponent's pieces
        # We need to convert to absolute board (not relative to current player)
        obs = np.array(state.observation)
        current = int(state.current_player)

        # Build absolute board: 0=empty, 1=player0, 2=player1
        board = np.zeros((6, 7), dtype=int)
        if not (bool(state.terminated) or bool(state.truncated)):
            # obs is from current player's perspective
            board = np.where(obs[:, :, 0] > 0, current + 1, board)
            board = np.where(obs[:, :, 1] > 0, (1 - current) + 1, board)
        else:
            # After terminal, observation may be stale; reconstruct from last known
            board = np.where(obs[:, :, 0] > 0, current + 1, board)
            board = np.where(obs[:, :, 1] > 0, (1 - current) + 1, board)

        mask = np.array(state.legal_action_mask)
        terminated = bool(state.terminated) or bool(state.truncated)

        legal_actions = []
        if not terminated:
            for col in range(7):
                if mask[col]:
                    legal_actions.append({"id": col, "col": col})

        rewards = np.array(state.rewards)
        result = None
        if terminated:
            if rewards[human_player] > 0:
                result = "win"
            elif rewards[human_player] < 0:
                result = "lose"
            else:
                result = "draw"

        return {
            "board": board.tolist(),
            "human_player": human_player,
            "legal_actions": legal_actions,
            "terminated": terminated,
            "result": result,
            "current_player": current,
        }
