"""Phantom Tic-Tac-Toe web adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from jaxpot.env.phantom_ttt import PhantomTTT
from jaxpot.web.adapter import WebAdapter
from jaxpot.web.server import load_model_from_checkpoint, register_game

CELL_NAMES = [
    "top-left", "top-center", "top-right",
    "mid-left", "center", "mid-right",
    "bot-left", "bot-center", "bot-right",
]


@register_game("phantom_ttt")
class PhantomTTTWebAdapter(WebAdapter):

    def __init__(self, checkpoint: str | None = None, **model_kwargs):
        self.env = PhantomTTT(observation_cls="default")
        self.model = None

        if checkpoint:
            from jaxpot.models.architectures.resnet import ResNetModel

            params = {
                "num_filters": model_kwargs.get("num_filters", 64),
                "num_blocks": model_kwargs.get("num_blocks", 4),
            }
            self.model = load_model_from_checkpoint(
                checkpoint,
                ResNetModel,
                {"action_dim": 9, "obs_shape": (3, 3, 3), **params},
            )
            print("Loaded trained Phantom TTT model")
        else:
            print("Using random baseline for Phantom TTT (no trained model)")

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "PHANTOM TIC-TAC-TOE", "num_players": 2}

    def get_frontend_path(self) -> Path:
        return Path(__file__).parent / "frontend.html"

    def action_to_display(self, action: int, state) -> str | None:
        return CELL_NAMES[action]

    def action_from_input(self, user_input: dict[str, Any], state) -> int:
        return int(user_input["action_id"])

    def state_to_json(self, state, human_player: int) -> dict[str, Any]:
        x = state._x
        board = np.array(x.board).tolist()  # true board: 0=empty, 1=X, 2=O
        x_view = np.array(x.x_view).tolist()
        o_view = np.array(x.o_view).tolist()

        # Determine which color the human is
        player_order = np.array(state._player_order)
        human_color = int(np.where(player_order == human_player)[0][0])
        human_view = x_view if human_color == 0 else o_view

        mask = np.array(state.legal_action_mask)
        terminated = bool(state.terminated) or bool(state.truncated)

        legal_actions = []
        if not terminated:
            for i in range(9):
                if mask[i]:
                    legal_actions.append({
                        "id": i,
                        "row": i // 3,
                        "col": i % 3,
                        "name": CELL_NAMES[i],
                    })

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
            "board": board,
            "human_view": human_view,
            "human_color": human_color,
            "human_player": human_player,
            "legal_actions": legal_actions,
            "terminated": terminated,
            "result": result,
            "current_player": int(state.current_player),
            "move_succeeded": bool(x.move_succeeded),
        }
