"""Dark Hex web adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from jaxpot.env.dark_hex import DarkHex
from jaxpot.web.adapter import WebAdapter
from jaxpot.web.server import load_model_from_checkpoint, register_game


@register_game("dark_hex")
class DarkHexWebAdapter(WebAdapter):

    def __init__(
        self,
        checkpoint: str | None = None,
        num_rows: int = 5,
        num_cols: int = 5,
        **model_kwargs,
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.env = DarkHex(num_rows=num_rows, num_cols=num_cols, observation_cls="default")
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
                {
                    "action_dim": num_rows * num_cols,
                    "obs_shape": (num_rows, num_cols, 3),
                    **params,
                },
            )
            print("Loaded trained Dark Hex model")
        else:
            print("Using random baseline for Dark Hex (no trained model)")

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "DARK HEX",
            "num_players": 2,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
        }

    def get_frontend_path(self) -> Path:
        return Path(__file__).parent / "frontend.html"

    def action_to_display(self, action: int, state) -> str | None:
        row = action // self.num_cols
        col = action % self.num_cols
        return f"{chr(ord('a') + col)}{row + 1}"

    def action_from_input(self, user_input: dict[str, Any], state) -> int:
        return int(user_input["action_id"])

    def state_to_json(self, state, human_player: int) -> dict[str, Any]:
        x = state._x
        board = np.array(x.board).tolist()  # true: 0=empty, 1=Black, 2=White
        black_view = np.array(x.black_view).tolist()
        white_view = np.array(x.white_view).tolist()

        player_order = np.array(state._player_order)
        human_color = int(np.where(player_order == human_player)[0][0])
        human_view = black_view if human_color == 0 else white_view

        mask = np.array(state.legal_action_mask)
        terminated = bool(state.terminated) or bool(state.truncated)

        legal_actions = []
        if not terminated:
            for i in range(len(mask)):
                if mask[i]:
                    row = i // self.num_cols
                    col = i % self.num_cols
                    legal_actions.append({
                        "id": i,
                        "row": row,
                        "col": col,
                        "name": f"{chr(ord('a') + col)}{row + 1}",
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
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "human_color": human_color,
            "human_player": human_player,
            "legal_actions": legal_actions,
            "terminated": terminated,
            "result": result,
            "current_player": int(state.current_player),
            "move_succeeded": bool(x.move_succeeded),
        }
