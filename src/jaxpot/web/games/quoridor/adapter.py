"""Quoridor web adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from jaxpot.env.quoridor import Quoridor, action_to_text, canonical_to_absolute
from jaxpot.env.quoridor.game import BOARD_SIZE, WALL_SIZE
from jaxpot.env.quoridor.notation import _move_target
from jaxpot.web.adapter import WebAdapter
from jaxpot.web.server import load_model_from_checkpoint, register_game


@register_game("quoridor")
class QuoridorWebAdapter(WebAdapter):

    def __init__(self, checkpoint: str | None = None, **model_kwargs):
        self.env = Quoridor(observation_cls="spatial_scalar")

        if checkpoint:
            from jaxpot.models.games.quoridor.resnet_lstm import QuoridorResNetLSTMModel

            params = {
                "num_filters": model_kwargs.get("num_filters", 128),
                "num_blocks": model_kwargs.get("num_blocks", 6),
                "lstm_hidden_size": model_kwargs.get("lstm_hidden_size", 256),
                "num_lstm_layers": model_kwargs.get("num_lstm_layers", 1),
            }
            self.model = load_model_from_checkpoint(
                checkpoint,
                QuoridorResNetLSTMModel,
                {"action_dim": 140, "obs_shape": (326,), **params},
            )
            print("Loaded trained Quoridor model (recurrent)")
        else:
            from jaxpot.models.games.quoridor.bfs_baseline import QuoridorBFSBaseline

            self.model = QuoridorBFSBaseline()
            print("Using BFS heuristic baseline")

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "QUORIDOR", "num_players": 2}

    def get_frontend_path(self) -> Path:
        return Path(__file__).parent / "frontend.html"

    def action_to_display(self, action: int, state) -> str | None:
        color = int(state._x.color)
        abs_action = canonical_to_absolute(action, color)
        pawn_pos = np.array(state._x.pawn_pos)
        my_pos = int(pawn_pos[color])
        opp_pos = int(pawn_pos[1 - color])
        my_r, my_c = my_pos // BOARD_SIZE, my_pos % BOARD_SIZE
        opp_r, opp_c = opp_pos // BOARD_SIZE, opp_pos % BOARD_SIZE
        return action_to_text(abs_action, my_r, my_c, opp_r, opp_c)

    def action_from_input(self, user_input: dict[str, Any], state) -> int:
        return int(user_input["action_id"])

    def state_to_json(self, state, human_player: int) -> dict[str, Any]:
        pawn_pos = np.array(state._x.pawn_pos)
        h_walls = np.array(state._x.h_walls)
        v_walls = np.array(state._x.v_walls)
        walls_remaining = np.array(state._x.walls_remaining)
        mask = np.array(state.legal_action_mask)

        pawns = []
        for i in range(2):
            pos = int(pawn_pos[i])
            pawns.append({"row": pos // BOARD_SIZE, "col": pos % BOARD_SIZE})

        h_wall_list = []
        v_wall_list = []
        for r in range(WALL_SIZE):
            for c in range(WALL_SIZE):
                if h_walls[r, c]:
                    h_wall_list.append({"row": r, "col": c})
                if v_walls[r, c]:
                    v_wall_list.append({"row": r, "col": c})

        legal_actions = []
        terminated = bool(state.terminated) or bool(state.truncated)
        legal_indices = np.where(mask)[0] if not terminated else []

        for a in legal_indices:
            a = int(a)
            notation = self.action_to_display(a, state)
            if a < 4:
                atype = "move"
            elif a < 8:
                atype = "jump"
            elif a < 12:
                atype = "diagonal"
            elif a < 76:
                atype = "h_wall"
            else:
                atype = "v_wall"

            action_info = {"id": a, "notation": notation, "type": atype}

            if a < 12:
                color = int(state._x.color)
                abs_action = canonical_to_absolute(a, color)
                my_pos = int(pawn_pos[color])
                opp_pos = int(pawn_pos[1 - color])
                my_r, my_c = my_pos // BOARD_SIZE, my_pos % BOARD_SIZE
                opp_r, opp_c = opp_pos // BOARD_SIZE, opp_pos % BOARD_SIZE
                tr, tc = _move_target(abs_action, my_r, my_c, opp_r, opp_c)
                action_info["target_row"] = tr
                action_info["target_col"] = tc
            else:
                color = int(state._x.color)
                abs_action = canonical_to_absolute(a, color)
                if abs_action < 76:
                    idx = abs_action - 12
                else:
                    idx = abs_action - 76
                action_info["wall_row"] = idx // WALL_SIZE
                action_info["wall_col"] = idx % WALL_SIZE

            legal_actions.append(action_info)

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
            "pawns": pawns,
            "h_walls": h_wall_list,
            "v_walls": v_wall_list,
            "walls_remaining": [int(walls_remaining[0]), int(walls_remaining[1])],
            "human_player": human_player,
            "legal_actions": legal_actions,
            "terminated": terminated,
            "result": result,
            "current_player": int(state.current_player),
        }
