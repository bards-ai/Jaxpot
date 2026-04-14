"""Liar's Dice web adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from jaxpot.env.liars_dice import LiarsDice
from jaxpot.env.liars_dice.game import DICE_SIDES
from jaxpot.web.adapter import WebAdapter
from jaxpot.web.server import load_model_from_checkpoint, register_game


def _bid_text(action_id: int, num_dice: int) -> str:
    """Convert action ID to human-readable bid text."""
    liar_id = 2 * num_dice * DICE_SIDES
    if action_id == liar_id:
        return "Liar!"
    quantity = action_id // DICE_SIDES + 1
    face = action_id % DICE_SIDES + 1
    face_str = str(face) if face < 6 else f"{face} (wild)"
    return f"{quantity}x {face_str}"


@register_game("liars_dice")
class LiarsDiceWebAdapter(WebAdapter):

    def __init__(self, checkpoint: str | None = None, num_dice: int = 5, **model_kwargs):
        self.num_dice = num_dice
        self.env = LiarsDice(num_dice=num_dice, observation_cls="default")
        self.model = None

        if checkpoint:
            from jaxpot.models.architectures.mlp import MLPModel

            total_dice = 2 * num_dice
            num_actions = total_dice * DICE_SIDES + 1
            obs_size = num_dice * DICE_SIDES + total_dice * DICE_SIDES
            self.model = load_model_from_checkpoint(
                checkpoint,
                MLPModel,
                {"action_dim": num_actions, "obs_shape": (obs_size,), **model_kwargs},
            )
            print("Loaded trained Liar's Dice model")
        else:
            print("Using random baseline for Liar's Dice (no trained model)")

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "LIAR'S DICE",
            "num_players": 2,
            "num_dice": self.num_dice,
        }

    def get_frontend_path(self) -> Path:
        return Path(__file__).parent / "frontend.html"

    def action_to_display(self, action: int, state) -> str | None:
        return _bid_text(action, self.num_dice)

    def action_from_input(self, user_input: dict[str, Any], state) -> int:
        return int(user_input["action_id"])

    def state_to_json(self, state, human_player: int) -> dict[str, Any]:
        x = state._x
        dice = np.array(x.dice)  # (2, num_dice)
        current_bid = int(x.current_bid)
        bidder = int(x.bidder)

        # Determine human's color from player order
        player_order = np.array(state._player_order)
        human_color = int(np.where(player_order == human_player)[0][0])

        # Human can only see their own dice
        human_dice = dice[human_color].tolist()

        mask = np.array(state.legal_action_mask)
        terminated = bool(state.terminated) or bool(state.truncated)

        liar_id = 2 * self.num_dice * DICE_SIDES
        legal_actions = []
        if not terminated:
            for i in range(len(mask)):
                if mask[i]:
                    legal_actions.append({
                        "id": i,
                        "text": _bid_text(i, self.num_dice),
                        "is_liar": i == liar_id,
                    })

        # Current bid info
        current_bid_info = None
        if current_bid >= 0:
            quantity = current_bid // DICE_SIDES + 1
            face = current_bid % DICE_SIDES + 1
            # Who made the bid in terms of human/bot
            bid_by_human = (player_order[bidder] == human_player) if bidder >= 0 else False
            current_bid_info = {
                "quantity": quantity,
                "face": face,
                "text": _bid_text(current_bid, self.num_dice),
                "by_human": bool(bid_by_human),
            }

        rewards = np.array(state.rewards)
        result = None
        if terminated:
            # On terminal, reveal all dice
            all_dice = [dice[0].tolist(), dice[1].tolist()]
            if rewards[human_player] > 0:
                result = "win"
            elif rewards[human_player] < 0:
                result = "lose"
            else:
                result = "draw"
        else:
            all_dice = None

        return {
            "human_dice": human_dice,
            "all_dice": all_dice,
            "num_dice": self.num_dice,
            "human_color": human_color,
            "human_player": human_player,
            "current_bid": current_bid_info,
            "legal_actions": legal_actions,
            "terminated": terminated,
            "result": result,
            "current_player": int(state.current_player),
        }
