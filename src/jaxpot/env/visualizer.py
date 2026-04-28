# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import jax
import svgwrite
from pgx.core import State

ColorTheme = Literal["light", "dark"]


@dataclass
class Config:
    color_theme: ColorTheme = "light"
    scale: float = 1.0
    frame_duration_seconds: float = 0.2


global_config = Config()


def set_visualization_config(
    *,
    color_theme: ColorTheme = "light",
    scale: float = 1.0,
    frame_duration_seconds: float = 0.2,
):
    global_config.color_theme = color_theme
    global_config.scale = scale
    global_config.frame_duration_seconds = frame_duration_seconds


@dataclass
class ColorSet:
    p1_color: str = "black"
    p2_color: str = "white"
    p1_outline: str = "black"
    p2_outline: str = "black"
    background_color: str = "white"
    grid_color: str = "black"
    text_color: str = "black"
    p1_wall_color: str = "#333333"
    p1_wall_outline: str = "black"
    p2_wall_color: str = "#cccccc"
    p2_wall_outline: str = "#666666"


class Visualizer:
    """The Pgx Visualizer

    color_theme: Default(None) is "light"
    scale: change image size. Default(None) is 1.0
    """

    def __init__(
        self,
        *,
        color_theme: Optional[ColorTheme] = None,
        scale: Optional[float] = None,
    ) -> None:
        color_theme = color_theme if color_theme is not None else global_config.color_theme
        scale = scale if scale is not None else global_config.scale

        self.config = {
            "GRID_SIZE": -1,
            "BOARD_WIDTH": -1,
            "BOARD_HEIGHT": -1,
            "COLOR_THEME": color_theme,
            "COLOR_SET": ColorSet(),
            "SCALE": scale,
        }
        self._make_dwg_group = None

    """
    notebook で可視化する際に、変数名のみで表示させる場合
    def _repr_html_(self) -> str:
        assert self.state is not None
        return self._to_dwg_from_states(states=self.state).tostring()
    """

    def get_dwg(
        self,
        states,
    ):
        try:
            SIZE = len(states.current_player)
            WIDTH = math.ceil(math.sqrt(SIZE - 0.1))
            if SIZE - (WIDTH - 1) ** 2 >= WIDTH:
                HEIGHT = WIDTH
            else:
                HEIGHT = WIDTH - 1
            if SIZE == 1:
                states = self._get_nth_state(states, 0)
        except TypeError:
            SIZE = 1
            WIDTH = 1
            HEIGHT = 1

        self._set_config_by_state(states)
        assert self._make_dwg_group is not None

        GRID_SIZE = self.config["GRID_SIZE"]
        BOARD_WIDTH = self.config["BOARD_WIDTH"]
        BOARD_HEIGHT = self.config["BOARD_HEIGHT"]
        SCALE = self.config["SCALE"]

        dwg = svgwrite.Drawing(
            "temp.svg",
            (
                (BOARD_WIDTH + 1) * GRID_SIZE * WIDTH * SCALE,
                (BOARD_HEIGHT + 1) * GRID_SIZE * HEIGHT * SCALE,
            ),
        )
        group = dwg.g()

        # background
        group.add(
            dwg.rect(
                (0, 0),
                (
                    (BOARD_WIDTH + 1) * GRID_SIZE * WIDTH,
                    (BOARD_HEIGHT + 1) * GRID_SIZE * HEIGHT,
                ),
                fill=self.config["COLOR_SET"].background_color,
            )
        )

        if SIZE == 1:
            g = self._make_dwg_group(dwg, states, self.config)
            g.translate(
                GRID_SIZE * 1 / 2,
                GRID_SIZE * 1 / 2,
            )
            group.add(g)
            group.scale(SCALE)
            dwg.add(group)
            return dwg

        for i in range(SIZE):
            x = i % WIDTH
            y = i // WIDTH
            _state = self._get_nth_state(states, i)
            g = self._make_dwg_group(
                dwg,
                _state,  # type:ignore
                self.config,
            )

            g.translate(
                GRID_SIZE * 1 / 2 + (BOARD_WIDTH + 1) * GRID_SIZE * x,
                GRID_SIZE * 1 / 2 + (BOARD_HEIGHT + 1) * GRID_SIZE * y,
            )
            group.add(g)
            group.add(
                dwg.rect(
                    (
                        (BOARD_WIDTH + 1) * GRID_SIZE * x,
                        (BOARD_HEIGHT + 1) * GRID_SIZE * y,
                    ),
                    (
                        (BOARD_WIDTH + 1) * GRID_SIZE,
                        (BOARD_HEIGHT + 1) * GRID_SIZE,
                    ),
                    fill="none",
                    stroke="gray",
                )
            )
        group.scale(SCALE)
        dwg.add(group)
        return dwg

    def _set_config_by_state(self, _state: State):  # noqa: C901
        if _state.env_id == "animal_shogi":
            from pgx._src.dwg.animalshogi import _make_animalshogi_dwg

            self.config["GRID_SIZE"] = 60
            self.config["BOARD_WIDTH"] = 4
            self.config["BOARD_HEIGHT"] = 4
            self._make_dwg_group = _make_animalshogi_dwg  # type:ignore
            if (
                self.config["COLOR_THEME"] is None and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "dimgray",
                    "black",
                    "whitesmoke",
                    "whitesmoke",
                    "#1e1e1e",
                    "white",
                    "",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "white",
                    "lightgray",
                    "black",
                    "black",
                    "white",
                    "black",
                    "",
                )
        elif _state.env_id == "backgammon":
            from pgx._src.dwg.backgammon import _make_backgammon_dwg

            self.config["GRID_SIZE"] = 25
            self.config["BOARD_WIDTH"] = 17
            self.config["BOARD_HEIGHT"] = 14
            self._make_dwg_group = _make_backgammon_dwg  # type:ignore
            if (
                self.config["COLOR_THEME"] is None and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "darkgray",
                    "white",
                    "white",
                    "#1e1e1e",
                    "silver",
                    "dimgray",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "white",
                    "black",
                    "lightgray",
                    "white",
                    "white",
                    "black",
                    "gray",
                )
        elif _state.env_id == "bridge_bidding":
            from pgx._src.dwg.bridge_bidding import _make_bridge_dwg

            self.config["GRID_SIZE"] = 50
            self.config["BOARD_WIDTH"] = 14
            self.config["BOARD_HEIGHT"] = 10
            self._make_dwg_group = _make_bridge_dwg  # type:ignore
            if (
                self.config["COLOR_THEME"] is None and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "gray",
                    "black",
                    "black",
                    "dimgray",
                    "#1e1e1e",
                    "gainsboro",
                    "white",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "white",
                    "black",
                    "lightgray",
                    "white",
                    "white",
                    "black",
                    "black",
                )
        elif _state.env_id == "chess":
            from pgx._src.dwg.chess import _make_chess_dwg

            self.config["GRID_SIZE"] = 50
            self.config["BOARD_WIDTH"] = 8
            self.config["BOARD_HEIGHT"] = 8
            self._make_dwg_group = _make_chess_dwg  # type:ignore
            if (
                self.config["COLOR_THEME"] is None and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "none",
                    "none",
                    "#404040",
                    "gray",
                    "#1e1e1e",
                    "silver",
                    "",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "none",
                    "none",
                    "gray",
                    "white",
                    "white",
                    "black",
                    "",
                )
        elif _state.env_id == "gardner_chess":
            from pgx._src.dwg.gardner_chess import _make_gardner_chess_dwg

            self.config["GRID_SIZE"] = 50
            self.config["BOARD_WIDTH"] = 5
            self.config["BOARD_HEIGHT"] = 5
            self._make_dwg_group = _make_gardner_chess_dwg  # type:ignore
            if (
                self.config["COLOR_THEME"] is None and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "none",
                    "none",
                    "#404040",
                    "gray",
                    "#1e1e1e",
                    "silver",
                    "",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "none",
                    "none",
                    "gray",
                    "white",
                    "white",
                    "black",
                    "",
                )
        elif _state.env_id == "connect_four":
            from pgx._src.dwg.connect_four import _make_connect_four_dwg

            self.config["GRID_SIZE"] = 35
            self.config["BOARD_WIDTH"] = 7
            self.config["BOARD_HEIGHT"] = 7
            self._make_dwg_group = _make_connect_four_dwg  # type:ignore
            if (
                self.config["COLOR_THEME"] is None and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "darkgray",
                    "white",
                    "white",
                    "#1e1e1e",
                    "silver",
                    "gray",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "white",
                    "black",
                    "black",
                    "white",
                    "black",
                    "gray",
                )
        elif _state.env_id in ("go_9x9", "go_19x19"):
            from pgx._src.dwg.go import _make_go_dwg

            self.config["GRID_SIZE"] = 25
            self.config["BOARD_WIDTH"] = int(_state._size)
            self.config["BOARD_HEIGHT"] = int(_state._size)
            self._make_dwg_group = _make_go_dwg
            if (
                self.config["COLOR_THEME"] is None and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "darkgray",
                    "white",
                    "white",
                    "#1e1e1e",
                    "silver",
                    "",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "white",
                    "black",
                    "black",
                    "white",
                    "black",
                    "",
                )
        elif _state.env_id == "hex":
            import jax.numpy as jnp
            from pgx._src.dwg.hex import _make_hex_dwg, four_dig

            self.config["GRID_SIZE"] = 30
            size = int(math.sqrt(_state._x.board.shape[-1]))
            self.config["BOARD_WIDTH"] = four_dig(size * 1.5)  # type:ignore
            self.config["BOARD_HEIGHT"] = four_dig(size * jnp.sqrt(3) / 2)  # type:ignore
            self._make_dwg_group = _make_hex_dwg  # type:ignore
            if (
                self.config["COLOR_THEME"] is None and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "darkgray",
                    "black",
                    "white",
                    "white",
                    "#1e1e1e",
                    "gray",
                    "#333333",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "white",
                    "black",
                    "black",
                    "white",
                    "black",
                    "lightgray",
                )
        elif _state.env_id == "kuhn_poker":
            from pgx._src.dwg.kuhn_poker import _make_kuhnpoker_dwg

            self.config["GRID_SIZE"] = 30
            self.config["BOARD_WIDTH"] = 8
            self.config["BOARD_HEIGHT"] = 8
            self._make_dwg_group = _make_kuhnpoker_dwg  # type:ignore
            if (
                self.config["COLOR_THEME"] is None and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "lightgray",
                    "white",
                    "lightgray",
                    "#1e1e1e",
                    "lightgray",
                    "lightgray",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "white",
                    "black",
                    "black",
                    "white",
                    "black",
                    "",
                )
        elif _state.env_id == "leduc_holdem":
            from pgx._src.dwg.leduc_holdem import _make_leducHoldem_dwg

            self.config["GRID_SIZE"] = 30
            self.config["BOARD_WIDTH"] = 8
            self.config["BOARD_HEIGHT"] = 8
            self._make_dwg_group = _make_leducHoldem_dwg  # type:ignore
            if (
                self.config["COLOR_THEME"] is None and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "gray",
                    "lightgray",
                    "",
                    "",
                    "#1e1e1e",
                    "lightgray",
                    "lightgray",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "gray",
                    "black",
                    "",
                    "",
                    "white",
                    "black",
                    "",
                )
        elif _state.env_id == "mahjong":
            from pgx._src.dwg.mahjong import _make_mahjong_dwg

            self.config["GRID_SIZE"] = 10
            self.config["BOARD_WIDTH"] = 70
            self.config["BOARD_HEIGHT"] = 70
            self._make_dwg_group = _make_mahjong_dwg  # type:ignore
            if (
                self.config["COLOR_THEME"] is None and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "white",
                    "black",
                    "black",
                    "white",
                    "black",
                    "black",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "white",
                    "black",
                    "black",
                    "white",
                    "black",
                    "black",
                )
        elif _state.env_id == "othello":
            from pgx._src.dwg.othello import _make_othello_dwg

            self.config["GRID_SIZE"] = 30
            self.config["BOARD_WIDTH"] = 8
            self.config["BOARD_HEIGHT"] = 8
            self._make_dwg_group = _make_othello_dwg  # type:ignore
            if (
                self.config["COLOR_THEME"] is None and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "darkgray",
                    "white",
                    "white",
                    "#1e1e1e",
                    "silver",
                    "",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "white",
                    "black",
                    "black",
                    "white",
                    "black",
                    "",
                )
        elif _state.env_id == "2048":
            from pgx._src.dwg.play2048 import _make_2048_dwg

            self.config["GRID_SIZE"] = 50
            self.config["BOARD_WIDTH"] = 4
            self.config["BOARD_HEIGHT"] = 4
            self._make_dwg_group = _make_2048_dwg  # type:ignore
            if (
                self.config["COLOR_THEME"] is None and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "lightgray",
                    "",
                    "",
                    "",
                    "#1e1e1e",
                    "black",
                    "white",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "#f0f0f0",
                    "",
                    "",
                    "white",
                    "black",
                    "black",
                )
        elif _state.env_id == "shogi":
            from pgx._src.dwg.shogi import _make_shogi_dwg

            self.config["GRID_SIZE"] = 50
            self.config["BOARD_WIDTH"] = 10
            self.config["BOARD_HEIGHT"] = 9
            self._make_dwg_group = _make_shogi_dwg  # type:ignore
            if (
                self.config["COLOR_THEME"] is None and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "gray", "black", "gray", "gray", "#1e1e1e", "gray", ""
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "white",
                    "lightgray",
                    "black",
                    "black",
                    "white",
                    "black",
                    "",
                )
        elif _state.env_id == "sparrow_mahjong":
            from pgx._src.dwg.sparrow_mahjong import _make_sparrowmahjong_dwg

            self.config["GRID_SIZE"] = 50
            self.config["BOARD_WIDTH"] = 15
            self.config["BOARD_HEIGHT"] = 10
            self._make_dwg_group = _make_sparrowmahjong_dwg  # type:ignore
            if (
                self.config["COLOR_THEME"] is None and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "lightgray",
                    "dimgray",
                    "#404040",
                    "gray",
                    "#1e1e1e",
                    "darkgray",
                    "whitesmoke",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "white",
                    "white",
                    "gray",
                    "white",
                    "white",
                    "silver",
                    "black",
                )
        elif _state.env_id == "tic_tac_toe":
            from pgx._src.dwg.tictactoe import _make_tictactoe_dwg

            self.config["GRID_SIZE"] = 60
            self.config["BOARD_WIDTH"] = 3
            self.config["BOARD_HEIGHT"] = 3
            self._make_dwg_group = _make_tictactoe_dwg  # type:ignore
            if (
                self.config["COLOR_THEME"] is None and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "gray",
                    "black",
                    "black",
                    "dimgray",
                    "#1e1e1e",
                    "gainsboro",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "white", "black", "lightgray", "white", "white", "black"
                )
        elif _state.env_id == "liars_dice":
            from jaxpot.env.liars_dice.dwg import _make_liars_dice_dwg

            self.config["GRID_SIZE"] = 40
            # Layout: num_dice * die + gaps + margins, vertically ~5 grid units
            num_dice = _state._x.dice.shape[-1]
            # Width: num_dice * 1.25 + 1 margin, Height: ~6 grid units
            self.config["BOARD_WIDTH"] = int(num_dice * 1.25 + 1.5)
            self.config["BOARD_HEIGHT"] = 6
            self._make_dwg_group = _make_liars_dice_dwg  # type:ignore
            if self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    p1_color="white",
                    p2_color="gray",
                    p1_outline="white",
                    p2_outline="dimgray",
                    background_color="#1e1e1e",
                    grid_color="gainsboro",
                    text_color="gainsboro",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    p1_color="black",
                    p2_color="white",
                    p1_outline="black",
                    p2_outline="black",
                    background_color="white",
                    grid_color="black",
                    text_color="black",
                )
        elif _state.env_id == "phantom_ttt":
            from jaxpot.env.phantom_ttt.dwg import _make_phantom_ttt_dwg

            self.config["GRID_SIZE"] = 50
            # 3 boards of 3 cells + 2 gaps (0.8 * GRID_SIZE each) ≈ 10.6 cells wide
            # BOARD_WIDTH/HEIGHT used for canvas sizing: (W+1)*GRID_SIZE per tile
            self.config["BOARD_WIDTH"] = 11
            self.config["BOARD_HEIGHT"] = 4
            self._make_dwg_group = _make_phantom_ttt_dwg  # type:ignore
            if self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    p1_color="gray",
                    p2_color="black",
                    p1_outline="black",
                    p2_outline="dimgray",
                    background_color="#1e1e1e",
                    grid_color="gainsboro",
                    text_color="gainsboro",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    p1_color="white",
                    p2_color="black",
                    p1_outline="lightgray",
                    p2_outline="white",
                    background_color="white",
                    grid_color="black",
                    text_color="black",
                )
        elif _state.env_id == "quoridor":
            from jaxpot.env.quoridor.dwg import _make_quoridor_dwg

            self.config["GRID_SIZE"] = 36
            self.config["BOARD_WIDTH"] = 12  # ~board_px / GRID_SIZE + padding
            self.config["BOARD_HEIGHT"] = 12
            self._make_dwg_group = _make_quoridor_dwg  # type:ignore
            if self.config["COLOR_THEME"] is not None and self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    p1_color="white",
                    p2_color="gray",
                    p1_outline="white",
                    p2_outline="dimgray",
                    background_color="#1e1e1e",
                    grid_color="gainsboro",
                    text_color="gainsboro",
                    p1_wall_color="#e0e0e0",
                    p1_wall_outline="white",
                    p2_wall_color="#555555",
                    p2_wall_outline="#888888",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    p1_color="black",
                    p2_color="white",
                    p1_outline="black",
                    p2_outline="#333333",
                    background_color="white",
                    grid_color="black",
                    text_color="black",
                    p1_wall_color="black",
                    p1_wall_outline="black",
                    p2_wall_color="white",
                    p2_wall_outline="#333333",
                )
        elif _state.env_id == "dark_hex":
            import math

            from jaxpot.env.dark_hex.dwg import (
                _board_pixel_size,
                _make_dark_hex_dwg,
            )

            num_cells = len(_state._x.board)
            num_rows = int(round(num_cells**0.5))
            num_cols = num_cells // max(1, num_rows)
            grid_size = 50
            hex_size = grid_size * 0.6
            pad = hex_size * 0.3
            bw, bh = _board_pixel_size(num_rows, num_cols, hex_size)
            panel_w = bw + 2 * pad
            panel_h = bh + 2 * pad
            spacing = grid_size * 0.6
            # Total width = 3 panels + 2 gaps; expressed in GRID_SIZE units
            total_w = 3 * panel_w + 2 * spacing
            total_h = panel_h + grid_size  # extra for labels
            self.config["GRID_SIZE"] = grid_size
            self.config["NUM_ROWS"] = num_rows
            self.config["NUM_COLS"] = num_cols
            self.config["BOARD_WIDTH"] = int(math.ceil(total_w / grid_size))
            self.config["BOARD_HEIGHT"] = int(math.ceil(total_h / grid_size))
            self._make_dwg_group = _make_dark_hex_dwg  # type:ignore
            if self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    p1_color="gray",
                    p2_color="black",
                    p1_outline="black",
                    p2_outline="dimgray",
                    background_color="#1e1e1e",
                    grid_color="gainsboro",
                    text_color="gainsboro",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    p1_color="white",
                    p2_color="black",
                    p1_outline="lightgray",
                    p2_outline="white",
                    background_color="white",
                    grid_color="black",
                    text_color="black",
                )
        else:
            assert False

    def _get_nth_state(self, states: State, i):
        return jax.tree_util.tree_map(lambda x: x[i], states)


def save_svg(
    state: State,
    filename: Union[str, Path],
    *,
    color_theme: Optional[Literal["light", "dark"]] = None,
    scale: Optional[float] = None,
) -> None:
    if state.env_id.startswith("minatar"):
        state.save_svg(filename=filename)
    else:
        v = Visualizer(color_theme=color_theme, scale=scale)
        v.get_dwg(states=state).saveas(filename)


def save_svg_animation(
    states: Sequence[State],
    filename: Union[str, Path],
    *,
    color_theme: Optional[Literal["light", "dark"]] = None,
    scale: Optional[float] = None,
    frame_duration_seconds: Optional[float] = None,
    player_labels: Optional[dict[int, str]] = None,
) -> None:
    """
    Save an animated SVG of game states.

    Parameters
    ----------
    states : Sequence[State]
        Sequence of game states to animate.
    filename : Union[str, Path]
        Output file path.
    color_theme : Optional[Literal["light", "dark"]]
        Color theme for visualization.
    scale : Optional[float]
        Scale factor for the visualization.
    frame_duration_seconds : Optional[float]
        Duration of each frame in seconds.
    player_labels : Optional[dict[int, str]]
        Dictionary mapping player index (0 or 1) to label string.
        E.g., {0: "Agent", 1: "Random"}
    """
    assert not states[0].env_id.startswith("minatar"), "MinAtar does not support svg animation."
    v = Visualizer(color_theme=color_theme, scale=scale)

    if frame_duration_seconds is None:
        frame_duration_seconds = global_config.frame_duration_seconds

    frame_groups = []
    dwg = None
    for i, state in enumerate(states):
        dwg = v.get_dwg(states=state)
        assert len([e for e in dwg.elements if type(e) is svgwrite.container.Group]) == 1, (
            "Drawing must contain only one group"
        )
        group: svgwrite.container.Group = dwg.elements[-1]
        group["id"] = f"_fr{i:x}"  # hex frame number
        group["class"] = "frame"
        frame_groups.append(group)

    assert dwg is not None
    del dwg.elements[-1]

    # Add player labels if provided
    label_height = 0.0
    if player_labels:
        label_height = 25 * (scale if scale else 1.0)
        # Get original dimensions
        orig_width, orig_height = dwg["width"], dwg["height"]
        # Convert to float if they're strings with units
        if isinstance(orig_width, str):
            orig_width = float(orig_width.replace("px", ""))
        if isinstance(orig_height, str):
            orig_height = float(orig_height.replace("px", ""))

        # Resize SVG to accommodate labels at top
        dwg["height"] = orig_height + label_height
        dwg["width"] = orig_width

        # Add label text with background (match game's color scheme)
        if color_theme == "dark":
            text_color = "white"
            bg_color = "#1e1e1e"
        else:
            text_color = "black"
            bg_color = "white"
        font_size = 14 * (scale if scale else 1.0)

        # Sort labels by content so "1st: ..." always appears before "2nd: ..."
        label_parts = sorted(player_labels.values())
        label_text = "  |  ".join(label_parts)

        # Add static label group (not animated)
        label_group = dwg.g(id="_labels")
        # Add background rectangle
        label_group.add(
            dwg.rect(
                insert=(0, 0),
                size=(orig_width, label_height),
                fill=bg_color,
            )
        )
        # Add text on top of background
        label_group.add(
            dwg.text(
                label_text,
                insert=(10, font_size + 4),
                fill=text_color,
                font_size=f"{font_size}px",
                font_family="monospace",
            )
        )
        dwg.add(label_group)

        # Shift all frame groups down to make room for labels
        for group in frame_groups:
            group.translate(0, label_height / (scale if scale else 1.0))

    total_seconds = frame_duration_seconds * len(frame_groups)

    style = f".frame{{visibility:hidden; animation:{total_seconds}s linear _k infinite;}}"
    style += f"@keyframes _k{{0%,{100 / len(frame_groups)}%{{visibility:visible}}{100 / len(frame_groups) * 1.000001}%,100%{{visibility:hidden}}}}"

    for i, group in enumerate(frame_groups):
        dwg.add(group)
        style += f"#{group['id']}{{animation-delay:{i * frame_duration_seconds}s}}"
    dwg.defs.add(svgwrite.container.Style(content=style))
    dwg.saveas(filename)


def save_gif(
    states: Sequence[State],
    filename: Union[str, Path],
    *,
    color_theme: Optional[Literal["light", "dark"]] = None,
    scale: Optional[float] = None,
    frame_duration_seconds: Optional[float] = None,
    player_labels: Optional[dict[int, str]] = None,
) -> None:
    """Save an animated GIF of game states."""
    import io

    import cairosvg
    from PIL import Image

    v = Visualizer(color_theme=color_theme, scale=scale)

    if frame_duration_seconds is None:
        frame_duration_seconds = global_config.frame_duration_seconds

    # Build label text once
    label_text = None
    if player_labels:
        label_parts = sorted(player_labels.values())
        label_text = "  |  ".join(label_parts)

    frames: list[Image.Image] = []
    for state in states:
        dwg = v.get_dwg(states=state)

        # Add player labels if provided
        if label_text is not None:
            _scale = scale if scale else 1.0
            label_height = 25 * _scale
            orig_width, orig_height = dwg["width"], dwg["height"]
            if isinstance(orig_width, str):
                orig_width = float(orig_width.replace("px", ""))
            if isinstance(orig_height, str):
                orig_height = float(orig_height.replace("px", ""))

            dwg["height"] = orig_height + label_height
            dwg["width"] = orig_width

            if color_theme == "dark":
                text_color, bg_color = "white", "#1e1e1e"
            else:
                text_color, bg_color = "black", "white"
            font_size = 14 * _scale

            label_group = dwg.g(id="_labels")
            label_group.add(dwg.rect(insert=(0, 0), size=(orig_width, label_height), fill=bg_color))
            label_group.add(
                dwg.text(
                    label_text,
                    insert=(10, font_size + 4),
                    fill=text_color,
                    font_size=f"{font_size}px",
                    font_family="monospace",
                )
            )
            # Insert label before the frame group
            dwg.elements.insert(-1, label_group)

            # Shift frame group down
            frame_group = dwg.elements[-1]
            frame_group.translate(0, label_height / _scale)

        svg_bytes = dwg.tostring().encode("utf-8")
        png_bytes = cairosvg.svg2png(bytestring=svg_bytes)
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        frames.append(img)

    if not frames:
        return

    frame_ms = int(frame_duration_seconds * 1000)
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
    )
