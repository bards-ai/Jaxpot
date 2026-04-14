#!/usr/bin/env python3
"""Evaluate a Quoridor checkpoint against an LLM (GPT, Claude, etc).

Uses a text interface: renders the board as ASCII, sends it to the LLM
with game rules, and parses the LLM's move response.

Saves per-game:
  - Animated GIF of the board
  - JSON log with every move, LLM prompts/responses, model policy details

Usage:
    python scripts/eval_vs_llm.py outputs/2026-03-04/Quoridor_resnet_lstm_10-18-56/checkpoints/005843 \
        --llm-model gpt-4o --num-games 10

    python scripts/eval_vs_llm.py outputs/2026-03-04/Quoridor_resnet_lstm_10-18-56/checkpoints/005843 \
        --print-prompt

Requires: OPENAI_API_KEY or ANTHROPIC_API_KEY env vars set (in .env or shell).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from dotenv import load_dotenv
from flax import nnx

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jaxpot.env.quoridor import Quoridor
from jaxpot.env.quoridor.game import BOARD_SIZE, NUM_ACTIONS, WALL_SIZE
from jaxpot.env.quoridor.notation import (
    action_to_text,
    canonical_to_absolute,
    text_to_action,
)
from jaxpot.models.architectures.resnet_lstm import ResNetLSTMModel

# ---------------------------------------------------------------------------
# Board rendering
# ---------------------------------------------------------------------------


def render_board_ascii(state) -> str:
    """Render the Quoridor board as ASCII text.

    B = Black (player 0), W = White (player 1).
    '==' = horizontal wall segment, '||' = vertical wall segment.
    """
    game_state = state._x
    p0_pos = int(game_state.pawn_pos[0][0])
    p1_pos = int(game_state.pawn_pos[0][1])
    p0_r, p0_c = p0_pos // BOARD_SIZE, p0_pos % BOARD_SIZE
    p1_r, p1_c = p1_pos // BOARD_SIZE, p1_pos % BOARD_SIZE
    h_walls = np.array(game_state.h_walls[0])
    v_walls = np.array(game_state.v_walls[0])
    walls_rem = np.array(game_state.walls_remaining[0])

    lines = []
    for r in range(BOARD_SIZE - 1, -1, -1):
        row_str = f"{r + 1:2d} "
        for c in range(BOARD_SIZE):
            if r == p0_r and c == p0_c:
                cell = "B"
            elif r == p1_r and c == p1_c:
                cell = "W"
            else:
                cell = "."
            row_str += cell
            if c < BOARD_SIZE - 1:
                has_vwall = False
                for wr in range(max(0, r - 1), min(WALL_SIZE, r + 1)):
                    if c < WALL_SIZE and v_walls[wr, c]:
                        if wr == r or wr == r - 1:
                            has_vwall = True
                            break
                row_str += "||" if has_vwall else "  "
        lines.append(row_str)

        if r > 0:
            wall_row = "   "
            for c in range(BOARD_SIZE):
                has_hwall = False
                wr = r - 1
                if wr < WALL_SIZE:
                    for wc in range(max(0, c - 1), min(WALL_SIZE, c + 1)):
                        if h_walls[wr, wc]:
                            if wc == c or wc == c - 1:
                                has_hwall = True
                                break
                wall_row += "==" if has_hwall else "  "
                if c < BOARD_SIZE - 1:
                    wall_row += " "
            lines.append(wall_row)

    lines.append("   a  b  c  d  e  f  g  h  i")
    lines.append(f"   Black(B) walls: {walls_rem[0]}  White(W) walls: {walls_rem[1]}")
    return "\n".join(lines)


def state_to_svg(state) -> str:
    """Render the board state as SVG string (unbatched, numpy-converted)."""
    single = jax.tree_util.tree_map(lambda x: np.asarray(x[0]), state)
    return single.to_svg()


def svg_to_pil(svg_str: str):
    """Convert SVG string to PIL Image."""
    import cairosvg
    from PIL import Image

    png_bytes = cairosvg.svg2png(bytestring=svg_str.encode("utf-8"))
    return Image.open(io.BytesIO(png_bytes))


def save_game_gif(frames: list[str], path: Path, duration_ms: int = 800):
    """Save a list of SVG frame strings as an animated GIF."""
    if not frames:
        return
    pil_frames = [svg_to_pil(svg) for svg in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def get_legal_moves_text(state, current_color: int) -> list[str]:
    """Get list of legal moves in algebraic notation."""
    game_state = state._x
    mask = np.array(state.legal_action_mask[0], dtype=bool)

    p0_pos = int(game_state.pawn_pos[0][0])
    p1_pos = int(game_state.pawn_pos[0][1])
    p0_r, p0_c = p0_pos // BOARD_SIZE, p0_pos % BOARD_SIZE
    p1_r, p1_c = p1_pos // BOARD_SIZE, p1_pos % BOARD_SIZE

    if current_color == 0:
        my_r, my_c = p0_r, p0_c
        opp_r, opp_c = p1_r, p1_c
    else:
        my_r, my_c = p1_r, p1_c
        opp_r, opp_c = p0_r, p0_c

    moves = []
    for action_idx in range(NUM_ACTIONS):
        if not mask[action_idx]:
            continue
        abs_action = canonical_to_absolute(action_idx, current_color)
        try:
            text = action_to_text(abs_action, my_r, my_c, opp_r, opp_c)
            moves.append(text)
        except (ValueError, IndexError):
            moves.append(f"action_{action_idx}")
    return moves


def get_placed_walls_text(state) -> tuple[list[str], list[str]]:
    """Return placed walls in algebraic notation.

    Returns
    -------
    tuple[list[str], list[str]]
        Horizontal walls, then vertical walls.
    """
    game_state = state._x
    h_walls = np.array(game_state.h_walls[0], dtype=bool)
    v_walls = np.array(game_state.v_walls[0], dtype=bool)
    wall_cols = "abcdefgh"

    horizontal_walls = [
        f"{wall_cols[col]}{row + 1}h"
        for row in range(WALL_SIZE)
        for col in range(WALL_SIZE)
        if h_walls[row, col]
    ]
    vertical_walls = [
        f"{wall_cols[col]}{row + 1}v"
        for row in range(WALL_SIZE)
        for col in range(WALL_SIZE)
        if v_walls[row, col]
    ]
    return horizontal_walls, vertical_walls


# ---------------------------------------------------------------------------
# LLM interface
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are playing the board game Quoridor on a 9x9 grid.

RULES:
- Two players: Black (B) and White (W).
- Black starts at row 1, column e (bottom center). Black's goal: reach row 9.
- White starts at row 9, column e (top center). White's goal: reach row 1.
- On your turn you can either:
  1. MOVE your pawn one step in any cardinal direction (north/south/east/west),
     or jump over the opponent if adjacent (straight jump or diagonal if blocked).
  2. PLACE A WALL to block movement. Each player has 10 walls.
     Walls span 2 cells. Walls cannot completely block either player's path to goal.

NOTATION:
- Pawn moves: target square, e.g. "e5" means move pawn to column e, row 5.
  Columns: a-i (left to right). Rows: 1-9 (bottom to top).
- Wall placement: "<col><row><h|v>", e.g. "e4h" = horizontal wall at e4,
  "d5v" = vertical wall at d5. Wall columns: a-h, wall rows: 1-8.
  A horizontal wall blocks north-south movement. A vertical wall blocks east-west movement.

BOARD DISPLAY:
- "B" = Black pawn, "W" = White pawn, "." = empty square
- "==" between rows = horizontal wall segment
- "||" between columns = vertical wall segment
- A separate "Placed walls" section lists all walls explicitly in notation.

Think about your strategy briefly, then give your move.
Format your final answer as: MOVE: <your move>
Example: MOVE: e5  or  MOVE: d3h"""


def build_game_prompt(
    state,
    current_color: int,
    move_history: list[str],
) -> str:
    """Build the user message for the LLM."""
    color_name = "Black (B)" if current_color == 0 else "White (W)"
    goal = "reach row 9" if current_color == 0 else "reach row 1"
    board = render_board_ascii(state)
    horizontal_walls, vertical_walls = get_placed_walls_text(state)

    parts = [f"You are playing as {color_name}. Your goal: {goal}.\n"]
    parts.append(f"Moves so far: {', '.join(move_history) if move_history else 'none'}")
    parts.append("")
    parts.append("Current board:")
    parts.append(board)
    parts.append("")
    parts.append("Placed walls:")
    parts.append(f"- Horizontal: {', '.join(horizontal_walls) if horizontal_walls else 'none'}")
    parts.append(f"- Vertical: {', '.join(vertical_walls) if vertical_walls else 'none'}")
    parts.append("")
    parts.append(
        "Think about your strategy, then give one legal move as MOVE: <move> "
        "(for example MOVE: e5 or MOVE: d3h)"
    )

    return "\n".join(parts)


def call_llm(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_retries: int = 3,
) -> dict:
    """Call an LLM and return response details.

    Returns dict with 'content', 'reasoning_content', 'usage'.
    For thinking/reasoning models (gpt-5.x, o1, o3), captures the
    chain-of-thought via reasoning_content.
    """
    import logging

    import litellm

    litellm.drop_params = True
    litellm.suppress_debug_info = True
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)

    # Detect reasoning models and use responses API prefix for OpenAI
    is_reasoning = any(tag in model_name for tag in ("gpt-5", "o1", "o3", "o4"))
    api_model = model_name
    if is_reasoning and not model_name.startswith("openai/responses/"):
        # Strip existing openai/ prefix if present, then add responses prefix
        base = model_name.removeprefix("openai/")
        api_model = f"openai/responses/{base}"

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": api_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": 16384 if is_reasoning else 512,
            }
            if is_reasoning:
                kwargs["reasoning_effort"] = {"effort": "medium", "summary": "detailed"}

            response = litellm.completion(**kwargs)
            msg = response.choices[0].message
            content = (msg.content or "").strip()
            reasoning = getattr(msg, "reasoning_content", None) or ""

            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                # Capture reasoning tokens if available
                reasoning_tokens = getattr(response.usage, "reasoning_tokens", None)
                if reasoning_tokens is not None:
                    usage["reasoning_tokens"] = reasoning_tokens

            return {
                "content": content,
                "reasoning_content": reasoning,
                "usage": usage,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  LLM call failed (attempt {attempt + 1}): {e}")
                time.sleep(2**attempt)
            else:
                raise


def parse_llm_move(response: str) -> tuple[str, str]:
    """Extract move and thinking from LLM response.

    Returns (move, thinking) where thinking is the text before the move.
    """
    text = response.strip()

    # Look for "MOVE: xxx" pattern
    match = re.search(r"MOVE:\s*([a-i][1-9][hv]?)", text, re.IGNORECASE)
    if match:
        move = match.group(1).lower()
        thinking = text[: match.start()].strip()
        return move, thinking

    # Fallback: find any move-like pattern
    match = re.search(r"\b([a-i][1-9][hv]?)\b", text.lower())
    if match:
        return match.group(1), text

    return text.lower(), text


# ---------------------------------------------------------------------------
# Model loading & inference
# ---------------------------------------------------------------------------


def load_model(ckpt_dir: str | Path, cfg: dict) -> ResNetLSTMModel:
    """Load ResNetLSTMModel from checkpoint."""
    ckpt_dir = Path(ckpt_dir).resolve()
    state_path = ckpt_dir / "state"

    abstract_model = nnx.eval_shape(
        lambda: ResNetLSTMModel(
            rngs=nnx.Rngs(0),
            action_dim=(NUM_ACTIONS,),
            obs_shape=(BOARD_SIZE, BOARD_SIZE, 4),
            num_filters=cfg["num_filters"],
            num_blocks=cfg["num_blocks"],
            lstm_hidden_size=cfg["lstm_hidden_size"],
            num_lstm_layers=cfg["num_lstm_layers"],
        )
    )
    graphdef, abstract_state = nnx.split(abstract_model)

    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    fallback_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    payload = checkpointer.restore(
        state_path,
        args=ocp.args.StandardRestore(fallback_sharding=fallback_sharding),
    )

    nnx.replace_by_pure_dict(abstract_state, payload["model"])
    model = nnx.merge(graphdef, abstract_state)
    return model


def model_select_action(
    model: ResNetLSTMModel,
    state,
    hidden: jnp.ndarray | None,
    key: jax.Array,
    deterministic: bool = True,
) -> tuple[int, jnp.ndarray | None, dict]:
    """Run model inference and return action + policy details for logging."""
    obs = state.observation[0]
    mask = state.legal_action_mask[0]

    output = model(obs, hidden_state=hidden) if hidden is not None else model(obs)
    new_hidden = output.hidden_state

    logits = output.policy_logits
    masked_logits = jnp.where(mask, logits, -1e9)
    probs = jax.nn.softmax(masked_logits)
    value = float(output.value.squeeze())

    if deterministic:
        action = int(jnp.argmax(masked_logits))
    else:
        action = int(jax.random.categorical(key, masked_logits))

    # Top-5 actions for logging
    top_k = min(5, int(mask.sum()))
    top_indices = np.argsort(np.array(probs))[::-1][:top_k]
    top_probs = {int(i): float(probs[i]) for i in top_indices}

    details = {
        "value": value,
        "action_prob": float(probs[action]),
        "top_actions": top_probs,
    }

    return action, new_hidden, details


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------


def play_one_game(
    model: ResNetLSTMModel,
    env: Quoridor,
    init_fn,
    step_fn,
    observe_fn,
    llm_model_name: str,
    model_seat: int,
    key: jax.Array,
    deterministic: bool = True,
    verbose: bool = True,
    max_steps: int = 200,
    temperature: float = 0.3,
) -> dict:
    """Play one game: trained model vs LLM.

    Returns dict with result info, move log, and SVG frames.
    """
    key, init_key = jax.random.split(key)
    state = init_fn(init_key[None])

    move_history = []
    move_log = []  # detailed per-move log
    svg_frames = []
    hidden = None
    llm_invalid_moves = 0
    step_count = 0

    # Capture initial board
    svg_frames.append(state_to_svg(state))

    while step_count < max_steps:
        if bool(state.terminated[0]) or bool(state.truncated[0]):
            break

        current_player = int(state.current_player[0])
        current_color = int(state._x.color[0])
        color_name = ["Black", "White"][current_color]

        game_state = state._x
        p0_pos = int(game_state.pawn_pos[0][0])
        p1_pos = int(game_state.pawn_pos[0][1])
        p0_r, p0_c = p0_pos // BOARD_SIZE, p0_pos % BOARD_SIZE
        p1_r, p1_c = p1_pos // BOARD_SIZE, p1_pos % BOARD_SIZE

        if current_color == 0:
            my_r, my_c = p0_r, p0_c
            opp_r, opp_c = p1_r, p1_c
        else:
            my_r, my_c = p1_r, p1_c
            opp_r, opp_c = p0_r, p0_c

        is_model_turn = current_player == model_seat
        step_entry = {
            "step": step_count,
            "player": "model" if is_model_turn else "llm",
            "color": color_name,
            "board_ascii": render_board_ascii(state),
        }

        if is_model_turn:
            obs = observe_fn(state, jnp.int32([model_seat]))
            model_state = state.replace(observation=obs)

            key, action_key = jax.random.split(key)
            canonical_action, hidden, model_details = model_select_action(
                model, model_state, hidden, action_key, deterministic
            )
            abs_action = canonical_to_absolute(canonical_action, current_color)
            try:
                move_text = action_to_text(abs_action, my_r, my_c, opp_r, opp_c)
            except (ValueError, IndexError):
                move_text = f"action_{canonical_action}"

            # Convert top action indices to text for logging
            top_actions_text = {}
            for aidx, prob in model_details["top_actions"].items():
                a = canonical_to_absolute(aidx, current_color)
                try:
                    t = action_to_text(a, my_r, my_c, opp_r, opp_c)
                except (ValueError, IndexError):
                    t = f"action_{aidx}"
                top_actions_text[t] = f"{prob:.3f}"

            step_entry["move"] = move_text
            step_entry["model_value"] = model_details["value"]
            step_entry["model_action_prob"] = model_details["action_prob"]
            step_entry["model_top_actions"] = top_actions_text

            if verbose:
                top_str = ", ".join(f"{m}({p})" for m, p in top_actions_text.items())
                print(
                    f"  Model ({color_name}): {move_text}  "
                    f"[V={model_details['value']:+.3f} P={model_details['action_prob']:.3f} "
                    f"top: {top_str}]"
                )

            action = jnp.int32([canonical_action])
        else:
            # LLM's turn
            legal_moves = get_legal_moves_text(state, current_color)
            prompt = build_game_prompt(state, current_color, move_history)
            step_entry["llm_prompt"] = prompt

            canonical_action = None
            llm_attempts = []
            for retry in range(5):
                try:
                    llm_result = call_llm(
                        llm_model_name, SYSTEM_PROMPT, prompt, temperature=temperature
                    )
                    raw_response = llm_result["content"]
                    reasoning_content = llm_result.get("reasoning_content", "")
                    move_text, thinking = parse_llm_move(raw_response)

                    # Prefer reasoning_content (chain-of-thought) over parsed thinking
                    if reasoning_content:
                        thinking = reasoning_content

                    attempt_entry = {
                        "raw_response": raw_response,
                        "reasoning_content": reasoning_content,
                        "parsed_move": move_text,
                        "thinking": thinking,
                        "usage": llm_result["usage"],
                    }

                    if verbose:
                        if reasoning_content:
                            print(f"  LLM ({color_name}) reasoning:")
                            for line in reasoning_content.strip().splitlines():
                                print(f"    | {line}")
                        elif thinking:
                            print(f"  LLM ({color_name}) thinks:")
                            for line in thinking.strip().splitlines():
                                print(f"    | {line}")
                        print(f"  LLM ({color_name}) plays: {move_text}")

                    abs_action = text_to_action(move_text, my_r, my_c, opp_r, opp_c)
                    if current_color == 1:
                        canonical_action = canonical_to_absolute(abs_action, 1)
                    else:
                        canonical_action = abs_action

                    mask = np.array(state.legal_action_mask[0], dtype=bool)
                    if not mask[canonical_action]:
                        if verbose:
                            print("    -> Illegal move! Retrying...")
                        attempt_entry["error"] = "illegal_move"
                        llm_attempts.append(attempt_entry)
                        llm_invalid_moves += 1
                        canonical_action = None
                        prompt += (
                            f"\n\n'{move_text}' is not a legal move. "
                            f"Legal moves are: {', '.join(legal_moves)}\n"
                            f"Think again and give your move as MOVE: <move>"
                        )
                        continue

                    attempt_entry["valid"] = True
                    llm_attempts.append(attempt_entry)
                    break
                except (ValueError, IndexError) as e:
                    if verbose:
                        print(f"    -> Parse error: {e}. Retrying...")
                    llm_attempts.append(
                        {
                            "raw_response": raw_response if "raw_response" in dir() else str(e),
                            "error": str(e),
                        }
                    )
                    llm_invalid_moves += 1
                    prompt += (
                        f"\n\nCould not understand your move. "
                        f"Legal moves: {', '.join(legal_moves)}\n"
                        f"Respond with MOVE: <move> (e.g. MOVE: e5 or MOVE: d3h)"
                    )

            if canonical_action is None:
                mask = np.array(state.legal_action_mask[0], dtype=bool)
                legal_indices = np.where(mask)[0]
                canonical_action = int(np.random.choice(legal_indices))
                abs_action = canonical_to_absolute(canonical_action, current_color)
                try:
                    move_text = action_to_text(abs_action, my_r, my_c, opp_r, opp_c)
                except (ValueError, IndexError):
                    move_text = f"action_{canonical_action}"
                if verbose:
                    print(f"    -> Fallback random move: {move_text}")
                llm_attempts.append({"fallback_random": move_text})

            step_entry["move"] = move_text
            step_entry["llm_attempts"] = llm_attempts

            action = jnp.int32([canonical_action])

        move_history.append(f"{['B', 'W'][current_color]}:{move_text}")
        move_log.append(step_entry)
        state = step_fn(state, action)
        step_count += 1

        # Capture frame after each move
        svg_frames.append(state_to_svg(state))

    # Determine result
    rewards = np.array(state.rewards[0])
    model_reward = float(rewards[model_seat])

    result = {
        "model_reward": model_reward,
        "model_win": model_reward > 0,
        "model_lose": model_reward < 0,
        "draw": model_reward == 0,
        "terminated": bool(state.terminated[0]),
        "truncated": bool(state.truncated[0]),
        "steps": step_count,
        "llm_invalid_moves": llm_invalid_moves,
        "move_history": move_history,
        "move_log": move_log,
        "svg_frames": svg_frames,
    }
    return result


def get_first_llm_prompt(
    model: ResNetLSTMModel,
    init_fn,
    step_fn,
    observe_fn,
    model_seat: int,
    key: jax.Array,
    deterministic: bool = True,
    max_steps: int = 200,
) -> dict:
    """Build the first LLM turn prompt and return it.

    Parameters
    ----------
    model : ResNetLSTMModel
        Model used to advance the game until the LLM's first turn.
    init_fn
        Batched environment init function.
    step_fn
        Batched environment step function.
    observe_fn
        Batched environment observe function.
    model_seat : int
        Seat played by the checkpointed model.
    key : jax.Array
        RNG key for environment initialization and optional stochastic policy sampling.
    deterministic : bool, default=True
        Whether to use argmax model actions.
    max_steps : int, default=200
        Safety cap on steps taken while searching for the first LLM turn.

    Returns
    -------
    dict
        Prompt metadata including system prompt, user prompt, seat, color, and move history.
    """
    key, init_key = jax.random.split(key)
    state = init_fn(init_key[None])
    hidden = None
    move_history = []

    for step_count in range(max_steps):
        if bool(state.terminated[0]) or bool(state.truncated[0]):
            break

        current_player = int(state.current_player[0])
        current_color = int(state._x.color[0])
        color_name = ["Black", "White"][current_color]
        is_model_turn = current_player == model_seat

        if not is_model_turn:
            prompt = build_game_prompt(state, current_color, move_history)
            return {
                "step": step_count,
                "color": color_name,
                "model_seat": model_seat,
                "move_history": move_history,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": prompt,
            }

        game_state = state._x
        p0_pos = int(game_state.pawn_pos[0][0])
        p1_pos = int(game_state.pawn_pos[0][1])
        p0_r, p0_c = p0_pos // BOARD_SIZE, p0_pos % BOARD_SIZE
        p1_r, p1_c = p1_pos // BOARD_SIZE, p1_pos % BOARD_SIZE

        if current_color == 0:
            my_r, my_c = p0_r, p0_c
            opp_r, opp_c = p1_r, p1_c
        else:
            my_r, my_c = p1_r, p1_c
            opp_r, opp_c = p0_r, p0_c

        obs = observe_fn(state, jnp.int32([model_seat]))
        model_state = state.replace(observation=obs)
        key, action_key = jax.random.split(key)
        # canonical_action, hidden, _ = model_select_action(
        #     model, model_state, hidden, action_key, deterministic
        # )
        canonical_action = 90
        abs_action = canonical_to_absolute(canonical_action, current_color)
        try:
            move_text = action_to_text(abs_action, my_r, my_c, opp_r, opp_c)
        except (ValueError, IndexError):
            move_text = f"action_{canonical_action}"
        move_history.append(f"{['B', 'W'][current_color]}:{move_text}")
        state = step_fn(state, jnp.int32([canonical_action]))

    raise RuntimeError("Failed to reach an LLM turn before the game ended.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Quoridor checkpoint against an LLM.")
    parser.add_argument("checkpoint_path", type=str, help="Path to checkpoint directory.")
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-5.4",
        help="LLM model name for litellm (e.g. gpt-4o, claude-opus-4-6). Default: gpt-5.4.",
    )
    parser.add_argument("--num-games", type=int, default=1, help="Number of games (default: 1).")
    parser.add_argument(
        "--model-seat",
        type=int,
        default=1,
        choices=[-1, 0, 1],
        help="Model seat: 0=first, 1=second, -1=alternate (default).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--temperature", type=float, default=0.3, help="LLM temperature (default: 0.3)."
    )
    parser.add_argument("--stochastic", action="store_true", help="Sample from model policy.")
    parser.add_argument("--quiet", action="store_true", help="Only print summary.")
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the first LLM system/user prompt and exit.",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--gif-duration", type=int, default=800, help="GIF frame duration in ms (default: 800)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results. Default: next to checkpoint.",
    )
    # Model architecture
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--num-lstm-layers", type=int, default=1)

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint_path).resolve()
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from: {ckpt_path}")
    model_cfg = {
        "num_filters": args.num_filters,
        "num_blocks": args.num_blocks,
        "lstm_hidden_size": args.lstm_hidden_size,
        "num_lstm_layers": args.num_lstm_layers,
    }
    model = load_model(ckpt_path, model_cfg)
    model.eval()

    env = Quoridor(observation_cls="default")
    init_fn = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(env.step))
    observe_fn = jax.jit(jax.vmap(env.observe))

    key = jax.random.key(args.seed)
    deterministic = not args.stochastic
    verbose = not args.quiet

    if args.print_prompt:
        model_seat = 0 if args.model_seat == -1 else args.model_seat
        prompt_info = get_first_llm_prompt(
            model=model,
            init_fn=init_fn,
            step_fn=step_fn,
            observe_fn=observe_fn,
            model_seat=model_seat,
            key=key,
            deterministic=deterministic,
            max_steps=args.max_steps,
        )
        seat_label = "Black (first)" if model_seat == 0 else "White (second)"
        print("Model loaded. Prompt-only mode.\n")
        print(f"Model seat: {seat_label}")
        print(f"LLM color: {prompt_info['color']}")
        print(f"LLM turn step: {prompt_info['step']}")
        if prompt_info["move_history"]:
            print(f"Moves before prompt: {', '.join(prompt_info['move_history'])}")
        else:
            print("Moves before prompt: none")
        print("\n[SYSTEM PROMPT]")
        print(prompt_info["system_prompt"])
        print("\n[USER PROMPT]")
        print(prompt_info["user_prompt"])
        return

    # Create output directory
    llm_tag = args.llm_model.replace("/", "_")
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = ckpt_path.parent.parent / f"llm_eval_{llm_tag}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model loaded. Output dir: {out_dir}\n")

    results = []
    print(f"Playing {args.num_games} games: Model vs {args.llm_model}")
    print(f"Model policy: {'stochastic' if args.stochastic else 'deterministic'}")
    print(f"LLM temperature: {args.temperature}")
    print("=" * 60)

    for game_idx in range(args.num_games):
        if args.model_seat == -1:
            model_seat = game_idx % 2
        else:
            model_seat = args.model_seat

        seat_label = "Black (first)" if model_seat == 0 else "White (second)"
        print(f"\nGame {game_idx + 1}/{args.num_games} — Model plays as {seat_label}")
        print("-" * 40)

        key, game_key = jax.random.split(key)
        t0 = time.perf_counter()
        result = play_one_game(
            model=model,
            env=env,
            init_fn=init_fn,
            step_fn=step_fn,
            observe_fn=observe_fn,
            llm_model_name=args.llm_model,
            model_seat=model_seat,
            key=game_key,
            deterministic=deterministic,
            verbose=verbose,
            max_steps=args.max_steps,
            temperature=args.temperature,
        )
        elapsed = time.perf_counter() - t0
        result["model_seat"] = model_seat
        result["elapsed"] = elapsed
        result["game_idx"] = game_idx

        outcome = "WIN" if result["model_win"] else ("LOSE" if result["model_lose"] else "DRAW")
        print(
            f"  Result: Model {outcome} in {result['steps']} steps "
            f"({elapsed:.1f}s, {result['llm_invalid_moves']} invalid LLM moves)"
        )

        # Save GIF
        svg_frames = result.pop("svg_frames")
        gif_path = out_dir / f"game_{game_idx + 1:03d}_{outcome.lower()}.gif"
        try:
            save_game_gif(svg_frames, gif_path, duration_ms=args.gif_duration)
            print(f"  GIF saved: {gif_path}")
        except Exception as e:
            print(f"  Warning: Failed to save GIF: {e}")

        # Save per-game JSON log
        log_path = out_dir / f"game_{game_idx + 1:03d}_{outcome.lower()}.json"
        with open(log_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    wins = sum(1 for r in results if r["model_win"])
    losses = sum(1 for r in results if r["model_lose"])
    draws = sum(1 for r in results if r["draw"])
    total = len(results)
    total_invalid = sum(r["llm_invalid_moves"] for r in results)
    avg_steps = np.mean([r["steps"] for r in results])

    print(f"  Model wins:  {wins}/{total} ({wins / total * 100:.0f}%)")
    print(f"  Model loses: {losses}/{total} ({losses / total * 100:.0f}%)")
    print(f"  Draws:       {draws}/{total} ({draws / total * 100:.0f}%)")
    print(f"  Avg steps:   {avg_steps:.1f}")
    print(f"  Total LLM invalid moves: {total_invalid}")

    for seat in [0, 1]:
        seat_results = [r for r in results if r["model_seat"] == seat]
        if not seat_results:
            continue
        sw = sum(1 for r in seat_results if r["model_win"])
        st = len(seat_results)
        label = "Black/first" if seat == 0 else "White/second"
        print(f"  As {label}: {sw}/{st} wins ({sw / st * 100:.0f}%)")

    # Save summary
    summary_path = out_dir / "summary.json"
    summary = {
        "checkpoint": str(ckpt_path),
        "llm_model": args.llm_model,
        "num_games": total,
        "model_wins": wins,
        "model_losses": losses,
        "draws": draws,
        "win_rate": wins / total if total > 0 else 0,
        "temperature": args.temperature,
        "deterministic": deterministic,
        "avg_steps": float(avg_steps),
        "total_llm_invalid_moves": total_invalid,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to: {out_dir}")


if __name__ == "__main__":
    main()
