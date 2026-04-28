"""Universal FastAPI web server for playing against a trained bot.

Usage:
    python -m jaxpot.web --env quoridor --checkpoint /path/to/ckpt
    python -m jaxpot.web --env connect4 --checkpoint /path/to/ckpt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel as PydanticModel

from jaxpot.web.adapter import WebAdapter


class MoveRequest(PydanticModel):
    action_id: int


# ── Game registry ─────────────────────────────────────────────────────

GAME_REGISTRY: dict[str, type] = {}


def register_game(name: str):
    """Decorator to register a WebAdapter subclass."""
    def wrapper(cls):
        GAME_REGISTRY[name] = cls
        return cls
    return wrapper


def get_adapter(name: str, **kwargs) -> WebAdapter:
    """Instantiate a registered game adapter."""
    if name not in GAME_REGISTRY:
        available = ", ".join(GAME_REGISTRY.keys())
        raise ValueError(f"Unknown game '{name}'. Available: {available}")
    return GAME_REGISTRY[name](**kwargs)


def _discover_games():
    """Import all game adapter modules to trigger registration."""
    games_dir = Path(__file__).parent / "games"
    for game_dir in games_dir.iterdir():
        if game_dir.is_dir() and (game_dir / "adapter.py").exists():
            game_name = game_dir.name
            __import__(f"jaxpot.web.games.{game_name}.adapter")


# ── Model loading ─────────────────────────────────────────────────────

def load_model_from_checkpoint(checkpoint_path: str, model_cls, model_kwargs: dict):
    """Load a model from an Orbax checkpoint."""
    import orbax.checkpoint as ocp
    from flax import nnx

    ckpt_dir = Path(checkpoint_path).resolve()
    state_path = ckpt_dir / "state"
    if not state_path.exists():
        if (ckpt_dir / "metadata").exists() or ckpt_dir.name == "state":
            state_path = ckpt_dir
        else:
            raise FileNotFoundError(f"Orbax state directory not found at {state_path}")

    metadata_path = ckpt_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"Checkpoint metadata: {json.dumps(metadata, indent=2)}")

    abstract_model = nnx.eval_shape(
        lambda: model_cls(rngs=nnx.Rngs(0), **model_kwargs)
    )
    graphdef, abstract_state = nnx.split(abstract_model)

    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    fallback_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    payload = checkpointer.restore(
        state_path,
        args=ocp.args.StandardRestore(fallback_sharding=fallback_sharding),
    )

    nnx.replace_by_pure_dict(abstract_state, payload["model"])
    loaded_model = nnx.merge(graphdef, abstract_state)
    loaded_model.eval()
    return loaded_model


# ── Game server ───────────────────────────────────────────────────────

class GameServer:
    """Encapsulates all mutable game state and bot logic."""

    def __init__(self, adapter: WebAdapter):
        self.adapter = adapter
        self.model = adapter.model
        self.jit_init = jax.jit(adapter.env.init)
        self.jit_step = jax.jit(adapter.env.step)
        self.game_state = None
        self.game_count = 0
        self.human_player = 0
        self.bot_hidden_state = None

    def warmup(self):
        """JIT-compile env functions with a dummy run."""
        print("Warming up JIT...")
        t0 = time.perf_counter()
        _state = self.jit_init(jax.random.PRNGKey(0))
        _state = self.jit_step(_state, jnp.int32(0))
        jax.block_until_ready(_state)
        print(f"JIT warmup done in {time.perf_counter() - t0:.1f}s")

    def get_bot_move(self, state):
        """Get bot's move. Falls back to random if no model is loaded."""
        mask = state.legal_action_mask

        if self.model is None:
            legal = np.where(np.array(mask))[0]
            return int(np.random.choice(legal))

        obs = state.observation

        if getattr(self.model, "is_recurrent", False):
            if self.bot_hidden_state is None:
                self.bot_hidden_state = self.model.init_state(batch_size=1)
            output = self.model(obs, hidden_state=self.bot_hidden_state)
            self.bot_hidden_state = output.hidden_state
        else:
            # Some models (e.g. Connect4Baseline) override __call__ and
            # always expect a batch dimension. Add it if missing.
            obs_shape = getattr(self.model, "obs_shape", None)
            needs_batch = obs_shape and obs.ndim == len(obs_shape)
            if needs_batch:
                obs = obs[None, ...]
            output = self.model(obs)
            if needs_batch:
                output = jax.tree.map(lambda x: x.squeeze(0) if x is not None else x, output)

        logits = output.policy_logits
        masked_logits = jnp.where(mask, logits, -jnp.inf)
        return int(jnp.argmax(masked_logits))

    def start_new_game(self) -> str | None:
        """Initialize a new game, resetting LSTM hidden state.

        Returns the bot's first move display string, or None.
        """
        self.game_count += 1
        key = jax.random.PRNGKey(self.game_count)
        self.game_state = self.jit_init(key)

        self.bot_hidden_state = None

        self.human_player = int(self.game_state.current_player)
        bot_player = 1 - self.human_player

        last_bot_move = None
        if int(self.game_state.current_player) == bot_player:
            bot_action = self.get_bot_move(self.game_state)
            last_bot_move = self.adapter.action_to_display(bot_action, self.game_state)
            self.game_state = self.jit_step(self.game_state, jnp.int32(bot_action))

        return last_bot_move

    def apply_human_move(self, action_id: int) -> dict:
        """Apply human move, get bot response, return board JSON.

        Handles games where a failed move keeps the turn with the same
        player (e.g. classical dark hex, phantom TTT). The bot only plays
        when it's actually the bot's turn.
        """
        t_total = time.perf_counter()

        self.game_state = self.jit_step(self.game_state, jnp.int32(action_id))

        bot_player = 1 - self.human_player
        last_bot_move = None

        # Bot plays as long as it's the bot's turn and game isn't over
        while (
            not (bool(self.game_state.terminated) or bool(self.game_state.truncated))
            and int(self.game_state.current_player) == bot_player
        ):
            t0 = time.perf_counter()
            bot_action = self.get_bot_move(self.game_state)
            t_think = time.perf_counter() - t0

            last_bot_move = self.adapter.action_to_display(bot_action, self.game_state)
            self.game_state = self.jit_step(self.game_state, jnp.int32(bot_action))

            print(f"[move] bot_think={t_think * 1000:.1f}ms", end=" | ")

        board = self.adapter.state_to_json(self.game_state, self.human_player)
        board["last_bot_move"] = last_bot_move

        t_total = time.perf_counter() - t_total
        print(f"total={t_total * 1000:.1f}ms")
        return board

    def get_current_state(self) -> dict:
        """Return current board state as JSON."""
        if self.game_state is None:
            last_bot_move = self.start_new_game()
            board = self.adapter.state_to_json(self.game_state, self.human_player)
            board["last_bot_move"] = last_bot_move
            return board
        board = self.adapter.state_to_json(self.game_state, self.human_player)
        board["last_bot_move"] = None
        return board


# ── FastAPI app factory ───────────────────────────────────────────────

def create_app(server: GameServer, dev: bool = False) -> FastAPI:
    """Create a FastAPI app wired to the given GameServer."""
    import hashlib

    app = FastAPI()

    def _page_hash() -> str:
        """Hash the frontend files for live-reload detection."""
        base_path = Path(__file__).parent / "templates" / "base.html"
        game_html_path = server.adapter.get_frontend_path()
        content = base_path.read_bytes() + game_html_path.read_bytes()
        return hashlib.md5(content).hexdigest()[:12]

    @app.get("/favicon.ico")
    async def favicon():
        return Response(status_code=204)

    if dev:
        @app.get("/api/content-hash")
        async def content_hash():
            return {"hash": _page_hash()}

    @app.get("/", response_class=HTMLResponse)
    async def index():
        base_path = Path(__file__).parent / "templates" / "base.html"
        game_html_path = server.adapter.get_frontend_path()
        metadata = server.adapter.get_metadata()

        base = base_path.read_text()
        game_html = game_html_path.read_text()

        page = base.replace("{{ game_name }}", metadata["name"])
        page = page.replace("{{ game_content }}", game_html)

        if dev:
            # Inject live-reload poller
            live_reload = """
<script>
(function() {
  let lastHash = null;
  setInterval(async () => {
    try {
      const r = await fetch('/api/content-hash');
      const {hash} = await r.json();
      if (lastHash && hash !== lastHash) location.reload();
      lastHash = hash;
    } catch(e) {}
  }, 1000);
})();
</script>"""
            page = page.replace("</body>", live_reload + "\n</body>")

        return page

    @app.get("/api/metadata")
    async def get_metadata():
        return server.adapter.get_metadata()

    @app.post("/api/new_game")
    async def new_game():
        last_bot_move = server.start_new_game()
        board = server.adapter.state_to_json(server.game_state, server.human_player)
        board["last_bot_move"] = last_bot_move
        return board

    @app.post("/api/move")
    async def make_move(req: MoveRequest):
        return server.apply_human_move(req.action_id)

    @app.get("/api/state")
    async def get_state():
        return server.get_current_state()

    return app


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Play a game against a trained bot in the browser."
    )
    parser.add_argument(
        "--env", type=str, required=True,
        help="Game environment name (e.g. quoridor, connect4).",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint. Uses baseline/heuristic if not provided.",
    )
    parser.add_argument("--port", type=int, default=8504)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--dev", action="store_true",
        help="Dev mode: auto-reload on Python file changes, "
             "auto-refresh browser on frontend changes.",
    )
    # Model architecture overrides
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--num-lstm-layers", type=int, default=1)
    return parser.parse_args()


def _build_server(args, dev: bool = False) -> tuple[GameServer, FastAPI]:
    """Build GameServer + FastAPI app from parsed args."""
    _discover_games()

    adapter = get_adapter(
        args.env,
        checkpoint=args.checkpoint,
        num_filters=args.num_filters,
        num_blocks=args.num_blocks,
        lstm_hidden_size=args.lstm_hidden_size,
        num_lstm_layers=args.num_lstm_layers,
    )

    server = GameServer(adapter)
    server.warmup()
    server.start_new_game()

    app = create_app(server, dev=dev)
    return server, app


def _make_app():
    """App factory for uvicorn --reload (reads config from env vars)."""
    import os

    class _Args:
        env = os.environ["JAXPOT_WEB_ENV"]
        checkpoint = os.environ.get("JAXPOT_WEB_CHECKPOINT") or None
        num_filters = int(os.environ.get("JAXPOT_WEB_NUM_FILTERS", "128"))
        num_blocks = int(os.environ.get("JAXPOT_WEB_NUM_BLOCKS", "6"))
        lstm_hidden_size = int(os.environ.get("JAXPOT_WEB_LSTM_HIDDEN", "256"))
        num_lstm_layers = int(os.environ.get("JAXPOT_WEB_LSTM_LAYERS", "1"))

    _, app = _build_server(_Args(), dev=True)
    return app


def main():
    import uvicorn

    args = parse_args()

    if args.dev:
        import os

        # Pass config via env vars so the app factory can read them on reload
        os.environ["JAXPOT_WEB_ENV"] = args.env
        if args.checkpoint:
            os.environ["JAXPOT_WEB_CHECKPOINT"] = args.checkpoint
        os.environ["JAXPOT_WEB_NUM_FILTERS"] = str(args.num_filters)
        os.environ["JAXPOT_WEB_NUM_BLOCKS"] = str(args.num_blocks)
        os.environ["JAXPOT_WEB_LSTM_HIDDEN"] = str(args.lstm_hidden_size)
        os.environ["JAXPOT_WEB_LSTM_LAYERS"] = str(args.num_lstm_layers)

        web_dir = str(Path(__file__).parent)
        uvicorn.run(
            "jaxpot.web.server:_make_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=[web_dir],
        )
    else:
        _, app = _build_server(args)
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
