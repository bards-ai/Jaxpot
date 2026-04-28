from typing import override
import haiku as hk
from flax import nnx
from pgx._src.baseline import _create_az_model_v0, _load_baseline_model

from jaxpot.models.base import ModelOutput, PolicyValueModel


class PGXBaselineModel(PolicyValueModel):
    """Wrap a PGX baseline model for policy/value inference."""

    def __init__(
        self, baseline_model_id: str, download_dir: str = "baselines", is_eval: bool = True
    ):
        super().__init__()
        if baseline_model_id not in (
            "animal_shogi_v0",
            "gardner_chess_v0",
            "go_9x9_v0",
            "hex_v0",
            "othello_v0",
        ):
            raise ValueError(f"Unsupported baseline model ID: {baseline_model_id}")
        model_args, model_params, model_state = _load_baseline_model(
            baseline_model_id, download_dir
        )

        self.model_params = nnx.data(model_params)
        self.model_state = nnx.data(model_state)
        self.model_args = nnx.data(model_args)

        self.is_eval = is_eval

        def forward_fn(x):
            net = _create_az_model_v0(**model_args)
            policy_out, value_out = net(x, is_training=not self.is_eval, test_local_stats=False)
            return policy_out, value_out

        self.baseline_fn = hk.without_apply_rng(hk.transform_with_state(forward_fn))

    @override
    def __call__(self, x, hidden_state=None) -> ModelOutput:
        (logits, value), _ = self.baseline_fn.apply(self.model_params, self.model_state, x)
        return ModelOutput(value=value, policy_logits=logits)
