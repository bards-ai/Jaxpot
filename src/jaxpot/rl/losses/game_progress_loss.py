import jax.numpy as jnp

from jaxpot.models.base import GameProgressModelOutput
from jaxpot.rl.losses.auxiliary_losses import AuxiliaryLoss
from jaxpot.rl.losses.mse_loss import mse_loss
from jaxpot.rollout.aux_target_hooks import AuxTargetHook, GameProgressTargetHook


class GameProgressMSELoss(AuxiliaryLoss[GameProgressModelOutput]):
    """MSE loss for game progress prediction head."""

    name: str = "game_progress_mse_loss"
    target_field: str = "game_progress_target"

    def __init__(self, coef: float = 1.0, max_steps: int = 42):
        self.coef = coef
        self.max_steps = max_steps

    def __call__(
        self,
        model_outputs: GameProgressModelOutput,
        target: jnp.ndarray,
    ) -> jnp.ndarray:
        pred = model_outputs.game_progress
        if pred is None:
            return jnp.array(0.0)
        return mse_loss(pred, target, loss_coeff=self.coef)

    def make_target_hook(self) -> AuxTargetHook:
        return GameProgressTargetHook(max_steps=self.max_steps)
