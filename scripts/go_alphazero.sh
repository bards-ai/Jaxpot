#!/bin/bash

/home/bard/Documents/factorio-jax/.venv/bin/python scripts/gtp_engine.py \
  --checkpoint outputs/2026-03-23/23-39-36/checkpoints \
  --agent mcts \
  --mcts-num-simulations 50 \
  --mcts-max-considered-actions 19 \
  --mcts-dirichlet-alpha 0.03 \
  --mcts-exploration-fraction 0.25 \
  --mcts-gamma 1.0 \
  --log-file logs/gtp_engine_$(date +%Y-%m-%d_%H-%M-%S).log \
  --log-illegal-moves