#!/bin/bash
python scripts/gtp_engine.py --checkpoint outputs/2026-02-23/09-39-35/09-39-35/checkpoints/002999/ --log-file logs/gtp_engine_$(date +%Y-%m-%d_%H-%M-%S).log --log-illegal-moves
