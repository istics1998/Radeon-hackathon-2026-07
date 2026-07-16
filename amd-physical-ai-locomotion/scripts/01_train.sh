#!/usr/bin/env bash
# Phase 1 · Train the locomotion policy on the AMD GPU.
#   ENV=Go1JoystickFlatTerrain (quadruped, default) or G1JoystickFlatTerrain (humanoid)
#   SMOKE=1 runs a tiny 200k-step run just to confirm the pipeline works.
set -euo pipefail
cd "$(dirname "$0")/.."

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"

ENV="${ENV:-Go1JoystickFlatTerrain}"
SEED="${SEED:-0}"

if [ "${SMOKE:-0}" = "1" ]; then
  echo "[01_train] SMOKE run (200k steps)"
  python3 -m src.train --env "$ENV" --seed "$SEED" --num-timesteps 200000
else
  echo "[01_train] FULL run (Playground tuned num_timesteps)"
  python3 -m src.train --env "$ENV" --seed "$SEED"
fi
