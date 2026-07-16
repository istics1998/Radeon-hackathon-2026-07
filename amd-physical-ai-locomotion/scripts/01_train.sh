#!/usr/bin/env bash
# Phase 1 · Train the locomotion policy on the AMD GPU.
#   ENV=Go1JoystickFlatTerrain (quadruped, default) or G1JoystickFlatTerrain (humanoid)
#   SMOKE=1 runs a tiny 200k-step run just to confirm the pipeline works.
#
# Uses our from-scratch single-GPU jit PPO (src/train_jax_ppo.py), NOT brax's
# PPO runner — the latter segfaults in libhsa-runtime64 on this gfx1100/ROCm
# stack (see docs/HANDOFF.md section 5). Set TRAINER=brax to use the old path.
set -euo pipefail
cd "$(dirname "$0")/.."

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"

ENV="${ENV:-Go1JoystickFlatTerrain}"
SEED="${SEED:-0}"
TRAINER="${TRAINER:-jax}"   # jax = our single-GPU PPO; brax = original brax runner

if [ "$TRAINER" = "brax" ]; then
  MODULE="src.train"
else
  MODULE="src.train_jax_ppo"
fi

if [ "${SMOKE:-0}" = "1" ]; then
  echo "[01_train] SMOKE run (200k steps) via $MODULE"
  python3 -m "$MODULE" --env "$ENV" --seed "$SEED" --num-timesteps 200000
else
  echo "[01_train] FULL run via $MODULE"
  python3 -m "$MODULE" --env "$ENV" --seed "$SEED"
fi
