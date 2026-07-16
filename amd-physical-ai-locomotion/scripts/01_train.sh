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

# Stable region for the jax trainer on this gfx1100/ROCm stack: the rocprofiler
# HSA segfault probability rises with dispatch size and count, so cap env count
# and fold a bounded number of iters into each single-dispatch chunk. 512 envs
# with 4-iter chunks is proven stable (docs/HANDOFF.md). Override via env vars.
NUM_ENVS="${NUM_ENVS:-512}"
ITERS_PER_CHUNK="${ITERS_PER_CHUNK:-4}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-50000000}"
MAX_RESTARTS="${MAX_RESTARTS:-200}"   # auto-resume budget across HSA segfaults

if [ "$TRAINER" = "brax" ]; then
  MODULE="src.train"
else
  MODULE="src.train_jax_ppo"
fi

if [ "${SMOKE:-0}" = "1" ]; then
  echo "[01_train] SMOKE run (200k steps) via $MODULE"
  python3 -m "$MODULE" --env "$ENV" --seed "$SEED" --num-timesteps 200000 \
    --num-envs "$NUM_ENVS" --iters-per-chunk "$ITERS_PER_CHUNK"
elif [ "$TRAINER" = "brax" ]; then
  echo "[01_train] FULL run via $MODULE"
  python3 -m "$MODULE" --env "$ENV" --seed "$SEED"
else
  # Auto-restart loop: each run trains chunk-by-chunk, checkpointing after every
  # chunk. If the ROCm HSA segfault kills a run mid-training, --resume picks up
  # from the last checkpoint, so total progress accumulates across crashes.
  echo "[01_train] FULL run via $MODULE (envs=$NUM_ENVS chunk=$ITERS_PER_CHUNK, auto-resume)"
  attempt=0
  while [ "$attempt" -le "$MAX_RESTARTS" ]; do
    resume_flag=""
    [ "$attempt" -gt 0 ] && resume_flag="--resume"
    set +e
    python3 -m "$MODULE" --env "$ENV" --seed "$SEED" \
      --num-timesteps "$NUM_TIMESTEPS" --num-envs "$NUM_ENVS" \
      --iters-per-chunk "$ITERS_PER_CHUNK" $resume_flag
    code=$?
    set -e
    if [ "$code" -eq 0 ]; then
      echo "[01_train] training completed (exit 0)"
      break
    fi
    attempt=$((attempt + 1))
    echo "[01_train] run exited $code (likely HSA segfault) — resume attempt $attempt/$MAX_RESTARTS"
    sleep 2
  done
  if [ "$attempt" -gt "$MAX_RESTARTS" ]; then
    echo "[01_train] hit MAX_RESTARTS=$MAX_RESTARTS; latest checkpoint is preserved. Re-run to continue."
    exit 1
  fi
fi
