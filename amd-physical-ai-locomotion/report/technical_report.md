# Technical Report — AMD Physical AI: Quadruped Locomotion RL on Radeon (ROCm)

> AMD AI DevMaster Hackathon · Track 3 (Physical AI)
> Team: **istics1998** (solo) · Submission title: `[Physical AI] istics1998 - Quadruped Locomotion RL on Radeon`

This report supplements the [README](../README.md) with the technical detail behind
what we did, the method we used, and the innovation. It is written to be honest
about scope: the training pipeline and PPO algorithm are complete and verified,
but a fully trained walking policy was blocked by a ROCm runtime bug (see §4/§7).

## 1. Target application — definition & description

Velocity-command locomotion for the Unitree Go1 quadruped
(`Go1JoystickFlatTerrain`): the robot should track a commanded body velocity
(joystick input) while staying upright on flat terrain. This is a core Physical AI
task — a velocity-tracking locomotion controller is the foundation for autonomous
inspection, logistics, and navigation in unstructured environments.

## 2. System architecture & solution design

- Simulation: MuJoCo Playground (MJX) — GPU-parallel physics via JAX/XLA.
- Learning: from-scratch single-GPU jit PPO (`src/train_jax_ppo.py`), built with
  `jax.jit` + `lax.scan`, no `pmap`. A Brax PPO path (`src/train.py`) is kept as a
  reference/comparison baseline.
- Network: asymmetric actor-critic (`src/nets.py`) — the policy sees the 48-dim
  `state`, the value function sees the 123-dim `privileged_state`.
- Closed loop: proprioceptive obs → policy → joint targets → PD control, wrapped
  for Brax training via `mujoco_playground.wrapper.wrap_for_brax_training`.
- Rendering: headless, for display-less server instances.

Pipeline: MJX env → PPO trainer (jit + lax.scan) → checkpoint → closed-loop eval
(`src/eval.py`) → rendered demo video (`src/render.py`).

## 3. Dataset(s) used for training / evaluation

Reinforcement learning from simulation only — there is no external dataset. The
environment comes directly from MuJoCo Playground's `registry`, so there is no
custom dataset to build, license, or download.

## 4. AMD Radeon GPU usage

- GPU: AMD Radeon gfx1100 · ROCm 7.2.1 · JAX 0.11 (`jax[rocm7-local]`).
- Verification: `scripts/00_verify_rocm.sh` passes 6/6 checks and prints
  `RocmDevice`, confirming the GPU and JAX ROCm stack are live.
- Simulation ran on the GPU: `env.reset()` / `env.step()` / `jax.vmap` all execute
  on the Radeon GPU via MJX.
- PPO ran on the GPU: on runs that did not hit the profiler race, training reached
  `EXIT=0` with finite (non-NaN) reward/loss values and saved checkpoints (see §5
  table). These were short 3–7 iteration correctness checks, not converged training.
- Blocker: full stable training segfaults inside `libhsa-runtime64.so.1` — see §7.

## 5. Algorithm validation

On non-crashing runs the from-scratch jit PPO reached `EXIT=0` with finite (non-NaN)
reward/loss and saved checkpoints. These are short correctness checks (3–7
iterations), not converged training — they confirm the algorithm runs correctly,
not that the robot learned to walk:

| Envs | Unroll | Iters | Result |
|------|--------|-------|--------|
| 256  | 10     | 7     | EXIT=0, checkpoint saved |
| 512  | 10     | 5     | EXIT=0, checkpoint saved |
| 1024 | 20     | 3     | EXIT=0, checkpoint saved |

Ablation — trainer choice on ROCm:

| Trainer | Execution path | Result on gfx1100 |
|---|---|---|
| Brax PPO (`src/train.py`) | `pmap` + `device_put_replicated` | Segfaults early — replication hits the profiler race almost immediately |
| From-scratch jit PPO (`src/train_jax_ppo.py`) | `jax.jit` + `lax.scan`, single device | Runs; `EXIT=0` on non-crashing runs; still susceptible under long dispatch-heavy runs |

## 6. Innovation & key technical contributions

- End-to-end GPU-parallel MJX rollouts + RL training on AMD Radeon via ROCm, with
  no CUDA-only dependency.
- A from-scratch single-GPU jit PPO (`jax.jit` + `lax.scan`, no `pmap`) that is
  portable across accelerators and sidesteps the fastest ROCm crash path.
- A provable GPU-execution guard: the pipeline asserts `RocmDevice` before running.
- The most substantive contribution is the upstream bug analysis in §7 — a precise,
  reproducible diagnosis of a ROCm runtime issue that blocks JAX GPU training on
  gfx1100.

## 7. Upstream open-source contribution

We identified and documented a ROCm runtime bug: `jax-rocm7-plugin`
(`xla_rocm_plugin.so`) has `librocprofiler-sdk.so.1` as a static NEEDED dependency.
The profiler intercepts every HIP kernel launch via GOTCHA hooks, and on gfx1100
that HSA injection has a non-deterministic race condition causing random segfaults.

We confirmed it cannot be disabled by environment variables
(`HSA_TOOLS_LIB=`, `ROCP_TOOL_LIB=`, `ROCPROFILER_DISABLE=1`), XLA flags,
`patchelf --remove-needed` (the plugin needs the `rocprofiler_force_configure`
symbol), or uninstalling `rocprofiler-sdk` (which destroys JAX GPU support).

The full report — reproduction steps, rocgdb stack trace, and ldd evidence — is in
[`docs/ROCM_BUG_REPORT.md`](../docs/ROCM_BUG_REPORT.md), ready to file to ROCm/JAX.
The actionable insight: rocprofiler-sdk should be dynamically loaded, not statically
linked into the XLA plugin.

## 8. Additional information

Because full training is blocked at the ROCm runtime layer (not in our code), the
demo video is a random-policy rollout on `Go1JoystickFlatTerrain`, shown to prove
the GPU simulation + rendering pipeline works on AMD Radeon. Per the track FAQ,
simulation-only results are acceptable for submission.

## 9. Team & contributions

| Member | Role | Contribution |
|---|---|---|
| istics1998 | Solo developer | PPO implementation, MJX integration, ROCm debugging, documentation |

---

### Reproducibility checklist (align with README)
- [x] `scripts/00_verify_rocm.sh` passes on the Radeon instance (RocmDevice)
- [x] From-scratch jit PPO reaches `EXIT=0` on non-crashing runs (checkpoint saved)
- [ ] Full training to convergence — blocked by the ROCm profiler race (§7)
- [x] `scripts/03_record_video.sh` produces the demo mp4 (random policy)
- [ ] Docker build + run verified on a fresh Radeon instance
- [ ] Upstream ROCm/JAX bug report filed and linked
