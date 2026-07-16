# AMD Physical AI — Quadruped/Humanoid Locomotion RL on Radeon (ROCm)

Track 3 (Physical AI) submission for the **AMD AI DevMaster Hackathon**.

We train a legged-locomotion control policy with reinforcement learning that
runs **end-to-end on AMD Radeon GPUs via ROCm**, using
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
(MJX) for GPU-parallel physics simulation and [Brax](https://github.com/google/brax)
PPO for training. The default robot is the **Unitree Go1 quadruped**; a humanoid
(**Unitree G1**) task is available by changing one flag.

Why this stack on AMD: MJX runs MuJoCo through the **XLA compiler via JAX**,
which supports AMD GPUs natively — no CUDA-only dependency. This is the same
path AMD documents in its
[ROCm + JAX + MuJoCo blog](https://rocm.blogs.amd.com/artificial-intelligence/rocm-jax-mujoco/README.html).
We deliberately avoid Isaac Gym/Lab (CUDA-only) and the MJX-*Warp* backend
(NVIDIA-only); we use the **MJX-JAX** path.

---

## What this demonstrates (mapped to the judging criteria)

| Criterion | Where |
|---|---|
| Robot capability (30) | PPO locomotion policy; closed-loop tracking metrics in `src/eval.py` |
| **AMD Radeon/ROCm adoption (20)** | Training **and** inference run on the GPU; `src/config.py:assert_gpu` hard-fails on CPU so it's provable |
| Innovation (20) | GPU-parallel MJX rollouts + domain randomization for robustness |
| Application value (20) | Legged locomotion control — inspection, logistics, last-mile robotics |
| Upstream OSS contribution (10) | See `report/` — ROCm install docs / compatibility fix PR |

---

## Repository layout

```
scripts/00_verify_rocm.sh   # RUN FIRST — proves JAX sees the AMD GPU + MJX rollout
scripts/01_train.sh         # train (SMOKE=1 for a quick pipeline check)
scripts/02_eval.sh          # closed-loop evaluation + metrics
scripts/03_record_video.sh  # render rollout to mp4 for the demo video
src/config.py               # paths + GPU assertion + headless render defaults
src/train.py                # Brax PPO on a Playground locomotion env
src/eval.py                 # load checkpoint, closed-loop rollout, metrics
src/render.py               # rollout -> mp4 (headless osmesa)
configs/go1_joystick.yaml   # default task settings
Dockerfile                  # ROCm JAX base image, reproducible build
report/                     # technical report template
```

---

## Setup on a Radeon Cloud instance

Provision an AMD Radeon GPU instance following the
[Radeon Cloud User Guide](https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07/blob/main/Radeon-Cloud-User%20Guide/README.md),
then open a JupyterLab terminal or SSH in.

### Option A — Docker (recommended, most reproducible)

The base image `rocm/jax-community` already ships a ROCm-enabled JAX build.

```bash
# 1. Pick the tag matching your instance's ROCm version (see below), then:
docker build --build-arg BASE_TAG=<tag> -t amd-locomotion .

# 2. Verify the toolchain (GPUs MUST be passed through):
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --security-opt seccomp=unconfined \
  -v "$PWD/outputs:/app/outputs" \
  amd-locomotion bash scripts/00_verify_rocm.sh
```

### Option B — Native pip

```bash
# 1. Install the ROCm build of JAX (NOT jax[cuda12]).
#    First-class ROCm entry point (uses the instance's system ROCm):
pip install "jax[rocm7-local]"
#    (older/newer ROCm: use the matching extra, or the ROCm JAX package index —
#     see https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/jax-install.html)

# 2. Install framework deps (does NOT re-install jax):
pip install -r requirements.txt

# 3. Verify:
bash scripts/00_verify_rocm.sh
```

### Choosing the base image / JAX ROCm version

Check the ROCm version on the instance:

```bash
cat /opt/rocm/.info/version    # or: rocm-smi --showdriverversion
```

Then pick a matching `rocm/jax-community` tag from
<https://hub.docker.com/r/rocm/jax-community/tags> (Option A), or the matching
`jax[rocmN-local]` extra (Option B). **Getting this version match right is the
single most important setup step** — a mismatch is what makes JAX silently fall
back to CPU.

---

## Reproduce the results (step by step)

```bash
# 0. Verify the AMD GPU toolchain (must print RocmDevice / gpu and pass all checks)
bash scripts/00_verify_rocm.sh

# 1. Smoke test the pipeline end-to-end (~minutes, tiny run)
SMOKE=1 bash scripts/01_train.sh

# 2. Full training on the AMD GPU (Go1 quadruped, Playground-tuned 200M steps)
bash scripts/01_train.sh
#    -> outputs/checkpoints/Go1JoystickFlatTerrain_seed0.pkl
#    -> outputs/logs/..._metrics.json  (reward curve for the report)

# 3. Closed-loop evaluation (reports reward / episode length)
bash scripts/02_eval.sh
#    -> outputs/logs/..._eval.json

# 4. Render the demo video
bash scripts/03_record_video.sh
#    -> outputs/videos/Go1JoystickFlatTerrain_seed0.mp4
```

### Switch to the humanoid (Unitree G1)

```bash
ENV=G1JoystickFlatTerrain bash scripts/01_train.sh
ENV=G1JoystickFlatTerrain bash scripts/02_eval.sh
ENV=G1JoystickFlatTerrain bash scripts/03_record_video.sh
```

Other confirmed envs: `Go1JoystickRoughTerrain`, `BerkeleyHumanoidJoystickFlatTerrain`,
`SpotJoystick`, `BarkourJoystick`.

---

## Dependency specification

- **JAX**: the **ROCm build** (installed out-of-band; NOT pinned in
  `requirements.txt` because the correct wheel is ROCm-version specific).
- Framework deps: see `requirements.txt` (mujoco, mujoco-mjx, brax,
  mujoco-playground, mediapy, ...).
- Python ≥ 3.10.

## Expected results

- `00_verify_rocm.sh`: all 6 checks pass; devices list shows `RocmDevice`.
- Training: `eval/episode_reward` rises over training; a Go1 flat-terrain
  joystick policy learns stable velocity-tracking gait.
- Eval: positive mean episode reward, episodes reach near the max length
  (robot stays upright / tracks commands).
- Video: an mp4 of the robot walking under commanded velocity.

## Troubleshooting

- **`assert_gpu` raised / backend is CPU**: the ROCm JAX build isn't active.
  Try `unset LD_LIBRARY_PATH`, confirm the JAX ROCm version matches the
  instance ROCm, and re-run `scripts/00_verify_rocm.sh`.
- **Rendering errors / no display**: ensure `MUJOCO_GL=osmesa` (scripts set
  this) and that `libosmesa6` is installed (the Docker image installs it).
- For local development without a GPU, pass `--allow-cpu` to the Python
  entrypoints (dry run only — do NOT use for submission numbers).
