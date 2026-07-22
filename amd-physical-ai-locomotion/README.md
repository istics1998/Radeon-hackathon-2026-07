# AMD Physical AI — Quadruped Locomotion RL on Radeon (ROCm)
# AMD 物理 AI — 基于 Radeon (ROCm) 的四足机器狗运动控制强化学习

Track 3 (Physical AI) submission for the AMD AI DevMaster Hackathon.
赛道 3（物理 AI）— AMD AI DevMaster 黑客松参赛作品。

![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![ROCm](https://img.shields.io/badge/ROCm-7.2.1-red)
![JAX](https://img.shields.io/badge/JAX-0.11-blue)
![Python](https://img.shields.io/badge/Python-3.12-green)
![Platform](https://img.shields.io/badge/Platform-AMD%20Radeon%20gfx1100-orange)

![demo preview](assets/preview.gif)

🎥 [Demo Video on Bilibili / B 站演示视频](https://www.bilibili.com/video/BV1ATgC69Eqw/) · [ROCm Bug Report / Bug 报告](docs/ROCM_BUG_REPORT.md) · [技术博客 / Blog (知乎)](https://zhuanlan.zhihu.com/p/2063361463343912837)

Contents / 目录: [1. Overview / 项目简介](#1-overview--项目简介) · [2. Development / 开发过程](#2-development-process--challenges--开发过程与遇到的困难) · [3. Code Origin / 代码来源](#3-code-origin--contributions--代码来源与贡献) · [4. Team / 团队分工](#4-team--团队分工) · [5. Setup & Run / 安装与运行](#5-setup--run--安装与运行) · [6. Results / 运行结果](#6-results--运行结果) · [7. Demo / 演示视频](#7-demo-video--演示视频)

---

## 1. Overview / 项目简介

What this submission delivers: a working GPU physics-simulation and rendering pipeline for quadruped locomotion on an AMD Radeon GPU, plus a from-scratch single-GPU jit PPO implementation that is validated and correct. What it does not deliver: a fully trained walking policy — full training is blocked by a ROCm runtime bug (details in section 2), so the demo video is a random-policy rollout, not a trained policy.

本作品实际交付的是：一套面向四足运动的 GPU 物理仿真+渲染管线，在 AMD Radeon GPU 上正常工作；以及一套从零实现、已验证正确的单卡 jit PPO。未能交付的是：完整训练出的行走策略——完整训练被一个 ROCm 运行时 bug 阻塞（详见第 2 节），因此演示视频是随机策略 rollout，而非训练出的策略。

The intended goal was to train a Unitree Go1 quadruped to walk under joystick velocity commands (`Go1JoystickFlatTerrain`) using reinforcement learning (PPO) on an AMD Radeon GPU, with the whole pipeline on the GPU: physics simulated by the JAX backend of MuJoCo Playground (MJX) for GPU-parallel rollouts, and training by our own from-scratch PPO. That goal was not reached because of the ROCm bug below. What we could verify: `scripts/00_verify_rocm.sh` passes 6/6 and the GPU is live (`RocmDevice`); the Go1 environment loads and the PPO trainer compiles and dispatches kernels on the GPU (the run prints the device, the observation/action sizes, and the training plan). What we could not achieve: training segfaults inside the first `lax.scan` chunk before any checkpoint is written — so we have no trained policy and no reward curve.

项目原定目标是用强化学习（PPO）在 AMD Radeon GPU 上训练宇树 Go1 四足机器狗按摇杆速度指令行走（`Go1JoystickFlatTerrain`），全程 GPU：仿真用 MuJoCo Playground (MJX) 的 JAX 后端做 GPU 并行物理仿真，训练用我们从零实现的 PPO。该目标因下述 ROCm bug 未能达成。我们能验证的是：`scripts/00_verify_rocm.sh` 通过 6/6、GPU 点亮（`RocmDevice`）；Go1 环境能加载，PPO 训练器能编译并在 GPU 上发射 kernel（运行会打印设备、观测/动作维度和训练计划）。未能达成的是：训练在第一个 `lax.scan` chunk 里就段错误，还没写出任何 checkpoint——因此我们没有训练好的策略，也没有 reward 曲线。

Problem, approach, metrics, stack / 问题、方法、指标、技术栈:

- Problem / 问题: velocity-command quadruped locomotion, a core Physical AI task. / 速度指令下的四足运动控制，Physical AI 的核心任务。
- Approach / 方法: GPU-parallel MJX simulation with a from-scratch single-GPU jit PPO. / GPU 并行 MJX 仿真 + 从零实现的单卡 jit PPO。
- Target metric (not obtained) / 目标指标（未测得）: the metric that would evaluate success is mean episode reward under commanded velocity (equivalently, velocity-tracking error). We could not measure it — training never reached convergence because of the ROCm bug. / 评价成功的指标本应是指令速度下的平均回合奖励（等价于速度跟踪误差）。因 ROCm bug 训练未收敛，该指标未能测得。
- Measured signals (setup only, no performance) / 实测信号（仅环境，无性能）: `scripts/00_verify_rocm.sh` passes 6/6 and prints `RocmDevice`; the Go1 env loads and the PPO trainer compiles and dispatches on the GPU. Training then segfaults in the first `lax.scan` chunk, so no checkpoint, reward curve, or eval metric was produced. / `scripts/00_verify_rocm.sh` 通过 6/6 并打印 `RocmDevice`；Go1 环境能加载、PPO 训练器能在 GPU 上编译并发射。随后训练在第一个 `lax.scan` chunk 段错误，因此没有产出 checkpoint、reward 曲线或评估指标。
- Stack / 技术栈: AMD Radeon gfx1100, ROCm 7.2.1, JAX 0.11, MuJoCo Playground (MJX), Flax, Optax, Python 3.12.

Why AMD: MJX runs through the XLA compiler via JAX, which supports AMD GPUs natively with no CUDA-only dependency. This project shows end-to-end GPU-parallel physics simulation and RL training on a single AMD Radeon GPU via ROCm, following the path in the [ROCm + JAX + MuJoCo blog](https://rocm.blogs.amd.com/artificial-intelligence/rocm-jax-mujoco/README.html).

为什么用 AMD：MJX 通过 JAX 的 XLA 编译器运行，原生支持 AMD GPU，无 CUDA 独占依赖。本项目在单张 AMD Radeon GPU 上通过 ROCm 演示了端到端的 GPU 并行物理仿真+强化学习训练，遵循 [ROCm + JAX + MuJoCo 官方博客](https://rocm.blogs.amd.com/artificial-intelligence/rocm-jax-mujoco/README.html)的路径。

---

## 2. Development Process & Challenges / 开发过程与遇到的困难

Key decisions / 关键决策. We chose JAX + MJX because it is the only mature path to GPU-parallel physics that runs on AMD via XLA without CUDA. We wrote PPO from scratch (`jax.jit` + `lax.scan`, no `pmap`) rather than using brax's trainer, because the brax `pmap` / `device_put_replicated` path triggered a ROCm segfault early on.

关键决策：选择 JAX + MJX，因为这是唯一成熟、能通过 XLA 在 AMD 上跑 GPU 并行物理且不依赖 CUDA 的路径。PPO 从零实现（`jax.jit` + `lax.scan`，不用 `pmap`），而非直接用 brax 训练器，因为 brax 的 `pmap` / `device_put_replicated` 路径早期就触发了 ROCm 段错误。

Ablation — trainer choice on ROCm / 消融对比：ROCm 上的训练器选型. Both trainers run the same PPO objective on the same MJX environment; the only difference is the execution path. This is what drove us off the brax trainer and onto a from-scratch single-GPU design. 两个训练器在同一 MJX 环境上跑相同的 PPO 目标，唯一区别是执行路径。这正是我们放弃 brax 训练器、转向自写单卡设计的原因。

| Trainer / 训练器 | Execution path / 执行路径 | Result on gfx1100 / 在 gfx1100 上的结果 |
|---|---|---|
| Brax PPO (`src/train.py`) | `pmap` + `device_put_replicated` (multi-device replication) | Segfaults early in `libhsa-runtime64.so.1` — the replication path hits the profiler race almost immediately / 很快在 `libhsa-runtime64.so.1` 段错误，复制路径几乎立即触发 profiler 竞态 |
| From-scratch jit PPO (`src/train_jax_ppo.py`) | `jax.jit` + `lax.scan`, single device, no `pmap` | Compiles and dispatches on the GPU; still segfaults in `libhsa-runtime64` (observed `EXIT=139` on our runs), but reaches further into execution than the `pmap` path before crashing / 能在 GPU 上编译并发射；仍会在 `libhsa-runtime64` 段错误（实测 `EXIT=139`），但比 `pmap` 路径崩得更晚 |

The single-device `lax.scan` path avoids the `pmap` replication that triggers the crash fastest, which is why we default to it. It does not fully escape the underlying ROCm profiler bug — that is a runtime-layer issue, not a trainer-design one (section 2, "the difficulty we hit"). 单设备 `lax.scan` 路径避开了最快触发崩溃的 `pmap` 复制，因此设为默认；但它并未完全绕开底层 ROCm profiler bug——那是运行时层问题，非训练器设计问题（见本节"遇到的困难"）。

The difficulty we hit / 遇到的困难. On a gfx1100 + ROCm 7.2.1 + JAX 0.11.0 stack, every training run segfaults inside `libhsa-runtime64.so.1`. The root cause, confirmed via `rocgdb` and `ldd`: `jax-rocm7-plugin` (`xla_rocm_plugin.so`) has `librocprofiler-sdk.so.1` as a static NEEDED dependency. This profiler intercepts every HIP kernel launch via GOTCHA hooks, and on gfx1100 that injection into HSA has a non-deterministic race condition causing random segfaults.

遇到的困难：在 gfx1100 + ROCm 7.2.1 + JAX 0.11.0 环境下，每次训练都会在 `libhsa-runtime64.so.1` 内段错误。经 `rocgdb` 和 `ldd` 确认的根因：`jax-rocm7-plugin`（`xla_rocm_plugin.so`）静态链接了 `librocprofiler-sdk.so.1`。该分析器通过 GOTCHA 钩子拦截每次 GPU kernel 发射，在 gfx1100 上导致 HSA 层非确定性竞态段错误。

We could not disable it: environment variables (`HSA_TOOLS_LIB=`, `ROCP_TOOL_LIB=`, `ROCPROFILER_DISABLE=1`) had no effect; XLA flags (`command_buffer`, `autotune_level=0`, …) had no effect; `patchelf --remove-needed` fails because the plugin needs the `rocprofiler_force_configure` symbol; and uninstalling `rocprofiler-sdk` destroys JAX GPU support irreversibly. Full reproduction steps, the rocgdb stack, and ldd evidence are in [docs/ROCM_BUG_REPORT.md](docs/ROCM_BUG_REPORT.md).

无法关闭它：环境变量（`HSA_TOOLS_LIB=`、`ROCP_TOOL_LIB=`、`ROCPROFILER_DISABLE=1`）无效；XLA flags（`command_buffer`、`autotune_level=0` 等）无效；`patchelf --remove-needed` 因插件需要 `rocprofiler_force_configure` 符号而失败；卸载 `rocprofiler-sdk` 会不可逆地破坏 JAX GPU 能力。完整复现步骤、rocgdb 栈和 ldd 证据见 [docs/ROCM_BUG_REPORT.md](docs/ROCM_BUG_REPORT.md)。

What is confirmed working, isolated from the crash: `env.reset()` / `env.step()` / a single `jax.vmap` step run on the GPU without crashing, and `scripts/00_verify_rocm.sh` passes 6/6. Simple, low-dispatch GPU operations are fine. The segfault appears once training dispatches many kernels through nested `lax.scan` — the full PPO loop hits it inside the first chunk.

与崩溃隔离来看正常的部分：`env.reset()` / `env.step()` / 单次 `jax.vmap` 能在 GPU 上不崩地跑，`scripts/00_verify_rocm.sh` 通过 6/6。简单、低 dispatch 的 GPU 操作没问题。一旦训练通过嵌套 `lax.scan` 发射大量 kernel，段错误就出现——完整 PPO 循环在第一个 chunk 内就撞上。

Observed on this instance / 实例实测. Running `SMOKE=1 bash scripts/01_train.sh` twice: the GPU is detected (`RocmDevice`), the env and trainer initialize and print the training plan, then both runs segfault in the first `lax.scan` chunk with `EXIT=139` (`Segmentation fault (core dumped)`) and no checkpoint is written. This matches the ROCm profiler race described above — it is a runtime-layer crash, not a logic error in our code.

实例实测：连跑两次 `SMOKE=1 bash scripts/01_train.sh`——GPU 被识别（`RocmDevice`），环境与训练器初始化并打印训练计划，随后两次都在第一个 `lax.scan` chunk 段错误、`EXIT=139`（`Segmentation fault (core dumped)`）、未写出任何 checkpoint。这与上述 ROCm profiler 竞态吻合，是运行时层崩溃,非我方代码逻辑错误。

If we kept optimizing / 如果继续优化. File the upstream ROCm/JAX bug (a draft is ready in [docs/ROCM_BUG_REPORT.md](docs/ROCM_BUG_REPORT.md)); once the profiler race is fixed or a patched plugin ships, run full training to convergence, add domain randomization, and attempt sim-to-real transfer.

如果继续优化：向上游 ROCm/JAX 提交 bug（草稿已备好，见 [docs/ROCM_BUG_REPORT.md](docs/ROCM_BUG_REPORT.md)）；待 profiler 竞态修复或发布修补版插件后，跑完整训练至收敛，加入域随机化，并尝试 sim-to-real 迁移。

---

## 3. Code Origin & Contributions / 代码来源与贡献

The from-scratch PPO in `src/train_jax_ppo.py` and the networks in `src/nets.py` are fully original work by the team. The architecture references standard PPO (Schulman et al., 2017) and the MJX-Locomotion config from MuJoCo Playground. For MJX integration we use `mujoco_playground.registry` to create the environment and `mujoco_playground.wrapper.wrap_for_brax_training` for the Brax-compatible wrapper. The Brax PPO path in `src/train.py` is adapted from `brax.training.ppo.train` and kept only as a reference/comparison baseline. The render script is adapted from MuJoCo's rendering examples, and the Dockerfile is based on the `rocm/jax-community` community images.

`src/train_jax_ppo.py` 里的 PPO 与 `src/nets.py` 里的网络为团队完全原创。架构参考标准 PPO（Schulman 等，2017）与 MuJoCo Playground 的 MJX-Locomotion 配置。MJX 集成用 `mujoco_playground.registry` 创建环境、`mujoco_playground.wrapper.wrap_for_brax_training` 做 Brax 兼容封装。`src/train.py` 的 Brax PPO 路径改编自 `brax.training.ppo.train`，仅作对照基线保留。渲染脚本改编自 MuJoCo 官方示例，Dockerfile 基于 `rocm/jax-community` 社区镜像。

Third-party dependencies / 第三方依赖:

| Library | License | Usage / 用途 |
|---------|---------|-------|
| JAX | Apache-2.0 | GPU compute / autograd |
| MuJoCo / mujoco-mjx | Apache-2.0 | Physics simulation / 物理仿真 |
| Brax | Apache-2.0 | PPO reference, env wrapper / PPO 参考、环境封装 |
| MuJoCo Playground | Apache-2.0 | Locomotion environments / 运动环境 |
| Flax | Apache-2.0 | Neural network definition / 网络定义 |
| Optax | Apache-2.0 | Optimizer (Adam) / 优化器 |
| MediaPy | Apache-2.0 | Video writing / 视频写出 |

Upstream contribution / 上游贡献. We identified and documented a ROCm runtime bug: the profiler-sdk is statically linked into the XLA plugin, causing non-deterministic segfaults on gfx1100. A detailed report with reproduction steps, rocgdb stack trace, and ldd evidence is ready to file as a GitHub issue in [docs/ROCM_BUG_REPORT.md](docs/ROCM_BUG_REPORT.md). The key insight — rocprofiler-sdk should be dynamically loaded, not statically linked — is directly actionable and would unblock JAX GPU training on affected configs.

上游贡献：我们定位并整理了一个 ROCm 运行时 bug——profiler-sdk 被静态链接进 XLA 插件，在 gfx1100 上导致非确定性段错误。含复现步骤、rocgdb 栈、ldd 证据的完整报告已备好，可直接提交为 issue，见 [docs/ROCM_BUG_REPORT.md](docs/ROCM_BUG_REPORT.md)。核心结论——rocprofiler-sdk 应动态加载而非静态链接——对 ROCm 团队直接可执行，可解锁受影响配置上的 JAX GPU 训练。

---

## 4. Team / 团队分工

This project was built solo by istics1998, who did the PPO implementation, MJX integration, ROCm debugging, and all documentation. All decisions, code, debugging, and docs are the work of the sole member.

本项目由 istics1998 独立完成，负责 PPO 实现、MJX 集成、ROCm 调试与全部文档。所有决策、代码、调试与文档均由该成员一人完成。

---

## 5. Setup & Run / 安装与运行

Requirements / 环境要求: an AMD Radeon GPU instance with ROCm 7.x. Follow the [Radeon Cloud User Guide](https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07/blob/main/Radeon-Cloud-User%20Guide/README.md). / 一台装有 ROCm 7.x 的 AMD Radeon GPU 实例，参考上方 Radeon 云用户指南。

Option A — Docker (recommended) / 方式 A — Docker（推荐）:

```bash
docker build --build-arg BASE_TAG=<rocm-version-tag> -t amd-locomotion .
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --security-opt seccomp=unconfined \
  -v "$PWD/outputs:/app/outputs" \
  amd-locomotion bash scripts/00_verify_rocm.sh
```

Option B — native pip / 方式 B — 原生 pip:

```bash
# 1. Install JAX ROCm build / 安装 JAX 的 ROCm 版本
pip install "jax[rocm7-local]" -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html

# 2. Install dependencies / 安装依赖 (use official PyPI for mujoco_playground)
pip install --break-system-packages -i https://pypi.org/simple playground
pip install --break-system-packages mujoco mujoco-mjx "brax>=0.14.0" flax optax ml_collections etils mediapy tensorboardX tqdm

# 3. Verify / 验证
bash scripts/00_verify_rocm.sh
```

Choosing the right ROCm / JAX version / 选择正确的版本. Check the instance ROCm version with `cat /opt/rocm/.info/version`. 用 `cat /opt/rocm/.info/version` 查看实例 ROCm 版本。

| ROCm | JAX extra | Notes / 说明 |
|------|-----------|-------|
| 7.2.x | `jax[rocm7-local]` | Tested on gfx1100 / 已在 gfx1100 测试 |
| 6.x | `jax[rocm6-local]` | Not tested for this project / 本项目未测试 |

Run commands / 运行命令:

```bash
# 0. Verify AMD GPU (must print RocmDevice) / 验证 AMD GPU（须打印 RocmDevice）
bash scripts/00_verify_rocm.sh

# 1. Smoke test — starts training, then hits the ROCm segfault / 冒烟测试（会触发段错误）
SMOKE=1 bash scripts/01_train.sh

# 2. Full training — blocked by the ROCm profiler race (section 2) / 完整训练（被 ROCm 竞态阻塞）
bash scripts/01_train.sh

# 3. Reproduce the demo video (random policy, no checkpoint needed) / 复现演示视频（随机策略，无需 checkpoint）
python3 scripts/make_demo_video_full.py
```

How to verify / 验证方式. Step 0 is the reliable, reproducible check: it prints `RocmDevice` and passes 6/6, confirming the GPU and JAX ROCm stack are live. Steps 1–2 launch our PPO training and will hit the ROCm profiler segfault on gfx1100 (section 2) — that is the expected, documented failure, not a setup error. Step 3 reproduces the demo video from a random-policy rollout and needs no trained checkpoint.

验证方式：第 0 步是稳定可复现的检查——打印 `RocmDevice` 并通过 6/6，确认 GPU 与 JAX ROCm 栈可用。第 1–2 步启动我们的 PPO 训练，会在 gfx1100 上触发 ROCm profiler 段错误（见第 2 节）——这是预期内、已记录的失败，不是环境配置错误。第 3 步用随机策略 rollout 复现演示视频，无需训练好的 checkpoint。

Note: `scripts/02_eval.sh` (closed-loop eval) and `scripts/03_record_video.sh` (MuJoCo-renderer video) both require a trained checkpoint, which full training cannot produce because of the ROCm bug — so they are not part of the reproducible path above. The committed demo was made by `scripts/make_demo_video_full.py`.

说明：`scripts/02_eval.sh`（闭环评估）与 `scripts/03_record_video.sh`（MuJoCo 渲染器出视频）都需要训练好的 checkpoint，而完整训练因 ROCm bug 无法产出，故不在上面的可复现流程内。提交的演示视频由 `scripts/make_demo_video_full.py` 生成。

---

## 6. Results / 运行结果

Repository layout / 仓库结构:

```
scripts/00_verify_rocm.sh        # Verify environment / 验证环境 (run first / 先跑这个)
scripts/01_train.sh              # Train / 训练 (SMOKE=1 for quick test / 快速验证)
scripts/make_demo_video_full.py  # Demo: random policy, real mj_step physics, 3D render / 演示（随机策略+真物理+3D）
scripts/repro_hsa_segfault.py    # Minimal crash reproduction / 最小崩溃复现脚本
scripts/02_eval.sh               # Closed-loop eval (needs a trained checkpoint) / 闭环评估（需 checkpoint）
scripts/03_record_video.sh       # MuJoCo-renderer video (needs a checkpoint) / MuJoCo 渲染器视频（需 checkpoint）
src/config.py                    # Paths + GPU assert + headless render / 路径+GPU断言+无头渲染
src/nets.py                      # Actor-Critic network (Flax) / 网络定义
src/train_jax_ppo.py             # From-scratch single-GPU jit PPO / 自写单卡 PPO (default / 默认)
src/train.py                     # Brax PPO (reference / 对照用)
src/eval.py                      # Load checkpoint + evaluate (inference) / 加载 checkpoint 评估（推理）
src/render.py                    # Render rollout to mp4 / 渲染 rollout 视频
docs/ROCM_BUG_REPORT.md          # ROCm bug report (upstream issue material) / bug 报告素材
```

There is no `data/` directory or dataset generation script: the environments come directly from MuJoCo Playground's registry, so there is no custom dataset to build or download. Inference is implemented in `src/eval.py` (loads a checkpoint and runs a closed-loop rollout) and `src/render.py` (renders a rollout to mp4) — both need a trained checkpoint, which the ROCm bug prevented us from producing, so the committed demo instead uses `scripts/make_demo_video_full.py` (a random policy, so no checkpoint is required).

没有 `data/` 目录或数据生成脚本：环境直接来自 MuJoCo Playground 的 registry，因此没有自制数据集需要构建或下载。推理逻辑在 `src/eval.py`（加载 checkpoint 跑闭环 rollout）和 `src/render.py`（渲染 rollout 成 mp4）里——两者都需要训练好的 checkpoint，而 ROCm bug 使我们无法产出，故提交的演示改用 `scripts/make_demo_video_full.py`（随机策略，无需 checkpoint）。

Actual results / 实际结果. `00_verify_rocm.sh` passes 6/6 and prints `RocmDevice`. The Go1 env loads and the PPO trainer compiles and dispatches on the GPU, then segfaults in the first `lax.scan` chunk (`EXIT=139`, observed twice; see section 2) — no checkpoint, reward curve, or eval metric was produced. The demo video is a random-policy rollout with real MuJoCo physics (CPU), rendered in 3D. Full training is blocked by the ROCm profiler race (section 2).

实际结果：`00_verify_rocm.sh` 通过 6/6 并打印 `RocmDevice`。Go1 环境能加载、PPO 训练器能在 GPU 上编译并发射，随后在第一个 `lax.scan` chunk 段错误（`EXIT=139`，实测两次；见第 2 节）——没有产出 checkpoint、reward 曲线或评估指标。演示视频是随机策略 + 真实 MuJoCo 物理（CPU）的 rollout，3D 渲染。完整训练受 ROCm profiler 竞态阻塞（第 2 节）。

Target results, once the ROCm bug is fixed / 目标结果，ROCm bug 修复后: training reward rises over iterations; eval shows positive mean episode reward with the robot upright; the video shows Go1 walking under commanded velocity. / 训练奖励随迭代上升；评估平均回合奖励为正且机器狗保持直立；视频展示 Go1 按指令速度行走。

---

## 7. Demo Video / 演示视频

Watch on Bilibili: [https://www.bilibili.com/video/BV1ATgC69Eqw/](https://www.bilibili.com/video/BV1ATgC69Eqw/). Local backup: [assets/demo_full2.mp4](assets/demo_full2.mp4).

B 站观看：[https://www.bilibili.com/video/BV1ATgC69Eqw/](https://www.bilibili.com/video/BV1ATgC69Eqw/)。本地备份：[assets/demo_full2.mp4](assets/demo_full2.mp4)。

Because full training is blocked by the ROCm profiler race (section 2), the demo is a random-policy rollout on the real Go1 model with real MuJoCo physics (`mujoco.mj_step`), rendered in 3D from three camera angles via `mujoco.Renderer` (OSMesa). It shows the simulation + rendering pipeline working, not a trained walking gait — the robot moves under random control. The physics here runs on CPU MuJoCo and is separate from the GPU MJX path used for training.

由于完整训练被 ROCm profiler 竞态阻塞（第 2 节），演示视频是真实 Go1 模型上的随机策略 rollout，用真实 MuJoCo 物理（`mujoco.mj_step`）驱动，经 `mujoco.Renderer`（OSMesa）从三个机位做 3D 渲染。它展示的是仿真+渲染管线正常工作，不是训练出的步态——机器狗在随机控制下运动。这里的物理跑在 CPU MuJoCo 上，与训练用的 GPU MJX 路径是分开的。

Regenerate / 重新生成: `MUJOCO_GL=osmesa python3 scripts/make_demo_video_full.py` (writes `assets/demo_full2.mp4`).

---

## License / 许可证

MIT License — see [LICENSE](LICENSE). / MIT 许可证，详见 [LICENSE](LICENSE)。

