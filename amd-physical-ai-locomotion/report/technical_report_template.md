# Technical Report — <Application Name>

> AMD AI DevMaster Hackathon · Track 3 (Physical AI)
> Team: **<Team Name>** · Submission title: `Track 3, <Team Name>, <Application Name>`
>
> Fill in every `<...>`. All submission materials MUST be in English.

## 1. Target application — definition & description
<What robot, what task, why it matters. E.g. "Velocity-tracking locomotion
controller for the Unitree Go1 quadruped, for autonomous inspection/logistics
in unstructured environments.">

## 2. System architecture & solution design
- Simulation: MuJoCo Playground (MJX) — GPU-parallel physics via JAX/XLA.
- Learning: Brax PPO, `<N>` parallel envs, policy net `<sizes>`.
- Closed loop: proprioceptive obs → policy → joint targets → PD control.
- Robustness: domain randomization (`registry.get_domain_randomizer`).
<Add an architecture diagram: sim env → PPO trainer → checkpoint → closed-loop eval → deployment/video.>

## 3. Dataset(s) used for training / evaluation
<RL from simulation — no external dataset, or list any assets/motion data.
Confirm licensing/ethics compliance per the rules.>

## 4. AMD Radeon GPU usage (training, inference, other stages)
- GPU model: `<e.g. Radeon ...>` · ROCm version: `<...>` · JAX ROCm build: `<...>`
- **Training** ran on the GPU: paste `scripts/00_verify_rocm.sh` output showing
  `RocmDevice`, and the `[device] Using AMD GPU` line from training.
- **Inference/eval** ran on the GPU: same, from `src/eval.py`.
- Throughput: `<env steps/s or wall-clock for N steps>` on the Radeon instance.
- Any ROCm-specific tuning: `<XLA flags, batch/env sizes, osmesa render, ...>`

## 5. Innovation & key technical contributions
<GPU-parallel MJX rollouts on AMD; robustness via domain randomization;
provable GPU execution guard; reproducible Docker build; ... what's novel.>

## 6. Final deliverables & output form
- Source repo (this repo) · Docker image · reproducible README.
- Trained checkpoint(s) · training/eval metrics · demo video (3–5 min).

## 7. Upstream open-source contribution (10 pts)
<Link the PR(s). Prefer AMD-platform support: ROCm install docs for
mujoco_playground / brax / jax, or a fix for a ROCm compatibility issue found in
Phase 0. Describe the problem, the fix, and the PR URL.>

## 8. Additional information
<Anything that highlights the strengths/uniqueness of the work.>

## 9. Team & contributions
| Member | Role | Contribution |
|---|---|---|
| `<name>` | `<role>` | `<what they did>` |

---

### Reproducibility checklist (align with README)
- [ ] `scripts/00_verify_rocm.sh` passes on the Radeon instance (RocmDevice)
- [ ] `SMOKE=1 bash scripts/01_train.sh` completes
- [ ] Full training checkpoint produced
- [ ] `scripts/02_eval.sh` reports metrics
- [ ] `scripts/03_record_video.sh` produces the demo mp4
- [ ] Docker build + run reproduces the above
- [ ] Upstream PR opened and linked
- [ ] All team members registered in the AMD AI Developer Program
