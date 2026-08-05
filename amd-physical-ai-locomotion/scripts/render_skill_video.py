#!/usr/bin/env python3
"""Render a demo clip driven by a TRAINED skill policy (Footstand / Handstand).

Skill envs have no joystick command (obs["state"] is 45-dim), so there is no
command to inject — we just reset and let the learned policy hold the pose for
the whole clip. Diagnostics report trunk height and body pitch (the angle the
torso lifts to), since "did it stand up?" is the pitch, not the ground path.

Reuses render_3d / title_card / concat from make_demo_video_full so the style
matches the locomotion demo. Runs on CPU (AMD GPU blocked by ROCm profiler
race, see docs/ROCM_BUG_REPORT.md).

Usage (on the instance, after a skill checkpoint exists):
    cd /workspace/demo
    SKILL_ENV=Go1Footstand MUJOCO_GL=egl python3 scripts/render_skill_video.py
    SKILL_ENV=Go1Handstand MUJOCO_GL=egl python3 scripts/render_skill_video.py checkpoints_go1handstand/latest.pkl
"""
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import jax
import subprocess

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

import make_demo_video_full as demo

ENV = os.environ.get("SKILL_ENV", "Go1Footstand")
LABELS = {"Go1Footstand": "Rear-leg stand (Footstand)",
          "Go1Handstand": "Front-leg handstand",
          "Go1Getup": "Get up after a fall"}
CKPT = (Path(sys.argv[1]) if len(sys.argv) > 1
        else Path(f"checkpoints_{ENV.lower()}/latest.pkl"))
OUT = Path(f"assets/demo_{ENV.lower()}.mp4")
NSTEPS = int(os.environ.get("SKILL_STEPS", 500))


def load_policy(env, ckpt_path):
    """Rebuild the PPO net with the SAME normalization used in training.

    Skill training also uses normalize_observations=True, so params[0] is a
    RunningStatisticsState; the net must preprocess obs the same way or the
    policy sees raw-scale obs and collapses (see render_policy_video.py notes).
    """
    ppo_params = locomotion_params.brax_ppo_config(ENV)
    nf = dict(ppo_params).get("network_factory", {})
    net = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize, **nf)
    make_policy = ppo_networks.make_inference_fn(net)
    with open(ckpt_path, "rb") as f:
        params = pickle.load(f)
    return make_policy(params, deterministic=True)


def _pitch(qpos):
    # body pitch from the trunk quaternion (rotation about the y-axis): how far
    # the torso tips from horizontal. ~90deg means fully stood up on two legs.
    w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
    return np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))


def rollout(env, policy, nsteps):
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    inference = jax.jit(policy)
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    snaps, heights, pitches = [], [], []
    for i in range(nsteps):
        act_rng, rng = jax.random.split(rng)
        act, _ = inference(state.obs, act_rng)
        state = jit_step(state, act)
        qpos = np.asarray(state.data.qpos)
        snaps.append((qpos.copy(), None))
        heights.append(float(qpos[2]))
        pitches.append(float(_pitch(qpos)))
    # report the settled pose: mean over the LAST 40% (after it has stood up)
    tail = slice(int(0.6 * nsteps), None)
    st = dict(h_final=float(np.mean(heights[tail])),
              h_max=float(np.max(heights)),
              pitch_final=float(np.mean(pitches[tail])),
              pitch_max=float(np.max(np.abs(pitches))))
    return snaps, st


def main():
    if not CKPT.exists():
        sys.exit(f"[render] checkpoint not found: {CKPT}\n"
                 f"         train first (SKILL_ENV={ENV}), or pass a .pkl path")
    print(f"[render] env={ENV} checkpoint: {CKPT}")

    cfg = registry.get_default_config(ENV)
    cfg.impl = "jax"
    env = registry.load(ENV, config=cfg)
    model = env.mj_model
    demo._hud = _patched_hud

    policy = load_policy(env, CKPT)
    label = LABELS.get(ENV, ENV)
    print(f"[render] rolling out '{label}' — {NSTEPS} steps ...")
    snaps, st = rollout(env, policy, NSTEPS)
    print(f"         trunk_h final={st['h_final']:.3f}m max={st['h_max']:.3f}m")
    print(f"         body_pitch final={st['pitch_final']:+.0f}deg "
          f"max|pitch|={st['pitch_max']:.0f}deg  (~90deg = fully upright)")

    # tracked camera, side-on (azimuth=90) so the vertical pose is clearly
    # visible; render_3d fixes elevation at -18 which shows the pitch well.
    frames = demo.render_3d(model, snaps, label, track=True, azimuth=90,
                            distance=3.0)
    seg = demo.save_segment(frames, "skill_" + ENV.lower())

    flist = [
        demo.title_card(
            f"Unitree Go1 — TRAINED skill policy\n{label} · real MuJoCo physics · CPU", 4),
        demo.title_card(label, 2),
        seg,
        demo.title_card(
            "GPU training blocked by a ROCm profiler race\nTrained on CPU — see docs/ROCM_BUG_REPORT.md", 4),
    ]
    tmp_txt = Path(f"/tmp/concat_{ENV.lower()}.txt")
    tmp_txt.write_text("".join(f"file '{p}'\n" for p in flist))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_bin = "ffmpeg"
    subprocess.run([ffmpeg_bin, "-y", "-f", "concat", "-safe", "0",
                    "-i", str(tmp_txt), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(OUT)], check=True)
    print(f"✅ {OUT}  ({OUT.stat().st_size // 1024} KB)")


def _patched_hud(img_arr, label, note=None):
    from PIL import Image, ImageDraw
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    f = demo._font(15)
    draw.text((10, 10), f"Unitree Go1 - trained skill policy, real MuJoCo physics - {label}",
              fill=(20, 20, 60), font=f)
    if note:
        draw.text((10, 32), note, fill=(120, 60, 60), font=f)
    return np.array(img)


if __name__ == "__main__":
    main()
