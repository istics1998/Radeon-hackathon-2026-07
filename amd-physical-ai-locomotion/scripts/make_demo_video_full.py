#!/usr/bin/env python3
"""
Demo video — RANDOM-POLICY rollout with REAL MuJoCo physics, rendered headless.

What this actually does (honest description):
  * loads the real Unitree Go1 model from mujoco_playground's registry;
  * samples smoothed RANDOM actions within the actuator control range and steps
    the TRUE MuJoCo dynamics (mujoco.mj_step) on CPU — a real physics rollout,
    not a scripted animation;
  * renders that one rollout from 3 camera angles (side / front / 3-4) and
    stitches title cards + segments into one short mp4.

This is a RANDOM policy (no trained checkpoint) on CPU MuJoCo physics. It shows
the simulation + rendering pipeline working — it is NOT a trained walking gait,
and the CPU physics here is separate from the GPU MJX path used for training.

Rendering: tries mujoco.Renderer (real 3D); if that is unavailable on this
Python/MuJoCo build it falls back to a 2D skeleton drawing driven by the SAME
real-physics trajectory.

Usage:
    MUJOCO_GL=osmesa python3 scripts/make_demo_video_full.py
    # or: MUJOCO_GL=egl python3 scripts/make_demo_video_full.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")

import subprocess
from pathlib import Path

import numpy as np
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont
from mujoco_playground import registry as reg

W, H = 640, 480
FPS = 30
NUM_STEPS = 200          # frames per camera view (~6.7s @ 30fps)
ACTION_SMOOTH = 0.85     # low-pass on random actions so it isn't pure white noise
SEED = 0
OUT = Path("assets/demo_bob.mp4")   # new bobbing-gait version; old one stays as demo_full2.mp4

# --- PIL fallback drawing constants (only used if mujoco.Renderer is unavailable)
SCALE = 200
LEG_BODIES = {
    'FR': ('FR_hip', 'FR_thigh', 'FR_calf'),
    'FL': ('FL_hip', 'FL_thigh', 'FL_calf'),
    'RR': ('RR_hip', 'RR_thigh', 'RR_calf'),
    'RL': ('RL_hip', 'RL_thigh', 'RL_calf'),
}
LEG_COLORS = {'FR': (200, 40, 40), 'FL': (40, 80, 200),
              'RR': (40, 180, 80), 'RL': (220, 130, 40)}

def _font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def rollout_real_physics(model, data, num_steps, rng, base_ctrl):
    """Step TRUE MuJoCo dynamics while driving a slow, gentle bob: the thigh/calf
    targets are modulated by a low-frequency sine so the robot rhythmically flexes
    and extends its legs (body rises and lowers), plus a tiny phase offset front vs
    back for a natural look. Real mj_step physics throughout; the motion is a scripted
    reference trajectory, not a trained policy. Returns mjData snapshots (copies).

    base_ctrl: the standing (home) position-control targets. Go1 leg joints per leg
    are [hip, thigh, calf]; extending a leg (thigh down, calf up) raises the body."""
    nu = model.nu
    lo, hi = model.actuator_ctrlrange[:, 0].copy(), model.actuator_ctrlrange[:, 1].copy()
    unlimited = lo >= hi
    lo = np.where(unlimited, -1.0, lo)
    hi = np.where(unlimited, 1.0, hi)

    A = 0.15                 # bob amplitude (rad) — modest, stays upright
    period = 60.0            # frames per cycle (~2s at 30fps)
    nlegs = nu // 3

    snaps = []
    for t in range(num_steps):
        ctrl = base_ctrl.copy()
        for leg in range(nlegs):
            # small front/back phase offset so it looks alive, not a rigid squat
            phase = 2.0 * np.pi * (t / period) + (0.0 if leg < 2 else np.pi * 0.25)
            s = A * np.sin(phase)
            thigh = 3 * leg + 1
            calf = 3 * leg + 2
            ctrl[thigh] = base_ctrl[thigh] - s          # thigh down -> extend
            ctrl[calf] = base_ctrl[calf] + 2.0 * s      # calf up   -> extend, foot stays under
        data.ctrl[:] = np.clip(ctrl, lo, hi)
        mujoco.mj_step(model, data)          # REAL dynamics integration
        snaps.append((data.qpos.copy(), data.xpos.copy()))
    return snaps


def render_3d(model, snaps, cam_name_or_id, label):
    """Render snapshots with the real mujoco.Renderer. Raises if unavailable."""
    frames = []
    renderer = mujoco.Renderer(model, height=H, width=W)
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 2.2
    cam.elevation = -18
    cam.azimuth = {"side": 90, "front": 0, "3q": 45}[label]
    for qpos, _ in snaps:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        frames.append(_hud(np.array(img), label))
    renderer.close()
    return frames


def render_pil(model, snaps, axis, label):
    """Fallback: 2D skeleton driven by the SAME real-physics trajectory."""
    n2id = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, j): j
            for j in range(model.nbody)}

    def project(pos):
        if axis == "front":
            ix = -pos[0] * SCALE + W // 2
        elif axis == "3q":
            ix = (pos[0] * 0.35 + pos[1] * 0.65) * SCALE + W // 2
        else:
            ix = pos[1] * SCALE + W // 2
        iy = H // 2 - pos[2] * SCALE
        return int(ix), int(iy)

    frames = []
    for _, xpos in snaps:
        img = Image.new("RGB", (W, H), (250, 250, 252))
        draw = ImageDraw.Draw(img)
        gy = H // 2
        draw.line([(0, gy), (W, gy)], fill=(60, 60, 60), width=2)
        for ln in ['FR', 'FL', 'RR', 'RL']:
            color = LEG_COLORS[ln]
            ids = [n2id[b] for b in LEG_BODIES[ln]]
            pts = [project(xpos[i]) for i in ids]
            draw.line([pts[0], pts[1]], fill=color, width=10)
            draw.line([pts[1], pts[2]], fill=color, width=8)
            fc = (220, 40, 40) if xpos[ids[2], 2] < 0.03 else color
            draw.ellipse([pts[2][0] - 8, pts[2][1] - 8, pts[2][0] + 8, pts[2][1] + 8], fill=fc)
        tp = project(xpos[n2id['trunk']])
        draw.rectangle([tp[0] - 80, tp[1] - 18, tp[0] + 80, tp[1] + 18],
                       fill=(20, 30, 50), outline=(0, 0, 0), width=2)
        frames.append(_hud(np.array(img), label, note="2D skeleton (renderer unavailable)"))
    return frames


def _hud(img_arr, label, note=None):
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    f = _font(15)
    view = {"side": "Side view", "front": "Front view", "3q": "3/4 view"}[label]
    draw.text((10, 10), f"Unitree Go1 — scripted bobbing gait, real MuJoCo physics — {view}",
              fill=(20, 20, 60), font=f)
    if note:
        draw.text((10, 32), note, fill=(120, 60, 60), font=f)
    return np.array(img)


def title_card(text, sec=3):
    frames = []
    font = _font(26)
    for _ in range(int(FPS * sec)):
        img = Image.new("RGB", (W, H), (40, 40, 60))
        draw = ImageDraw.Draw(img)
        lines = text.split("\n")
        y = (H - len(lines) * 40) // 2
        for line in lines:
            b = draw.textbbox((0, 0), line, font=font)
            draw.text(((W - b[2]) // 2, y), line, fill=(220, 220, 240), font=font)
            y += 40
        frames.append(np.array(img))
    out = Path(f"/tmp/title_{abs(hash(text))}.mp4")
    imageio.mimsave(str(out), np.stack(frames), fps=FPS, codec="libx264", quality=8)
    return out


def save_segment(frames, name):
    out = Path(f"/tmp/demo_seg_{name}.mp4")
    imageio.mimsave(str(out), np.stack(frames), fps=FPS,
                    codec="libx264", quality=8, pixelformat="yuv420p")
    print(f"  segment {name}: {len(frames)} frames -> {out} ({out.stat().st_size // 1024} KB)")
    return out


def main():
    rng = np.random.default_rng(SEED)
    print("[demo] loading Go1JoystickFlatTerrain ...")
    cfg = reg.get_default_config("Go1JoystickFlatTerrain")
    cfg.impl = "jax"
    env = reg.load("Go1JoystickFlatTerrain", config=cfg)
    model = env.mj_model

    # Standing start pose from the model's home keyframe (correct crouch + height),
    # then let real physics take over.
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)   # home keyframe = proper stance
    else:
        # fallback stance: trunk at standing height, legs in a bent crouch
        data.qpos[0:3] = [0, 0, 0.28]
        data.qpos[3:7] = [1, 0, 0, 0]
        # hip=0, thigh~0.9, calf~-1.8 per leg (typical Go1 stand)
        for leg in range(4):
            data.qpos[7 + leg * 3: 7 + leg * 3 + 3] = [0.0, 0.9, -1.8]
    mujoco.mj_forward(model, data)

    # base control = the standing joint targets. Prefer the home keyframe's ctrl;
    # if that is absent/all-zero, fall back to the standing joint angles (qpos).
    # (A zero base_ctrl would straighten the legs and make the robot collapse.)
    base_ctrl = None
    if model.nkey > 0 and getattr(model, "key_ctrl", None) is not None \
            and model.key_ctrl.shape[1] == model.nu:
        kc = model.key_ctrl[0].copy()
        if np.any(np.abs(kc) > 1e-6):
            base_ctrl = kc
    if base_ctrl is None:
        base_ctrl = data.qpos[7:7 + model.nu].copy()

    # Gravity-compensation feedforward. The actuators are pure P servos (kp=35, no
    # gravity comp), so commanding the exact standing angles yields ~zero torque and
    # the robot sags to a crouch. Fix: settle once, MEASURE the steady-state sag,
    # then add it back to the target. Since torque = kp*(ctrl - q) is linear, one
    # correction lands the equilibrium at the intended standing pose.
    nu = model.nu
    for _ in range(120):                       # settle at raw target -> sags
        data.ctrl[:] = base_ctrl
        mujoco.mj_step(model, data)
    sag = base_ctrl - data.qpos[7:7 + nu]      # steady-state position error
    base_ctrl = base_ctrl + sag                # feedforward: command extra extension
    lo = model.actuator_ctrlrange[:, 0]; hi = model.actuator_ctrlrange[:, 1]
    base_ctrl = np.clip(base_ctrl, lo, hi)
    for _ in range(120):                       # re-settle at compensated target -> stands
        data.ctrl[:] = base_ctrl
        mujoco.mj_step(model, data)
    print(f"[demo] gravity-comp done; trunk height now {float(data.qpos[2]):.3f} m (target ~0.278)")

    print(f"[demo] rolling out {NUM_STEPS} steps of REAL physics with a scripted bobbing gait ...")
    snaps = rollout_real_physics(model, data, NUM_STEPS, rng, base_ctrl)

    use_3d = True
    try:
        _ = mujoco.Renderer(model, height=8, width=8)  # probe
        _.close()
    except Exception as e:
        use_3d = False
        print(f"[demo] mujoco.Renderer unavailable ({e}); falling back to 2D skeleton.")

    segs = {}
    for label in ("side", "front", "3q"):
        print(f"[demo] rendering {label} view ...")
        if use_3d:
            frames = render_3d(model, snaps, None, label)
        else:
            frames = render_pil(model, snaps, label, label)
        segs[label] = save_segment(frames, label)

    cards = {
        "t0": title_card("Unitree Go1\nScripted bobbing gait\nReal MuJoCo physics · AMD Radeon", 4),
        "t1": title_card("Side view", 2),
        "t2": title_card("Front view", 2),
        "t3": title_card("3/4 view", 2),
        "t4": title_card("Training blocked by a ROCm profiler race\nSee docs/ROCM_BUG_REPORT.md", 4),
    }

    flist = [cards["t0"], cards["t1"], segs["side"],
             cards["t2"], segs["front"],
             cards["t3"], segs["3q"], cards["t4"]]

    tmp_txt = Path("/tmp/concat_list.txt")
    tmp_txt.write_text("".join(f"file '{p}'\n" for p in flist))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(f"ffmpeg -y -f concat -safe 0 -i {tmp_txt} -c copy {OUT}",
                   shell=True, check=True)
    print(f"✅ {OUT}  ({OUT.stat().st_size // 1024} KB)  render={'3D' if use_3d else '2D-skeleton'}")
    print("   Upload to YouTube/Bilibili for the PR submission.")


if __name__ == "__main__":
    main()

