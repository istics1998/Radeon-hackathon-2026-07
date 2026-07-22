#!/usr/bin/env python3
"""
Demo video renderer — PIL 2D with PD controller for stable standing.

Reads body positions from CPU MuJoCo, applies a simple PD position controller
to maintain a stable quadruped standing pose. Draws robot as 2D side view.

Usage:
    python3 scripts/make_demo_video.py
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import mujoco
import imageio
from mujoco_playground import registry as reg
from pathlib import Path

W, H = 640, 480
SCALE = 200

# Body names for each leg (hip, thigh, calf)
LEG_BODIES = {
    'FR': ('FR_hip', 'FR_thigh', 'FR_calf'),
    'FL': ('FL_hip', 'FL_thigh', 'FL_calf'),
    'RR': ('RR_hip', 'RR_thigh', 'RR_calf'),
    'RL': ('RL_hip', 'RL_thigh', 'RL_calf'),
}
LEG_COLORS = {
    'FR': (200, 40, 40), 'FL': (40, 80, 200),
    'RR': (40, 180, 80), 'RL': (220, 130, 40),
}


def find_body_ids(model):
    """Build name→id map for all bodies."""
    name2id = {}
    for j in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, j)
        name2id[nm] = j
    return name2id


def draw_ground(draw, extent=0.8):
    gy = H // 2
    draw.line([(0, gy), (W, gy)], fill=(60, 60, 60), width=2)
    for v in np.linspace(-extent, extent, 7):
        ix = int(v * SCALE + W / 2)
        draw.line([(W // 2, gy), (ix, gy - 30)], fill=(180, 180, 180), width=1)


def draw_torso(draw, xpos, name2id):
    cy = int(xpos[name2id['trunk'], 1] * SCALE + W / 2)
    cz = int(H / 2 - xpos[name2id['trunk'], 2] * SCALE)
    draw.rectangle([cy-80, cz-18, cy+80, cz+18],
                   fill=(20, 30, 50), outline=(0, 0, 0), width=2)


def draw_leg(draw, xpos, name2id, leg_name, color):
    hid = name2id[LEG_BODIES[leg_name][0]]
    tid = name2id[LEG_BODIES[leg_name][1]]
    cid = name2id[LEG_BODIES[leg_name][2]]

    hx = int(xpos[hid, 1] * SCALE + W / 2)
    hy = int(H / 2 - xpos[hid, 2] * SCALE)
    tx = int(xpos[tid, 1] * SCALE + W / 2)
    ty = int(H / 2 - xpos[tid, 2] * SCALE)
    cx = int(xpos[cid, 1] * SCALE + W / 2)
    cy = int(H / 2 - xpos[cid, 2] * SCALE)

    # Hip→thigh (thick)
    draw.line([(hx, hy), (tx, ty)], fill=color, width=10)
    # Thigh→calf (thick)
    draw.line([(tx, ty), (cx, cy)], fill=color, width=8)
    # Joint circles
    for x, y in [(hx, hy), (tx, ty), (cx, cy)]:
        draw.ellipse([x-5, y-5, x+5, y+5], fill=color)
    # Foot (ground contact indicator)
    red = (220, 40, 40)
    foot_color = red if xpos[cid, 2] < 0.03 else color
    draw.ellipse([cx-8, cy-8, cx+8, cy+8], fill=foot_color)


def draw_hud(draw, step, total, height=0.0):
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    state = "falling..." if height > 0.35 else "standing"
    ts = f"step {step}/{total}  {state}  time: {step/30:.1f}s"
    draw.text((10, 10), "Unitree Go1  drop & stand", fill=(20, 20, 60), font=font)
    draw.text((10, 30), ts, fill=(20, 20, 60), font=font)
    draw.text((10, H - 30), "AMD Radeon  ROCm 7.2.1  JAX 0.11.0", fill=(80, 80, 80), font=font)


def main():
    env_name = "Go1JoystickFlatTerrain"
    num_frames = 250
    fps = 30
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "demo.mp4"

    print(f"[make_demo_video] Loading {env_name} ...")
    cfg = reg.get_default_config(env_name)
    cfg.impl = "jax"
    env = reg.load(env_name, config=cfg)
    model = env.mj_model
    name2id = find_body_ids(model)

    data = mujoco.MjData(model)
    # 只设关节角度为 0 (站立姿态), 保持机身高度不变
    data.qpos[7:19] = 0
    mujoco.mj_forward(model, data)

    frames = []
    for i in range(num_frames):
        # 逆向动力学 (重力补偿) + PD 阻尼 (防止累积漂移)
        mujoco.mj_inverse(model, data)
        for j in range(12):
            data.ctrl[j] = data.qfrc_inverse[j] - 5.0 * data.qvel[6 + j]

        mujoco.mj_step(model, data)
        xpos = data.xpos

        img = Image.new("RGB", (W, H), (250, 250, 252))
        draw = ImageDraw.Draw(img)
        draw_ground(draw)

        for leg_name in ['FR', 'FL', 'RR', 'RL']:
            draw_leg(draw, xpos, name2id, leg_name, LEG_COLORS[leg_name])

        draw_torso(draw, xpos, name2id)
        draw_hud(draw, i + 1, num_frames, xpos[name2id['trunk'], 2])
        frames.append(np.array(img))

        if (i + 1) % 10 == 0:
            print(f"  rendered {i + 1}/{num_frames}")

    print(f"[make_demo_video] Writing {out_path} ...")
    imageio.mimsave(str(out_path), np.stack(frames), fps=fps,
                    codec="libx264", quality=10, pixelformat="yuv420p")
    size_kb = out_path.stat().st_size / 1024
    print(f"✅ {out_path} ({size_kb:.0f} KB, {num_frames} frames)")


if __name__ == "__main__":
    main()
