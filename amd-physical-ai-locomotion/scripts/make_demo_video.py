#!/usr/bin/env python3
"""
Demo video renderer — 2D side-view via PIL, thick lines guaranteed visible.

Reads body positions from CPU MuJoCo, projects to 2D (side view),
draws with PIL rectangles. No matplotlib 3D rendering quirks.

Usage:
    python3 scripts/make_demo_video.py
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import mujoco
import imageio
from mujoco_playground import registry as reg
from pathlib import Path

# Image dimensions
W, H = 640, 480
SCALE = 200  # pixels per meter
# Camera looks from +x axis, so we plot (y, z) — side view
# y goes to image x, z goes to image y (inverted, so up is up)


def project(pos, camera_origin=(2.0, 0.0, 0.3)):
    """Project 3D world to 2D image coords (y→x, z→y, scaled)."""
    # Simple orthographic: drop the x-axis, use y for image x, z for image y
    img_x = pos[1] * SCALE + W / 2
    img_y = H / 2 - pos[2] * SCALE
    return int(img_x), int(img_y)


def draw_ground(draw, extent=0.8):
    """Draw ground line + grid."""
    gy = H // 2
    # Main ground line
    draw.line([(0, gy), (W, gy)], fill=(60, 60, 60), width=2)
    # Subtle perspective lines (radiating from center to horizon)
    for v in np.linspace(-extent, extent, 7):
        ix, _ = project([0, v, 0])
        draw.line([(W // 2, gy), (ix, gy - 30)], fill=(180, 180, 180), width=1)


def draw_torso(draw, xpos, xmat):
    """Draw torso as a thick dark rectangle (side view)."""
    # Get torso center and orientation in 2D
    cy, cz = project(xpos[1])
    # Use body xmat to get rotation around y/z axis
    # For side view, we just draw a horizontal rectangle for the torso
    torso_w, torso_h = 100, 30  # pixels
    draw.rectangle(
        [cy - torso_w // 2, cz - torso_h // 2,
         cy + torso_w // 2, cz + torso_h // 2],
        fill=(20, 30, 50), outline=(0, 0, 0), width=2
    )
    # Head indicator (small light square on top-front)
    draw.rectangle(
        [cy - 40, cz - torso_h // 2 - 12,
         cy - 10, cz - torso_h // 2 + 4],
        fill=(180, 180, 200), outline=(0, 0, 0), width=1
    )


def draw_leg(draw, xpos, hip, knee, foot, color):
    """Draw one leg as thick lines (hip→knee→foot) with joint circles."""
    hy, hz = project(xpos[hip])
    ky, kz = project(xpos[knee])
    fy, fz = project(xpos[foot])

    # Thigh
    draw.line([(hy, hz), (ky, kz)], fill=color, width=10)
    # Shin
    draw.line([(ky, kz), (fy, fz)], fill=color, width=8)
    # Joints
    for x, y in [(hy, hz), (ky, kz), (fy, fz)]:
        r = 6
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(0, 0, 0))
    # Foot dot
    foot_color = (220, 40, 40) if fz > H // 2 - 10 else color
    r = 8
    draw.ellipse([fy - r, fz - r, fy + r, fz + r], fill=foot_color, outline=(0, 0, 0))


def find_legs(model):
    children = {i: [] for i in range(model.nbody)}
    for j in range(1, model.nbody):
        children[model.body_parentid[j]].append(j)

    legs = []
    # Take first 4 children of torso (body 1) as legs
    for c in children[1][:4]:
        knee = children[c][0] if children[c] else c
        foot = children[knee][0] if children[knee] else knee
        legs.append((c, knee, foot))
    return legs


def draw_hud(draw, step, total):
    """Draw info overlay."""
    txt = f"step {step}/{total}"
    # PIL default font is small but works
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 10), "Unitree Go1  random policy", fill=(20, 20, 60), font=font)
    draw.text((10, 30), txt, fill=(20, 20, 60), font=font)
    draw.text((10, H - 30), "AMD Radeon  ROCm 7.2.1  JAX 0.11.0",
              fill=(80, 80, 80), font=font)


def main():
    env_name = "Go1JoystickFlatTerrain"
    num_frames = 100
    fps = 30
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "demo.mp4"

    print(f"[make_demo_video] Loading {env_name} ...")
    cfg = reg.get_default_config(env_name)
    cfg.impl = "jax"
    env = reg.load(env_name, config=cfg)
    model = env.mj_model

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    legs = find_legs(model)
    leg_colors = [(200, 40, 40), (40, 80, 200), (40, 180, 80), (220, 130, 40)]

    print(f"[make_demo_video] Bodies: {model.nbody}, Legs: {len(legs)}")

    frames = []
    for i in range(num_frames):
        # 前 30 帧: 0 动作(让它站住, 展示初始姿态)
        # 后 70 帧: 小随机动作(模拟扰动, 但不至于立刻翻)
        if i < 30:
            data.ctrl[:] = 0.0
        else:
            data.ctrl[:] = np.random.uniform(-0.2, 0.2, 12)
        mujoco.mj_step(model, data)
        xpos = data.xpos

        img = Image.new("RGB", (W, H), (250, 250, 252))
        draw = ImageDraw.Draw(img)
        draw_ground(draw)
        draw_torso(draw, xpos, data.xmat)

        for (hip, knee, foot), color in zip(legs, leg_colors):
            draw_leg(draw, xpos, hip, knee, foot, color)

        draw_hud(draw, i + 1, num_frames)
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
