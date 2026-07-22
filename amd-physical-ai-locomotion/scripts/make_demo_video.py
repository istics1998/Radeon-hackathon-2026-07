#!/usr/bin/env python3
"""
Improved demo video renderer for Route C deliverable.

Generates `outputs/demo.mp4` using matplotlib (Agg) + CPU MuJoCo + imageio.
Draws a proper Go1 quadruped: torso box, leg segments, ground plane, info overlay.

Usage:
    python3 scripts/make_demo_video.py

Dependencies: matplotlib imageio imageio-ffmpeg mujoco mujoco-mjx playground
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import mujoco
import imageio
from mujoco_playground import registry as reg
from pathlib import Path

# Go1 body hierarchy (from MuJoCo model)
# 0:world  1:torso  2:FR_thigh  3:FR_shin  4:FL_thigh  5:FL_shin
# 6:RR_thigh  7:RR_shin  8:RL_thigh  9:RL_shin  10:FR_foot  11:FL_foot  12:RR_foot  13:RL_foot
TORSO = 1
# (name, parent, color) for each leg segment
LEG_SEGMENTS = [
    # FR leg
    (2, 1, 'orangered'),    (3, 2, 'tomato'),      (10, 3, 'lightsalmon'),
    # FL leg
    (4, 1, 'royalblue'),    (5, 4, 'cornflowerblue'), (11, 5, 'lightsteelblue'),
    # RR leg
    (6, 1, 'seagreen'),     (7, 6, 'mediumseagreen'),  (12, 7, 'palegreen'),
    # RL leg
    (8, 1, 'darkorange'),   (9, 8, 'orange'),          (13, 9, 'moccasin'),
]

# Geoms that are part of the torso (box)
TORSO_GEOM_IDS = {0, 1, 2, 3}  # approximate, will refine


def draw_ground(ax, xlim=(-0.8, 0.8), ylim=(-0.8, 0.8)):
    """Draw a semi-transparent ground plane with grid."""
    xx = np.linspace(xlim[0], xlim[1], 9)
    yy = np.linspace(ylim[0], ylim[1], 9)
    for x in xx:
        ax.plot([x, x], [ylim[0], ylim[1]], [0, 0], color='gray', lw=0.3, alpha=0.4)
    for y in yy:
        ax.plot([xlim[0], xlim[1]], [y, y], [0, 0], color='gray', lw=0.3, alpha=0.4)


def draw_torso(ax, xpos, model):
    """Draw the torso as a box using torso geom positions."""
    # Collect geom positions belonging to the torso
    torso_pts = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == TORSO:
            torso_pts.append(xpos[model.geom_bodyid[g]])
    if torso_pts:
        center = np.mean(torso_pts, axis=0)
        # Draw a simple box representation
        w, l, h = 0.12, 0.35, 0.10
        corners = np.array([
            [-w/2, -l/2, -h/2], [ w/2, -l/2, -h/2], [ w/2,  l/2, -h/2], [-w/2,  l/2, -h/2],
            [-w/2, -l/2,  h/2], [ w/2, -l/2,  h/2], [ w/2,  l/2,  h/2], [-w/2,  l/2,  h/2],
        ]) + center
        for i, j in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            ax.plot([corners[i,0], corners[j,0]],
                    [corners[i,1], corners[j,1]],
                    [corners[i,2], corners[j,2]], color='#2c3e50', lw=2.5)


def draw_legs(ax, xpos):
    """Draw leg segments as thick colored lines."""
    for child, parent, color in LEG_SEGMENTS:
        ax.plot(
            [xpos[child, 0], xpos[parent, 0]],
            [xpos[child, 1], xpos[parent, 1]],
            [xpos[child, 2], xpos[parent, 2]],
            color=color, lw=4, solid_capstyle='round',
        )


def draw_foot_contacts(ax, xpos, contact_dist):
    """Highlight feet near the ground."""
    # Foot bodies
    foot_bodies = [10, 11, 12, 13]
    for fb in foot_bodies:
        z = xpos[fb, 2]
        if z < 0.02:  # near ground
            ax.scatter(*xpos[fb], color='red', s=60, zorder=10)


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

    frames = []
    for i in range(num_frames):
        # Random action (12-dim: 3 per leg × 4 legs)
        data.ctrl[:] = np.random.uniform(-1, 1, 12)
        mujoco.mj_step(model, data)

        fig = plt.figure(figsize=(7.2, 5.4), facecolor='white')
        ax = fig.add_subplot(111, projection='3d', facecolor='white')

        # Camera
        ax.view_init(elev=-25, azim=90)
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_zlim(-0.05, 0.6)

        # Remove axis tick labels for cleaner look
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        xpos = data.xpos

        # Draw layers in order
        draw_ground(ax)
        draw_torso(ax, xpos, model)
        draw_legs(ax, xpos)
        draw_foot_contacts(ax, xpos, data.contact)

        # Info overlay
        ax.set_title(f"Unitree Go1 — Random Policy Rollout", fontsize=13, pad=8)
        # Add text box with step info
        text_str = f"Step {i+1}/{num_frames}"
        ax.text2D(0.02, 0.98, text_str, transform=ax.transAxes,
                  fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype="uint8").reshape(
            (h, w, 4)
        )[:, :, 1:]
        frames.append(img)
        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"  rendered {i + 1}/{num_frames}")

    print(f"[make_demo_video] Writing {out_path} ...")
    imageio.mimsave(
        str(out_path),
        np.stack(frames),
        fps=fps,
        codec="libx264",
        quality=10,
        pixelformat="yuv420p",
    )

    size_kb = out_path.stat().st_size / 1024
    print(f"✅ {out_path} ({size_kb:.0f} KB, {num_frames} frames)")


if __name__ == "__main__":
    main()
