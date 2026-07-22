#!/usr/bin/env python3
"""
Demo video renderer — stable body-center + thick-line robot render.

Reads body positions (xpos) from CPU MuJoCo data and draws the Go1 as:
  - Torso: large semi-transparent box
  - 4 legs: thick colored lines (hip→knee→foot)
  - Feet: colored dots (red when touching ground)
  - Ground: grid plane
  - Info overlay

Usage:
    python3 scripts/make_demo_video.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import mujoco
import imageio
from mujoco_playground import registry as reg
from pathlib import Path

# Go1 body hierarchy (typical nbody layout for quadruped with 6-dof free joint)
# body[0]=world, body[1]=torso, then 4 legs each with thigh+shin+foot
# We'll discover it at runtime


def find_leg_bodies(model):
    """Return dict mapping leg_name -> (hip_body_id, knee_body_id, foot_body_id)."""
    # Strategy: starting from body 1 (torso), walk through body tree
    # to find 4 legs with 3 segments each
    children = {i: [] for i in range(model.nbody)}
    for j in range(1, model.nbody):
        p = model.body_parentid[j]
        children[p].append(j)

    legs = {}
    leg_names = ["FR", "FL", "RR", "RL"]
    torso_children = children[1]  # direct children of torso
    for idx, child in enumerate(torso_children[:4]):  # typically 4 in correct order
        knee_children = children.get(child, [])
        foot = knee_children[0] if knee_children else None
        foot_children = children.get(foot, [])
        foot2 = foot_children[0] if foot_children else None
        legs[leg_names[idx] if idx < len(leg_names) else f"leg{idx}"] = {
            "hip": child, "knee": foot if foot else child, "foot": foot2 if foot2 else foot if foot else child
        }

    return legs, children


def draw_torso_box(ax, xpos, xmat):
    """Draw the torso as a box from body[1] position+orientation."""
    center = xpos[1]
    # Use the body's rotation matrix (3x3)
    rot = xmat[1].reshape(3, 3) if xmat[1].size == 9 else xmat[1]

    # Torso approximate size (Go1)
    sx, sy, sz = 0.20, 0.42, 0.12
    corners_local = np.array([
        [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
        [-sx, -sy,  sz], [sx, -sy,  sz], [sx, sy,  sz], [-sx, sy,  sz],
    ])
    corners = corners_local @ rot.T + center

    # Draw edges (wireframe box — always works)
    edges = [
        (0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)
    ]
    for i, j in edges:
        ax.plot([corners[i,0], corners[j,0]],
                [corners[i,1], corners[j,1]],
                [corners[i,2], corners[j,2]],
                color='#1a1a2e', lw=3, solid_capstyle='round')


def draw_leg(ax, xpos, leg, color):
    """Draw one leg as hip→knee→foot thick lines + joint dots."""
    hip = leg["hip"]
    knee = leg["knee"]
    foot = leg["foot"]

    # Draw segments
    for (a, b) in [(hip, knee), (knee, foot)]:
        ax.plot([xpos[a, 0], xpos[b, 0]],
                [xpos[a, 1], xpos[b, 1]],
                [xpos[a, 2], xpos[b, 2]],
                color=color, lw=6, solid_capstyle='round')

    # Joint dots
    ax.scatter(*xpos[hip], color=color, s=50, zorder=5)
    ax.scatter(*xpos[knee], color=color, s=40, zorder=5)
    ax.scatter(*xpos[foot], color='red' if xpos[foot,2] < 0.03 else color,
               s=60, zorder=5)


def draw_ground(ax, extent=0.9):
    """Draw ground plane + subtle grid."""
    # Semi-transparent plane
    xx, yy = np.meshgrid([-extent, extent], [-extent, extent])
    ax.plot_surface(xx, yy, np.zeros_like(xx),
                    color='#f0f0f0', alpha=0.5, linewidth=0)
    # Grid
    for v in np.linspace(-extent, extent, 7):
        ax.plot([v, v], [-extent, extent], [0, 0], color='gray', lw=0.3, alpha=0.3)
        ax.plot([-extent, extent], [v, v], [0, 0], color='gray', lw=0.3, alpha=0.3)


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

    legs, children = find_leg_bodies(model)
    leg_colors = {"FR": "orangered", "FL": "royalblue",
                  "RR": "seagreen", "RL": "darkorange"}

    print(f"[make_demo_video] Bodies: {model.nbody}, Legs: {len(legs)}")
    for name, l in legs.items():
        print(f"  {name}: hip={l['hip']} knee={l['knee']} foot={l['foot']}")

    frames = []
    for i in range(num_frames):
        data.ctrl[:] = np.random.uniform(-1, 1, 12)
        mujoco.mj_step(model, data)
        xpos = data.xpos
        xmat = data.xmat

        fig = plt.figure(figsize=(8, 6), facecolor='white')
        ax = fig.add_subplot(111, projection='3d', facecolor='white')
        ax.view_init(elev=-20, azim=60)
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_zlim(-0.05, 0.65)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

        draw_ground(ax)
        draw_torso_box(ax, xpos, xmat)

        for name, leg in legs.items():
            color = leg_colors.get(name, "gray")
            draw_leg(ax, xpos, leg, color)

        # Title + step counter
        ax.set_title("Unitree Go1 — random policy", fontsize=12, pad=6)
        ax.text2D(0.02, 0.97, f"step {i+1}/{num_frames}",
                  transform=ax.transAxes, fontsize=9, va='top',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype="uint8").reshape(
            (h, w, 4))[:, :, 1:]
        frames.append(img)
        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"  rendered {i + 1}/{num_frames}")

    print(f"[make_demo_video] Writing {out_path} ...")
    h_fix = (16 - frames[0].shape[0] % 16) % 16
    w_fix = (16 - frames[0].shape[1] % 16) % 16
    if h_fix or w_fix:
        frames = [np.pad(f, ((0,h_fix),(0,w_fix),(0,0)), mode='edge') for f in frames]

    imageio.mimsave(str(out_path), np.stack(frames), fps=fps,
                    codec="libx264", quality=10, pixelformat="yuv420p")
    size_kb = out_path.stat().st_size / 1024
    print(f"✅ {out_path} ({size_kb:.0f} KB, {num_frames} frames)")


if __name__ == "__main__":
    main()
