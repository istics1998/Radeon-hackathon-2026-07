#!/usr/bin/env python3
"""
Improved demo video renderer — uses real MuJoCo geom data.

Draws proper Go1 quadruped with body boxes, leg capsules, and ground plane.
Avoids MuJoCo Renderer (broken in 3.10) — uses matplotlib Agg + CPU MuJoCo.

Usage:
    python3 scripts/make_demo_video.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import mujoco
import imageio
from mujoco_playground import registry as reg
from pathlib import Path

GEOM_BOX = mujoco.mjtGeom.mjGEOM_BOX
GEOM_CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE
GEOM_ELLIPSOID = mujoco.mjtGeom.mjGEOM_ELLIPSOID
GEOM_CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER
GEOM_SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE
GEOM_PLANE = mujoco.mjtGeom.mjGEOM_PLANE


def box_faces(center, size, rot):
    sx, sy, sz = size
    corners_local = np.array([
        [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
        [-sx, -sy,  sz], [sx, -sy,  sz], [sx, sy,  sz], [-sx, sy,  sz],
    ])
    return corners_local @ rot.T + center


def get_world_geom(model, data, g):
    body_id = model.geom_bodyid[g]
    bpos = data.xpos[body_id].copy()
    brot = data.xmat[body_id].reshape(3, 3).copy()
    gpos_local = model.geom_pos[g]
    gpos = bpos + brot @ gpos_local
    return gpos, brot, model.geom_size[g].copy(), model.geom_rgba[g].copy(), model.geom_type[g]


def draw_geom(ax, pos, rot, size, rgba, gtype):
    color = rgba[:3]
    alpha = 0.92

    if gtype == GEOM_BOX:
        corners = box_faces(pos, size, rot)
        faces = [
            [corners[0], corners[1], corners[2], corners[3]],
            [corners[4], corners[5], corners[6], corners[7]],
            [corners[0], corners[1], corners[5], corners[4]],
            [corners[2], corners[3], corners[7], corners[6]],
            [corners[0], corners[3], corners[7], corners[4]],
            [corners[1], corners[2], corners[6], corners[5]],
        ]
        coll = Poly3DCollection(faces, alpha=alpha)
        coll.set_facecolor(color)
        coll.set_edgecolor('black')
        coll.set_linewidth(0.4)
        ax.add_collection3d(coll)

    elif gtype in (GEOM_CAPSULE, GEOM_CYLINDER):
        axis = rot[:, 0] * size[0]
        p1 = pos - axis
        p2 = pos + axis
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=color, lw=10, solid_capstyle='round', alpha=alpha)
        r = size[1] if gtype == GEOM_CAPSULE else size[0] * 0.8
        for p in (p1, p2):
            u, v = np.mgrid[0:2*np.pi:6j, 0:np.pi:3j]
            x = r * np.cos(u) * np.sin(v)
            y = r * np.sin(u) * np.sin(v)
            z = r * np.cos(v)
            pts = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=0) + p[:, None]
            ax.plot_surface(pts[0].reshape(x.shape), pts[1].reshape(x.shape),
                            pts[2].reshape(x.shape), color=color, alpha=alpha, linewidth=0)

    elif gtype in (GEOM_ELLIPSOID, GEOM_SPHERE):
        u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:5j]
        x = size[0] * np.cos(u) * np.sin(v)
        y = size[0] * np.sin(u) * np.sin(v)
        z = size[0] * np.cos(v)
        if gtype == GEOM_ELLIPSOID:
            scale = np.array([size[0], size[1], size[2]])
            pts = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=0) * scale[:, None]
        else:
            pts = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=0)
        pts = rot @ pts + pos[:, None]
        ax.plot_surface(pts[0].reshape(x.shape), pts[1].reshape(x.shape),
                        pts[2].reshape(x.shape), color=color, alpha=alpha, linewidth=0)


def draw_ground(ax, extent=1.0):
    xx, yy = np.meshgrid([-extent, extent], [-extent, extent])
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color='#e8e8e8', alpha=0.7, linewidth=0)
    for v in np.linspace(-extent, extent, 9):
        ax.plot([v, v], [-extent, extent], [0, 0], color='gray', lw=0.3, alpha=0.4)
        ax.plot([-extent, extent], [v, v], [0, 0], color='gray', lw=0.3, alpha=0.4)


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

    plane_geom_ids = {g for g in range(model.ngeom)
                      if model.geom_type[g] == GEOM_PLANE}

    frames = []
    for i in range(num_frames):
        data.ctrl[:] = np.random.uniform(-1, 1, 12)
        mujoco.mj_step(model, data)

        fig = plt.figure(figsize=(8, 6), facecolor='white')
        ax = fig.add_subplot(111, projection='3d', facecolor='white')
        ax.view_init(elev=-20, azim=70)
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_zlim(-0.05, 0.7)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')

        draw_ground(ax, extent=0.8)

        for g in range(model.ngeom):
            if g in plane_geom_ids:
                continue
            pos, rot, size, rgba, gtype = get_world_geom(model, data, g)
            try:
                draw_geom(ax, pos, rot, size, rgba, gtype)
            except Exception as e:
                # Skip problematic geoms instead of crashing
                continue

        ax.set_title("Unitree Go1 — Random Policy Rollout (matplotlib CPU render)",
                     fontsize=11, pad=8)
        ax.text2D(0.02, 0.98, f"step {i+1}/{num_frames}",
                  transform=ax.transAxes, fontsize=9, va='top',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype="uint8").reshape(
            (h, w, 4))[:, :, 1:]
        frames.append(img)
        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"  rendered {i + 1}/{num_frames}")

    print(f"[make_demo_video] Writing {out_path} ...")
    # Pad to 16-aligned dimensions for x264 macro_block_size
    h, w = frames[0].shape[:2]
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h or pad_w:
        frames = [np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)),
                         mode='edge') for f in frames]

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
