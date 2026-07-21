#!/usr/bin/env python3
"""
Standalone demo video renderer for Route C deliverable.

Generates `outputs/demo.mp4` using matplotlib (Agg) + CPU MuJoCo + imageio.
Avoids MuJoCo Renderer (OSMesa/EGL/glfw are broken in 3.10) and
avoids GPU→CPU copies (triggers rocprofiler segfault).

Usage:
    python3 scripts/make_demo_video.py

Dependencies:
    pip install --break-system-packages matplotlib imageio imageio-ffmpeg mujoco mujoco-mjx playground
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import mujoco
import imageio
from mujoco_playground import registry as reg
from pathlib import Path


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
        data.ctrl[:] = np.random.uniform(-1, 1, 12)
        mujoco.mj_step(model, data)

        fig, ax = plt.subplots(figsize=(6.4, 4.8), subplot_kw={"projection": "3d"})
        xpos = data.xpos
        ax.scatter(xpos[:, 0], xpos[:, 1], xpos[:, 2], c="blue", s=20)
        for j in range(model.nbody):
            p = model.body_parentid[j]
            ax.plot(
                [xpos[j, 0], xpos[p, 0]],
                [xpos[j, 1], xpos[p, 1]],
                [xpos[j, 2], xpos[p, 2]],
                "k-", lw=1,
            )
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 1)
        ax.set_title(f"Go1 — step {i}/{num_frames}")

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
        quality=8,
        pixelformat="yuv420p",
    )

    size_kb = out_path.stat().st_size / 1024
    print(f"✅ {out_path} ({size_kb:.0f} KB, {num_frames} frames)")


if __name__ == "__main__":
    main()
