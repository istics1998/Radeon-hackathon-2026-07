#!/usr/bin/env python3
"""
Demo video — 3 camera angles + title cards, stitched into one ~40s video.

Usage:
    python3 scripts/make_demo_video_full.py
"""

import numpy as np
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont
from mujoco_playground import registry as reg
from pathlib import Path
import tempfile, os, subprocess

W, H = 640, 480
SCALE = 200

LEG_BODIES = {
    'FR': ('FR_hip', 'FR_thigh', 'FR_calf'),
    'FL': ('FL_hip', 'FL_thigh', 'FL_calf'),
    'RR': ('RR_hip', 'RR_thigh', 'RR_calf'),
    'RL': ('RL_hip', 'RL_thigh', 'RL_calf'),
}
LEG_COLORS = {'FR': (200,40,40), 'FL': (40,80,200), 'RR': (40,180,80), 'RL': (220,130,40)}


def find_body_ids(model):
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, j): j
            for j in range(model.nbody)}


def frame(imageio_frames, model, data, axis):
    """Render one frame, append to imageio_frames list."""
    n2id = find_body_ids(model)
    def project(pos):
        if axis == "x":
            ix = -pos[0] * SCALE + W//2
        elif axis == "xy":
            ix = (pos[0]*0.35 + pos[1]*0.65) * SCALE + W//2
        else:  # y (side)
            ix = pos[1] * SCALE + W//2
        iy = H//2 - pos[2] * SCALE
        return int(ix), int(iy)

    img = Image.new("RGB", (W, H), (250, 250, 252))
    draw = ImageDraw.Draw(img)

    # Ground
    gy = H//2
    draw.line([(0,gy),(W,gy)], fill=(60,60,60), width=2)
    for v in np.linspace(-0.8, 0.8, 7):
        ix = int(v*SCALE + W//2)
        draw.line([(W//2, gy), (ix, gy-30)], fill=(180,180,180), width=1)

    xpos = data.xpos
    # Legs
    for ln in ['FR','FL','RR','RL']:
        color = LEG_COLORS[ln]
        hi = n2id[LEG_BODIES[ln][0]]; ti = n2id[LEG_BODIES[ln][1]]; ci = n2id[LEG_BODIES[ln][2]]
        hp = project(xpos[hi]); tp = project(xpos[ti]); cp = project(xpos[ci])
        draw.line([hp, tp], fill=color, width=10)
        draw.line([tp, cp], fill=color, width=8)
        for p in [hp, tp, cp]:
            draw.ellipse([p[0]-5,p[1]-5,p[0]+5,p[1]+5], fill=color)
        fc = (220,40,40) if xpos[ci,2] < 0.03 else color
        draw.ellipse([cp[0]-8,cp[1]-8,cp[0]+8,cp[1]+8], fill=fc)

    # Torso
    tp = project(xpos[n2id['trunk']])
    draw.rectangle([tp[0]-80,tp[1]-18,tp[0]+80,tp[1]+18], fill=(20,30,50), outline=(0,0,0), width=2)

    # HUD
    try: f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except: f = ImageFont.load_default()
    draw.text((10,10), f"Unitree Go1 — {axis.upper()} view", fill=(20,20,60), font=f)
    draw.text((10,H-30), "AMD Radeon  ROCm 7.2.1  JAX 0.11.0", fill=(80,80,80), font=f)

    imageio_frames.append(np.array(img))


def render_segment(model, data, axis, num_frames, fps=30):
    """Render a segment with sine breathing."""
    frames = []
    for i in range(num_frames):
        phase = i * 0.10
        for j in range(12):
            data.qpos[7+j] = 0.04 * np.sin(phase + j * 0.3)
        mujoco.mj_forward(model, data)
        frame(frames, model, data, axis)
        if (i+1) % 30 == 0:
            print(f"  {axis} view: {i+1}/{num_frames}")

    out = Path(f"/tmp/demo_seg_{axis}.mp4")
    imageio.mimsave(str(out), np.stack(frames), fps=fps,
                    codec="libx264", quality=10, pixelformat="yuv420p")
    print(f"  ✅ segment saved ({out.stat().st_size//1024} KB)")
    return out


def title_card(text, sec=3):
    frames = []
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    except: font = ImageFont.load_default()
    for _ in range(int(30*sec)):
        img = Image.new("RGB", (W, H), (40, 40, 60))
        draw = ImageDraw.Draw(img)
        lines = text.split("\n")
        y = (H - len(lines)*40)//2
        for line in lines:
            b = draw.textbbox((0,0), line, font=font)
            x = (W - b[2])//2
            draw.text((x, y), line, fill=(220,220,240), font=font)
            y += 40
        frames.append(np.array(img))
    out = Path(f"/tmp/title_{abs(hash(text))}.mp4")
    imageio.mimsave(str(out), np.stack(frames), fps=30, codec="libx264", quality=10)
    return out


def main():
    print("[make_demo_video_full] Loading model...")
    cfg = reg.get_default_config("Go1JoystickFlatTerrain"); cfg.impl = "jax"
    env = reg.load("Go1JoystickFlatTerrain", config=cfg)
    model = env.mj_model
    data = mujoco.MjData(model)
    data.qpos[0:3] = [0, 0, 0.213]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:19] = 0
    mujoco.mj_forward(model, data)

    N = 200  # ~6.7s each @ 30fps
    segs = [
        render_segment(model, data, "y", N),
        render_segment(model, data, "x", N),
        render_segment(model, data, "xy", N),
    ]

    # Title cards
    cards = {
        "t0": title_card("Unitree Go1\nAMD Physical AI Simulation", 4),
        "t1": title_card("Side View", 2),
        "t2": title_card("Front View", 2),
        "t3": title_card("3/4 View", 2),
        "t4": title_card("ROCm Bug Report\nSee docs/ROCM_BUG_REPORT.md", 4),
    }

    # Build file list for ffmpeg concat
    flist = [
        cards["t0"], cards["t1"], segs[0],
        cards["t2"], segs[1],
        cards["t3"], segs[2],
        cards["t4"],
    ]

    tmp_txt = Path("/tmp/concat_list.txt")
    with open(tmp_txt, "w") as f:
        for p in flist:
            f.write(f"file '{p}'\n")

    out_path = Path("outputs/demo_full2.mp4")
    cmd = f"ffmpeg -y -f concat -safe 0 -i {tmp_txt} -c copy {out_path}"
    print(f"[make_demo_video_full] Concatenating {len(flist)} segments...")
    subprocess.run(cmd, shell=True, check=True)

    print(f"✅ {out_path}  ({out_path.stat().st_size//1024} KB)")
    print(f"   Upload to YouTube/Bilibili for PR submission")


if __name__ == "__main__":
    main()
