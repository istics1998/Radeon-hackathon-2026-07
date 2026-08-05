#!/usr/bin/env python3
"""Read-only probe: is the Footstand policy STANDING or SITTING?

Rolls out a Footstand checkpoint headless (no video) and reports the settled
pose so we can tell a true rear-leg stand (torso lifted to ~0.53m, ONLY the
rear feet touching the floor) from the sitting local optimum (torso low, rear
feet + haunches/base pressing the ground).

Usage (on the instance):
    cd /workspace/demo
    MUJOCO_GL=egl python3 scripts/probe_footstand_pose.py checkpoints_go1footstand/latest.pkl
    MUJOCO_GL=egl python3 scripts/probe_footstand_pose.py checkpoints_go1footstand/final.pkl
"""
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import jax

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

ENV = "Go1Footstand"
CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints_go1footstand/latest.pkl")
NSTEPS = int(os.environ.get("SKILL_STEPS", 400))


def load_policy(env, ckpt_path):
    ppo_params = locomotion_params.brax_ppo_config(ENV)
    nf = dict(ppo_params).get("network_factory", {})
    net = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize, **nf)
    make_policy = ppo_networks.make_inference_fn(net)
    with open(ckpt_path, "rb") as f:
        params = pickle.load(f)
    return make_policy(params, deterministic=True)


def main():
    if not CKPT.exists():
        sys.exit(f"[probe] not found: {CKPT}")
    print(f"[probe] env={ENV} ckpt={CKPT} steps={NSTEPS}")

    cfg = registry.get_default_config(ENV)
    cfg.impl = "jax"
    env = registry.load(ENV, config=cfg)
    m = env.mj_model

    # all four foot geoms; front (FR/FL) should be UP, rear (RR/RL) DOWN in a stand
    foot_geoms = {n: m.geom(n).id for n in ("FR", "FL", "RR", "RL")}
    floor_id = m.geom("floor").id
    imu_z = m.site("imu").id

    policy = load_policy(env, CKPT)
    jit_reset, jit_step, inf = jax.jit(env.reset), jax.jit(env.step), jax.jit(policy)
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)

    torso_h, base_z = [], []
    contact_counts = {n: 0 for n in foot_geoms}
    ncon_nonfoot = []  # contacts involving neither floor-foot pair: butt/thigh/base on floor

    tail0 = int(0.6 * NSTEPS)
    for i in range(NSTEPS):
        act_rng, rng = jax.random.split(rng)
        act, _ = inf(state.obs, act_rng)
        state = jit_step(state, act)
        d = state.data
        torso_h.append(float(np.asarray(d.site_xpos[imu_z])[2]))
        base_z.append(float(np.asarray(d.qpos)[2]))
        if i >= tail0:
            g1 = np.asarray(d.contact.geom1)
            g2 = np.asarray(d.contact.geom2)
            dist = np.asarray(d.contact.dist)
            active = dist < 0
            pairs = set(zip(g1[active].tolist(), g2[active].tolist()))
            foot_ids = set(foot_geoms.values())
            nonfoot_floor = 0
            for a, b in pairs:
                s = {a, b}
                if floor_id in s:
                    other = (s - {floor_id}).pop() if len(s) == 2 else floor_id
                    if other not in foot_ids:
                        nonfoot_floor += 1  # something that ISN'T a foot touching floor
            ncon_nonfoot.append(nonfoot_floor)
            for n, gid in foot_geoms.items():
                if (gid, floor_id) in pairs or (floor_id, gid) in pairs:
                    contact_counts[n] += 1

    tail = slice(tail0, None)
    ntail = NSTEPS - tail0
    print(f"  torso_h (imu z): final={np.mean(torso_h[tail]):.3f}m  max={np.max(torso_h):.3f}m  (target z_des=0.53)")
    print(f"  base_z  (qpos2): final={np.mean(base_z[tail]):.3f}m")
    print(f"  foot-floor contact frac over settled tail ({ntail} steps):")
    for n in ("FR", "FL", "RR", "RL"):
        print(f"      {n}: {contact_counts[n]/ntail:5.0%}")
    print(f"  NON-foot parts touching floor (butt/thigh/base): mean {np.mean(ncon_nonfoot):.2f} contacts/step")
    verdict = ("SITTING (non-foot ground contact + low torso)"
               if (np.mean(ncon_nonfoot) > 0.2 or np.mean(torso_h[tail]) < 0.42)
               else "STANDING-ish (torso high, only feet on floor)")
    print(f"  => {verdict}")


if __name__ == "__main__":
    main()
