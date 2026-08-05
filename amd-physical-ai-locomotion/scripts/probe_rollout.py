#!/usr/bin/env python3
"""Decisive A/B: is the fall caused by our command override, or by physics/backend?

Test 1: roll out letting the env use its OWN sampled command (no override).
Test 2: roll out with override cmd=[1,0,0] (forward).
For each, print trunk height every 50 steps so we see WHEN/IF it collapses.
Repeat both under impl='jax' and impl='warp' if warp is importable on CPU.
"""
import sys, pickle
from pathlib import Path
import numpy as np
import jax, jax.numpy as jp

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import networks as ppo_networks

ENV = "Go1JoystickFlatTerrain"
CKPT = Path("checkpoints/latest.pkl")


def load_policy(env):
    ppo_params = locomotion_params.brax_ppo_config(ENV)
    nf = dict(ppo_params).get("network_factory", {})
    net = ppo_networks.make_ppo_networks(env.observation_size, env.action_size, **nf)
    make_policy = ppo_networks.make_inference_fn(net)
    with open(CKPT, "rb") as f:
        params = pickle.load(f)
    return make_policy(params, deterministic=True)


def run(impl, override):
    cfg = registry.get_default_config(ENV)
    cfg.impl = impl
    env = registry.load(ENV, config=cfg)
    policy = load_policy(env)
    reset, step, inf = jax.jit(env.reset), jax.jit(env.step), jax.jit(policy)
    rng = jax.random.PRNGKey(0)
    state = reset(rng)
    cmd = jp.array([1.0, 0.0, 0.0], jp.float32)
    traj = []
    for i in range(400):
        obs = state.obs
        if override:
            obs = {**obs, "state": obs["state"].at[-3:].set(cmd)}
        act_rng, rng = jax.random.split(rng)
        act, _ = inf(obs, act_rng)
        state = step(state, act)
        h = float(np.asarray(state.data.qpos)[2])
        if i % 50 == 0 or h < 0.15:
            traj.append((i, h))
            if h < 0.05:
                break
    tag = f"impl={impl} override={override}"
    print(f"[roll] {tag}: " + " ".join(f"{i}:{h:.2f}" for i, h in traj[:12]))


def main():
    for impl in ["jax", "warp"]:
        try:
            run(impl, override=False)
            run(impl, override=True)
        except Exception as e:
            print(f"[roll] impl={impl} FAILED: {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
