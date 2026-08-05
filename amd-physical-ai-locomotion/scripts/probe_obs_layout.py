#!/usr/bin/env python3
"""Diagnostic: find WHERE the joystick command lives inside obs['state'].

We never trust the '-3:' slice again without proof. This resets the env,
reads state.info['command'], then locates those exact values inside
obs['state'] so we know the true index. Also checks whether the saved
params include a normalizer (which would sit in front of the actor).
"""
import pickle
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jp

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import networks as ppo_networks

ENV = "Go1JoystickFlatTerrain"
CKPT = Path("checkpoints/latest.pkl")

cfg = registry.get_default_config(ENV)
cfg.impl = "jax"
env = registry.load(ENV, config=cfg)

print("[probe] observation_size:", env.observation_size)
print("[probe] action_size:", env.action_size)

rng = jax.random.PRNGKey(0)
state = jax.jit(env.reset)(rng)

print("[probe] obs keys:", list(state.obs.keys()) if isinstance(state.obs, dict) else "ARRAY")
obs_state = np.asarray(state.obs["state"] if isinstance(state.obs, dict) else state.obs)
print("[probe] obs['state'].shape:", obs_state.shape)

info = state.info
print("[probe] info keys:", list(info.keys()))
cmd = np.asarray(info["command"]) if "command" in info else None
print("[probe] info['command']:", cmd)

if cmd is not None:
    # locate each command component inside obs['state']
    for ci, cv in enumerate(cmd):
        hits = np.where(np.isclose(obs_state, cv, atol=1e-5))[0]
        print(f"[probe]   cmd[{ci}]={cv:+.4f} found at obs indices: {hits.tolist()}")
    print("[probe] last 3 obs values:", obs_state[-3:].tolist())
    print("[probe] does obs[-3:] == info['command']?",
          np.allclose(obs_state[-3:], cmd, atol=1e-5))

# check params structure for a normalizer
with open(CKPT, "rb") as f:
    params = pickle.load(f)
print("[probe] params type:", type(params).__name__,
      "len:" , len(params) if isinstance(params, (tuple, list)) else "n/a")

# step once WITHOUT any override, see if command persists in info
step = jax.jit(env.step)
zero_act = jp.zeros(env.action_size)
s2 = step(state, zero_act)
print("[probe] info['command'] after 1 step (no override):",
      np.asarray(s2.info["command"]))
