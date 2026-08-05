#!/usr/bin/env python3
"""ROOT-CAUSE test: does the policy need observation NORMALIZATION at inference?

Training used brax ppo.train, whose config almost certainly sets
normalize_observations=True. That makes params[0] a running mean/std normalizer
and wires preprocess_observations_fn=running_statistics.normalize into the net.

The render script rebuilt the net WITHOUT that preprocess fn, so it fed RAW obs
to a policy trained on NORMALIZED obs -> scale mismatch -> falls ~step 70.

This rebuilds the net WITH normalization and rolls out forward. If it stays up,
that's the fix and I patch render_policy_video.py to match.
"""
import pickle
from pathlib import Path
import numpy as np
import jax, jax.numpy as jp

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

ENV = "Go1JoystickFlatTerrain"
CKPT = Path("checkpoints/latest.pkl")

ppo_params = locomotion_params.brax_ppo_config(ENV)
print("[norm] normalize_observations in config:",
      dict(ppo_params).get("normalize_observations"))
with open(CKPT, "rb") as f:
    params = pickle.load(f)
print("[norm] params[0] type:", type(params[0]).__name__)
# params[0] is the normalizer state; show it has real mean/std, not identity
try:
    mean = jax.tree_util.tree_leaves(params[0])
    print("[norm] normalizer leaf count:", len(mean),
          "first leaf shape:", np.asarray(mean[0]).shape if mean else None)
except Exception as e:
    print("[norm] normalizer introspect failed:", e)


def build(env, normalize):
    nf = dict(ppo_params).get("network_factory", {})
    pre = running_statistics.normalize if normalize else None
    kw = dict(nf)
    if pre is not None:
        kw["preprocess_observations_fn"] = pre
    net = ppo_networks.make_ppo_networks(env.observation_size, env.action_size, **kw)
    return ppo_networks.make_inference_fn(net)(params, deterministic=True)


def rollout(env, policy):
    reset, step, inf = jax.jit(env.reset), jax.jit(env.step), jax.jit(policy)
    rng = jax.random.PRNGKey(0)
    state = reset(rng)
    cmd = jp.array([1.0, 0.0, 0.0], jp.float32)
    fell = None
    for i in range(400):
        obs = {**state.obs, "state": state.obs["state"].at[-3:].set(cmd)}
        r, rng = jax.random.split(rng)
        act, _ = inf(obs, r)
        state = step(state, act)
        h = float(np.asarray(state.data.qpos)[2])
        if h < 0.15:
            fell = i; break
    dx = float(np.asarray(state.data.qpos)[0])
    return fell, dx


def main():
    cfg = registry.get_default_config(ENV); cfg.impl = "jax"
    env = registry.load(ENV, config=cfg)
    for norm in [False, True]:
        try:
            policy = build(env, norm)
            fell, dx = rollout(env, policy)
            v = f"FELL@{fell}" if fell is not None else f"OK 400 steps dx={dx:+.2f}"
            print(f"[norm] normalize={norm}: {v}")
        except Exception as e:
            print(f"[norm] normalize={norm} EXC {type(e).__name__}: {str(e)[:140]}")


if __name__ == "__main__":
    main()
