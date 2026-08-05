#!/usr/bin/env python3
"""LAST attempt: close the warp->mjx contact gap by raising solver iterations.

Training ran on the `warp` backend; rendering runs on `jax`(mjx). mjx defaults
to fewer solver / line-search iterations, giving softer contacts that let the
trained policy slide into a fall after ~70 steps. We reload the env, bump
model.opt.iterations / ls_iterations to several levels, and roll out the
forward command to see which (if any) keeps trunk height up past 400 steps.
"""
import pickle
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


def run(iters, ls_iters):
    cfg = registry.get_default_config(ENV)
    cfg.impl = "jax"
    env = registry.load(ENV, config=cfg)
    # bump solver iterations on the underlying mj_model if we can reach it
    changed = False
    try:
        env.mj_model.opt.iterations = iters
        env.mj_model.opt.ls_iterations = ls_iters
        changed = True
        # rebuild the mjx model from the patched mj_model if the env exposes it
        import mujoco.mjx as mjx
        if hasattr(env, "_mjx_model"):
            env._mjx_model = mjx.put_model(env.mj_model)
    except Exception as e:
        print(f"[solver] iters={iters}: cannot set opt ({type(e).__name__})")

    policy = load_policy(env)
    reset, step, inf = jax.jit(env.reset), jax.jit(env.step), jax.jit(policy)
    rng = jax.random.PRNGKey(0)
    state = reset(rng)
    cmd = jp.array([1.0, 0.0, 0.0], jp.float32)
    last_h, fell_at = 0.29, None
    for i in range(400):
        obs = {**state.obs, "state": state.obs["state"].at[-3:].set(cmd)}
        act_rng, rng = jax.random.split(rng)
        act, _ = inf(obs, act_rng)
        state = step(state, act)
        last_h = float(np.asarray(state.data.qpos)[2])
        if last_h < 0.15 and fell_at is None:
            fell_at = i
            break
    dx = float(np.asarray(state.data.qpos)[0])
    verdict = f"FELL@{fell_at}" if fell_at is not None else f"OK survived 400 (final_h={last_h:.2f} dx={dx:+.2f})"
    print(f"[solver] iters={iters} ls={ls_iters} set={changed}: {verdict}")


def main():
    for it, ls in [(1, 5), (4, 8), (10, 20), (50, 50)]:
        try:
            run(it, ls)
        except Exception as e:
            print(f"[solver] iters={it} EXC {type(e).__name__}: {str(e)[:100]}")


if __name__ == "__main__":
    main()
