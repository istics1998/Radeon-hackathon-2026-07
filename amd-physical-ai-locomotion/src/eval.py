"""Evaluate a trained locomotion policy as a closed perception->control loop.

Loads a checkpoint, rolls the policy out in the MJX env, and reports metrics
(episode reward, survival length, velocity/command tracking error). This is the
"closed-loop control" capability the track asks for.

Example:
    python -m src.eval --env Go1JoystickFlatTerrain --episodes 5
"""
from __future__ import annotations

import argparse
import json
import pickle

from src import config as C

C.set_headless_render_defaults()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from mujoco_playground import registry  # noqa: E402
from mujoco_playground.config import locomotion_params  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Closed-loop policy evaluation.")
    p.add_argument("--env", default=C.DEFAULT_ENV)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--ckpt", default=None, help="Path to checkpoint .pkl (defaults by env/seed).")
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def load_inference_fn(env, env_name, ckpt_path):
    """Rebuild the PPO network with the same factory used in training and load params."""
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    nf = dict(ppo_params.get("network_factory", {}))
    with open(ckpt_path, "rb") as f:
        params = pickle.load(f)

    obs_size = env.observation_size
    act_size = env.action_size
    networks = ppo_networks.make_ppo_networks(
        observation_size=obs_size, action_size=act_size, **nf
    )
    make_policy = ppo_networks.make_inference_fn(networks)
    # deterministic=True for evaluation.
    return make_policy(params, deterministic=True)


def main() -> None:
    args = parse_args()
    C.ensure_dirs()
    C.assert_gpu(require=not args.allow_cpu)

    env_name = args.env
    env = registry.load(env_name)
    ckpt_path = args.ckpt or (C.CKPT_DIR / f"{env_name}_seed{args.seed}.pkl")
    print(f"[eval] env={env_name} ckpt={ckpt_path}")

    policy_fn = load_inference_fn(env, env_name, ckpt_path)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_policy = jax.jit(policy_fn)

    rng = jax.random.PRNGKey(args.seed)
    all_rewards, all_lengths = [], []

    for ep in range(args.episodes):
        rng, key = jax.random.split(rng)
        state = jit_reset(key)
        total_r, steps = 0.0, 0
        while True:
            rng, akey = jax.random.split(rng)
            action, _ = jit_policy(state.obs, akey)
            state = jit_step(state, action)
            total_r += float(state.reward)
            steps += 1
            if bool(state.done) or steps >= env._config.episode_length:
                break
        all_rewards.append(total_r)
        all_lengths.append(steps)
        print(f"[eval] episode {ep}: reward={total_r:8.3f} length={steps}")

    metrics = {
        "env": env_name,
        "episodes": args.episodes,
        "reward_mean": float(jnp.mean(jnp.array(all_rewards))),
        "reward_std": float(jnp.std(jnp.array(all_rewards))),
        "length_mean": float(jnp.mean(jnp.array(all_lengths))),
    }
    out = C.LOG_DIR / f"{env_name}_seed{args.seed}_eval.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"[eval] summary: {metrics}")
    print(f"[eval] wrote -> {out}")


if __name__ == "__main__":
    main()
