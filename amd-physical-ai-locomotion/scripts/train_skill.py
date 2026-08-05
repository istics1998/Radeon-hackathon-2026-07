#!/usr/bin/env python3
"""Train a Go1 SKILL policy (Footstand / Handstand) with brax PPO on this instance.

Same machinery as train_go1.py, but parameterised for the skill environments,
which have no joystick command (obs["state"] is 45-dim, not 48). Pick the env
with the SKILL_ENV var; checkpoints go to a per-skill dir so they never clash
with the flat-terrain locomotion checkpoints.

The AMD Radeon GPU is blocked by a ROCm profiler race (see docs/ROCM_BUG_REPORT.md),
so JAX falls back to CPU. The upstream default is 100M steps; that is impractical
on CPU, so TRAIN_STEPS defaults to a smaller amount that still produces a clear
pose. Checkpoints are written periodically so the demo can render the best one.

Usage (on the instance):
    cd /workspace/demo
    # footstand (rear-leg stand):
    SKILL_ENV=Go1Footstand MUJOCO_GL=egl nohup python3 scripts/train_skill.py > train_footstand.log 2>&1 &
    # handstand (front-leg stand):
    SKILL_ENV=Go1Handstand MUJOCO_GL=egl nohup python3 scripts/train_skill.py > train_handstand.log 2>&1 &
    tail -f train_footstand.log        # reward should climb; ckpt every ~2M steps
"""
import os
import time
import pickle
import functools
from pathlib import Path

import jax
import jax.numpy as jnp

# --- shim: JAX 0.11 removed device_put_replicated; brax 0.14.2 still calls it.
# CPU has a single device, so a plain stack + device_put is equivalent. -------
if not hasattr(jax, "device_put_replicated"):
    def _dpr(x, devices):
        n = len(devices)
        stacked = jax.tree_util.tree_map(
            lambda l: jnp.broadcast_to(jnp.asarray(l), (n,) + jnp.asarray(l).shape), x)
        if n == 1:
            return jax.device_put(stacked, devices[0])
        return jax.device_put_sharded(
            [jax.tree_util.tree_map(jnp.asarray, x)] * n, devices)
    jax.device_put_replicated = _dpr
# ---------------------------------------------------------------------------

from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

# for the RSI (reference-state init) Footstand subclass
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import handstand as go1_handstand

ENV = os.environ.get("SKILL_ENV", "Go1Footstand")
assert ENV in ("Go1Footstand", "Go1Handstand", "Go1Getup"), f"unexpected SKILL_ENV={ENV}"
# Reference-state init: fraction of episodes reset directly to the balanced
# 'footstand' keyframe so the policy escapes the sitting local optimum. Set
# RSI_PROB=0 to reproduce the vanilla (sits) behaviour.
RSI_PROB = float(os.environ.get("RSI_PROB", 0.5))


class FootstandRSI(go1_handstand.Footstand):
    """Go1 Footstand + reference-state initialization.

    Vanilla Footstand plateaus in a "sit": torso ~0.22 m, all four feet in the
    air, haunches/base resting on the floor (verified with probe_footstand_pose).
    The reward (smooth height + orientation, no fall while seated) makes sitting
    a stable high-value basin that pure PPO exploration never leaves.

    Fix: on reset, with probability RSI_PROB start from the model's own
    'footstand' keyframe (a balanced rear-leg stand, already stored as
    self._handstand_q by Footstand._post_init) instead of the home pose. The
    policy then experiences the standing state, learns to hold it, and the value
    function pulls home-started rollouts toward it too. Reward is left untouched.
    """

    def _post_init(self) -> None:
        super()._post_init()
        self._rsi_prob = RSI_PROB

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, ref_rng = jax.random.split(rng)
        from_ref = jax.random.bernoulli(ref_rng, self._rsi_prob)
        # _handstand_q is the FOOTSTAND keyframe for this subclass (set in
        # Footstand._post_init); _init_q is the home stance.
        qpos = jnp.where(from_ref, self._handstand_q, self._init_q)

        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = mjx_math.axis_angle_to_quat(jnp.array([0, 0, 1]), yaw)
        qpos = qpos.at[3:7].set(mjx_math.quat_mul(qpos[3:7], quat))

        # home starts get a velocity kick (as upstream); reference-stand starts
        # begin at rest so the policy first learns to simply hold the pose.
        qvel_kick = jnp.zeros(self.mjx_model.nv)
        rng, key = jax.random.split(rng)
        qvel_kick = qvel_kick.at[0:6].set(
            jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5))
        qvel = jnp.where(from_ref, jnp.zeros(self.mjx_model.nv), qvel_kick)

        data = mjx_env.make_data(
            self.mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:],
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax, njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        info = {"step": 0, "rng": rng, "last_act": jnp.zeros(self.mjx_model.nu)}
        metrics = {k: jnp.zeros(())
                   for k in self._config.reward_config.scales.keys()}
        contact = jnp.array([
            data.sensordata[self._mj_model.sensor_adr[sid]] > 0
            for sid in self._fullcollision_floor_found_sensor
        ])
        obs = self._get_obs(data, info, contact)
        reward, done = jnp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)
TOTAL_STEPS = int(os.environ.get("TRAIN_STEPS", 40_000_000))
NUM_ENVS = int(os.environ.get("TRAIN_ENVS", 1024))
# per-skill checkpoint dir so we never overwrite the locomotion checkpoints
CKPT_DIR = Path(f"checkpoints_{ENV.lower()}"); CKPT_DIR.mkdir(exist_ok=True)


def main():
    print(f"[train] jax devices: {jax.devices()}", flush=True)
    print(f"[train] env={ENV} total_steps={TOTAL_STEPS} num_envs={NUM_ENVS} "
          f"ckpt_dir={CKPT_DIR}", flush=True)

    cfg = registry.get_default_config(ENV)
    cfg.impl = "jax"                        # force JAX-MJX (no Warp/GPU)
    if ENV == "Go1Footstand" and RSI_PROB > 0:
        print(f"[train] Footstand with reference-state init, RSI_PROB={RSI_PROB}", flush=True)
        env = FootstandRSI(config=cfg)
    else:
        env = registry.load(ENV, config=cfg)

    ppo_params = locomotion_params.brax_ppo_config(ENV)
    ppo_params.num_timesteps = TOTAL_STEPS
    ppo_params.num_envs = NUM_ENVS
    ppo_params.num_evals = max(2, TOTAL_STEPS // 2_000_000)   # ckpt every ~2M steps

    kw = dict(ppo_params)
    nf = kw.pop("network_factory", None)
    network_factory = (functools.partial(ppo_networks.make_ppo_networks, **nf)
                       if nf else ppo_networks.make_ppo_networks)

    t0 = time.time()

    def progress(step, metrics):
        r = metrics.get("eval/episode_reward")
        rl = metrics.get("eval/episode_reward_std")
        dt = time.time() - t0
        sps = step / dt if dt > 0 else 0
        print(f"[train] step={step:>10}  reward={r}  std={rl}  "
              f"elapsed={dt/60:.1f}min  {sps:.0f} steps/s", flush=True)

    def save_ckpt(step, make_policy, params):
        path = CKPT_DIR / f"{ENV.lower()}_{int(step)}.pkl"
        with open(path, "wb") as f:
            pickle.dump(params, f)
        # keep a stable 'latest' pointer for the renderer
        with open(CKPT_DIR / "latest.pkl", "wb") as f:
            pickle.dump(params, f)
        print(f"[train] saved checkpoint {path}", flush=True)

    train_fn = functools.partial(
        ppo.train, **kw,
        network_factory=network_factory,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        randomization_fn=registry.get_domain_randomizer(ENV),
        progress_fn=progress,
        policy_params_fn=save_ckpt,
    )

    make_inference_fn, params, _ = train_fn(environment=env)

    with open(CKPT_DIR / "final.pkl", "wb") as f:
        pickle.dump(params, f)
    print(f"[train] DONE in {(time.time()-t0)/60:.1f} min. "
          f"Final params -> {CKPT_DIR/'final.pkl'}", flush=True)


if __name__ == "__main__":
    main()
