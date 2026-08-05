#!/usr/bin/env python3
"""Dump the Go1 joystick default config so we can find the perturbation /
push knobs to disable for a clean eval rollout (training pushes the robot;
eval should not)."""
from mujoco_playground import registry

ENV = "Go1JoystickFlatTerrain"
cfg = registry.get_default_config(ENV)
print("[cfg] ===== full default config =====")
print(cfg)
