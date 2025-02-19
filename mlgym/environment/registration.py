"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Environment registration module.

This module provides functionality to register ML tasks as Gymnasium environments.
It allows for the registration of environments with unique IDs and entry points.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from mlgym.environment.env import EnvironmentArguments, MLGymEnv


# TODO: this should accept a EnvironmentConfig. This environment config should already have the task config loaded so that we can register the task with the correct entrypoint.
def register_task(env_config: EnvironmentArguments, *args, nondeterministic: bool = True, **kwargs):
    """
    Registers a ML task as a gym environment with its unique id.

    Args:
        task_config_path: the path to the task config file
        nondeterministic: whether the task cannot be guaranteed deterministic transitions.
        *args: additional arguments for the mlgym environment.
        *kwargs: additional arguments for the mlgym environment.
    """
    task_id = env_config.task.id  # type: ignore
    # ! FIXME: Cannot skip environment registration here for some reason. It causes the environment to be initialized with default configs.

    gym.register(
        id=f"mlgym/{task_id}",
        entry_point=lambda *env_args, **env_kwargs: MLGymEnv(env_config, *env_args, **env_kwargs),
        nondeterministic=nondeterministic,
        *args,
        **kwargs,
    )
