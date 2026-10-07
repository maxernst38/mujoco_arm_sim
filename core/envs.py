from functools import partial

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


class FixedResetOptions(gym.Wrapper):
    def __init__(self, env, options):
        super().__init__(env)
        self.options = options

    def reset(self, **kwargs):
        options = dict(kwargs.pop("options", None) or {})
        options.update(self.options)
        return self.env.reset(options=options, **kwargs)


def make_env(project, evaluation=False):
    env = project.env_cls()
    if evaluation and project.eval_reset_options:
        env = FixedResetOptions(env, project.eval_reset_options)
    env = TimeLimit(env, max_episode_steps=project.max_episode_steps)
    return Monitor(env)


def make_train_env(project, n_envs):
    if n_envs == 1:
        return make_env(project)
    return SubprocVecEnv([partial(make_env, project) for _ in range(n_envs)])
