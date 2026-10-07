from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym


@dataclass
class Project:
    name: str
    env_cls: Callable[[], gym.Env]
    model_path: Path
    max_episode_steps: int
    # reset options used for evaluation, e.g. always start from the hard case
    eval_reset_options: dict = field(default_factory=dict)
    # (infos for one episode, env dt) -> (success, {metric name: formatted value})
    summarize_episode: Optional[Callable[[list, float], tuple]] = None
