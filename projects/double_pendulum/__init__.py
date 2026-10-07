import numpy as np

from core.project import Project

from .env import MAX_EPISODE_STEPS, MODEL_PATH, PendubotEnv

HOLD_SECONDS = 3.0  # success = balanced for the whole last HOLD_SECONDS


def summarize_episode(infos, dt):
    balanced = np.array([info["is_balanced"] for info in infos])
    tip = np.array([info["tip_height"] for info in infos])
    hold_steps = int(HOLD_SECONDS / dt)
    success = bool(len(balanced) >= hold_steps and balanced[-hold_steps:].all())
    first = f"{np.argmax(balanced) * dt:.1f}s" if balanced.any() else "never"
    return success, {
        "balanced": f"{balanced.mean():5.1%}",
        "first balanced": f"{first:>6}",
        "max tip": f"{tip.max():.2f}",
    }


PROJECT = Project(
    name="double_pendulum",
    env_cls=PendubotEnv,
    model_path=MODEL_PATH,
    max_episode_steps=MAX_EPISODE_STEPS,
    eval_reset_options={"hanging": True},
    summarize_episode=summarize_episode,
)
