"""
    python eval.py                                    # newest run of the default project
    python eval.py --project double_pendulum --which final
    python eval.py --model projects/double_pendulum/pretrained/best_model.zip
    python eval.py --render --episodes 3
"""
import argparse
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
from stable_baselines3 import SAC

from core.envs import make_env
from projects import DEFAULT_PROJECT, PROJECTS, get_project


def latest_model(runs_dir, which):
    runs = [d for d in Path(runs_dir).iterdir() if d.is_dir()] if Path(runs_dir).is_dir() else []
    if not runs:
        raise SystemExit(f"no run directories found in {runs_dir}")
    newest = max(runs, key=lambda d: max((f.stat().st_mtime for f in d.rglob("*") if f.is_file()), default=0))
    order = ["best/best_model.zip", "final_model.zip"]
    if which == "final":
        order.reverse()
    for name in order:
        if (newest / name).exists():
            return newest / name
    checkpoints = sorted((newest / "checkpoints").glob("*.zip"), key=lambda f: f.stat().st_mtime)
    if checkpoints:
        return checkpoints[-1]
    raise SystemExit(f"no saved model found in {newest}")


def run_episode(model, env, seed, viewer=None):
    obs, _ = env.reset(seed=seed)
    returns, infos = 0.0, []
    dt = env.unwrapped.dt
    done = False
    while not done:
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        returns += reward
        infos.append(info)
        done = terminated or truncated
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
            if not viewer.is_running():
                break
    return returns, infos, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=DEFAULT_PROJECT, choices=sorted(PROJECTS))
    parser.add_argument("--model", type=Path, default=None, help="explicit model path")
    parser.add_argument("--runs-dir", type=Path, default=None, help="default: runs/<project>")
    parser.add_argument("--which", choices=["best", "final"], default="best")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    project = get_project(args.project)
    runs_dir = args.runs_dir or Path("runs") / project.name
    model_path = args.model or latest_model(runs_dir, args.which)
    print(f"project: {project.name} | model: {model_path}")
    model = SAC.load(str(model_path), device="cpu")
    env = make_env(project, evaluation=True)

    viewer_cm = (
        mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data)
        if args.render
        else None
    )
    viewer = viewer_cm.__enter__() if viewer_cm else None

    successes, returns = 0, []
    try:
        for ep in range(args.episodes):
            ret, infos, dt = run_episode(model, env, args.seed + ep, viewer)
            returns.append(ret)
            line = f"ep {ep:2d} | return {ret:8.1f}"
            if project.summarize_episode:
                success, metrics = project.summarize_episode(infos, dt)
                successes += success
                line += "".join(f" | {k}: {v}" for k, v in metrics.items())
                line += f" | {'SUCCESS' if success else 'fail'}"
            print(line)
            if viewer is not None and not viewer.is_running():
                break
    finally:
        if viewer_cm:
            viewer_cm.__exit__(None, None, None)

    summary = f"\nmean return {np.mean(returns):.1f} over {len(returns)} episodes"
    if project.summarize_episode:
        summary += f" | {successes}/{len(returns)} successful"
    print(summary)


if __name__ == "__main__":
    main()
