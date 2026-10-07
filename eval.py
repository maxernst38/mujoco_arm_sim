"""
    python eval.py
    python eval.py --which final
    python eval.py --model runs/sac/final_model.zip
    python eval.py --render --episodes 3
"""
import argparse
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
from stable_baselines3 import SAC

from env import FRAME_SKIP
from train import make_env

HOLD_SECONDS = 3.0  # success = balanced for the whole last HOLD_SECONDS


def latest_model(runs_dir, which):
    runs = [d for d in Path(runs_dir).iterdir() if d.is_dir()]
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
    returns, balanced, tip_heights = 0.0, [], []
    dt = FRAME_SKIP * env.unwrapped.model.opt.timestep
    done = False
    while not done:
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        returns += reward
        balanced.append(info["is_balanced"])
        tip_heights.append(info["tip_height"])
        done = terminated or truncated
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
            if not viewer.is_running():
                break
    return returns, np.array(balanced), np.array(tip_heights), dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=None, help="explicit model path")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--which", choices=["best", "final"], default="best")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    model_path = args.model or latest_model(args.runs_dir, args.which)
    print(f"model: {model_path}")
    model = SAC.load(str(model_path), device="cpu")
    env = make_env(hanging_only=True)

    viewer_cm = (
        mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data)
        if args.render
        else None
    )
    viewer = viewer_cm.__enter__() if viewer_cm else None

    successes, rows = 0, []
    try:
        for ep in range(args.episodes):
            ret, balanced, tip, dt = run_episode(model, env, args.seed + ep, viewer)
            hold_steps = int(HOLD_SECONDS / dt)
            success = len(balanced) >= hold_steps and balanced[-hold_steps:].all()
            first = f"{np.argmax(balanced) * dt:5.1f}s" if balanced.any() else "never"
            successes += success
            rows.append((ret, balanced.mean(), first, tip.max(), success))
            print(
                f"ep {ep:2d} | return {ret:8.1f} | balanced {balanced.mean():5.1%} "
                f"| first balanced: {first:>6} | max tip {tip.max():.2f} "
                f"| {'SUCCESS' if success else 'fail'}"
            )
            if viewer is not None and not viewer.is_running():
                break
    finally:
        if viewer_cm:
            viewer_cm.__exit__(None, None, None)

    n = len(rows)
    print(
        f"\n{successes}/{n} successful (balanced for the last {HOLD_SECONDS:.0f}s) | "
        f"mean return {np.mean([r[0] for r in rows]):.1f}"
    )


if __name__ == "__main__":
    main()
