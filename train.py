"""
    python train.py                                   # default project (double_pendulum)
    python train.py --project double_pendulum --steps 2000000 --seed 1
    python train.py --n-envs 4
"""
import argparse
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from core.envs import make_env, make_train_env
from projects import DEFAULT_PROJECT, PROJECTS, get_project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=DEFAULT_PROJECT, choices=sorted(PROJECTS))
    parser.add_argument("--steps", type=int, default=500_000, help="total env steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="default: runs/<project>/sac_<date>_<time>_s<seed>")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-envs", type=int, default=1, help="parallel environments")
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="updates per env.step() of the vec env; -1 = one per transition collected",
    )
    parser.add_argument("--threads", type=int, default=2, help="torch CPU threads")
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    args = parser.parse_args()

    project = get_project(args.project)
    if args.out is None:
        args.out = Path("runs") / project.name / f"sac_{datetime.now():%Y%m%d_%H%M%S}_s{args.seed}"
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"project: {project.name} | run directory: {args.out}")

    torch.set_num_threads(args.threads)

    train_env = make_train_env(project, args.n_envs)
    eval_env = make_env(project, evaluation=True)

    model = SAC(
        "MlpPolicy",
        train_env,
        gamma=args.gamma,
        learning_rate=args.lr,
        batch_size=256,
        learning_starts=args.learning_starts,
        gradient_steps=args.gradient_steps,
        seed=args.seed,
        device="cpu",
        tensorboard_log=str(args.out / "tb"),
        verbose=1,
    )

    # callback frequencies are counted in vec-env steps, i.e. n_envs transitions each
    callbacks = CallbackList(
        [
            EvalCallback(
                eval_env,
                n_eval_episodes=5,
                eval_freq=max(args.eval_freq // args.n_envs, 1),
                deterministic=True,
                best_model_save_path=str(args.out / "best"),
                log_path=str(args.out / "eval"),
            ),
            CheckpointCallback(
                save_freq=max(args.steps // 10 // args.n_envs, 1),
                save_path=str(args.out / "checkpoints"),
                name_prefix="sac",
            ),
        ]
    )

    model.learn(total_timesteps=args.steps, callback=callbacks)
    model.save(str(args.out / "final_model"))


if __name__ == "__main__":
    main()
