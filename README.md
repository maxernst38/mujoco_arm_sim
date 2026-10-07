# mujoco_arm_sim

A place to build, train and simulate robot arms in MuJoCo. Each robot/task lives
in its own folder under `projects/`, and the shared scripts (`view.py`,
`train.py`, `eval.py`) work on any of them via `--project`.

The default project is `double_pendulum`: a pendubot (motor on the shoulder only)
that learns to swing up from hanging and balance upright with SAC.

## Setup

```bash
conda env create -f environment.yml
conda activate mujoco-arm
```

## Usage

```bash
python view.py                         # open the model in the MuJoCo viewer
python train.py                        # train SAC, saves to runs/<project>/...
python eval.py                         # evaluate the newest run's best model
python eval.py --render --episodes 3   # watch it
tensorboard --logdir runs

./run.sh double_pendulum 10000000 10 4 # 4 seeds x 10 envs x 10M steps, in the background
```

All scripts default to `--project double_pendulum`. `python train.py --help` lists the options.

`--n-envs N` runs N environments in parallel, but with the default
`--gradient-steps 1` that also means N times fewer network updates per
transition, so it's faster per step but learns less per step.

## Layout

```
core/                  shared code
  project.py           Project definition (what a project has to provide)
  envs.py              env construction: TimeLimit, Monitor, parallel envs
projects/
  __init__.py          registry of available projects
  double_pendulum/     example project
    double_pendulum.xml
    env.py
    __init__.py        PROJECT definition
    pretrained/        saved models
train.py  eval.py  view.py  run.sh
runs/                  training output (gitignored), one folder per project
```

## Adding a project

1. Make `projects/<name>/` with the MJCF model and an `env.py` containing a
   `gymnasium.Env`. The env should set `self.dt` (seconds per `step()`), which
   `eval.py` uses for real-time playback.
2. In `projects/<name>/__init__.py`, define `PROJECT = Project(...)`:
   - `name`, `env_cls`, `model_path`, `max_episode_steps`
   - `eval_reset_options` (optional): reset options forced during evaluation
   - `summarize_episode` (optional): `(infos, dt) -> (success, metrics)`, used
     by `eval.py` to print per-episode metrics and a success count
3. Add it to the tuple in `projects/__init__.py`.

Then `python train.py --project <name>` etc. works. `projects/double_pendulum`
is the reference for all of the above.
