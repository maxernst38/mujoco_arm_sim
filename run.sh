#!/usr/bin/env bash
# usage: ./run.sh [project] [steps] [n_envs] [n_seeds]
# logs go to runs/<project>/vec<n_envs>_s<seed>.log; stop with: pkill -f "train.py --project"

PROJECT=${1:-double_pendulum}
STEPS=${2:-10000000}
N_ENVS=${3:-10}
N_SEEDS=${4:-4}

cd "$(dirname "$0")"
mkdir -p "runs/$PROJECT"
for ((s = 0; s < N_SEEDS; s++)); do
  out="runs/$PROJECT/vec${N_ENVS}_s$s"
  nohup python train.py --project "$PROJECT" --n-envs "$N_ENVS" --steps "$STEPS" --seed "$s" \
    --eval-freq 100000 --out "$out" > "$out.log" 2>&1 &
done
