#!/usr/bin/env bash
# logs go to runs/vec10_10M_s<seed>.log; stop with: pkill -f "train.py --n-envs 10"

cd /home/maxer/mujoco_arm_sim
for s in 0 1 2 3; do
  nohup python train.py --n-envs 10 --steps 10000000 --seed $s --eval-freq 100000 \
    --out runs/vec10_10M_s$s > runs/vec10_10M_s$s.log 2>&1 &
done
