"""
Pendubot swing-up: motor on the base joint only, elbow is passive.
qpos = 0 is hanging, upright is base = pi, elbow = 0. MuJoCo doesn't wrap
hinge angles, so wrap before comparing.
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

MODEL_PATH = Path(__file__).parent / "models" / "double_pendulum.xml"

FRAME_SKIP = 10           # 50 Hz control at dt = 0.002
MAX_EPISODE_STEPS = 1000  # 20 s, applied by TimeLimit in train.make_env

UPRIGHT_START_PROB = 0.4
UPRIGHT_POS_NOISE = 0.2
UPRIGHT_VEL_NOISE = 0.5
HANG_POS_NOISE = 0.05
HANG_VEL_NOISE = 0.05

# must match the XML
PIVOT_HEIGHT = 1.0
LINK_LENGTH = 0.4

BALANCE_ANGLE_TOL = 0.2
BALANCE_VEL_TOL = 1.0

TORQUE_LIMIT = 4.0        # must match ctrlrange in the XML
W_HEIGHT = 1.0
W_BALANCE = 2.0
W_TORQUE = 0.05
W_VEL = 0.01
SIGMA_ANGLE = 0.2
SIGMA_VEL = 2.0


def compute_reward(q, dq, torque):
    tip_z = PIVOT_HEIGHT - LINK_LENGTH * np.cos(q[0]) - LINK_LENGTH * np.cos(q[0] + q[1])
    height = (tip_z - (PIVOT_HEIGHT - 2 * LINK_LENGTH)) / (4 * LINK_LENGTH)

    err0 = q[0] % (2 * np.pi) - np.pi
    err1 = (q[1] + np.pi) % (2 * np.pi) - np.pi

    close = np.exp(-(err0**2 + err1**2) / SIGMA_ANGLE**2)
    speed_sq = float(np.dot(dq, dq))
    slow = np.exp(-speed_sq / SIGMA_VEL**2)
    balance = close * slow

    torque_cost = (torque / TORQUE_LIMIT) ** 2
    vel_cost = close * speed_sq

    reward = (
        W_HEIGHT * height
        + W_BALANCE * balance
        - W_TORQUE * torque_cost
        - W_VEL * vel_cost
    )
    return float(reward)


class PendubotEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.torque_scale = self.model.actuator_ctrlrange
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)

    def _get_obs(self):
        # qpos/qvel rather than sensordata, which lags a step behind
        q = self.data.qpos
        dq = self.data.qvel
        return np.array(
            [np.sin(q[0]), np.cos(q[0]), np.sin(q[1]), np.cos(q[1]), dq[0], dq[1]],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        force_hanging = bool(options and options.get("hanging", False))
        nq, nv = self.model.nq, self.model.nv

        if not force_hanging and self.np_random.random() < UPRIGHT_START_PROB:
            upright = np.array([np.pi, 0.0])
            self.data.qpos[:] = upright + self.np_random.uniform(
                -UPRIGHT_POS_NOISE, UPRIGHT_POS_NOISE, size=nq
            )
            self.data.qvel[:] = self.np_random.uniform(
                -UPRIGHT_VEL_NOISE, UPRIGHT_VEL_NOISE, size=nv
            )
        else:
            self.data.qpos[:] = self.np_random.uniform(
                -HANG_POS_NOISE, HANG_POS_NOISE, size=nq
            )
            self.data.qvel[:] = self.np_random.uniform(
                -HANG_VEL_NOISE, HANG_VEL_NOISE, size=nv
            )

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        torque = float(action[0]) * self.torque_scale[0, 1]
        self.data.ctrl[0] = torque

        mujoco.mj_step(self.model, self.data, nstep=FRAME_SKIP)

        obs = self._get_obs()
        q, dq = self.data.qpos, self.data.qvel
        reward = compute_reward(q, dq, torque)
        terminated = False
        truncated = False

        # computed from angles because site_xpos is stale right after mj_step
        tip_z = PIVOT_HEIGHT - LINK_LENGTH * np.cos(q[0]) - LINK_LENGTH * np.cos(q[0] + q[1])
        err0 = q[0] % (2 * np.pi) - np.pi
        err1 = (q[1] + np.pi) % (2 * np.pi) - np.pi
        is_balanced = bool(
            abs(err0) < BALANCE_ANGLE_TOL
            and abs(err1) < BALANCE_ANGLE_TOL
            and np.abs(dq).max() < BALANCE_VEL_TOL
        )
        info = {"tip_height": float(tip_z), "is_balanced": is_balanced}

        return obs, reward, terminated, truncated, info
