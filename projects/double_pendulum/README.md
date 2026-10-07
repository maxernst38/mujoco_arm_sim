# double_pendulum

Pendubot swing-up: two 0.4 m links, a ±4 N·m motor on the base joint, passive elbow.
Starts hanging; the goal is to swing up and stay balanced upright.

- Observation: `[sin q0, cos q0, sin q1, cos q1, dq0, dq1]`
- Action: one value in [-1, 1], scaled to the motor's `ctrlrange`
- Episode: 1000 steps at 50 Hz (20 s), no early termination
- Training resets: 40% near upright (to practice the hold), the rest hanging.
  Evaluation always starts hanging.
- Success (in `eval.py`): balanced for the whole last 3 s

If you change the geometry or motor limit in the XML, update `PIVOT_HEIGHT`,
`LINK_LENGTH` and `TORQUE_LIMIT` in `env.py` to match.

```bash
python view.py --project double_pendulum
python eval.py --model projects/double_pendulum/pretrained/best_model.zip --render --episodes 3
```
