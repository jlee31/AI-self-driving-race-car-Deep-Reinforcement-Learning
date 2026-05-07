# Car Reinforcement Learning

A self-driving race car trained with Deep Reinforcement Learning on the [CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/#starting-state) environment.

The goal is for the agent to score an average of **900+ points over 100 consecutive races** — the benchmark for "solving" the environment.

---

## What this is

This is a personal rewrite of [this TF1/Gym DQN project](tf1_original/README.md), rebuilt from scratch using modern libraries:

| Original (`tf1_original/`) | This rewrite (`src/`) |
| --- | --- |
| TensorFlow 1.x | PyTorch |
| `gym==0.17.3` (deprecated) | `gymnasium>=1.0.0` |
| 5 hardcoded discrete actions | TBD — continuous or discrete |
| DQN | TBD — exploring PPO / DQN |

---

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** `gymnasium[box2d]` requires `swig`. If the install fails, run `brew install swig` first (macOS).

---
