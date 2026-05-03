# How This Repo Works — A Complete Explanation

This project trains an AI agent to drive a race car using **Deep Q-Network (DQN)** reinforcement learning. The goal is to score an average of 900 points over 100 consecutive races — the OpenAI benchmark for "solving" the CarRacing environment.

---

## Is Pygame Used?

**No.** The car and track visuals are rendered by **Pyglet** (a Python graphics library), which OpenAI Gym uses internally for its CarRacing-v0 environment. You never write any rendering code yourself — Gym handles it. `requirements.txt` lists `pyglet==1.5.11` but no `pygame`.

---

## The Big Picture

The AI learns by trial and error, just like a human would:

1. It watches the game screen (camera input)
2. It picks an action (steer left, gas, brake, etc.)
3. The game gives it a reward or penalty
4. It remembers what happened and learns from it over time

After thousands of episodes, it figures out what actions lead to good scores.

---

## The Car Environment

The car is **not coded from scratch**. It comes from **OpenAI Gym's `CarRacing-v0`** environment.

```python
import gym
env = gym.make('CarRacing-v0')
```

Gym provides:
- A procedurally generated race track every episode
- A top-down car with realistic physics (via the **Box2D** physics engine)
- A 96×96 pixel RGB screenshot of the track as the "observation"
- A **continuous** action space: `[steering, gas, brake]` each as a float

The only job of this repo's code is to **watch** those screenshots and **decide** what actions to take.

---

## File-by-File Breakdown

### [main.py](main.py) — The Entry Point

This is where you run everything. It:

- Creates the Gym environment
- Creates the DQN agent with a config dictionary of all hyperparameters
- Runs a loop for 15,000 episodes (training) or 150 (testing with a loaded checkpoint)
- Saves the best models and logs scores to a text file

Key config values:
| Parameter | Value | What it does |
|---|---|---|
| `batchsize` | 64 | How many experiences to learn from at once |
| `gamma` | 0.95 | How much the AI values future rewards vs immediate ones |
| `frame_skip` | 3 | Each action is repeated 3 game frames |
| `initial_epsilon` | 1.0 | Starts fully random (exploring) |
| `min_epsilon` | 0.05 | Eventually only 5% random (mostly exploiting learned knowledge) |
| `epsilon_decay_steps` | 100,000 | How many frames until epsilon reaches its minimum |
| `target_network_update_freq` | 1,000 | How often to sync the two neural networks |
| `experience_capacity` | 150,000 | How many past experiences to remember |
| `max_negative_rewards` | 8 | Quit the episode early if stuck |

---

### [car_dqn.py](car_dqn.py) — The Racing-Specific Agent

This file wraps the generic DQN with car-specific logic.

**The Action Space Problem:**

The real CarRacing environment has a *continuous* action space — you could steer to any angle, apply any amount of gas, etc. Continuous actions are much harder to learn. This project solves that by converting to **5 discrete actions**:

```python
actions = [
    [-1, 0,   0  ],  # Turn left
    [ 0, 1,   0  ],  # Accelerate (gas)
    [ 0, 0,   0.5],  # Brake
    [ 0, 0,   0  ],  # Do nothing
    [ 1, 0,   0  ],  # Turn right
]
```

The AI now just picks one of these 5 options each step. Much simpler.

**Biased Exploration:**

When exploring randomly, the code makes the car *more likely* to press gas (14x more likely than any other action). Without this, the car would sit still or brake a lot during random exploration, learning nothing useful.

**Early Stopping:**

If the car gets more than 8 consecutive negative rewards (it's probably off-track or stuck), the episode ends early with a -20 point penalty. This stops the AI from wasting time in hopeless situations.

---

### [dqn.py](dqn.py) — The Core Learning Algorithm

This is the heart of the project. It implements **Deep Q-Network (DQN)**, a classic reinforcement learning algorithm.

#### What is DQN?

A **Q-value** is the expected total future reward for taking a specific action in a specific state. The AI learns a function `Q(state, action) → expected_reward`.

During play:
- It picks the action with the **highest Q-value** (exploitation)
- Sometimes it picks a **random action** (exploration, controlled by epsilon)

#### Two Neural Networks

The algorithm uses **two identical neural networks** — a key trick for stability:

| Network | Role |
|---|---|
| **Train network** | Gets updated constantly via backprop |
| **Target network** | Frozen copy, only updated every 1,000 steps |

The target network provides stable "ground truth" Q-values to learn from. If you only had one network, the targets would shift every step, causing training to oscillate and diverge.

#### Neural Network Architecture

Both networks have the same structure:

```
Input: 96×96 grayscale image (×3 stacked frames) = (96, 96, 3)
  ↓
Conv2D: 8 filters, 7×7 kernel, stride 4
  ↓ ReLU
MaxPool: 2×2, stride 2
  ↓
Conv2D: 16 filters, 3×3 kernel, stride 1
  ↓ ReLU
MaxPool: 2×2, stride 2
  ↓
Flatten
  ↓
Dense: 400 units, ReLU
  ↓
Dense: 5 units (one Q-value per action)
```

The conv layers detect track features (edges, curves). The dense layers combine those features into action decisions.

#### The Training Step

Every 3 game steps, if there are enough experiences stored:

1. Sample 64 random past experiences from memory
2. For each experience, compute the **target Q-value**:
   ```
   Q_target = reward + 0.95 * max(Q_target_network(next_state)) * (1 - done)
   ```
   (If done=1, future reward is 0 — episode ended)
3. Compute **current Q-estimate** from train network
4. Minimize the squared error between target and estimate using Adam optimizer

#### Reward Shaping

For positive rewards mid-episode, a small bonus is added:
```
reward += 0.2 * current_frame_number
```
This encourages the car to make progress deeper into the track (later frames = higher bonus).

---

### [exp_replay.py](exp_replay.py) — Experience Replay Buffer

The replay buffer is how the AI remembers past experiences and learns from them later.

**Why not just learn from each step immediately?**

If you trained on each game frame as it happened, consecutive frames are almost identical and highly correlated. The network would overfit to recent experiences and forget older ones. Randomly sampling from a large buffer of past experiences breaks those correlations.

**How it works:**

- Stores up to 150,000 experiences as a circular buffer (oldest gets overwritten)
- Each experience: `(state_frames, action, reward, next_state_frames, done)`
- Frame stacking: stores raw frames and builds states by combining the last 3 frames

**Frame Stacking** gives the AI a sense of motion — it can infer the car's velocity and direction from how the track moves across 3 consecutive frames.

---

### [processimage.py](processimage.py) — Image Preprocessing

Raw RGB screenshots from Gym are converted to a compact, normalized grayscale format:

1. **Grayscale conversion** using `skimage.color.rgb2gray`
2. **Mask the HUD** — sets the score/reward display area (rows 84-95, cols 0-12) to black so the AI ignores the UI
3. **Road segmentation** — thresholds specific pixel values to highlight track vs. grass
4. **Normalize to [-1, 1]**: `output = 2 * gray - 1`

This reduces the input size and removes noise, making learning faster and more reliable.

---

## Training Results

The `Train_Test_Data/` folder contains logs from 24 training runs:

- **CPU tests** (train05–14): Early experiments
- **GPU tests** (train16–24): Faster training with better results
- **train24**: The best checkpoint, saved at step 1,444,919

Results are logged per episode and visualized with matplotlib/seaborn. The final model achieves strong mean scores across 100 test episodes.

---

## How to Read the Learning Curve

As training progresses you'd see:
- **Epsilon** dropping from 1.0 → 0.05 (less random exploration over time)
- **100-episode average score** slowly rising
- High variance early on, more consistent later

DQN on CarRacing is notoriously slow — it takes hundreds of thousands of steps before the car learns to reliably stay on track.

---

## If You Want to Make Your Own Version

Here are the key levers to change:

| What to change | Where | Effect |
|---|---|---|
| Action set | `car_dqn.py` lines ~10-16 | Add more fine-grained steering angles |
| Network depth | `dqn.py` build methods | More layers = more capacity but slower training |
| Gamma | `main.py` config | Higher = cares more about long-term reward |
| Frame stack | `main.py` config | More frames = more temporal context |
| Reward shaping | `dqn.py` `play_episode()` | Change what behavior you encourage |
| Algorithm | Replace `dqn.py` | Try PPO, SAC, or A3C for better performance |

The biggest improvement you could make is switching from **DQN** to a modern algorithm like **PPO** (Proximal Policy Optimization) using a library like `stable-baselines3`. Modern algorithms handle continuous action spaces natively and converge much faster.

---

## Key Terms Glossary

| Term | Meaning |
|---|---|
| **Episode** | One full race from start to crash/completion |
| **State** | What the AI sees (3 stacked grayscale frames) |
| **Action** | One of 5 discrete driving commands |
| **Reward** | Points earned each step (positive on track, negative off) |
| **Q-value** | Predicted future reward for a (state, action) pair |
| **Epsilon-greedy** | Pick random action with probability ε, best action otherwise |
| **Experience replay** | Random sampling from past memories to train on |
| **Frame skip** | Repeat one action for N frames to reduce decision frequency |
| **Frame stacking** | Stack N consecutive frames as one "state" so AI can see motion |
| **Target network** | A frozen copy of the network used for stable training targets |
