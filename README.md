<div align="center">

# 🐦 Flappy Bird — Deep Q-Network RL Agent

### An AI agent that learns to play Flappy Bird from scratch using Deep Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0081A5?style=for-the-badge&logo=openai&logoColor=white)](https://gymnasium.farama.org)
[![DQN](https://img.shields.io/badge/Algorithm-DQN-blueviolet?style=for-the-badge)](https://arxiv.org/abs/1312.5602)
[![Trained](https://img.shields.io/badge/Status-Trained_%2810K%2B_episodes%29-1D9E75?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An AI agent trained using **Double DQN** to play Flappy Bird — starting with zero knowledge and learning entirely through reward signals after **10,000+ training episodes**. Reward improved from **-8.10 → 14.70**.

</div>

---

## 🎮 Live Demo

> 🔗 **Live Demo:** *(paste your link here after deployment)*

---

## 📈 Training Progress

The agent starts knowing absolutely nothing — it dies immediately. After thousands of episodes of trial and error, it learns to navigate pipes consistently.

| Episode | Best Reward | Behaviour |
|---|---|---|
| 1 | -8.10 | Dies instantly |
| 75 | -3.30 | Slightly better random flapping |
| 240 | 0.30 | First pipe cleared |
| 390 | 4.80 | Consistently clearing a few pipes |
| 2027 | 7.50 | Navigating multiple pipes |
| 6553 | 10.80 | Strong consistent performance |
| 10082+ | **14.70** | Best recorded — agent plays well |

> Trained for **10,000+ episodes**. Pre-trained model weights included in `runs/flappybirdv0.pt`.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Architecture](#-architecture)
- [Hyperparameters](#-hyperparameters)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Train From Scratch](#-train-from-scratch)
- [Run Pre-trained Agent](#-run-pre-trained-agent)
- [Key Design Decisions](#-key-design-decisions)
- [Future Roadmap](#-future-roadmap)
- [Author](#-author)

---

## 🔍 Overview

This project implements a **Deep Q-Network (DQN)** agent for Flappy Bird using the `flappy-bird-gymnasium` environment. Unlike pixel-based approaches, this agent uses the game's **12 state features** (bird position, velocity, pipe distances) directly — making it faster to train while still demonstrating core RL principles.

The full system includes:
- A **DQN neural network** for Q-value approximation
- A **Double DQN** setup with policy and target networks
- An **Experience Replay Buffer** to break temporal correlation
- **Epsilon-greedy exploration** with exponential decay
- **YAML-based hyperparameter management** for clean experimentation
- **Automatic model checkpointing** — saves only when a new best reward is achieved

---

## ⚙️ How It Works

```
Game State (12 features)
─────────────────────────────────────────
Bird y position, velocity
Next pipe: x distance, top gap y, bottom gap y
Next-next pipe: same 3 features
...and more
        │
        ▼
DQN Policy Network
─────────────────────────────────────────
Linear(12 → 256) → ReLU → Linear(256 → 2)
Output: Q-values for [do nothing, flap]
        │
        ▼
ε-Greedy Action Selection
─────────────────────────────────────────
• With probability ε   → random action (explore)
• With probability 1-ε → argmax Q-value (exploit)
• ε starts at 1.0, decays × 0.995 each episode → min 0.05
        │
        ▼
Environment Step
─────────────────────────────────────────
Returns: next_state, reward, terminated
        │
        ▼
Experience Replay Buffer (maxlen=100,000)
─────────────────────────────────────────
Store: (state, action, next_state, reward, done)
Sample random mini-batch of 32
        │
        ▼
Optimize (Bellman Equation)
─────────────────────────────────────────
target_Q = reward + γ * max Q(next_state)   [if not done]
target_Q = reward                            [if done]
loss = MSE(current_Q, target_Q)
Backpropagate → update policy network
        │
        ▼
Sync Target Network every 1000 steps
─────────────────────────────────────────
target_dqn.load_state_dict(policy_dqn.state_dict())
```

---

## 🧠 Architecture

### DQN Network (`dqn.py`)

```python
DQN(
  Linear(12  → 256)  # Input: 12 game state features
  ReLU()
  Linear(256 → 2)    # Output: Q-value for [nothing, flap]
)
```

Simple but effective. Since the state space is already structured (not raw pixels), a shallow MLP outperforms a CNN here and trains significantly faster.

### Double DQN Setup

```
Policy Network   → selects actions, updated every step
Target Network   → provides stable Q-value targets
                 → synced with policy every 1000 steps
```

Using two networks solves the "moving target" problem — if you use the same network to both select and evaluate actions, the target keeps shifting and training becomes unstable.

### Experience Replay (`experience_replay.py`)

```python
ReplayMemory(maxlen=100_000)
# FIFO deque — oldest experiences dropped when full
# Random sampling of mini-batches breaks temporal correlation
# Without replay: agent forgets old experiences and overfits to recent ones
```

---

## ⚙️ Hyperparameters

Managed cleanly via `parameters.yaml` — no hardcoded values in code.

```yaml
flappybirdv0:
  epsilon_init: 1        # Start fully random
  epsilon_min: 0.05      # Always keep 5% exploration
  epsilon_decay: 0.995   # Multiply epsilon by this each episode
  replay_memory_size: 100000
  mini_batch_size: 32
  network_sync_rate: 1000  # Sync target network every 1000 steps
  alpha: 0.001           # Adam learning rate
  gamma: 0.99            # Discount factor — values future rewards highly
  reward_threshold: 1000 # Cap episode length
```

---

## 📁 Project Structure

```
FLAPPYBIRD_RL/
│
├── 📄 agent.py                # Main Agent class — training loop, action selection, optimization
├── 📄 dqn.py                  # DQN neural network (MLP: 12 → 256 → 2)
├── 📄 experience_replay.py    # ReplayMemory — FIFO buffer with random sampling
├── 📄 parameters.yaml         # All hyperparameters — no hardcoded values
├── 📄 requirements.txt        # Python dependencies
│
└── 📂 runs/
    ├── flappybirdv0.pt        # Pre-trained model weights (10K+ episodes)
    └── flappybirdv0.log       # Full training log — reward per best episode
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/jay51211/FlappyBird-RL.git
cd FlappyBird-RL
```

### Step 2 — Install dependencies

```bash
pip install torch gymnasium flappy-bird-gymnasium pyyaml
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train From Scratch

```bash
python agent.py flappybirdv0 --train
```

**With live rendering** (watch the agent learn in real time — slower):

```bash
python agent.py flappybirdv0 --train --render
```

**What you will see:**

```
Using device: cpu
State size: 12, Action size: 2
Episode 0    | Reward = -8.10  | Epsilon = 0.9950
Episode 1    | Reward = -7.80  | Epsilon = 0.9900
...
Best reward = 0.30 at episode 240
Best reward = 4.80 at episode 390
...
Best reward = 14.70 at episode 10082
```

Model is saved automatically to `runs/flappybirdv0.pt` every time a new best reward is achieved.

---

## 🤖 Run Pre-trained Agent

A pre-trained model (10K+ episodes, best reward 14.70) is included in `runs/flappybirdv0.pt`.

```bash
python agent.py flappybirdv0
```

This loads the saved weights and runs the agent with rendering enabled — watch the trained AI play Flappy Bird.

---

## 💡 Key Design Decisions

**Why MLP instead of CNN?**
The `flappy-bird-gymnasium` environment exposes structured game state features (12 numbers) rather than raw pixels. An MLP on structured features trains in hours instead of days and achieves better performance with less compute.

**Why save only on best reward?**
Checkpointing only when a new best is achieved means `runs/flappybirdv0.pt` always contains the strongest model, not just the most recent one.

**Why YAML for hyperparameters?**
Clean separation of config from code. Swap parameter sets without touching source files. Makes experimentation systematic.

**Why Double DQN?**
Single-network DQN overestimates Q-values because the same network both selects and evaluates actions. The target network provides stable evaluation targets, which dramatically improves training stability.

---

## 🗺️ Future Roadmap

- [ ] Record and upload a gameplay video of the trained agent
- [ ] Add reward curve visualization with Matplotlib
- [ ] Implement Dueling DQN architecture for comparison
- [ ] Try Prioritized Experience Replay (weight important transitions more)
- [ ] Add TensorBoard logging for real-time training visualization
- [ ] Experiment with pixel-based input using CNN encoder

---

## 📦 Dependencies

```txt
torch
gymnasium
flappy-bird-gymnasium
pyyaml
```

```bash
pip install torch gymnasium flappy-bird-gymnasium pyyaml
```

---

## 👤 Author

<div align="center">

**Jay Kumbhar**

[![GitHub](https://img.shields.io/badge/GitHub-jay51211-181717?style=for-the-badge&logo=github)](https://github.com/jay51211)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-jaykumbhar5121-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/jaykumbhar5121)
[![Email](https://img.shields.io/badge/Email-jaykumbhar518@gmail.com-EA4335?style=for-the-badge&logo=gmail)](mailto:jaykumbhar518@gmail.com)

</div>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

*Built with ❤️ by Jay Kumbhar*

⭐ **Star this repo if you found it useful!** ⭐

</div>
