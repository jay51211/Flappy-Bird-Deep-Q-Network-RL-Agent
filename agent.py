import argparse
import os
import random
import itertools
import yaml

import flappy_bird_gymnasium
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from dqn import DQN
from experience_replay import ReplayMemory

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent:
    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        # Hyperparameters
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.network_sync_rate = params["network_sync_rate"]
        self.reward_threshold = params["reward_threshold"]

        # Loss / optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        # Files
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")

    def run(self, is_training=True, render=False):
        env = gym.make(
            "FlappyBird-v0",
            render_mode="human" if render else None
        )

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        print(f"State size: {num_states}, Action size: {num_actions}")

        # Policy network
        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            # Target network
            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)

            steps = 0
            best_reward = float("-inf")

            # Clear old log file
            with open(self.LOG_FILE, "w") as f:
                f.write(f"Training started for param set: {self.param_set}\n")

        else:
            if not os.path.exists(self.MODEL_FILE):
                raise FileNotFoundError(
                    f"Model file not found: {self.MODEL_FILE}\n"
                    f"Train first using:\n"
                    f"python main.py {self.param_set} --train"
                )

            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_dqn.eval()
            print(f"Loaded model from {self.MODEL_FILE}")

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)

            episode_reward = 0
            done = False

            while not done and episode_reward < self.reward_threshold:

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()  # random action
                else:
                    with torch.no_grad():
                        q_values = policy_dqn(state.unsqueeze(0))  # shape: [1, num_actions]
                        action = q_values.argmax(dim=1).item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_reward += reward

                # Convert to tensors
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device)
                reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)

                # Store transition
                if is_training:
                    memory.append((state, action, next_state_tensor, reward_tensor, done))
                    steps += 1

                # Move to next state
                state = next_state_tensor

                if is_training and len(memory) >= self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Sync target network
                    if steps >= self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps = 0


            if is_training:
                print(f"Episode {episode} | Reward = {episode_reward:.2f} | Epsilon = {epsilon:.4f}")
            else:
                print(f"Episode {episode} | Reward = {episode_reward:.2f}")


            if is_training:
                # Decay epsilon
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                # Save best model only
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    log_msg = f"Best reward = {episode_reward:.2f} at episode {episode + 1}"
                    print(log_msg)

                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

        # Optional manual close if needed
        # env.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float32, device=device)

        with torch.no_grad():
            next_q_values = target_dqn(next_states).max(dim=1)[0]
            target_q = rewards + (1 - terminations) * self.gamma * next_q_values

        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test Flappy Bird DQN model.")
    parser.add_argument("hyperparameters", help="Name of parameter set in parameters.yaml")
    parser.add_argument("--train", help="Training mode", action="store_true")
    parser.add_argument("--render", help="Render environment", action="store_true")

    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True, render=args.render)
    else:
        dql.run(is_training=False, render=True)