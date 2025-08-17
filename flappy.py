import gymnasium as gym
import flappy_bird_gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# --- CNN-based DQN ---
class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ReLU()
        )


        # Dynamically compute feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 84, 84)
            conv_out = self.conv(dummy_input)
            conv_out_size = conv_out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# --- Replay Memory ---
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.stack(s), a, r, np.stack(s2), d)
    def __len__(self):
        return len(self.buffer)

# --- Agent ---
class Agent:
    def __init__(self, n_actions):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = DQN(n_actions).to(self.device)
        self.target = DQN(n_actions).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.memory = ReplayBuffer()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.gamma = 0.99
        self.batch_size = 64
        self.sync_rate = 1000
        self.steps = 0
        self.epsilon = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.9995

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(2)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.policy(state)
        return torch.argmax(q_values).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        s, a, r, s2, d = self.memory.sample(self.batch_size)

        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)

        q_values = self.policy(s).gather(1, a).squeeze()
        with torch.no_grad():
            max_next = self.target(s2).max(1)[0]
            target = r + self.gamma * max_next * (1 - d)

        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.sync_rate == 0:
            self.target.load_state_dict(self.policy.state_dict())

        self.steps += 1
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

# --- State Preprocessing (grayscale & resize) ---
def preprocess(obs):
    import cv2
    if len(obs.shape) == 3 and obs.shape[2] == 3:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # only convert if RGB
    resized = cv2.resize(obs, (84, 84))
    return np.expand_dims(resized, axis=0) / 255.0  # shape: (1, 84, 84)


# --- Reward Shaping ---
def shaped_reward(base_reward, info, prev_score):
    # Reward for staying alive
    reward = 0.01
    if info.get("score", 0) > 0:  # passed a pipe
        reward += 10

    if info.get("score",0) > prev_score:
        reward += 10
    reward += base_reward
    return reward

# --- Training ---
def train(episodes=500):
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    agent = Agent(env.action_space.n)
    rewards = []
    scores = []

    for ep in range(episodes):
        obs, _ = env.reset()
        state = preprocess(obs)
        done = False
        total_reward = 0
        prev_score = 0

        while not done:
            action = agent.act(state)
            next_obs, base_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            reward = shaped_reward(base_reward, info, prev_score)
            next_state = preprocess(next_obs)
            agent.memory.push(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        scores.append(info.get('score', 0))
        print(f"Episode {ep} | Score: {info.get('score',0)} | TotalReward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    env.close()
    torch.save(agent.policy.state_dict(), "flappy_dqn_.pth")
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Stats")
    plt.title("Training Progress")
    plt.plot(scores)
    plt.legend(["Total Reward", "Score"])
    plt.show()

if __name__ == "__main__":
    print("cuda" if torch.cuda.is_available() else "cpu")
    train(episodes=1000)
