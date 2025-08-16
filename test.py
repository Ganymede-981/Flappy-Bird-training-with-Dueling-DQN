import gymnasium as gym
import flappy_bird_gymnasium
import torch
import cv2
import numpy as np
from collections import deque

# --- must match training config ---
FRAME_SIZE = (84, 84)
STACK_N = 4

# -----------------------------
# Preprocessing
# -----------------------------
def to_gray84(obs):
    """Convert RGB -> grayscale (84x84), normalize to [0,1]."""
    if len(obs.shape) == 3 and obs.shape[2] == 3:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    return obs.astype(np.float32) / 255.0

def make_initial_stack(frame):
    return np.stack([frame] * STACK_N, axis=0)

def update_stack(stack, new_frame):
    stack = np.roll(stack, shift=-1, axis=0)
    stack[-1] = new_frame
    return stack

# -----------------------------
# Model (must match training)
# -----------------------------
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_actions, in_channels=STACK_N):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, FRAME_SIZE[0], FRAME_SIZE[1])
            conv_out = self.features(dummy)
            flat = int(np.prod(conv_out.shape[1:]))
        self.fc = nn.Sequential(
            nn.Linear(flat, 512), nn.ReLU(),
        )
        self.value = nn.Linear(512, 1)
        self.adv = nn.Linear(512, n_actions)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        V = self.value(z)
        A = self.adv(z)
        return V + (A - A.mean(dim=1, keepdim=True))

# -----------------------------
# Test loop
# -----------------------------
@torch.no_grad()
def test(model_path, episodes=5, render=True):
    env = gym.make("FlappyBird-v0", render_mode="human" if render else None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init model
    n_actions = env.action_space.n
    policy = DQN(n_actions).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    for ep in range(episodes):
        obs, _ = env.reset()
        frame = to_gray84(obs)
        state = make_initial_stack(frame)
        done = False
        score = 0
        steps = 0

        while not done:
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = policy(s).argmax(dim=1).item()

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            frame = to_gray84(obs)
            state = update_stack(state, frame)

            score = info.get("score", score)
            steps += 1

        print(f"Episode {ep+1} | Score: {score} | Steps survived: {steps}")

    env.close()

if __name__ == "__main__":
    # Change this to your model filename
    test("flappy_dueling_dqn_step200000.pth", episodes=5, render=True)
