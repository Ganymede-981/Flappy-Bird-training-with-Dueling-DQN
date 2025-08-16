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
import cv2
import time
import os

# -----------------------------
# Global toggles
# -----------------------------
RENDER = False          # set True to watch (slows training a lot)
SEED = 42               # reproducibility
FRAME_SIZE = (84, 84)   # input size
STACK_N = 4             # number of stacked frames
WARMUP_STEPS = 50_000   # fill replay before learning
TRAINING_STEPS = 750_000  # total env steps (not episodes)
TARGET_SYNC = 1_000     # target network sync rate (steps)
EVAL_EVERY = 25_000     # quick eval (no learning) every X steps
SAVE_EVERY = 100_000    # save checkpoint every X steps

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

# -----------------------------
# Utils
# -----------------------------
def to_gray84(obs):
    """Convert RGB obs -> grayscale 84x84 float32 in [0,1], shape (84,84)."""
    if len(obs.shape) == 3 and obs.shape[2] == 3:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    return (obs.astype(np.float32) / 255.0)

def make_initial_stack(first_frame, n=STACK_N):
    """Create a (n,84,84) stack initialized with first_frame."""
    stack = np.stack([first_frame for _ in range(n)], axis=0)
    return stack

def update_stack(stack, new_frame):
    """Append new_frame to (n,84,84) stack along axis 0, drop oldest."""
    new_stack = np.roll(stack, shift=-1, axis=0)
    new_stack[-1] = new_frame
    return new_stack

# -----------------------------
# Dueling DQN (CNN) for stacks
# -----------------------------
class DQN(nn.Module):
    def __init__(self, n_actions, in_channels=STACK_N):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        # compute conv output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, FRAME_SIZE[0], FRAME_SIZE[1])
            conv_out = self.features(dummy)
            flat = int(np.prod(conv_out.shape[1:]))

        self.fc = nn.Sequential(
            nn.Linear(flat, 512), nn.ReLU(),
        )
        # Dueling heads
        self.value = nn.Linear(512, 1)
        self.adv = nn.Linear(512, n_actions)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        V = self.value(z)                # [B, 1]
        A = self.adv(z)                  # [B, n_actions]
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q

# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity=500_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        # store as float32, int, etc. for speed/memory
        self.buffer.append((s.astype(np.float32), int(a), float(r), s2.astype(np.float32), bool(d)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return np.stack(s), np.array(a), np.array(r, dtype=np.float32), np.stack(s2), np.array(d, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

# -----------------------------
# Agent with Double DQN updates
# -----------------------------
class Agent:
    def __init__(self, n_actions):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_actions = n_actions

        self.policy = DQN(n_actions).to(self.device)
        self.target = DQN(n_actions).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optim = optim.Adam(self.policy.parameters(), lr=5e-5)
        self.gamma = 0.99
        self.batch_size = 256

        # epsilon schedule (slow decay)
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.99995  # ~100k steps to ~0.05

        self.memory = ReplayBuffer(capacity=500_000)
        self.learn_steps = 0

    @torch.no_grad()
    def act(self, state_stack):
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        s = torch.tensor(state_stack, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.policy(s)
        return int(q.argmax(dim=1).item())

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        s, a, r, s2, d = self.memory.sample(self.batch_size)

        s  = torch.tensor(s, dtype=torch.float32, device=self.device)   # [B, C, H, W]
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        a  = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)  # [B,1]
        r  = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)# [B,1]
        d  = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)# [B,1]

        # Current Q(s,a)
        q = self.policy(s).gather(1, a)  # [B,1]

        with torch.no_grad():
            # Double DQN: action via policy, value via target
            next_actions = self.policy(s2).argmax(dim=1, keepdim=True)       # [B,1]
            next_q = self.target(s2).gather(1, next_actions)                 # [B,1]
            target = r + self.gamma * next_q * (1.0 - d)

        loss = F.smooth_l1_loss(q, target)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
        self.optim.step()

        self.learn_steps += 1
        return loss.item()

    def update_eps(self):
        self.eps = max(self.eps * self.eps_decay, self.eps_min)

    def sync_target(self):
        self.target.load_state_dict(self.policy.state_dict())

# -----------------------------
# Reward shaping (safe/guarded)
# -----------------------------
def shaped_reward(base_reward, info, prev_score):
    """
    - +0.1 per survival step
    - +10 when score increases
    - small penalty if env exposes distance from gap center (guarded)
    """
    reward = 0.1  # survival
    score = info.get("score", 0)
    if score > prev_score:
        reward += 10.0

    # Optional: penalty for vertical deviation from gap center, if available
    top = info.get("next_pipe_top_y", None)
    bot = info.get("next_pipe_bottom_y", None)
    py  = info.get("player_y", None)
    if top is not None and bot is not None and py is not None:
        gap_c = 0.5 * (top + bot)
        reward -= 0.002 * abs(py - gap_c)

    reward += float(base_reward)
    return reward, score

# -----------------------------
# Training Loop (step-based)
# -----------------------------
def train():
    env = gym.make("FlappyBird-v0", render_mode="human" if RENDER else None)
    n_actions = env.action_space.n

    agent = Agent(n_actions)
    print("Device:", agent.device)

    episode = 0
    step = 0
    scores_history = []
    reward_history = []
    loss_history = []

    best_score = -1
    t0 = time.time()

    while step < TRAINING_STEPS:
        obs, _ = env.reset(seed=SEED + episode)
        f = to_gray84(obs)
        state = make_initial_stack(f, n=STACK_N)
        done = False
        ep_reward = 0.0
        prev_score = 0
        ep_score = 0

        while not done and step < TRAINING_STEPS:
            if RENDER:
                env.render()

            # Act
            action = agent.act(state)

            # Step
            next_obs, base_r, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            f2 = to_gray84(next_obs)
            next_state = update_stack(state, f2)

            # Reward shaping
            rew, score_now = shaped_reward(base_r, info, prev_score)
            prev_score = score_now
            ep_score = score_now

            # Store
            agent.memory.push(state, action, rew, next_state, done)

            # Learn (after warm-up)
            loss_val = None
            if step > WARMUP_STEPS:
                loss_val = agent.learn()
                if (step % TARGET_SYNC) == 0:
                    agent.sync_target()
                agent.update_eps()

            if loss_val is not None:
                loss_history.append(loss_val)

            ep_reward += rew
            state = next_state
            step += 1

            # Periodic eval (brief, no learning)
            if (step % EVAL_EVERY) == 0:
                print(f"[Eval @ step {step}] eps={agent.eps:.3f} mem={len(agent.memory)} time={(time.time()-t0)/60:.1f}m")
                # quick greedy rollout (1 episode)
                _score = quick_eval(env, agent)
                print(f"  -> greedy score: {_score}")

            # Save checkpoints
            if (step % SAVE_EVERY) == 0:
                path = f"flappy_dueling_dqn_step{step}.pth"
                torch.save(agent.policy.state_dict(), path)
                print(f"Saved checkpoint: {path}")

        episode += 1
        scores_history.append(ep_score)
        reward_history.append(ep_reward)

        if ep_score > best_score:
            best_score = ep_score
            torch.save(agent.policy.state_dict(), "flappy_dueling_dqn_best.pth")

        print(f"Episode {episode} | Steps {step}/{TRAINING_STEPS} | Score: {ep_score} | EpReward: {ep_reward:.2f} | Eps: {agent.eps:.3f} | Mem: {len(agent.memory)}")

    env.close()

    # Final save
    torch.save(agent.policy.state_dict(), "flappy_dueling_dqn_final.pth")
    print("Training done. Saved model: flappy_dueling_dqn_final.pth")

    # Plots
    plt.figure()
    plt.plot(reward_history, label="Episode Reward")
    plt.plot(scores_history, label="Episode Score")
    plt.xlabel("Episodes"); plt.ylabel("Value"); plt.legend(); plt.title("Training Curves")
    plt.show()

    if len(loss_history) > 0:
        plt.figure()
        plt.plot(loss_history, label="Loss (smooth L1)")
        plt.xlabel("Optimization Steps"); plt.ylabel("Loss"); plt.legend(); plt.title("Critic Loss")
        plt.show()

# -----------------------------
# Quick greedy eval (1 episode)
# -----------------------------
@torch.no_grad()
def quick_eval(env, agent):
    # Greedy play (eps=0) for one episode to gauge progress
    obs, _ = env.reset()
    f = to_gray84(obs)
    state = make_initial_stack(f, n=STACK_N)
    done = False
    prev_score = 0
    score = 0
    steps = 0
    while not done and steps < 10_000:
        s = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        a = agent.policy(s).argmax(dim=1).item()
        next_obs, base_r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        f2 = to_gray84(next_obs)
        state = update_stack(state, f2)
        score = info.get("score", score)
        steps += 1
    return score

# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
    os.makedirs(".", exist_ok=True)
    train()
    print("Training complete.")