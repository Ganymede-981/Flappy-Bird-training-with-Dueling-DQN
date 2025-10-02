# Flappy-Bird-training-with-Dueling-DQN
#  Flappy Bird Training with Dueling DQN

This repository contains a Reinforcement Learning (RL) implementation of the **Flappy Bird game** using **Dueling Deep Q-Networks (Dueling DQN)**.  
The agent learns to play Flappy Bird by interacting with the game environment, improving its policy over time using experience replay, target networks, and epsilon-greedy exploration.

---

##  Features
-  Flappy Bird environment implemented in `flappy.py`  
-  **Dueling DQN** implementation (`duel.py`) for improved value/action separation  
-  Pre-trained model checkpoint: `flappy_dueling_dqn_step200000.pth`  
-  Training script (`flappy.py`) and testing script (`test.py`)  
-  Replay buffer and epsilon-greedy policy  
-  Visualization of agent performance  

---

## 🧠 What is Dueling DQN?
Dueling DQN is an extension of the standard Deep Q-Network.  
It splits the Q-value estimation into two streams:
- **Value Stream (V(s))** → how good it is to be in a state  
- **Advantage Stream (A(s,a))** → how good it is to take a specific action in that state  

The final Q-value is computed as:
\[
Q(s,a) = V(s) + (A(s,a) - \frac{1}{|A|} \sum_{a'} A(s,a'))
\]

This helps the agent learn **which states are valuable** even before knowing the best action.

---

## 📂 Repository Structure
```
📦 Flappy-Bird-training-with-Dueling-DQN
┣ 📜 LICENSE 
┣ 📜 README.md
┣ 📜 requirements.txt # Python dependencies
┣ 📜 duel.py # Dueling DQN model architecture
┣ 📜 flappy.py # Training script
┣ 📜 test.py # Evaluation / Testing script
┣ 📜 flappy_dueling_dqn_step200000.pth # Pre-trained weights
┣ 📂 models # (Optional) model saving folder
```

---

## ⚙️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Flappy-Bird-training-with-Dueling-DQN.git
   cd Flappy-Bird-training-with-Dueling-DQN
2. Install dependencies  
   ```bash
   pip install -r requirements.txt
