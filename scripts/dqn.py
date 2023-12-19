import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random

# Replay Buffer
class ReplayBuffer:
    # Initialize a deque with maximum length of buffer_limit
    def __init__(self, buffer_limit, device):
        self.device = device
        self.buffer_limit = buffer_limit
        self.buffer = deque(maxlen=self.buffer_limit)

    # Add a transition to the buffer
    def put(self, transition):
        self.buffer.append(transition)

    # Sample a mini batch from the buffer
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        state_list, action_list, reward_list, next_state_list, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            state, action, reward, next_state, done_mask = transition
            state_list.append(state)
            action_list.append([action])
            reward_list.append([reward])
            next_state_list.append(next_state)
            done_mask_lst.append([done_mask])

        # Convert the lists to PyTorch tensors and move them to the specified device
        return (
            torch.tensor(state_list, dtype=torch.float).to(self.device),
            torch.tensor(action_list).to(self.device),
            torch.tensor(reward_list).to(self.device),
            torch.tensor(next_state_list, dtype=torch.float).to(self.device),
            torch.tensor(done_mask_lst).to(self.device),
        )

    # Return the current size of the buffer
    def size(self):
        return len(self.buffer)

    # Clear the buffer
    def clear(self):
        self.buffer.clear()

# Q Network
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 4)

    # Forward pass
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)

# DQN Agent
class DQN:
    def __init__(self, buffer_limit, learning_rate, batch_size, gamma, device):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.buffer_limit = buffer_limit
        self.device = device

        self.q_net = QNetwork().to(device)
        self.target_net = QNetwork().to(device)
        self.optimizer = Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.buffer = ReplayBuffer(self.buffer_limit, self.device)
        self.update_count = 0

    # Get action from the Q network
    def get_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.randint(0, 4)
        else:
            state = torch.FloatTensor(state).to(self.device)
            q_value = self.q_net(state)
            return q_value.argmax().item()

    # Train the Q network
    def train(self):
        for _ in range(10):
            state, action, reward, next_state, done_mask = self.buffer.sample(self.batch_size)

            q_values = self.q_net(state)
            q_value = q_values.gather(1, action)

            next_q_values = self.target_net(next_state)
            next_q_value = next_q_values.max(1)[0].unsqueeze(1)
            target = reward + self.gamma * next_q_value * done_mask

            loss = F.smooth_l1_loss(q_value, target.detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # Update the target network
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    # Save and load the model
    def save(self, save_path):
        torch.save(self.q_net.state_dict(), save_path)

    # Load the model
    def load(self, load_path):
        self.q_net.load_state_dict(torch.load(load_path))
