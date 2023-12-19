import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import os
from gymnasium.wrappers.monitoring import video_recorder

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32
max_episodes = 50000
max_timesteps = 300
exploration_noise = 0.1
exploration_noise_decay = 0.9999
exploration_noise_min = 0.01
tau = 0.005
update_interval = 1
test = False
log_interval = 10
save_interval = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Replay Buffer
class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

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

        return (
            torch.tensor(state_list, dtype=torch.float).to(device),
            torch.tensor(action_list).to(device),
            torch.tensor(reward_list).to(device),
            torch.tensor(next_state_list, dtype=torch.float).to(device),
            torch.tensor(done_mask_lst).to(device),
        )

    def size(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()

# Q Network
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)

# DQN Agent
class DQN:
    def __init__(self):
        self.q_net = QNetwork().to(device)
        self.target_net = QNetwork().to(device)
        self.optimizer = Adam(self.q_net.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer()
        self.update_count = 0

    def get_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.randint(0, 4)
        else:
            state = torch.FloatTensor(state).to(device)
            q_value = self.q_net(state)
            return q_value.argmax().item()

    def train(self):
        for _ in range(10):
            state, action, reward, next_state, done_mask = self.buffer.sample(batch_size)

            q_values = self.q_net(state)
            q_value = q_values.gather(1, action)

            next_q_values = self.target_net(next_state)
            next_q_value = next_q_values.max(1)[0].unsqueeze(1)
            target = reward + gamma * next_q_value * done_mask

            loss = F.smooth_l1_loss(q_value, target.detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, save_path):
        torch.save(self.q_net.state_dict(), save_path)

    def load(self, load_path):
        self.q_net.load_state_dict(torch.load(load_path))

# Main
if __name__ == "__main__":

    if test:
        agent = DQN()

        env = gym.make("LunarLander-v2", render_mode='rgb_array')
        agent.load('./model_22300.pth')
        vid = video_recorder.VideoRecorder(env, path="./assets/{}.mp4".format("LunarLander"))
        state, info = env.reset()
        done = False
        while not done:
            frame = env.render()
            vid.capture_frame()
            
            action = agent.get_action(state, 0.0001)

            state, reward, done, _, info = env.step(action)
        vid.close()
        env.close()

    else:
        env = gym.make("LunarLander-v2")
        agent = DQN()
        score = 0.0
        print_interval = 20
        save_path = "./save_model"
        load_path = "./save_model"

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        if os.path.isfile(load_path):
            agent.load(load_path)

        for episode_num in range(max_episodes):
            epsilon = max(0.01, exploration_noise * (exploration_noise_decay ** episode_num))
            done = False
            state, info = env.reset()

            for time_step in range(max_timesteps):
                action = agent.get_action(state, epsilon)
                next_state, reward, done, _, info = env.step(action)

                done_mask = 0.0 if done else 1.0
                agent.buffer.put((state, action, reward / 100.0, next_state, done_mask))

                score += reward
                state = next_state

                if done:
                    break

            if agent.buffer.size() >= 2000:
                agent.train()
                agent.update_count += 1

                if agent.update_count % update_interval == 0:
                    agent.update_target()

            if episode_num % print_interval == 0 and episode_num != 0:
                print(
                    "Episode: {}, Steps: {}, Exploration: {:.4f}, Score: {:.1f}".format(
                        episode_num, time_step, epsilon, score / print_interval
                    )
                )

                writer = SummaryWriter('./train_logs')
                writer.add_scalar('Score', score/print_interval, episode_num)
                writer.close()

                score = 0.0

            if episode_num % save_interval == 0 and episode_num != 0:
                agent.save(os.path.join(save_path, "model_{}.pth".format(episode_num)))

        env.close()