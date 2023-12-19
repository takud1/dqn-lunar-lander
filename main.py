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
from scripts.dqn import DQN

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

# Main
if __name__ == "__main__":

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Test the model
    if test:
        agent = DQN(buffer_limit, learning_rate, batch_size, gamma, device)

        env = gym.make("LunarLander-v2", render_mode='rgb_array')
        agent.load('./weights/model_22300.pth')
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

    # Train the model
    else:
        env = gym.make("LunarLander-v2")
        agent = DQN(buffer_limit, learning_rate, batch_size, gamma, device)
        score = 0.0
        print_interval = 20
        save_path = "./save_model"
        load_path = "./save_model"

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        if os.path.isfile(load_path):
            agent.load(load_path)

        # Train the model for max_episodes
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

            # Print the score
            if episode_num % print_interval == 0 and episode_num != 0:
                print(
                    "Episode: {}, Steps: {}, Exploration: {:.4f}, Score: {:.1f}".format(
                        episode_num, time_step, epsilon, score / print_interval
                    )
                )

                # Save data to tensorboard
                writer = SummaryWriter('./train_logs')
                writer.add_scalar('Score', score/print_interval, episode_num)
                writer.close()

                score = 0.0

            # Save the model
            if episode_num % save_interval == 0 and episode_num != 0:
                agent.save(os.path.join(save_path, "model_{}.pth".format(episode_num)))

        env.close()