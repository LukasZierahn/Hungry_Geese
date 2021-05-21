import numpy as np

import torch
torch.manual_seed(42)

import torch.nn as nn
import torch.optim as optim

from kaggle_environments import make

from shared.generate_episode import generate_episode

def train(agent, memory_manager, opponents, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    criterion = nn.MSELoss()
    agent.model.to(device)
    agent.model.device = device
    best_params = 0
    highest_reward = -np.inf

    learning_rate = params["learning_rate"]
    sampeling_count = params["sampeling_count"]
    batch_size = params["batch_size"]
    training_time = params["training_time"]

    #optimizer = optim.SGD(agent.model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(agent.model.parameters(), lr=learning_rate)

    batch = False
    while not batch:
        generate_episode(agent, memory_manager, opponents)
        batch = memory_manager.get_batch(batch_size)

    episode_rewards = []
    for i in range(training_time):
        episode_rewards.append(generate_episode(agent, memory_manager, opponents))

        print(f"Current Iteration {i}/{training_time}, {100 * i/training_time:.2f}% ")
        print(episode_rewards[-1])
        print()

        if episode_rewards[-1] > highest_reward:
            highest_reward = episode_rewards[-1]
            best_params = agent.model.state_dict()

        for j in range(sampeling_count):
            batch = memory_manager.get_batch(batch_size)

            if batch == False:
                print("skipped", i)
                break

            reward = torch.tensor([x[memory_manager.INDEX.reward] for x in batch], device=device)

            old_observation = [x[memory_manager.INDEX.old_observation] for x in batch]
            actions = torch.tensor([x[memory_manager.INDEX.action] for x in batch], device=device, dtype=int)
            done = torch.tensor([x[memory_manager.INDEX.done] for x in batch], device=device)
            observation = [x[memory_manager.INDEX.observation] for x in batch]

            optimizer.zero_grad()

            policy_old, value_old = agent.model.forward(old_observation)
            policy, value = agent.model.forward(observation)

            # Policy Gradient
            target = torch.zeros(batch_size, 4)      
            for index_batch in range(batch_size):
                target[index_batch, actions[index_batch]] = reward[index_batch] + memory_manager.gamma * value[index_batch] - value_old[index_batch]

            estimate = torch.zeros(batch_size, 4)
            for index_batch in range(batch_size):
                estimate[index_batch, actions[index_batch]] = policy_old[index_batch, actions[index_batch]]

            loss = criterion(target, estimate)

            # Value Gradient
            target = reward.reshape(-1, 1) + memory_manager.gamma * value - value_old
            loss = criterion(target.double(), value_old.double())
            loss.backward()

            optimizer.step()
    return episode_rewards, best_params