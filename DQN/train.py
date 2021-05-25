import numpy as np
from numpy.core.fromnumeric import mean, reshape

import torch
torch.manual_seed(42)

import torch.nn as nn
import torch.nn.functional as F

from shared.generate_episode import generate_episode

def train(agent, memory_manager, optimizer, device, trainer, params):
    best_params = 0
    highest_reward = -np.inf
    
    
    criterion = nn.MSELoss()
    sampeling_count = params["sampeling_count"]
    batch_size = params["batch_size"]
    training_time = params["training_time"]

    batch = False
    while (not batch) and memory_manager.size != -1:
        generate_episode(agent, memory_manager, trainer)
        batch = memory_manager.get_batch(batch_size)

    episode_rewards = []
    for i in range(training_time):
        episode_rewards.append(generate_episode(agent, memory_manager, trainer))

        print(f"Current Iteration {i}/{training_time}, {100 * i/training_time:.2f}% ")
        print(episode_rewards[-1])

        if episode_rewards[-1] > highest_reward:
            highest_reward = episode_rewards[-1]
            best_params = agent.model.state_dict()

        for j in range(sampeling_count):
            batch = memory_manager.get_batch(batch_size)
            #batch = memory_manager.last_episode

            if batch == False:
                print("skipped", i)
                break

            reward = torch.tensor([x[memory_manager.INDEX.reward] for x in batch], device=device, dtype=torch.float32).reshape(-1, 1)

            old_observation = [x[memory_manager.INDEX.old_observation] for x in batch]
            actions = torch.tensor([x[memory_manager.INDEX.action] for x in batch], device=device, dtype=int)
            invalid_actions = [x[memory_manager.INDEX.invalid_action] for x in batch]
            done = torch.tensor([x[memory_manager.INDEX.done] for x in batch], device=device)
            observation = [x[memory_manager.INDEX.observation] for x in batch]

            optimizer.zero_grad()

            target = torch.zeros(batch_size, 4)      
            for index_batch in range(batch_size):
                if done[index_batch]:
                    target[index_batch, actions[index_batch]] = reward[index_batch]
                else:
                    policy, value = agent.model.forward([observation[index_batch]], [(actions[index_batch] + 2) % 4])
                    target[index_batch, actions[index_batch]] = reward[index_batch] + memory_manager.gamma * torch.max(policy[0])

            estimate = torch.zeros(batch_size, 4)
            policy_old, value_old = agent.model.forward(old_observation, invalid_actions)

            for index_batch in range(batch_size):
                estimate[index_batch, actions[index_batch]] = policy_old[index_batch, actions[index_batch]]

            loss = criterion(target, estimate)
            loss.backward()
            optimizer.step()
        
        print()

    return episode_rewards, best_params