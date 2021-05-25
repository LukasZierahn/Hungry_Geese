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
            #batch = memory_manager.get_batch(batch_size)
            batch = memory_manager.last_episode

            if batch == False:
                print("skipped", i)
                break

            reward = torch.tensor([x[memory_manager.INDEX.reward] for x in batch], device=device, dtype=torch.float32).reshape(-1, 1)

            old_observation = [x[memory_manager.INDEX.old_observation] for x in batch]
            actions = torch.tensor([x[memory_manager.INDEX.action] for x in batch], device=device, dtype=int)
            invalid_actions = [x[memory_manager.INDEX.invalid_action] for x in batch]
            done = torch.tensor([x[memory_manager.INDEX.done] for x in batch], device=device)
            observation = [x[memory_manager.INDEX.observation] for x in batch]

            if len(old_observation) == 0:
                continue

            policy_old, value_old = agent.model.forward(old_observation, invalid_actions)
            #policy, value = agent.model.forward(observation)

            #reward = (reward - reward.mean()) / (reward.std() + 1e-7)
            advantages = (reward - value_old).reshape(-1)

            policy_losses = []
            value_losses = []
            for k in range(len(policy_old)):
                #print(policy_old[k][actions[k]])
                #print(-torch.log(policy_old[k][actions[k]]) * advantages[k])
                categorical = torch.distributions.Categorical(policy_old[k])
                policy_losses.append(-categorical.log_prob(actions[k]) * advantages[k]) 

                value_losses.append(F.smooth_l1_loss(value_old[k], reward[k]))

            optimizer.zero_grad()

            loss = torch.stack(policy_losses).float().sum() + torch.stack(value_losses).float().sum()
            print("loss", loss)
            loss.backward()

            optimizer.step()
        
        print()

    return episode_rewards, best_params