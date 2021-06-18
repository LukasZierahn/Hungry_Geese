import numpy as np

import torch
torch.manual_seed(42)

import torch.nn.functional as F

from shared.generate_episode import generate_episode, possible_moves

def train(agent, memory_manager, optimizer, device, trainer, params):
    sampeling_count = params["sampeling_count"]
    batch_size = params["batch_size"]
    training_time = params["training_time"]
    entropy_weight = params["entropy_weight"]

    batch = False
    while (not batch) and memory_manager.size != -1:
        generate_episode(agent, memory_manager, trainer)
        batch = memory_manager.get_batch(batch_size)

    episode_rewards = []
    for i in range(training_time):

        episode_rewards.append(generate_episode(agent, memory_manager, trainer))

        for j in range(sampeling_count):
            #batch = memory_manager.get_batch(batch_size)
            batch = memory_manager.last_episode

            if batch == False:
                break

            step_reward = torch.tensor([x[memory_manager.INDEX.reward]["step_reward"] for x in batch], device=device, dtype=torch.float32).reshape(-1, 1)
            mc_reward = torch.tensor([x[memory_manager.INDEX.reward]["mc_reward"] for x in batch], device=device, dtype=torch.float32).reshape(-1, 1)

            old_observation = [x[memory_manager.INDEX.old_observation] for x in batch]
            actions = torch.tensor([x[memory_manager.INDEX.action] for x in batch], device=device, dtype=int)
            invalid_actions = [x[memory_manager.INDEX.invalid_action] for x in batch]
            done = torch.tensor([x[memory_manager.INDEX.done] for x in batch], device=device)
            observation = [x[memory_manager.INDEX.observation] for x in batch]

            policy, value = agent.model.forward(old_observation, invalid_actions)

            policy_losses = []
            entropy_losses = []
            value_losses = []

            # This iterates through the last episode back to front
            for k in range(len(policy)):
                categorical = torch.distributions.Categorical(policy[k])
                
                advantage = mc_reward[k] - value[k]
                policy_losses.append(-categorical.log_prob(actions[k]) * advantage)
                entropy_losses.append(torch.sum(policy[k] * torch.log(policy[k] + 1e-7)))
                value_losses.append(F.smooth_l1_loss(value[k], mc_reward[k]))


            optimizer.zero_grad()

            loss = torch.stack(policy_losses).float().sum() + entropy_weight * torch.stack(entropy_losses).float().sum() + torch.stack(value_losses).float().sum()
            #loss = torch.stack(value_losses).float().sum()

            memory_manager.policy_loss.append(torch.stack(policy_losses).mean().item())
            memory_manager.entropy_loss.append(entropy_weight * torch.stack(entropy_losses).mean().item())
            memory_manager.value_loss.append(torch.stack(value_losses).mean().item())

            if 0:
                print("mc_reward", mc_reward)
                print("value_old", value)
                print("value loss", value_losses)

            loss.backward()

            optimizer.step()        
