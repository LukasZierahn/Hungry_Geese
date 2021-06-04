import numpy as np

import torch
torch.manual_seed(42)

import torch.nn.functional as F

from shared.generate_episode import generate_episode, possible_moves

def train(agent, memory_manager, optimizer, device, trainer, params):
    sampeling_count = params["sampeling_count"]
    batch_size = params["batch_size"]
    training_time = params["training_time"]

    batch = False
    while (not batch) and memory_manager.size != -1:
        generate_episode(agent, memory_manager, trainer)
        batch = memory_manager.get_batch(batch_size)

    episode_rewards = []
    for i in range(training_time):

        if i % 100 == 0:
            torch.save(agent.model.state_dict(), "DQN/model")
            torch.save(optimizer.state_dict(), "DQN/opzimizer")


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
            value_losses = []
            for k in range(len(policy)):
                advantage = 0
                #advantage = mc_reward[k] - value[k]
                if done[k]:
                    advantage = step_reward[k] - value[k]
                    value_losses.append(F.smooth_l1_loss(value[k], step_reward[k]))
                else:
                    policy_next, value_next = agent.model.forward([observation[k]], [(actions[k] + 2) % 4])
                    advantage = step_reward[k] + memory_manager.gamma * value_next[0] - value[k]
                    value_losses.append(F.smooth_l1_loss(value[k], step_reward[k] + memory_manager.gamma * value_next[0]))

                categorical = torch.distributions.Categorical(policy[k])

                entropy_weight = 1000

                policy_losses.append(-categorical.log_prob(actions[k]) * advantage + entropy_weight * torch.sum(policy[k] * torch.log(policy[k] + 1e-7))) 
                #value_losses.append(F.smooth_l1_loss(value[k], mc_reward[k]))


            optimizer.zero_grad()

            loss = torch.stack(policy_losses).float().sum() + torch.stack(value_losses).float().sum()
            #loss = torch.stack(value_losses).float().sum()

            chose_percentage = np.zeros(len(possible_moves))
            for move in range(len(possible_moves)):
                chose_percentage[move] = np.sum(actions.cpu().numpy() == move) / len(actions)


            print(f"\rCurrent Iteration {i}/{training_time}, {100 * i/training_time:.2f}%, Value loss: {torch.stack(value_losses).float().mean().item():.2f}, last value loss: {value[0].item():.2f}, {mc_reward[0].item():.2f}, first value loss: {value[-1].item():.2f}, {mc_reward[-1].item():.2f}, Highest picked action: {possible_moves[np.argmax(chose_percentage)]}, {100 * np.max(chose_percentage):.2f}", end="")
            memory_manager.value_loss.append(torch.stack(value_losses).mean().item())

            if 0:
                print("mc_reward", mc_reward)
                print("value_old", value)
                print("value loss", value_losses)

            loss.backward()

            optimizer.step()        
