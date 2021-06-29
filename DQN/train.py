import numpy as np
import torch.nn as nn
from DQN.model import Model

import time

import torch
torch.manual_seed(42)

import torch.nn.functional as F

from shared.generate_episode import generate_episode, possible_moves

def train(agent, memory_manager, optimizer, device, trainer, params):
    sampling_count = params["sampling_count"]
    episode_count = params["episode_count"]
    batch_size = params["batch_size"]
    training_time = params["training_time"]
    target_update = params["target_update"]

    criterion = nn.MSELoss()

    target_model = Model()
    target_model.set_config(agent.model.config)

    target_model.load_state_dict(agent.model.state_dict())
    target_model.eval()

    target_model.to(device)
    target_model.device = device

    while memory_manager.size != len(memory_manager.memory) and memory_manager.size != -1:
        generate_episode(agent, memory_manager, trainer)
        print(f"\r filling memory: {len(memory_manager.memory)}/{100 * len(memory_manager.memory) / memory_manager.size:.2f}%", end="")

    start_time = time.time()
    episode_rewards = []
    for i in range(training_time):

        if i % 100 == 0 or i == (training_time - 1):
            torch.save(agent.model.state_dict(), f"backup/{i}_model")
            torch.save(optimizer.state_dict(), f"backup/{i}_optimizer")


        for episode_index in range(episode_count):
            episode_rewards.append(generate_episode(agent, memory_manager, trainer))

        loss_this_sample = 0
        for j in range(sampling_count):
            batch = memory_manager.get_batch(batch_size)
            #batch = memory_manager.last_episode

            if batch == False:
                break

            step_reward = torch.tensor([x[memory_manager.INDEX.reward]["step_reward"] for x in batch], device=device, dtype=torch.float32).reshape(-1, 1)
            mc_reward = torch.tensor([x[memory_manager.INDEX.reward]["mc_reward"] for x in batch], device=device, dtype=torch.float32).reshape(-1, 1)

            observation = [x[memory_manager.INDEX.observation] for x in batch]
            actions = torch.tensor([x[memory_manager.INDEX.action] for x in batch], device=device, dtype=int)
            invalid_actions = [x[memory_manager.INDEX.invalid_action] for x in batch]
            done = torch.tensor([x[memory_manager.INDEX.done] for x in batch], device=device)
            next_observation = [x[memory_manager.INDEX.next_observation] for x in batch]


            q_values_next = target_model.forward(next_observation, (actions + 2) % 4)
            target = step_reward + torch.reshape(~done * memory_manager.gamma * torch.max(q_values_next, dim=1).values, (-1, 1))

            q_values = agent.model.forward(observation, invalid_actions)
            loss = criterion(torch.gather(q_values, 1, actions.unsqueeze(1)), target)

            loss_this_sample += loss.item()

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
        
        memory_manager.policy_loss.append(loss_this_sample / sampling_count)

        if target_update < 1:
            updated_state_dict = {}
            model_state_dict = agent.model.state_dict()
            target_state_dict = target_model.state_dict()
            for key in model_state_dict.keys():
                updated_state_dict[key] = target_update * model_state_dict[key] + (1 - target_update) * target_state_dict[key]

            target_model.load_state_dict(updated_state_dict)
            target_model.eval()
        else:
            if i % target_update == 0:
                target_model.load_state_dict(agent.model.state_dict())
                target_model.eval()


        time_delta = time.time() - start_time
        time_predicted = (time_delta / (i + 1))  * (training_time - (i + 1))
        debug_output = "\r"
        debug_output += f"Current Iteration {i}/{training_time}, {100 * i/training_time:.2f}%, {int(time_predicted // 60)}:{int(time_predicted % 60)}"
        debug_output += f", Loss: {memory_manager.policy_loss[-1]:.2f}"

        if (len(memory_manager.places) >= 50):
            debug_output += f", place: {memory_manager.places[-1]}/{np.mean(memory_manager.places[-50:])}"
        else:
            debug_output += f", place: {memory_manager.places[-1]}"

        debug_output += f", rewards: {memory_manager.rewards[-1]:.2f}"
        debug_output += f", memory position: {memory_manager.memory_position}"

        debug_output += f", time {int(time_delta // 60)}:{int(time_delta % 60)}"
        print(debug_output, end="")