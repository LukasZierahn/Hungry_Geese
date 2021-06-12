from A3C.train import train
import multiprocessing as mp

from A3C.agent import Agent
from A3C.model import Model

from shared.memory_manager import MemoryManager

import torch
import torch.optim as optim

from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Configuration

import numpy as np

def a3c_helper(inp_dict):
    device = inp_dict["device"]
    learning_rate = inp_dict["learning_rate"]
    state_dict = inp_dict["state_dict"]
    opponents = inp_dict["opponents"]

    memory_manager = MemoryManager(-1, 0.99)

    model = Model()
    model.set_config(Configuration({"columns": 11, "rows": 7}))

    model.load_state_dict(state_dict)
    model.eval()

    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0)

    agent = Agent(model, lambda x: 0)
    agent.model.device = device

    params = {}
    params["sampeling_count"] = 1
    params["batch_size"] = -1
    params["training_time"] = 1

    env = make("hungry_geese", debug=False)
    trainer = env.train([None, *opponents])
    
    train(agent, memory_manager, optimizer, device, trainer, params)

    return agent.model.state_dict(), {"reward": memory_manager.rewards[0], "place": memory_manager.places[0], "value_loss": memory_manager.value_loss[0], "loss": memory_manager.loss[0]}

def combine_updates(updates):
    output = {}
    for key in updates[0].keys():
        output[key] = torch.mean(torch.stack([update[key] for update in updates]), dim=0)
    return output

def a3c_train(state_dict, steps, learning_rate, opponents, devices):
    rewards = []
    places = []
    value_loss = []
    loss = []

    if state_dict == None:
        state_dict = Model().state_dict()

    with mp.Pool(len(devices)) as pool:
        for i in range(steps):

            arguments = []
            for j in range(len(devices)):
                inp_dict = {}
                inp_dict["device"] = devices[j]
                inp_dict["learning_rate"] = learning_rate
                inp_dict["state_dict"] = state_dict
                inp_dict["opponents"] = opponents
                arguments.append(inp_dict)

            result = pool.map(a3c_helper, arguments)
            state_updates, infos = [x[0] for x in result], [x[1] for x in result]
            state_dict = combine_updates(state_updates)

            rewards.append(np.mean([info["reward"] for  info in infos]))
            places.append(np.mean([info["place"] for  info in infos]))
            value_loss.append(np.mean([info["value_loss"] for  info in infos]))
            loss.append(np.mean([info["loss"] for  info in infos]))

            print(f"\rCurrent Iteration {i}/{steps}, {100 * i/steps:.2f}%, Value loss: {value_loss[-1]:.2f}, place: {places[-1]}", end="")

    return state_dict, rewards, places, value_loss, loss
