import sys
import os.path
from kaggle_environments.envs.hungry_geese.hungry_geese import Configuration, Action, row_col
import torch

kaggel = os.path.isfile("/kaggle_simulations/agent/DQN/model") 

# if you have many scripts add this line before you import them
if kaggel:
    sys.path.append('/kaggle_simulations/agent/') 
from DQN.DQN_agent import Agent
from DQN.model import Model

working_directory = "tournament/A6/DQN/model"
if kaggel:
    working_directory = "/kaggle_simulations/agent/" + working_directory

model = Model()
model.load_state_dict(torch.load(working_directory, map_location="cpu"))
model.eval()

agent_class = Agent(model, lambda x: 0, greedy=True)

invalid_move = None

def agent(obs_dict, config_dict):
    model.set_config(Configuration(config_dict))
    possible_moves = [Action.NORTH.name, Action.EAST.name, Action.SOUTH.name, Action.WEST.name]

    global invalid_move
    action = agent_class.get_move(obs_dict, invalid_move)
    invalid_move = (action + 2) % 4

    return possible_moves[action]