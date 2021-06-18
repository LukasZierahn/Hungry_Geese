import sys
import os.path
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
import torch

import numpy as np

kaggel = os.path.isfile("/kaggle_simulations/agent/DQN/model") 

# if you have many scripts add this line before you import them
if kaggel:
    sys.path.append('/kaggle_simulations/agent/') 
from A3C.model import Model

invalid_action = None

def agent(obs_dict, config_dict):
    global invalid_action
    action = None
    while action == None or action == invalid_action:
        action = np.random.choice(4)
    
    invalid_action = (action + 2) % 4
    
    possible_moves = [Action.NORTH.name, Action.EAST.name, Action.SOUTH.name, Action.WEST.name]
    return possible_moves[action]