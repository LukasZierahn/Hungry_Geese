import numpy as np
import torch

class Agent(object):
    def __init__(self, model):
        self.model = model

    def get_model_move(self, observation):
        advantages, _ = self.model.forward([observation])
        advantages = advantages[0]

        return torch.distributions.Categorical(advantages).sample().item()

    def get_move(self, observation, invalid_action):
        action = self.get_model_move(observation)
        return action

