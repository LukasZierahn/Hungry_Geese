import numpy as np
import torch

from shared.generate_episode import possible_moves

class Agent(object):
    def __init__(self, model, epsilon):
        self.model = model

        self.epsilon = epsilon
        self.epsilon_counter = 0

        self.last_dir = None

    def get_greedy_move(self, observation):
        advantages, _ = self.model.forward([observation])
        advantages = advantages[0]
        
        advantages[self.last_dir] = -np.inf
        return torch.argmax(advantages)

    def get_move(self, observation):
        self.epsilon_counter += 1
        
        action = None
        if np.random.random() < self.epsilon(self.epsilon_counter):
            action = np.random.choice(len(possible_moves))
        else:
            action = self.get_greedy_move(observation)
        
        self.last_dir = action
        return action

