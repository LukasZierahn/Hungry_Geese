import numpy as np
import torch

class Agent(object):
    def __init__(self, model, epsilon, greedy=False):
        self.model = model

        self.greedy = greedy

        self.epsilon = epsilon
        self.epsilon_counter = 0

    def get_greedy_move(self, observation, invalid_action):
        advantages = self.model.forward([observation], [invalid_action])
        
        return torch.argmax(advantages[0])

    def get_move(self, observation, invalid_action):
        self.epsilon_counter += 1
        
        if not self.greedy:
            if np.random.random() < self.epsilon(self.epsilon_counter):
                action = None
                while action == None or action == invalid_action:
                    action = np.random.choice(4)
                
                return action
        
        return self.get_greedy_move(observation, invalid_action)

