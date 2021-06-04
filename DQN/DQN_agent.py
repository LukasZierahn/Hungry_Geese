import numpy as np
import torch

class Agent(object):
    def __init__(self, model, epsilon, greedy=False):
        self.model = model

        self.greedy = greedy

        self.epsilon = epsilon
        self.epsilon_counter = 0

    def get_model_move(self, observation, invalid_action):
        advantages, _ = self.model.forward([observation], [invalid_action])
        advantages = advantages[0]

        return torch.distributions.Categorical(advantages).sample().item()

    def get_greedy_move(self, observation, invalid_action):
        advantages, _ = self.model.forward([observation], [invalid_action])
        return torch.argmax(advantages[0])

    def get_move(self, observation, invalid_action):
        self.epsilon_counter += 1
        
        action = None
        if not self.greedy:
            if np.random.random() < self.epsilon(self.epsilon_counter):
                while action == None or action == invalid_action:
                    action = np.random.choice(4)
            else:
                action = self.get_model_move(observation, invalid_action)
        else:
            action = self.get_model_move(observation, invalid_action)
        
        return action

