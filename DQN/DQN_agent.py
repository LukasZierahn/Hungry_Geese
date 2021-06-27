import numpy as np
import torch

from shared.map import Map

class Agent(object):
    def __init__(self, model, epsilon, greedy=False, training_wheels=False):
        self.model = model

        self.greedy = greedy
        self.training_wheels = training_wheels

        self.epsilon = epsilon
        self.epsilon_counter = 0

    def get_illegal_moves(self, observation, invalid_action):
        map = Map(observation, 11)

        return map.get_suicide(observation, invalid_action)

    def get_greedy_move(self, observation, invalid_action):
        advantages = self.model.forward([observation], [invalid_action])[0]
        
        if self.training_wheels:
            suicides = self.get_illegal_moves(observation, invalid_action)
            for i in range(len(suicides)):
                if suicides[i]:
                    advantages[i] = -np.inf

        return torch.argmax(advantages)

    def get_move(self, observation, invalid_action):
        self.epsilon_counter += 1
        
        if not self.greedy:
            if np.random.random() < self.epsilon(self.epsilon_counter):
                suicides = []
                if self.training_wheels:        
                    suicides = self.get_illegal_moves(observation, invalid_action)
                
                if len(suicides) != 0 and all(suicides):
                    return np.random.choice(4)

                action = None
                while action == None or action == invalid_action or action in suicides:
                    action = np.random.choice(4)
                
                return action
        
        return self.get_greedy_move(observation, invalid_action)

