from matplotlib.pyplot import axis
import numpy as np

from shared.map import Map

import torch
import torch.nn as nn
import torch.nn.functional as F

from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, row_col

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.config = None
        self.device = "cpu"

        self.shared = nn.Sequential(
            nn.Linear(937, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            )

        self.values = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


        self.advantages = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )


    def set_config(self, config_dict):
        self.config = Configuration(config_dict)

    def transform_input_single(self, observation, invalid_action):
        # This is a done final state and will be ignored later on
        if len(observation.geese[observation.index]) == 0:
            return [0] * 937

        observation = Observation(observation)

        output = []
        output.append([observation.step])
        output.append([invalid_action == 0, invalid_action == 1, invalid_action == 2, invalid_action == 3])
        output.append([(40 - observation.step % 40) / 40])
        output.append([len(observation.geese[observation.index])])

        for i in range(4):
            if i != observation.index:
                output.append([len(observation.geese[i]) == 0])
                output.append([len(observation.geese[i])])

        map = Map(observation, self.config.columns)
        maps = map.build_maps()

        """
        print("output", output)
        print("heads_tails", heads_tails)
        print("maps", maps)
        """

        return np.concatenate([np.concatenate(output), np.concatenate(maps)]), map.get_suicide(invalid_action)
        #return np.concatenate([np.concatenate(output), np.concatenate(heads_tails)])


    def transform_input(self, observations, invalid_actions):
        inputs = []
        suicides = []
        for i in range(len(observations)):
            transformed_single, suicide = self.transform_input_single(observations[i], invalid_actions[i])
            inputs.append(transformed_single)
            suicides.append(suicide)
        
        return torch.tensor(inputs, device=self.device, dtype=torch.float), suicides

    def forward(self, observation, invalid_actions):
        transformed_input, suicides = self.transform_input(observation, invalid_actions)
        out = self.shared(transformed_input)

        advantages = self.advantages(out)
        values = self.values(out)

        means = advantages.sum(dim=1)

        # Doing this is okay and will not create bad gradient update
        # as the random action will never pick invalid actions
        for i in range(len(suicides)):
            if all(suicides[i]):
                means[i] -= advantages[i][invalid_actions[i]]
                advantages[i][invalid_actions[i]] = -np.inf
                means[i] /= 3
                continue

            for j in range(len(suicides[0])):
                if suicides[i][j]:
                    means[i] -= advantages[i][j]
                    advantages[i][j] = -np.inf
            means[i] /= np.sum(~suicides[i])

        return values + (advantages - means.reshape(-1, 1))


    def random_move(self, observation, invalid_action):
        map = Map(observation, self.config.columns)
        suicides = map.get_suicide(invalid_action)

        if all(suicides):
            action = None
            while action == None or action == invalid_action:
                action = np.random.choice(4)
            
            return action
    
        # little bit hackey, I'll admit
        action = None
        while action == None or action in suicides:
            action = np.random.choice(4)
        
        return action


