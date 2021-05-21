import numpy as np

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
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.values = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


        self.advantages = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )


    def set_config(self, config_dict):
        self.config = Configuration(config_dict)

    def translate_coordinates(self, player_head, position):
        row, column = row_col(position, self.config.columns)

        head_row, head_column = row_col(player_head, self.config.columns)

        return row - head_row, column - head_column

    def transform_input_single(self, observation):
        output = []

        observation = Observation(observation)
        player_index = observation.index
        player_head = observation.geese[player_index][0]

        output.append([observation.step])
        output.append([40 - observation.step % 40])

        for i in range(4):
            if i != player_index:
                if len(observation.geese[i]) == 0:
                    observation.geese[i].append(38) #38 is the middle

                output.append(self.translate_coordinates(player_head, observation.geese[i][0]))
                output.append(self.translate_coordinates(player_head, observation.geese[i][-1]))
        
        output.append(self.translate_coordinates(player_head, observation.geese[player_index][-1]))

        for i in range(2):
            output.append(self.translate_coordinates(player_head, observation.food[i]))

        return np.concatenate(output)


    def transform_input(self, observations):
        output = []
        for i in range(len(observations)):
            output.append(self.transform_input_single(observations[i]))
        
        return torch.tensor(output, device=self.device, dtype=torch.float)

    def forward(self, observation):
        transformed_input = self.transform_input(observation)
        out = self.shared(transformed_input)

        advantages = self.advantages(out)
        values = self.values(out)

        return advantages, values
