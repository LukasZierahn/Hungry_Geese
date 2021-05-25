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
            nn.Linear(329, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            )

        self.values = nn.Sequential(
            nn.Linear(64, 1)
        )


        self.advantages = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )


    def set_config(self, config_dict):
        self.config = Configuration(config_dict)

    def transform_input_single(self, observation):
        observation = Observation(observation)

        output = []
        output.append([observation.step])
        output.append([40 - observation.step % 40])
        output.append([len(observation.geese[observation.index])])

        map = Map(observation, self.config.columns)
        for i in range(2):
            output.append(map.translate(observation.food[i]))

        heads_tails = map.get_heads_tails()
        maps = map.build_maps()

        return np.concatenate([np.concatenate(output), np.concatenate(heads_tails), maps])
        #return np.concatenate([np.concatenate(output), np.concatenate(heads_tails)])


    def transform_input(self, observations):
        output = []
        for i in range(len(observations)):
            output.append(self.transform_input_single(observations[i]))
        
        return torch.tensor(output, device=self.device, dtype=torch.float)

    def forward(self, observation, invalid_actions):
        transformed_input = self.transform_input(observation)
        out = self.shared(transformed_input)


        advantages = self.advantages(out)
        values = self.values(out)

        for i in range(len(invalid_actions)):
            if invalid_actions[i] != None:
                advantages[i][invalid_actions[i]] = -np.inf

        advantages = F.softmax(advantages, dim=1)

        return (advantages - advantages.mean() + values).float(), values.float()
        #return advantages.float(), values.float()
