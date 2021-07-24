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
        self.flipping = False

        self.map_preprocessing = nn.Sequential(
            nn.Linear(231, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            )

        self.shared = nn.Sequential(
            nn.Linear(1012, 512),
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

    def transform_input_single(self, observation, invalid_action, flips):
        # This is a done final state and will be ignored later on
        if len(observation.geese[observation.index]) == 0:
            return [0] * 244, [[0] * 231, [0] * 231, [0] * 231]

        opponent_order = np.arange(4)
        np.random.shuffle(opponent_order)

        observation = Observation(observation)
        map = Map(observation, self.config.columns, self.config.rows, flips[0], flips[1], opponent_order)

        output = []
        output.append([observation.step])

        adjusted_invalid_action = map.transform_move_wrt_flipping(invalid_action)
        output.append([adjusted_invalid_action == 0, adjusted_invalid_action == 1, adjusted_invalid_action == 2, adjusted_invalid_action == 3])
        output.append([(40 - observation.step % 40) / 40])
        output.append([len(observation.geese[observation.index])])


        for i in opponent_order:
            if i != observation.index:
                output.append([len(observation.geese[i]) == 0])
                output.append([len(observation.geese[i])])

        maps, opponent_maps = map.build_maps()

        """
        print("output", output)
        print("heads_tails", heads_tails)
        print("maps", maps)
        """

        return np.concatenate([np.concatenate(output), np.concatenate(maps)]), np.concatenate(opponent_maps, axis=1)


    def transform_input(self, observations, invalid_actions, flips):
        output = []
        opponent_maps =  []
        for i in range(len(observations)):
            single_inp, opp_maps = self.transform_input_single(observations[i], invalid_actions[i], flips[i])
            output.append(single_inp)
            opponent_maps.append(opp_maps)

        return torch.tensor(output, device=self.device, dtype=torch.float), torch.tensor(opponent_maps, device=self.device, dtype=torch.float)

    def forward(self, observation, invalid_actions):
        flips = np.zeros((len(observation), 2), dtype=np.bool)
        if self.flipping:
            flips = np.random.choice([True, False], size=(len(observation), 2))

        transformed_input, opponent_maps = self.transform_input(observation, invalid_actions, flips)

        latent_maps = []
        for i in range(len(opponent_maps[0])):
            latent_maps.append(self.map_preprocessing(opponent_maps[:, i]))

        transformed_input = torch.cat([transformed_input, *latent_maps], dim=1)

        out = self.shared(transformed_input)

        advantages = self.advantages(out)
        values = self.values(out)

        for i in range(len(advantages)):
            if flips[i][0]:
                advantages[i][1], advantages[i][3] = advantages[i][3], advantages[i][1]

            if flips[i][1]:
                advantages[i][0], advantages[i][2] = advantages[i][2], advantages[i][0]

        means = advantages.mean(dim=1)

        # Doing this is okay and will not create bad gradient update
        # as the random action will never pick invalid actions
        for i in range(len(invalid_actions)):
            if invalid_actions[i] == None:
                continue

            means[i] -= advantages[i][invalid_actions[i]] / 4
            advantages[i][invalid_actions[i]] = -np.inf

        return values + (advantages - means.reshape(-1, 1))
