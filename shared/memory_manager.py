import numpy as np
import random

from kaggle_environments.envs.hungry_geese.hungry_geese import Action
from numpy.lib.function_base import place

possible_moves = [Action.NORTH.name, Action.EAST.name, Action.SOUTH.name, Action.WEST.name]

class INDEX(object):
    observation         = 0
    reward              = 1
    action              = 2
    invalid_action      = 3
    done                = 4
    next_observation    = 5

class MemoryManager(object):
    def __init__(self, size, gamma) -> None:
        self.size = size
        self.memory = []
        self.last_episode = []
        self.memory_position = 0

        self.places = []
        self.rewards = []

        self.policy_loss = []
        self.entropy_loss = []
        self.value_loss = []

        self.gamma = gamma
        self.INDEX = INDEX()

    
    def add_memory(self, new_memory, step_reward, mc_reward):
        observation, _, action, invalid_action, done, next_observation = new_memory
        
        memory = [observation, {"step_reward": step_reward, "mc_reward": mc_reward}, action, invalid_action, done, next_observation]
        self.last_episode.append(memory)

        if self.size != -1:
            if len(self.memory) < self.size:
                self.memory.append(memory)
            else:
                self.memory[self.memory_position] = memory
                self.memory_position = (self.memory_position + 1) % self.size
            

    @staticmethod
    def get_place(episode):
        places = np.zeros(4) - 1
        current_place = 0

        for i in reversed(range(len(episode))):
            geese = episode[i]["geese"]

            lengths = np.where(places == -1, [len(x) for x in geese], np.zeros(4))
            sorted_lengths = np.argsort(lengths)
            
            for j in reversed(range(len(sorted_lengths))):
                current_agent = sorted_lengths[j]

                if lengths[current_agent] == 0:
                    break

                places[current_agent] = current_place
                if j != 0 and lengths[sorted_lengths[j - 1]] != lengths[current_agent]:
                    current_place += np.sum(lengths == lengths[current_agent])

            if all(places != -1):
                return places
        
        return places

    def add_memories(self, new_memories):
        agent_index = new_memories[-1][self.INDEX.next_observation].index

        place = MemoryManager.get_place([x[self.INDEX.next_observation] for x in new_memories])[agent_index]
        self.places.append(place)

        mc_reward = 100 * (3 - place)

        self.last_episode = []
        self.add_memory(new_memories[-1], mc_reward, mc_reward)
        for i in range(2, len(new_memories)):
            got_bigger = len(new_memories[-i][self.INDEX.next_observation].geese[agent_index]) > len(new_memories[-i][self.INDEX.observation].geese[agent_index])
            step_reward = 0.25 * len(new_memories[-i][self.INDEX.next_observation].geese[agent_index]) + got_bigger * 10
            mc_reward = mc_reward * self.gamma + step_reward

            self.add_memory(new_memories[-i], step_reward, mc_reward)
        self.rewards.append(mc_reward)

    def get_batch(self, batch_size):
        if len(self.memory) < batch_size:
            return False

        return random.sample(self.memory, batch_size)


            
