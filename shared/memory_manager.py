import numpy as np
import random

from kaggle_environments.envs.hungry_geese.hungry_geese import Action

possible_moves = [Action.NORTH.name, Action.EAST.name, Action.SOUTH.name, Action.WEST.name]

class INDEX(object):
    old_observation = 0
    reward          = 1
    action          = 2
    invalid_action  = 3
    done            = 4
    observation     = 5

class MemoryManager(object):
    def __init__(self, size, gamma) -> None:
        self.size = size
        self.memory = []
        self.last_episode = []
        self.memory_position = 0

        self.places = []
        self.rewards = []

        self.gamma = gamma
        self.INDEX = INDEX()

    
    def add_memory(self, new_memory, adjusted_reward):
        old_observation, _, action, invalid_action, done, observation = new_memory
        
        memory = [old_observation, adjusted_reward, action, invalid_action, done, observation]
        self.last_episode.append(memory)

        if self.size != -1:
            if len(self.memory) < self.size:
                self.memory.append(memory)
            else:
                self.memory[self.memory_position] = memory
                self.memory_position = (self.memory_position + 1) % self.size
            


    def add_memories(self, new_memories):
        last_observation = new_memories[-1][self.INDEX.observation]

        survived = len(last_observation.geese[last_observation.index]) != 0
        others_survived = np.sum([len(x) != 0 for x in last_observation.geese]) - survived

        self.places.append(others_survived)

        actions = np.array([x[self.INDEX.action] for x in new_memories])
        print(f"survived: {survived}, others alive: {others_survived}, timesteps: {last_observation.step}")
        for i in range(len(possible_moves)):
            num_chosen = np.sum(actions == i)
            print(f"{possible_moves[i]}: {num_chosen * 100 / len(actions):.2f}% ({num_chosen}), ", end="")
        print()

        adjusted_reward = 100 * (-others_survived) + survived * 100
        #adjusted_reward = len(new_memories)
        self.rewards.append(adjusted_reward)

        self.last_episode = []
        self.add_memory(new_memories[-1], adjusted_reward)
        for i in range(2, len(new_memories)):
            adjusted_reward = adjusted_reward * self.gamma
            self.add_memory(new_memories[-i], len(new_memories[-i][self.INDEX.observation].geese[last_observation.index]))
        
        return adjusted_reward
    
    def get_batch(self, batch_size):
        if len(self.memory) < batch_size:
            return False

        return random.sample(self.memory, batch_size)


            
