import numpy as np
import random

class INDEX(object):
    old_observation = 0
    reward          = 1
    action          = 2
    done            = 3
    observation     = 4

class MemoryManager(object):
    def __init__(self, size, gamma) -> None:
        self.size = size
        self.memory = []
        self.memory_position = 0

        self.places = []
        self.rewards = []

        self.gamma = gamma
        self.INDEX = INDEX()

    
    def add_memory(self, new_memory, adjusted_reward):
        old_observation, _, action, done, observation = new_memory
        
        if done:
            return

        memory = [old_observation, adjusted_reward, action, done, observation]
        if len(self.memory) < self.size:
            self.memory.append(memory)
        else:
            self.memory[self.memory_position] = memory
            self.memory_position = (self.memory_position + 1) % self.size


    def add_memories(self, new_memories):
        last_observation = new_memories[-1][self.INDEX.observation]

        survived = len(last_observation.geese[last_observation.index]) != 0
        others_survived = np.sum([len(x) for x in last_observation.geese]) - survived

        self.places.append(others_survived)
        print(f"survived: {survived}, others alive: {others_survived}")

        adjusted_reward = 100 * (3 - others_survived) + survived * 100
        self.rewards.append(adjusted_reward)

        self.add_memory(new_memories[-1], adjusted_reward)
        for i in range(2, len(new_memories)):
            adjusted_reward *= self.gamma
            self.add_memory(new_memories[-i], adjusted_reward)
        
        return adjusted_reward
    
    def get_batch(self, batch_size):
        if len(self.memory) < batch_size:
            return False

        return random.sample(self.memory, batch_size)


            
