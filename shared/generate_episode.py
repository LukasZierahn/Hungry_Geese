import torch
torch.manual_seed(42)

from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col


possible_moves = [Action.NORTH.name, Action.EAST.name, Action.SOUTH.name, Action.WEST.name]

def generate_episode(agent, memory_manager, opponents):

    env = make("hungry_geese", debug=False)

    trainer = env.train([None, *opponents])

    done = False
    observation = trainer.reset()
    old_observation = observation

    short_memory = []
    while not done:
        
        action = agent.get_move(observation)
        
        observation, reward, done, info = trainer.step(possible_moves[action])

        short_memory.append([old_observation, reward, action, done, observation])

        old_observation = observation
    
    return memory_manager.add_memories(short_memory)


