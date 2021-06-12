import torch
torch.manual_seed(42)

from kaggle_environments.envs.hungry_geese.hungry_geese import Action


possible_moves = [Action.NORTH.name, Action.EAST.name, Action.SOUTH.name, Action.WEST.name]

def generate_episode(agent, memory_manager, trainer):
    done = False
    observation = trainer.reset()
    old_observation = observation

    short_memory = []
    invalid_action = None
    while not done:
        
        action = agent.get_move(observation, invalid_action)
        
        observation, reward, done, info = trainer.step(action)
        
        short_memory.append([old_observation, reward, action, invalid_action, done, observation])
        old_observation = observation
    
    return memory_manager.add_memories(short_memory)


