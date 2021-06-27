from kaggle_environments import make
import numpy as np


def get_place_from_index(index, inp):
    for i in range(len(inp)):
        if len(inp[i][0]["observation"]["geese"][index]) == 0:
            survived = len(inp[i][0]["observation"]["geese"][index]) != 0
            others_survived = np.sum([len(x) != 0 for x in inp[i][0]["observation"]["geese"]]) - survived
            return others_survived

    survived = len(inp[-1][0]["observation"]["geese"][index]) != 0
    others_survived = np.sum([len(x) != 0 for x in inp[-1][0]["observation"]["geese"]]) - survived
    return others_survived

def evaluate_agents(agents, num_episodes=1000, debug=True):
    env = make("hungry_geese", debug=False)

    results = []

    for i in range(num_episodes):
        if debug:
            print(f"\rEvaluating: {i}/{100 * i/num_episodes:.2f}", end="")
        states = env.run(agents)

        results.append(get_place_from_index(0, states))
    return results

def evaluate_agent(agent, opponent, num_episodes=1000):
    result = evaluate_agents([agent, opponent, opponent, opponent], num_episodes, False)
    return np.mean(result), np.std(result)

def evaluate_agent_against_ensemble(agent, opponents, num_episodes=50):
    point_estimate = np.zeros(len(opponents))
    std_dev = np.zeros(len(opponents))
    for i in reversed(range(len(opponents))):
            print(f"\r{i}/{100 - 100 * i / len(opponents):.2f}, {agent} vs {opponents[i]}", end="")
            point_estimate[i], std_dev[i] = evaluate_agent(agent, opponents[i], num_episodes)

    return point_estimate, std_dev

def tournament(agents, num_episodes=50):
    point_estimate = np.zeros((len(agents), len(agents)))
    std_dev = np.zeros((len(agents), len(agents)))
    for i in reversed(range(len(agents))):
        for j in reversed(range(len(agents))):
            print(f"\r{i * len(agents) + j}/{100 - 100 * (i * len(agents) + j)/(len(agents)**2):.2f}, {agents[i]} vs {agents[j]}", end="")
            point_estimate[i][j], std_dev[i][j] = evaluate_agent(agents[i], agents[j], num_episodes)

    return point_estimate, std_dev