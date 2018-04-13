import argparse

import SARSA
import environment
import numpy as np

def read_argument():

    parser = argparse.ArgumentParser('Reinforcement Learning')

    parser.add_argument('goal_state_reward', type=float, help='The reward for reaching the goal state')
    parser.add_argument('pit_fall_reward', type=float, help='The reward for falling into a pit')
    parser.add_argument('move_reward', type=float, help='The reward for moving')
    parser.add_argument('give_up_reward', type=float, help='The reward for giving up')
    parser.add_argument('number_of_trials', type=int, help='The number of learning trials to run')
    parser.add_argument('exploration_epsilon', type=float, help='The weight for exploration')

    args = vars(parser.parse_args())

    env = environment.Environment(
        args['goal_state_reward'],
        args['pit_fall_reward'],
        args['move_reward'],
        args['give_up_reward'])

    sarsa = SARSA.SARSA(
        env,
        args['number_of_trials'],
        args['exploration_epsilon']
    )

    return env, sarsa



def main():

    env, sarsa = read_argument()
    print ("Goal Reward \t", env._goal_reward, "\n" )
    print ("Pit Reward \t", env._pit_reward, "\n" )
    print ("Move Reward \t", env._move_reward, "\n" )
    print ("Give up Reward \t", env._give_up_reward, "\n" )

    Q_table = sarsa.initialize_Qtable(env)
    Qtab, avrew = sarsa.implement_sarsa(env)
    # print ("Initialized Q table \n", Q_table, "\n")

    sarsa.final_world(Qtab)
    sarsa.plot_graphs(avrew)


if __name__ == '__main__':
    main()
