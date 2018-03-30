import SARSA
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Reinforcement Learning')

    parser.add_argument('goal_state_reward', type=float, help='The reward for reaching the goal state')
    parser.add_argument('pit_fall_reward', type=float, help='The reward for falling into a pit')
    parser.add_argument('move_reward', type=float, help='The reward for moving')
    parser.add_argument('give_up_reward', type=float, help='The reward for giving up')
    parser.add_argument('number_of_trials', type=int, help='The number of learning trials to run')
    parser.add_argument('exploration_epsilon', type=float, help='The weight for exploration')

    args = vars(parser.parse_args())

    sarsa = SARSA.SARSA(
        args['goal_state_reward'],
        args['pit_fall_reward'],
        args['move_reward'],
        args['give_up_reward'],
        args['number_of_trials'],
        args['exploration_epsilon']
    )

    sarsa.learn()

