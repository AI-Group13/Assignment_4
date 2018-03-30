class SARSA:
    def __init__(self, goal_state_reward, pit_reward, move_reward,
                 give_up_reward, num_trails, exploration_epsilon):

        self._goal_reward = goal_state_reward
        self._pit_reward = pit_reward
        self._move_reward = move_reward
        self._give_up_reward = give_up_reward
        self._num_trials = num_trails
        self._exploration_epsilon = exploration_epsilon

    def learn(self):
        pass
