import random
import environment


class SARSA:
    def __init__(self, env_cls, num_trails, exploration_epsilon):

        self._env = env_cls

        self._num_trials = num_trails
        self._exploration_epsilon = exploration_epsilon



    def learn(self):
        for ii in range(self._num_trials):
            # do the trails, once we figure out how to do it
            pass

        self.print_recommended_actions()

    def print_recommended_actions(self):
        pass
