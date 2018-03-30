import random


class SARSA:
    def __init__(self, goal_state_reward, pit_reward, move_reward,
                 give_up_reward, num_trails, exploration_epsilon):
        self._goal_reward = goal_state_reward
        self._pit_reward = pit_reward
        self._move_reward = move_reward
        self._give_up_reward = give_up_reward
        self._num_trials = num_trails
        self._exploration_epsilon = exploration_epsilon

        # randomize the seed to current system time
        random.seed(None)

    def learn(self):
        for ii in range(self._num_trials):
            # do the trails, once we figure out how to do it
            pass

        self.print_recommended_actions()

    def print_recommended_actions(self):
        pass

    '''
    requested move = [0, 1, 2, 3] for [up, right, down, left]
    returns (move, is_double) where 'move' is the same mapping as the input, and 'is_double' is a bool flag
    '''
    @staticmethod
    def get_actual_movement(requested_move):

        def rotate_right(wanted_move):
            wanted_move += 1
            if wanted_move == 4:
                wanted_move = 0
            return wanted_move

        def rotate_left(wanted_move):
            wanted_move -= 1
            if wanted_move == -1:
                wanted_move = 4
            return wanted_move

        val = random.randint(0, 10)

        if val <= 7:
            return requested_move, False
        elif 7 < val <= 8:
            return rotate_right(requested_move), False
        elif 8 < val <= 9:
            return rotate_left(requested_move), False
        else:
            return requested_move, True
