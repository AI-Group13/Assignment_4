import random

import numpy as np


class Environment:

    def __init__(self, goal_state_reward, pit_reward, move_reward, give_up_reward):

        # randomize the seed to current system time
        random.seed(None)

        # 0 is empty area, 1 is goal, 2 is pit
        # self._grid[y][x]
        self._grid = np.matrix([[0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 2, 2, 0, 0, 0],
                                [0, 2, 1, 0, 0, 2, 0],
                                [0, 0, 2, 2, 2, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]])

        self._goal_reward = goal_state_reward
        self._pit_reward = pit_reward
        self._move_reward = move_reward
        self._give_up_reward = give_up_reward

        self._x_size = self._grid.shape[1]
        self._y_size = self._grid.shape[0]

        self._current_x = None
        self._current_y = None

        self.random_start()

    def step(self, action):
        direction, double = self.get_actual_movement(action)
        x, y = self.get_final_location(self._current_x, self._current_y, direction, double)

        reward = 0
        is_done = False

        self._current_x = x
        self._current_y = y

        return (x, y), reward, is_done

    def random_start(self):

        self._current_x = None
        self._current_y = None

        while self._current_x is None and self._current_y is None:
            self._current_x = random.randint(0, 6)
            self._current_y = random.randint(0, 5)

            if self._grid[self._current_y, self._current_x] != 0:
                self._current_x = None
                self._current_y = None

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

        val = random.randint(0, 9)

        if val <= 6:
            return requested_move, False
        elif 6 < val <= 7:
            return rotate_right(requested_move), False
        elif 7 < val <= 8:
            return rotate_left(requested_move), False
        else:
            return requested_move, True

    def get_final_location(self, start_x, start_y, direction, is_double):

        def clamp(val, minn, maxn):
            return min(max(val, minn), maxn)

        current_x = start_x
        current_y = start_y

        if direction == 0:
            if is_double:
                current_y += 2
            else:
                current_y += 1
        elif direction == 1:
            if is_double:
                current_x += 2
            else:
                current_x += 1
        elif direction == 2:
            if is_double:
                current_y -= 2
            else:
                current_y -= 1
        elif direction == 3:
            if is_double:
                current_x -= 2
            else:
                current_x -= 1

        current_x = clamp(current_x, 0, self._x_size)
        current_y = clamp(current_y, 0, self._y_size)

        return current_x, current_y
