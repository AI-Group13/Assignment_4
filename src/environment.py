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

        self._running_reward = 0

        self._x_size = self._grid.shape[1]
        self._y_size = self._grid.shape[0]

        self._x = None
        self._y = None

        self.random_start()

    '''
    takes in an action and determines the effect that action has on the environment
    '''

    def step(self, action):
        direction, double = self.get_actual_movement(action)

        # special check for double movements to make sure it didn't hit an ending location 1 move away
        if double:
            step_x, step_y = self.get_skipped_location(self._x, self._y, direction)
            reward, is_done = self.get_step_reward(step_x, step_y, False)
            # if we finished (hit goal or pit, either or), set the variables, and return that were finished
            if is_done:
                self._x = step_x
                self._y = step_y

                return (self._x, self._y), reward, is_done

        # if we get here, the movement was only one step, or the movement was double but we didn't hit an end goal
        self._x, self._y = self.get_final_location(self._x, self._y, direction, double)
        reward, is_done = self.get_step_reward(self._x, self._y, double)

        return (self._x, self._y), reward, is_done

    '''
    mirror of step method that instead takes in the starting location instead of changing the objects values
    '''
    def future_step(self, action, future_x, future_y):
        direction, double = self.get_actual_movement(action)

        # special check for double movements to make sure it didn't hit an ending location 1 move away
        if double:
            step_x, step_y = self.get_skipped_location(future_x, future_y, direction)
            reward, is_done = self.get_step_reward(step_x, step_y, False)
            # if we finished (hit goal or pit, either or), set the variables, and return that were finished
            if is_done:
                return (future_x, future_y), reward, is_done

        # if we get here, the movement was only one step, or the movement was double but we didn't hit an end goal
        x, y = self.get_final_location(future_x, future_y, direction, double)
        reward, is_done = self.get_step_reward(x, y, double)

        return (x, y), reward, is_done

    '''
    generates a random location. Checks to make sure that location is not a pit, otherwise it trys again
    '''

    def random_start(self):

        self._x = None
        self._y = None

        while self._x is None and self._y is None:
            self._x = random.randint(0, 6)
            self._y = random.randint(0, 5)

            if self._grid[self._y, self._x] != 0:
                self._x = None
                self._y = None

    '''
    resets the environment 
    sets the running reward to 0, then chooses a random start location
    '''

    def reset(self):

        self._running_reward = 0
        self.random_start()

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

    '''
    finds the final location of the movement
    '''

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

    '''
    finds the stepped over location for use with pit/goals checks
    '''

    def get_skipped_location(self, start_x, start_y, direction):
        def clamp(val, minn, maxn):
            return min(max(val, minn), maxn)

        current_x = start_x
        current_y = start_y

        if direction == 0:
            current_y += 1
        elif direction == 1:
            current_x += 1
        elif direction == 2:
            current_y -= 1
        elif direction == 3:
            current_x -= 1

        current_x = clamp(current_x, 0, self._x_size)
        current_y = clamp(current_y, 0, self._y_size)

        return current_x, current_y

    '''
    computes the step reward result and whether or not its done
    '''

    def get_step_reward(self, x, y, is_double):
        reward = 0
        is_done = False

        current_grid = self._grid[y][x]

        if current_grid == 0:
            reward += self._move_reward * (lambda _: 2 if is_double else 1)
        elif current_grid == 1:
            reward += self._goal_reward
            is_done = True
        elif current_grid == 2:
            reward += self._pit_reward
            is_done = True

        self._running_reward += reward

        return reward, is_done
