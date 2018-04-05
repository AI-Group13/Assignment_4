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


'''
Pseudocode/actual code

location, reward, is_done = self._env.step(action)

# location is a tuple of (x,y)
# reward is the numerical reward given for that step
# is_done bool for if you hit an end state or not, whether or not it was a pit/goal is denoted by reward
# action is something you'll have to cycle through yourself to find the best one

f_location, f_reward, f_is_done = self._env.future_step(f_action, f_x, f_y)

# f_location is not needed for SARSA
# f_reward used with total reward calculation
# f_is_done is not needed for SARSA
# f_x, f_y are the x & ys taken from location in the previous step
# f_action is something you'll have to cycle through yourself to find the best one

# nested for loops anyone?????
# remember, choose the first action from JUST reward, not reward + the avg of all the f_rewards


'''
