import numpy as np



class SARSA:
    def __init__(self, env_cls, num_trails, exploration_epsilon):
        self._env = env_cls

        self._num_trials = num_trails
        self._exploration_epsilon = exploration_epsilon
        self.Q_table = []

    # def learn(self):
    #     for ii in range(self._num_trials):
    #         # do the trails, once we figure out how to do it
    #         pass

    #     self.print_recommended_actions()

    # def print_recommended_actions(self):
    #     pass

    def initialize_Qtable(self, env):

        Q_table = np.zeros((env.state_size, env.action_size))
        # print ("Declared Q-table", Q_table)

        # print ("State size ", env.state_size)
        # print ("Action Space and size", env.action_space, env.action_size)

        grid_list = env._grid.tolist()
        # print ("Length State Space converted to list", len(grid_list[0]))

        s_index = 0

        # print (env._grid[4,0])

        for col_index in range(0, len(grid_list)):
            for row_index in range(0, len(grid_list[0])):

                env.set_start_location(row_index, col_index)  

                print ("State  is ", (env._y, env._x))
     
                if (env._grid[env._y, env._x] == 0):

                    for a_index, act in enumerate(env.action_space):
                        print ("Action", act)
                        ns, rew, _ = env.step(act, True)
                        print ("Reward Obtained", rew)
                        Q_table[s_index, a_index] = rew
                        env.set_start_location(row_index, col_index) 
                    # print (Q_table)
                s_index+=1

        self.Q_table = Q_table

        return self.Q_table

    def implement_sarsa(self):


        # IMPLEMENT THE SARSA CODE HERE
        Q_tabl = self.Q_table 
        print (Q_tabl)   


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
