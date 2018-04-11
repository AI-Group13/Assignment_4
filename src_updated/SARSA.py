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

    def implement_sarsa(self,env):


        # IMPLEMENT THE SARSA CODE HERE
        Q_tabl = self.Q_table 
        print (Q_tabl) 
        
        #ns,rew,status = env.step(0)
        #print(status)

        running_counter = 0
        giveUp_count = 50
        epsilon = self._exploration_epsilon
        learn_rate = 0.1
        gamma = 0.9

        for i in range(0,self._num_trials):
            running_counter = 0
            env.random_start()
            while(running_counter<giveUp_count):
                current_state = ((np.shape(env._grid)[1]*env._y) + env._x)
                random_factor = np.random.uniform(0.0,1.0)
                
                current_move = 0
                if random_factor < epsilon:
                    current_move = np.random.randint(5)
                    Q_cur = Q_tabl[current_state][current_move] 
                    Q_new, Q_prev_rew, status = env.step(current_move)
                else:
                    unique_test = np.unique(Q_tabl[current_state])
                    if np.size(unique_test) == 1:
                        current_move = np.random.randint(5)
                        Q_cur = Q_tabl[current_state][current_move] 
                        Q_new, Q_prev_rew, status = env.step(current_move)
                    else:
                        current_move = np.argmax(Q_tabl[current_state])
                        Q_cur = Q_tabl[current_state][current_move] 
                        Q_new, Q_prev_rew, status = env.step(current_move)
                
                
                if status == True:
                    break
                    
                new_state = ((np.shape(env._grid)[1]*env._y) + env._x)
                random_factor = np.random.uniform(0.0,1.0)
                
                if random_factor < epsilon:
                    rand_move = np.random.randint(5)
                    Q_next = Q_tabl[new_state][rand_move] 
                else:
                    unique_test = np.unique(Q_tabl[new_state])
                    if np.size(unique_test) == 1:
                        rand_move = np.random.randint(5)
                        Q_next = Q_tabl[new_state][rand_move] 
                    else:
                        move = np.argmax(Q_tabl[new_state])
                        Q_next = Q_tabl[new_state][move]
                        
                Update_Q = Q_cur + learn_rate*(Q_prev_rew + (gamma*Q_next) - Q_cur)
                Q_tabl[current_state][current_move] = Update_Q
                running_counter += 1
                
        print(Q_tabl)

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
