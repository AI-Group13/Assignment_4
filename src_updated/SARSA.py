import numpy as np
import random
import matplotlib.pyplot as plt

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

    def initialize_Qtable(self, env, clever= True):


        Q_table = np.zeros((env.state_size, env.action_size))
        # print ("Declared Q-table", Q_table)

        # If the initialization is wanted to be initial optimal estimates only !!

        # print ("State size ", env.state_size)
        # print ("Action Space and size", env.action_space, env.action_size)

        grid_list = env._grid.tolist()
        # print ("Length State Space converted to list", len(grid_list[0]))

        s_index = 0

        # print (env._grid[4,0])

        for col_index in range(0, len(grid_list)):
            for row_index in range(0, len(grid_list[0])):

                env.set_start_location(row_index, col_index)

                # print ("State  is ", (env._y, env._x))

                if (env._grid[env._y, env._x] == 0):

                    for a_index, act in enumerate(env.action_space):
                        # print ("Action", act)
                        ns, rew, _ = env.step(act, True)
                        # print ("Reward Obtained", rew)
                        Q_table[s_index, a_index] = rew
                        env.set_start_location(row_index, col_index)
                    # print (Q_table)
                s_index+=1

        self.Q_table = Q_table

        if clever == False:
            Q_table[Q_table != 0] = 0.5
            self.Q_table = Q_table
            return (self.Q_table)

        return self.Q_table

    def implement_sarsa(self,env, epsilon_decay = False, alpha_decay = False):


        # IMPLEMENT THE SARSA CODE HERE
        Q_tabl = self.Q_table
        print ("Initialized Q table", Q_tabl)

        #ns,rew,status = env.step(0)
        #print(status)

        running_counter = 0
        giveUp_count = 50
        epsilon = self._exploration_epsilon
        learn_rate = 0.9
        gamma = 0.95

        # Indicator for the termination state reached OR NOT
        status = False

        act_space_list = env.action_space.tolist()
        # print ("Action Space converted to list", act_space)

        # Running the algorithm for the number of trials sought
        # print ("Number of trials", self._num_trials)

        avg_reward = []

        for i in range(0,self._num_trials):

            # Initializing the counter value to 0 and making the agent spawn in a completely random state at the start of every trial using the reset condition
            running_counter = 0
            env.reset()
            status = False

            # Decaying epsilon OR learning rate that can arguably improve the performance

            if epsilon_decay == True: epsilon = epsilon*0.995

            # Need to check if alpha is supposed to increase or decrease
            # if alpha_decay == True: alpha = ( 1 - alpha)*0.995

            print ("Trial number: ", i)

            # Running the loop until fixed iterations in the trial are not over - Generally, the episode/trial would terminate
            # before this as the agent would encounter a terminal state

            # while True:
            # while status == False:
            while(running_counter<giveUp_count):

                # print ("Running in the while loop with counter value", running_counter)

                # print ("Status is", status)
                if status == True:
                    # print ("Breaking the while loop with status ",status, "and while loop number", running_counter)
                    break

                # Defining the current state with appropriate indexing for the Q table reference

                current_state = ((np.shape(env._grid)[1]*env._y) + env._x)
                random_factor = np.random.uniform(0.0,1.0)

                # current_move = 0

                # Taking a random action with probability of epsilon

                if random_factor < epsilon:
                    current_move = act_space_list.index(random.choice(act_space_list))

                    Q_cur = Q_tabl[current_state][current_move]
                    s_new, a_prev_rew, status = env.step(act_space_list[current_move])

                # Taking the greedy action

                else:

                    # Unique test to check if all the action values are same or not
                    unique_test = np.unique(Q_tabl[current_state])

                    # If all the action values are same, then the agent is again supposed to
                    # take an action, randomly out of all the equally good actions.

                    if np.size(unique_test) == 1:
                        current_move = act_space_list.index(random.choice(act_space_list))
 
                        Q_cur = Q_tabl[current_state][current_move]
                        s_new, a_prev_rew, status = env.step(act_space_list[current_move])

                    # If the action values are unequal, then take the action with highest action value

                    else:
                        current_move = np.argmax(Q_tabl[current_state])

                        Q_cur = Q_tabl[current_state][current_move]
                        s_new, a_prev_rew, status = env.step(act_space_list[current_move])

                # print ("Status is", status)

                if status == True:
                    # print ("Breaking the while loop with status", status, "and while loop number", running_counter)
                    break

                new_state = ((np.shape(env._grid)[1]*env._y) + env._x)
                random_factor = np.random.uniform(0.0,1.0)

                # Repeating the same procedure for the new state where we are supposed to take another epsilon-greedy policy

                if random_factor < epsilon:
                    rand_move = act_space_list.index(random.choice(act_space_list))
                    Q_next = Q_tabl[new_state][rand_move]

                    _, _, status = env.step(act_space_list[rand_move])

                else:

                    unique_test = np.unique(Q_tabl[new_state])

                    if np.size(unique_test) == 1:

                        rand_move = act_space_list.index(random.choice(act_space_list))
                        Q_next = Q_tabl[new_state][rand_move]
                        _, _, status = env.step(act_space_list[rand_move])

                    else:
                        move = np.argmax(Q_tabl[new_state])
                        Q_next = Q_tabl[new_state][move]
                        _, _, status  = env.step(act_space_list[move])


                Update_Q = Q_cur + learn_rate*(a_prev_rew + (gamma*Q_next) - Q_cur)

                Q_tabl[current_state][current_move] = Update_Q

                running_counter += 1
            if running_counter !=0:
               avg_reward.append(env.running_reward/running_counter)

        print ("Running reward", env.running_reward)
        # print ("Average rewards", avg_reward)
        print(Q_tabl)

        return  Q_tabl, avg_reward

    def final_world(self, Qtable):

        actions = {0:'^', 1:'>', 2:'v', 3:'<', 4:'G', 10:'N'}

        final_world_values = []
        final_world_actions = []

        for i in range(0,6):

            maxxs = []
            acs = []
            for j in range(0,7):

                table_index = i*7 + j
                max_act_value = np.amax(Qtable[table_index])
                max_act_value_act = np.argmax(Qtable[table_index])

                maxxs.append(max_act_value)
                if max_act_value == 0:
                    max_act_value_act = 10
                acs.append(actions[max_act_value_act])

            final_world_values.append(maxxs)
            final_world_actions.append(acs)

        print ("Final Action Values in the world \n", np.matrix(final_world_values), "\n\n" )
        print ("Final Actions in the world \n ", np.matrix(final_world_actions), "\n\n")

    def plot_graphs(self, avgrew):

        newrew = [sum(avgrew[i:i+50])/50 for i in range(0, len(avgrew), 50)]
        plt.plot(newrew)
        plt.xlabel("Number of trials - 1 step = 50 trials")
        plt.ylabel("Average reward for the trials")
        plt.show()




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
