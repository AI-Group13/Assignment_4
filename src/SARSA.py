import numpy as np
import random
import matplotlib.pyplot as plt


class SARSA:
    def __init__(self, env_cls, num_trails, exploration_epsilon):
        self.env = env_cls

        self._num_trials = num_trails
        self._exploration_epsilon = exploration_epsilon
        self.Q_table = []

    def initialize_Qtable(self, env, clever=True):

        Q_table = np.zeros((env.state_size, env.action_size))
        # print ("Declared Q-table", Q_table)

        # print ("State size ", env.state_size)
        # print ("Action Space and size", env.action_space, env.action_size)

        grid_list = env._grid.tolist()
        # print ("Length State Space converted to list", len(grid_list[0]))

        s_index = 0

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
                s_index += 1

        self.Q_table = Q_table

        # If the initialization is wanted to be initial optimal estimates only !!

        if clever == False:
            Q_table[Q_table > 0] = 10
            self.Q_table = Q_table
            return (self.Q_table)

        return self.Q_table

    def implement_sarsa(self, env, epsilon_decay=False, alpha_decay=False):

        # IMPLEMENT THE SARSA CODE HERE

        Q_tabl = self.Q_table
        print("Initialized Q table", Q_tabl)

        giveUp_count = 50
        epsilon = self._exploration_epsilon
        learn_rate = 0.1
        gamma = 0.95

        # Indicator for the termination state reached OR NOT
        status = False

        act_space_list = env.action_space.tolist()
        # print ("Action Space converted to list", act_space_list)

        # Running the algorithm for the number of trials sought
        # print ("Number of trials", self._num_trials)

        total_episode_reward = []

        for i in range(0, self._num_trials):

            # Initializing the counter value to 0 and making the agent spawn in a completely random state at the start of every trial using the reset function
            running_counter = 0
            env.reset()
            status = False

            # Decaying epsilon OR learning rate that can arguably improve the performance

            if epsilon_decay: epsilon = epsilon * 0.9995
            if alpha_decay: learn_rate = learn_rate * 0.9995

            print("Trial number: ", i)

            # Running the loop until fixed iterations in the trial are not over - Generally, the episode/trial would terminate
            # before this as the agent would encounter a terminal state

            # while True:
            # while status == False:
            while (running_counter < giveUp_count):

                # print ("Running in the while loop with counter value", running_counter)

                # print ("Status is", status)
                if status == True:
                    # print ("Breaking the while loop with status ",status, "and while loop number", running_counter)
                    break

                # Defining the current state with appropriate indexing for the Q table reference

                current_state = ((np.shape(env._grid)[1] * env._y) + env._x)
                random_factor = np.random.uniform(0.0, 1.0)

                # Taking a random action with probability of epsilon
                if random_factor < epsilon:
                    current_move = act_space_list.index(random.choice(act_space_list))
                    # print ("Randomly taken action is ", act_space_list[current_move], "\n\n")

                    Q_cur = Q_tabl[current_state][current_move]
                    s_new, a_prev_rew, status = env.step(act_space_list[current_move])
                    # print ("Reward Obtained for the action", a_prev_rew, "\n\n")

                # Taking the greedy action
                else:

                    # If the action values are unequal, then take the action with highest action value
                    current_move = np.argmax(Q_tabl[current_state])
                    # print ("Greedily taken action is", act_space_list[current_move], "\n\n" )

                    Q_cur = Q_tabl[current_state][current_move]
                    s_new, a_prev_rew, status = env.step(act_space_list[current_move])

                # print ("Status is", status)
                if status == True:
                    # print ("Breaking the while loop with status", status, "and while loop number", running_counter)
                    Q_next = 0
                    Update_Q = Q_cur + learn_rate * (a_prev_rew + (gamma * Q_next) - Q_cur)
                    Q_tabl[current_state][current_move] = Update_Q
                    break

                new_state = ((np.shape(env._grid)[1] * env._y) + env._x)
                random_factor = np.random.uniform(0.0, 1.0)

                # Repeating the same procedure for the new state where we are supposed to take another epsilon-greedy policy action
                if random_factor < epsilon:
                    rand_move = act_space_list.index(random.choice(act_space_list))
                    Q_next = Q_tabl[new_state][rand_move]
                    # print ("Randomly taken action is ", act_space_list[rand_move], "\n\n")

                    _, _, status = env.step(act_space_list[rand_move])
                    # print ("Reward Obtained for the action", _, "\n\n")

                else:
                    move = np.argmax(Q_tabl[new_state])
                    Q_next = Q_tabl[new_state][move]
                    _, _, status = env.step(act_space_list[move])

                Update_Q = Q_cur + learn_rate * (a_prev_rew + (gamma * Q_next) - Q_cur)
                Q_tabl[current_state][current_move] = Update_Q

                running_counter += 1
                # print ("Total cumulative reward for the episode", env.running_reward, "\n")

            total_episode_reward.append(env.running_reward)

        print("Final Q-table \n", Q_tabl, "\n\n")
        print("Final epsilon", epsilon, "\n\n")
        print("Final Learning rate", learn_rate, "\n\n")

        return Q_tabl, total_episode_reward

    def final_world(self, Qtable):

        actions = {0: '^', 1: '>', 2: 'v', 3: '<', 4: 'Gi', 10: 'N'}

        final_world_values = []
        final_world_actions = []

        for i in range(0, 6):

            maxxs = []
            acs = []
            for j in range(0, 7):

                table_index = i * 7 + j
                max_act_value = np.amax(Qtable[table_index])
                max_act_value_act = np.argmax(Qtable[table_index])

                maxxs.append(max_act_value)
                if max_act_value == 0:
                    max_act_value_act = 10
                acs.append(actions[max_act_value_act])

            final_world_values.append(maxxs)
            final_world_actions.append(acs)

        print("Final Action Values in the gridworld \n", np.matrix(final_world_values), "\n\n")
        print("Final Recommended Actions in the gridworld \n ", np.matrix(final_world_actions), "\n\n")
        print("Life is GOOD")

    def plot_graphs(self, totalrew):

        newrew = [sum(totalrew[i:i + 50]) / 50 for i in range(0, len(totalrew), 50)]
        plt.plot(newrew)
        plt.title(
            "SARSA | Epsilon = 0.1 | Learning Rate = 0.1 | Rewards - Goal = 5  Pit = 5 Move = -0.1 Give up = 3 \n The agent goes into"
            "a pit for some states and gives up for certain other states")
        plt.xlabel("Number of trials - 1 step = 50 trials")
        plt.ylabel("Average reward for the trials")
        plt.show()
