"""
I made use of the OpenAI Gym to provide the environment of 
a simple game called Frozen Lake. Using Q-learning, 
I then trained an agent to play the game along with the 
visualization of how the agent does while being trained.
The motivation is to learn and explore more about machine learning. 
Reinforcement learning algorithms intrigue me a lot and so I found 
myself implementing one.
"""



#importing all the libraries
import numpy as np
import gym
import random
import time
from IPython.display import clear_output
# to create our environment
#With this env object, we’re able to query for information about the environment, sample states and actions, retrieve rewards, and have our agent navigate the frozen lake.
env = gym.make("FrozenLake-v0")
#creating the q-table
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))
#the number of rows in the table is equivalent to the size of the state space in the environment,
# and the number of columns is equivalent to the size of the action space

#print(q_table)

# create and initialize all the parameters needed to implement the Q-learning algorithm.
num_episodes = 10000#total number of episodes we want the agent to play during training
max_steps_per_episode = 100#maximum number of steps that our agent is allowed to take within a single episode

learning_rate = 0.1
discount_rate = 0.99

#these four parameters below are related to the exploration-exploitation trade-off
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

#list to hold all of the rewards we’ll get from each episode
rewards_all_episodes = []

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()

    done = False#done variable keeps track of whether or not our episode is finished
    rewards_current_episode = 0#start out with no rewards at the beginning of each episode

    for step in range(max_steps_per_episode):

        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)#determine whether our agent will explore or exploit the environment in this time-step
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()

        """If the threshold is greater than the exploration_rate, which remember, is initially set to 1, 
        then our agent will exploit the environment and choose the action that has the highest Q-value in 
        the Q-table for the current state. If, on the other hand, the threshold is less than or equal to 
        the exploration_rate, then the agent will explore the environment, and sample an action randomly."""

        new_state, reward, done, info = env.step(action)
        #After our action is chosen, we then take that action by calling step() on our env object and passing our action to it

        # Updated Q-table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                                 learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        state = new_state
        rewards_current_episode += reward

        if done == True:
            break
            # if our last action ended the episode for us then that means
            # our agent stepped in a hole or reached the goal
            # then we jump out of this loop and move on to the next episode. Otherwise, we transition to the next time-step.

    # Exploration rate decay
    exploration_rate = min_exploration_rate + \
                       (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    #decay the exploration_rate using the above formula

    rewards_all_episodes.append(rewards_current_episode)# append the rewards from the current episode to the list of rewards from all episodes

# Calculates and prints the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
count = 1000
print("********Average reward per thousand episodes********\n")

for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r / 1000)))
    count += 1000

# Prints updated Q-table
print("\n\n********Q-table********\n")
print(q_table)

##########
'''to visualize what is happening'''
##########

# Watch the agent play Frozen Lake by playing the best action
# from each state according to the Q-table

for episode in range(3):
    state = env.reset()
    done = False#keeps track whether or not our last action ended the episode-just like in the training loop
    print("*****EPISODE ", episode+1, "*****\n\n\n\n")
    time.sleep(1)#so that there is time to actually read that printout before it disappears from the screen

    for step in range(max_steps_per_episode):
        clear_output(wait=True)#clears the output from the current cell until there is another printout to overwrite it
        #for smoothness while watching the agent play
        env.render()
        time.sleep(0.3)
        action = np.argmax(q_table[state, :])
        new_state, reward, done, info = env.step(action)
        #set action to be the action that has the highest Q-value from the
        #Q-table for the current state, and then we take that action with env.step()

        if done:
            clear_output(wait=True)
            env.render()#to see where the agent ended up from our last time-step

            if reward == 1:
                print("****You reached the goal!****")
                time.sleep(3)
            else:
                print("****You fell through a hole!****")
                time.sleep(3)
                clear_output(wait=True)
            break

        state = new_state
        # if the last action didn’t complete the episode,
        # then we skip over the conditional, transition to the new state,
        # and move on to the next time step

env.close()
#after all three episodes are done, close the environment
