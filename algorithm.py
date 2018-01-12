#!/usr/bin/env python
#
# Implements the Deep Q-Learning Algorithm as shown in the DeepMind paper

import sys
import atari_py
import numpy as np
from action_value_function import ActionValueFunction
from network import network
import random
import matplotlib.pyplot as plt

# set parameters, these are in the paper
REPLAY_MEMORY_SIZE = 10000
REPLAY_MINIBATCH_SIZE = 32
AGENT_HISTORY_LENGTH = 4
TARGET_NETWORK_UPDATE_FREQUENCY = 10000
DISCOUNT_FACTOR = 0.99
INITIAL_EXPLORATION = 1.0
FINAL_EXPLORATION = 0.1
FINAL_EXPLORATION_FRAME = 1000000
REPLAY_START_SIZE = REPLAY_MEMORY_SIZE / 50
NUM_EPISODES = 100

# initialize ALE interface
ale = atari_py.ALEInterface()
pong_path = atari_py.get_game_path('pong')
ale.loadROM(pong_path)
legal_actions = ale.getMinimalActionSet()
num_of_actions = len(legal_actions)
(screen_width, screen_height) = ale.getScreenDims()
screen_data = np.zeros((screen_height, screen_width, 3),
                       dtype=np.uint8)  # Using RGB

state1 = np.zeros((AGENT_HISTORY_LENGTH, screen_height,
                   screen_width, 3), dtype=np.uint8)
state2 = np.zeros((AGENT_HISTORY_LENGTH, screen_height,
                   screen_width, 3), dtype=np.uint8)

# observe initial state
a = legal_actions[np.random.randint(legal_actions.size)]
for i in range(AGENT_HISTORY_LENGTH):
    ale.act(a)
    ale.getScreenRGB(screen_data)
    state1[i] = np.copy(screen_data)

# initialize replay memory D
D = []
for i in range(REPLAY_START_SIZE):
    is_game_over = 0
    a = legal_actions[np.random.randint(legal_actions.size)]
    for j in range(AGENT_HISTORY_LENGTH):
        r = ale.act(a)
        if (ale.game_over()):
            is_game_over = 1
        ale.getScreenRGB(screen_data)
        state2[j] = np.copy(screen_data)
    D.append((np.copy(state1), a, r, np.copy(state2), is_game_over))
    state1 = np.copy(state2)

# initialize action-value function Q with random weights and its target clone
net = network(screen_height, screen_height, num_of_actions)

# main loop
episode = 0
step = 0
e = INITIAL_EXPLORATION
e_decrease = (INITIAL_EXPLORATION - FINAL_EXPLORATION) / \
    FINAL_EXPLORATION_FRAME
while (episode < NUM_EPISODES):
    is_game_over = 0

    # select an action a
    if (np.random.sample() < e):
        action = legal_actions[np.random.randint(legal_actions.size)]
    else:
        action = np.argmax(net.evaluate(state1))

    # carry out action and observe reward
    for i in range(AGENT_HISTORY_LENGTH):
        reward = ale.act(action)
        if (ale.game_over()):
            is_game_over = 1
        ale.getScreenRGB(screen_data)
        state2[i] = np.copy(screen_data)

    # store transition <s, a, r, s'> in replay memory D
    if (len(D) == REPLAY_MEMORY_SIZE):
        D.pop(0)
    D.append((np.copy(state1), action, reward, np.copy(state2), is_game_over))

    # sample random transitions <ss, aa, rr, ss'> from replay memory D
    D_sample = random.sample(D, REPLAY_MINIBATCH_SIZE)

    # calculate target for each minibatch transition
    for sample in D_sample:
        q_target = net.evaluate(sample[3])
        r = sample[2]
        if (sample[4]):
            t = r
        else:
            t = r + DISCOUNT_FACTOR * max(q_target)
        # network.backpropagate((t - values[sample[1]]) ** 2)
        network.backpropagate(sample[0], q_target)

    state1 = np.copy(state2)

    if e > FINAL_EXPLORATION:
        e = e - e_decrease
    step = step + 1

    # if (step == TARGET_NETWORK_UPDATE_FREQUENCY):
    #     Q_target = Q.copy()
    #     step = 0

    if (is_game_over):
        ale.reset_game()
        episode = episode + 1
