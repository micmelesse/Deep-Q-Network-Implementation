#!/usr/bin/env python

# Plays games using neural net restored from saved folder

import os
import sys
import atari_py
import numpy as np
import pygame
from network import network
import matplotlib.pyplot as plt


AGENT_HISTORY_LENGTH = 4
NUM_EPISODES = 10
learning_rate = 0.0000001
# os.makedirs("screen_shots")

if (len(sys.argv) != 2):
    print("Usage: \"python main.py [save folder name]\"")
    quit()


# init ALE
ale = atari_py.ALEInterface()
pong_path = atari_py.get_game_path('pong')
ale.loadROM(pong_path)
legal_actions = ale.getMinimalActionSet()
print("available actions {}".format(legal_actions))
(screen_width, screen_height) = ale.getScreenDims()
print("width/height: " + str(screen_width) + "/" + str(screen_height))

# init network
net = network(learning_rate,screen_height, screen_width, len(legal_actions))
net.restore(os.path.join('aws_models', sys.argv[1]))

# init pygame
pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Arcade Learning Environment Random Agent Display")
pygame.display.flip()

screen_data = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
state1 = np.zeros((AGENT_HISTORY_LENGTH, screen_height,
                   screen_width, 3), dtype=np.uint8)
state2 = np.zeros((AGENT_HISTORY_LENGTH, screen_height,
                   screen_width, 3), dtype=np.uint8)

# observe initial state
# a = np.random.choice(legal_actions)
a = legal_actions[1]
for i in range(AGENT_HISTORY_LENGTH):
    ale.act(a)
    ale.getScreenRGB(screen_data)
    state1[i] = np.copy(screen_data)

episode = 0
n_frame = 0
total_reward = 0.0
while (episode < NUM_EPISODES):
    n_frame += 1
    exit = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit = True
            break
    if (exit):
        break

    q_values = net.evaluate(net.preprocess(state1))
    # q_values[0][2] = 0.99 * q_values[0][2]
    print(q_values)
    action = legal_actions[np.argmax(q_values)]
    # print ("action %d" % action)
    reward = ale.act(action)
    total_reward += reward
    ale.getScreenRGB(screen_data)
    # plt.imsave("screen_shots/screen_shot_{}.png".format(n_frame), screen_data)
    screen_data_rot = np.flipud(np.rot90(screen_data))
    screen.blit(pygame.pixelcopy.make_surface(screen_data_rot), (0, 0))
    pygame.display.flip()

    for i in range(AGENT_HISTORY_LENGTH - 1):
        state2[i] = np.copy(state1[i + 1])
    state2[AGENT_HISTORY_LENGTH - 1] = np.copy(screen_data)
    if (np.array_equal(state1, state2)):
        ale.act(legal_actions[1])
    state1 = np.copy(state2)

    if (ale.game_over()):
        episode_frame_number = ale.getEpisodeFrameNumber()
        frame_number = ale.getFrameNumber()
        print("Frame Number: " + str(frame_number) +
              " Episode Frame Number: " + str(episode_frame_number))
        print("Episode " + str(episode) +
              " ended with score: " + str(total_reward))
        ale.reset_game()
        total_reward = 0.0
        episode = episode + 1
