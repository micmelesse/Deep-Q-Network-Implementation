import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
# Michael will implement this part


class network:
    # initializes dqn
    def __init__(self, screen_height, screen_width, num_of_actions):
        # self.net_graph = tf.Graph()
        # with self.net_graph.as_default():
        # build graph for network
        # raw input from atari game
        self.net_input = tf.placeholder(
            tf.float32, [None, screen_height, screen_width, 4])

        # first conv block
        self.conv2d_1 = tf.contrib.layers.conv2d(
            self.net_input, 32, 8, stride=4, activation_fn=None)
        self.conv2d_1 = tf.nn.relu(self.conv2d_1)
        print(self.conv2d_1.shape)

        # second conv block
        self.conv2d_2 = tf.contrib.layers.conv2d(
            self.conv2d_1, 64, 4, stride=2, activation_fn=None)
        self.conv2d_2 = tf.nn.relu(self.conv2d_2)
        print(self.conv2d_2.shape)

        # third conv block
        self.conv2d_3 = tf.contrib.layers.conv2d(
            self.conv2d_2, 64, 3, stride=1, activation_fn=None)
        self.conv2d_3 = tf.nn.relu(self.conv2d_3)
        print(self.conv2d_3.shape)

        # flatten out the tensor to a vector
        self.flat_output = tf.contrib.layers.flatten(self.conv2d_3)
        print(self.flat_output.shape)

        # final hidden layer
        self.fc_output = tf.contrib.layers.fully_connected(
            self.flat_output, 512, activation_fn=None)
        self.fc_output = tf.nn.relu(self.fc_output)
        print(self.fc_output.shape)

        # output layer for an arbitry number n of actions
        self.q_predicted = tf.contrib.layers.fully_connected(
            self.fc_output, num_of_actions, activation_fn=None)

        # loss

        self.q_target = tf.placeholder(dtype=tf.float32)
        self.ind = tf.placeholder(tf.int32)

        self.loss = tf.reduce_sum(
            tf.square(self.q_target - self.q_predicted[0, self.ind]))
        self.optimizing_op = tf.train.GradientDescentOptimizer(
            0.0000001).minimize(self.loss)

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        tf.global_variables_initializer().run()

    # returns a 1-D array whose indices correspond to actions, and values
    # correspond to the Q values of their respective actions
    def evaluate(self, state):
        fd = {self.net_input: state}
        return self.sess.run([self.q_predicted], fd)[0]

     # trains the Q network using SGD
    def backpropagate(self, cur_frames, ind, target_Q):
        fd = {self.net_input: cur_frames,
              self.q_target: target_Q, self.ind: ind}
        return self.sess.run([self.loss, self.optimizing_op], fd)[0]

    def preprocess(self, rgb_frames):
        frames = []
        for f in rgb_frames:
            f = Image.fromarray(f).convert('L')
            frames.append(np.array(f))
        return np.expand_dims(np.stack(frames, axis=-1), axis=0)

    def save(self, losses, rewards, scores):

        save_dir = "model_{}".format(str(datetime.now().time()))
        os.makedirs(save_dir)
        plt.plot(losses)
        plt.savefig('{}/loss_plot.png'.format(save_dir), bbox_inches='tight')
        plt.clf()
        plt.plot(rewards)
        plt.savefig('{}/reward_plot.png'.format(save_dir), bbox_inches='tight')
        plt.clf()
        plt.plot(scores)
        plt.savefig('{}/score_plot.png'.format(save_dir), bbox_inches='tight')
        self.saver.save(self.sess, "{}/model.ckpt".format(save_dir))
