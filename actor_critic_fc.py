import tensorflow as tf
import numpy as np
from tf_func import *


class Actor(object):
    def __init__(self, sess, dim, name, action_dim=14, num_subgoals=15, lr=0.0001):
        self.sess = sess
        self.vars = []
        self.target_vars = []
        self.name = name
        self.action_dim = action_dim
        self.lr = lr
        self.dim = dim
        self.num_subgoals = num_subgoals

        self.net_s_input, self.net_goal_input, self.net_output, self.vars = self.create_network()
        self.target_s_input, self.target_goal_input, self.target_output, \
            self.target_update, self.target_vars = self.create_target()

        # Create Training method
        self.q_gradient_input = tf.placeholder("float", [None, self.action_dim])
        self.parameters_gradients = tf.gradients(self.net_output, self.vars, grad_ys=-self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(lr).apply_gradients(zip(self.parameters_gradients, self.vars))

        self.sess.run(tf.initialize_all_variables())

        self.update_target()

        self.load_network()

    def actions(self, states, sub_goal):
        states = np.reshape(states,[-1]+self.dim+[1])
        return self.sess.run(self.net_output,
                             feed_dict={self.net_s_input: states, self.net_goal_input: sub_goal})

    def target_actions(self, states, sub_goal):
        states = np.reshape(states, [-1]+self.dim+[1])
        return self.sess.run(self.target_output,
                             feed_dict={self.target_s_input: states, self.target_goal_input: sub_goal})

    def train(self, q_gradient_batch, state_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.net_s_input: state_batch
        })

    def update_target(self):
        self.sess.run(self.target_update)

    def create_network(self):
        # Shape: -1,28,28,1
        with tf.variable_scope('actor_net'):
            # Placeholders
            s_input = tf.placeholder(tf.float32, shape=[None] + self.dim + [2], name='net_s_input')
            goal_input = tf.placeholder(tf.float32, shape=[None] + [self.num_subgoals], name='net_goal_input')
            # Layer1
            conv1_w = tf.get_variable('conv1_w', [7, 7, 1, 32],initializer=tf.contrib.layers.xavier_initializer())
            conv1_b = tf.get_variable('conv1_b', [32])
            C1 = tf.nn.conv2d(input=s_input, filter=conv1_w, strides=[1,1,1,1], padding='VALID') #22
            B1 = tf.nn.bias_add(C1, conv1_b)
            A1 = tf.nn.relu(B1)
            #print(A1)
            P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID') #11
            #print(P1)

            # Layer2
            conv2_w = tf.get_variable('conv2_w', [4, 4, 32, 64],initializer=tf.contrib.layers.xavier_initializer())
            conv2_b = tf.get_variable('conv2_b', [64])
            C2 = tf.nn.conv2d(input=P1, filter=conv2_w, strides=[1,2,2,1], padding='VALID') #5
            B2 = tf.nn.bias_add(C2, conv2_b)
            A2 = tf.nn.relu(B2)
            #print(A2)

            # Flatten
            A2 = tf.layers.flatten(A2)

            # Concat sub-goal
            A2 = tf.concat([A2, goal_input],axis=1)
            #print(A2)

            # Layer3
            fc1_w = tf.get_variable('fc1_w', shape=[A2.shape[1], 512],
                                    initializer=tf.contrib.layers.xavier_initializer())
            fc1_b = tf.get_variable('fc1_b', shape=[512])
            Z3 = tf.matmul(A2, fc1_w)
            B3 = tf.nn.bias_add(Z3, fc1_b)
            A3 = tf.tanh(B3)
            #print(A3)

            # Layer4
            fc2_w = tf.get_variable('fc2_w', shape=[512, 14], initializer=tf.contrib.layers.xavier_initializer())
            fc2_b = tf.get_variable('fc2_b', shape=[14])
            Z4 = tf.matmul(A3, fc2_w)
            B4 = tf.nn.bias_add(Z4, fc2_b)
            A4 = tf.tanh(B4)
            #print(A4)

        return s_input, goal_input, A4, \
            [conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b]

    def create_target(self, tau=0.001):
        # placeholders
        s_input = tf.placeholder(tf.float32, shape=[None] + self.dim + [2], name='target_s_input')
        goal_input = tf.placeholder(tf.float32, shape=[None] + [self.num_subgoals], name='target_goal_input')

        ema = tf.train.ExponentialMovingAverage(decay=1 - tau)
        target_update = ema.apply(self.vars)
        target_vars = [ema.average(x) for x in self.vars]

        # Layer1
        C1 = tf.nn.conv2d(input=s_input, filter=target_vars[0], strides=[1, 1, 1, 1], padding='VALID')
        B1 = tf.nn.bias_add(C1, target_vars[1])
        A1 = tf.nn.relu(B1)
        P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Layer2
        C2 = tf.nn.conv2d(input=P1, filter=target_vars[2], strides=[1, 2, 2, 1], padding='VALID')  # 5
        B2 = tf.nn.bias_add(C2, target_vars[3])
        A2 = tf.nn.relu(B2)

        # Flatten
        A2 = tf.layers.flatten(A2)

        # Concat sub-goal
        A2 = tf.concat([A2, goal_input], axis=1)

        # Layer3
        Z3 = tf.matmul(A2, target_vars[4])
        B3 = tf.nn.bias_add(Z3, target_vars[5])
        A3 = tf.tanh(B3)

        # Layer4
        Z4 = tf.matmul(A3, target_vars[6])
        B4 = tf.nn.bias_add(Z4, target_vars[7])
        A4 = tf.tanh(B4)

        return s_input, goal_input, A4, target_update, target_vars

    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def save_network(self, time_step):
        print('save actor-network...', time_step)
        self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step=time_step)


class Critic(object):
    def __init__(self, sess, dim, name, action_dim=14, num_subgoals=15, lr=0.0001, trainable=True):
        self.sess = sess
        self.vars = {}
        self.name = name
        self.lr = lr
        self.action_dim = action_dim

        self.s_input = tf.placeholder(tf.float32, shape=[None] + dim + [2], name='s_input')
        self.sub_goal = tf.placeholder(tf.float32, shape=[None] + [num_subgoals], name='sub_goal')
        self.action_q = tf.placeholder(tf.float32, shape=[None] + [14], name='action_q')  # Action to be evaluated

        with tf.variable_scope(name):

            self.conv1, self.vars['conv1_w'], self.vars['conv1_b'] = conv2d(self.s_input, 32, [4, 4],
                                                                            [2, 2], name='conv1', trainable=trainable)

            self.conv2, self.vars['conv2_w'], self.vars['conv2_b'] = conv2d(self.conv1, 64, [4, 4],
                                                                            [2, 2], name='conv2', trainable=trainable)

            # Flatten
            self.conv2 = tf.layers.flatten(self.conv2)

            # Concatenate sub goal and action_q
            self.conv2 = tf.concat([self.conv2, self.sub_goal, self.action_q], axis=1)

            self.fc1, self.vars['fc1_w'], self.vars['fc1_b'] = linear(self.conv2, 1024, tf.nn.relu,
                                                                      trainable=trainable, name='fc1')

            self.fc2, self.vars['fc2_w'], self.vars['fc2_b'] = linear(self.fc1, 512, tf.nn.relu,
                                                                      trainable=trainable, name='fc1')

            self.fc3, self.vars['fc3_w'], self.vars['fc3_b'] = linear(self.fc2, 1, tf.nn.relu,
                                                                      trainable=trainable, name='fc1')
            # Attributes
            self.output = self.fc3

    def get_q(self, state, sub_goal, action):
        return self.sess.run(self.output,
                             feed_dict={self.s_input: state, self.sub_goal: sub_goal, self.action_q: action})

    def create_update_op(self, network, tau=0.001):
        with tf.variable_scope(self.name):
            ops = []

            for name in self.vars.keys():
                op = self.vars[name].assign(tau * network.vars[name] + (1-tau) * self.vars[name])
                ops.append(op)

            self.update_op = tf.group(*ops, name='update_op')

    def update_target(self):
        assert self.update_op is not None
        self.sess.run(self.update_op)

