import tensorflow as tf
import numpy as np
from tf_func import *


class Actor(object):
    def __init__(self, sess, dim, name, action_dim=14, num_subgoals=1, lr=0.0001):
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
        self.q_gradient_input = tf.placeholder("float", [None, self.action_dim], 'temp_grad_holder')
        self.parameters_gradients = tf.gradients(self.net_output, self.vars, grad_ys=-self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(lr).apply_gradients(zip(self.parameters_gradients, self.vars))

        self.load_network()

        # Init
        self.sess.run(tf.global_variables_initializer())
        self.update_target()

    def actions(self, states, sub_goal):
        states = np.reshape(states, [-1]+self.dim+[1])
        return self.sess.run(self.net_output,
                             feed_dict={self.net_s_input: states, self.net_goal_input: sub_goal})

    def action(self, state, sub_goal):
        states = np.reshape(state, [-1]+self.dim+[1])
        return self.sess.run(self.net_output,
                             feed_dict={self.net_s_input: states, self.net_goal_input: sub_goal})[0]

    def target_actions(self, states, sub_goal):
        states = np.reshape(states, [-1]+self.dim+[1])
        return self.sess.run(self.target_output,
                             feed_dict={self.target_s_input: states, self.target_goal_input: sub_goal})

    def train(self, q_gradient_batch, goal_batch, state_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.net_s_input: state_batch,
            self.net_goal_input: goal_batch
        })

    def update_target(self):
        self.sess.run(self.target_update)

    def create_network(self):
        # Shape: -1,28,28,1
        with tf.variable_scope('actor_net'):
            # Placeholders
            s_input = tf.placeholder(tf.float32, shape=[None] + self.dim + [1], name='net_s_input')
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
            A3 = tf.nn.relu(B3)
            #print(A3)

            # Layer4
            fc2_w = tf.get_variable('fc2_w', shape=[512, 256],
                                    initializer=tf.contrib.layers.xavier_initializer())
            fc2_b = tf.get_variable('fc2_b', shape=[256])
            Z4 = tf.matmul(A3, fc2_w)
            B4 = tf.nn.bias_add(Z4, fc2_b)
            A4 = tf.nn.relu(B4)
            # print(A4)

            # Layer5
            fc3_w = tf.get_variable('fc3_w', shape=[256, 14], initializer=tf.contrib.layers.xavier_initializer())
            fc3_b = tf.get_variable('fc3_b', shape=[14])
            Z5 = tf.matmul(A4, fc3_w)
            B5 = tf.nn.bias_add(Z5, fc3_b)
            #A5 = tf.tanh(B5)
            A5 = tf.identity(B5)
            #print(A5)

        return s_input, goal_input, A5, \
            [conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b]

    def create_target(self, tau=0.001):
        # placeholders
        s_input = tf.placeholder(tf.float32, shape=[None] + self.dim + [1], name='target_s_input')
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

        # Layer5
        Z5 = tf.matmul(A4, target_vars[8])
        B5 = tf.nn.bias_add(Z5, target_vars[9])
        A5 = tf.identity(B5)

        return s_input, goal_input, A5, target_update, target_vars

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
    def __init__(self, sess, dim, name, action_dim=14, num_subgoals=1, lr=0.001):
        self.sess = sess
        self.vars = []
        self.target_vars = []
        self.name = name
        self.action_dim = action_dim
        self.lr = lr
        self.dim = dim
        self.num_subgoals = num_subgoals
        self.global_step = tf.train.create_global_step()

        self.L2 = 0.01
        self.tau = 0.001

        self.net_s_input, \
            self.net_goal_input, \
            self.net_action_input, \
            self.net_output, \
            self.vars = self.create_network()

        self.target_s_input, \
            self.target_goal_input, \
            self.target_action_input,\
            self.target_output, \
            self.target_update, \
            self.target_vars = self.create_target()

        self.load_network()

        # Create Training ops
        self.y_input = tf.placeholder("float", [None, 1], 'tmp_y_input')
        weight_decay = tf.add_n([self.L2 * tf.nn.l2_loss(var) for var in self.vars])
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.net_output)) + weight_decay
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        self.action_gradients = tf.gradients(self.net_output, self.net_action_input)

        # For summary
        self.q_mean = tf.reduce_mean(self.net_output)

        # Init
        self.sess.run(tf.global_variables_initializer())
        self.update_target()

    def create_network(self):
        with tf.variable_scope('critic_net'):

            s_input = tf.placeholder(tf.float32, shape=[None] + self.dim + [1], name='net_s_input')
            goal_input = tf.placeholder(tf.float32, shape=[None] + [self.num_subgoals], name='net_goal_input')
            action_input = tf.placeholder(tf.float32, shape=[None] + [14], name='net_action_input')  # Action to be evaluated

            # Layer1
            conv1_w = tf.get_variable('conv1_w', [7, 7, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
            conv1_b = tf.get_variable('conv1_b', [32])
            C1 = tf.nn.conv2d(input=s_input, filter=conv1_w, strides=[1, 1, 1, 1], padding='VALID')  # 22
            B1 = tf.nn.bias_add(C1, conv1_b)
            A1 = tf.nn.relu(B1)
            # print(A1)
            P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # 11
            # print(P1)

            # Layer2
            conv2_w = tf.get_variable('conv2_w', [4, 4, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
            conv2_b = tf.get_variable('conv2_b', [64])
            C2 = tf.nn.conv2d(input=P1, filter=conv2_w, strides=[1, 2, 2, 1], padding='VALID')  # 5
            B2 = tf.nn.bias_add(C2, conv2_b)
            A2 = tf.nn.relu(B2)
            # print(A2)

            # Flatten
            A2 = tf.layers.flatten(A2)

            # Concat
            A2 = tf.concat([A2, goal_input, action_input], axis=1)

            # Layer3
            fc1_w = tf.get_variable('fc1_w', shape=[A2.shape[1], 512],
                                    initializer=tf.contrib.layers.xavier_initializer())
            fc1_b = tf.get_variable('fc1_b', shape=[512])
            Z3 = tf.matmul(A2, fc1_w)
            B3 = tf.nn.bias_add(Z3, fc1_b)
            A3 = tf.nn.relu(B3)
            # print(A3)

            # Layer4
            fc2_w = tf.get_variable('fc2_w', shape=[512, 256],
                                    initializer=tf.contrib.layers.xavier_initializer())
            fc2_b = tf.get_variable('fc2_b', shape=[256])
            Z4 = tf.matmul(A3, fc2_w)
            B4 = tf.nn.bias_add(Z4, fc2_b)
            A4 = tf.nn.relu(B4)
            # print(A4)

            # Layer5
            fc3_w = tf.get_variable('fc3_w', shape=[256, 1], initializer=tf.contrib.layers.xavier_initializer())
            fc3_b = tf.get_variable('fc3_b', shape=[1])
            Z5 = tf.matmul(A4, fc3_w)
            B5 = tf.nn.bias_add(Z5, fc3_b)
            A5 = tf.identity(B5)
            # print(A5)
        return s_input, goal_input, action_input, A5, \
            [conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b]

    def create_target(self):
        s_input = tf.placeholder(tf.float32, shape=[None] + self.dim + [1], name='target_s_input')
        goal_input = tf.placeholder(tf.float32, shape=[None] + [self.num_subgoals], name='target_goal_input')
        action_input = tf.placeholder(tf.float32, shape=[None] + [14], name='target_action_input')

        ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau)
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
        A2 = tf.concat([A2, goal_input, action_input], axis=1)

        # Layer3
        Z3 = tf.matmul(A2, target_vars[4])
        B3 = tf.nn.bias_add(Z3, target_vars[5])
        A3 = tf.tanh(B3)

        # Layer4
        Z4 = tf.matmul(A3, target_vars[6])
        B4 = tf.nn.bias_add(Z4, target_vars[7])
        A4 = tf.tanh(B4)

        # Layer5
        Z5 = tf.matmul(A4, target_vars[8])
        B5 = tf.nn.bias_add(Z5, target_vars[9])
        A5 = tf.identity(B5)

        return s_input, goal_input, action_input, A5, target_update, target_vars

    def get_q(self, state, sub_goal, action):
        state = np.reshape(state, [-1]+self.dim+[1])
        return self.sess.run(self.net_output,
                             feed_dict={self.net_s_input: state, self.net_goal_input: sub_goal,
                                        self.net_action_input: action})

    def update_target(self):
        self.sess.run(self.target_update)

    def gradients(self, state_batch, action_batch, goal_batch):
        return self.sess.run(self.action_gradients, feed_dict={
            self.net_s_input: state_batch,
            self.net_action_input: action_batch,
            self.net_goal_input: goal_batch
        })[0]

    def target_q(self, state, sub_goal, action):
        return self.sess.run(self.target_output,
                             feed_dict={self.target_s_input: state, self.target_goal_input: sub_goal,
                                        self.target_action_input: action})

    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_critic_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            self.global_step = tf.train.get_global_step()
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")


    def save_network(self, time_step):
        print('save critic-network...', time_step)
        self.saver.save(self.sess, 'saved_critic_networks/' + 'critic-network', global_step=time_step)

    def train(self, ys, states, goals, actions):
        self.sess.run(self.global_step.assign_add(1))
        self.sess.run(self.optimizer, feed_dict={self.y_input: ys,
                                            self.net_s_input: states,
                                            self.net_goal_input: goals,
                                            self.net_action_input: actions})



