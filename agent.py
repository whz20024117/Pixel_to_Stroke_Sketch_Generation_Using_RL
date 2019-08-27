from reply_buffer import ReplyBuffer
from actor_critic import *
from env import CanvasEnvironment
import random
import numpy as np


GAMMA = 0.9


class StudentAgent(object):
    def __init__(self, sess, t_train_start, env=CanvasEnvironment(),
                 memory_size=50000):
        self.sess = sess
        self.env = env
        self.train_start_step = t_train_start

        self.axis_bounds = [[0 + 4, self.env.dim[0] - 4],
                            [0 + 4, self.env.dim[1] - 4]]
        self.curve_weight_bounds = [1.0, 6.0]
        self.lower = np.array([-1000,-1000,-1000,
                               self.axis_bounds[0][0],self.axis_bounds[1][0],
                               self.axis_bounds[0][0], self.axis_bounds[1][0],
                               self.axis_bounds[0][0], self.axis_bounds[1][0],
                               self.axis_bounds[0][0], self.axis_bounds[1][0],
                               self.axis_bounds[0][0], self.axis_bounds[1][0],
                               self.curve_weight_bounds[0]])

        self.upper = np.array([1000,1000,1000,
                               self.axis_bounds[0][1], self.axis_bounds[1][1],
                               self.axis_bounds[0][1], self.axis_bounds[1][1],
                               self.axis_bounds[0][1], self.axis_bounds[1][1],
                               self.axis_bounds[0][1], self.axis_bounds[1][1],
                               self.axis_bounds[0][1], self.axis_bounds[1][1],
                               self.curve_weight_bounds[1]])

        self.actor_network = Actor(sess, self.env.dim, 'actor')
        self.critic_network = Critic(sess, self.env.dim, 'critic')

        self.memory_size = memory_size
        self.experience = ReplyBuffer(self.memory_size)
        self.last_terminal = None

        self.score = np.zeros(self.env.classifier.number_of_goals)
        self.summary = tf.summary
        with tf.name_scope('summary'):
            self.q_mean = self.summary.scalar('batch_Q', self.critic_network.q_mean)
            self.critic_cost = self.summary.scalar('cost_critic', self.critic_network.cost)
            self.score_max = self.summary.scalar('mean_score', tf.reduce_max(self.score))
        self.merged = self.summary.merge_all()
        self.writer = self.summary.FileWriter('./summary', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def train(self, batch_size=64):
        self.score = self.env.get_score()

        minibatch = self.experience.get_batch(batch_size)
        goal_batch = np.asanyarray([data[0] for data in minibatch])
        state_batch = np.asarray([data[1] for data in minibatch])
        action_batch = np.asarray([data[2] for data in minibatch])
        reward_batch = np.asarray([data[3]for data in minibatch])
        next_state_batch = np.asarray([data[4] for data in minibatch])
        terminal_batch = np.asarray([data[5] for data in minibatch])

        np.resize(action_batch, [batch_size, self.actor_network.action_dim])

        # Calculate Q-value using Bellman's equation
        next_action_batch = self.actor_network.target_actions(next_state_batch, goal_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch, goal_batch, next_action_batch)
        y_batch = []

        for i in range(batch_size):
            if terminal_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])

        y_batch = np.resize(y_batch, [batch_size, 1])

        # Update Critic network
        self.critic_network.train(y_batch, state_batch, goal_batch, action_batch)

        # Update Actor network
        action_batch_for_gradients = self.actor_network.actions(state_batch, goal_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch, action_batch_for_gradients, goal_batch)
        #print(q_gradient_batch.shape)
        #print(action_batch_for_gradients.shape)

        # Inverting Gradients
        inverting_gradients = []
        for dq_das, actions in zip(q_gradient_batch, action_batch_for_gradients):
            #print('a')
            inverting_gradient = []
            for idx, tmp in enumerate(zip(dq_das, actions)):
                dq_da, action = tmp
                if dq_da >= 0.0:
                    inverting_gradient.append(dq_da * (self.upper[idx] - action) / (self.upper[idx] - self.lower[idx]))
                else:
                    inverting_gradient.append(dq_da * (action - self.lower[idx]) / (self.upper[idx] - self.lower[idx]))
            inverting_gradients.append(inverting_gradient)
        #print(np.array(inverting_gradients).shape)

        inverting_gradients = np.array(inverting_gradients).reshape(-1, self.actor_network.action_dim)
        #print(inverting_gradients.shape)

        ###############################
        self.actor_network.train(inverting_gradients, goal_batch, state_batch)

        # Update Target
        self.actor_network.update_target()
        self.critic_network.update_target()
        summary = self.sess.run(self.merged, feed_dict={self.critic_network.net_s_input: state_batch,
                                                        self.critic_network.net_goal_input: goal_batch,
                                                        self.critic_network.net_action_input: action_batch,
                                                        self.critic_network.y_input: y_batch})

        self.c_summary = self.sess.run([self.critic_network.q_mean,
                                        self.critic_network.cost,
                                        tf.reduce_max(self.score)],
                                       feed_dict={self.critic_network.net_s_input: state_batch,
                                                  self.critic_network.net_goal_input: goal_batch,
                                                  self.critic_network.net_action_input: action_batch,
                                                  self.critic_network.y_input: y_batch})
        self.writer.add_summary(summary, self.global_step())

    def predict(self, s_t, sub_goal, ep):
        if random.random() < ep:
            # How to sample explore part of the epsilon greedy policy
            action = self.sample_explore_action()
        else:
            action = self.action(s_t, sub_goal)

        return action

    def sample_explore_action(self):
        # do_nothing, q_line, q_curve, x0_line, y0_line, x1_line ,y1_line,
        # x0_c, y0_c, x1_c, y1_c, x2_c, y2_c, c
        action = np.zeros(14)
        category = random.randrange(3)
        action[category] = 1
        action[3] = np.random.randint(self.axis_bounds[0][0],self.axis_bounds[0][1])
        action[4] = np.random.randint(self.axis_bounds[1][0],self.axis_bounds[1][1])
        action[5] = np.random.randint(self.axis_bounds[0][0],self.axis_bounds[0][1])
        action[6] = np.random.randint(self.axis_bounds[1][0],self.axis_bounds[1][1])
        action[7] = np.random.randint(self.axis_bounds[0][0],self.axis_bounds[0][1])
        action[8] = np.random.randint(self.axis_bounds[1][0],self.axis_bounds[1][1])
        action[9] = np.random.randint(self.axis_bounds[0][0],self.axis_bounds[0][1])
        action[10] = np.random.randint(self.axis_bounds[1][0],self.axis_bounds[1][1])
        action[11] = np.random.randint(self.axis_bounds[0][0],self.axis_bounds[0][1])
        action[12] = np.random.randint(self.axis_bounds[1][0],self.axis_bounds[1][1])

        action[13] = np.random.uniform(self.curve_weight_bounds[0],self.curve_weight_bounds[1])

        return action

    def action(self, s_t, sub_goal):
        output = self.actor_network.action(s_t, sub_goal)

        return output

    def global_step(self):
        return self.sess.run(self.critic_network.global_step)

    def perceive(self, goal, state, action, reward, new_state, terminal):
        self.experience.add(goal, state, action, reward, new_state, terminal)

        if self.experience.count() > self.train_start_step:
            self.train()

        t = self.global_step()

        if (t-1) % 20 ==0:
            pass
            #print('Global Step：',t)

        if (t-1) % 1000 == 0 and t >= 500:
            self.actor_network.save_network(t)
            self.critic_network.save_network(t)

        if terminal:
            self.last_terminal = (state, goal)

    def get_summary(self):
        return self.c_summary


        







