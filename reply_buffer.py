from collections import deque
import random
import numpy as np


class ReplyBuffer(object):

    def __init__(self, buffer_size, goal_dim = 1, canvas_dim=[28,28]):
        self.buffer_size = buffer_size

        self.buffer = deque()
        self.num_experience = 0
        self.canvas_dim = canvas_dim
        self.goal_dim = goal_dim

    def get_batch(self, batch_size=64):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, goal, state, action, reward, new_state, terminal):
        goal = np.reshape(goal, self.goal_dim)
        state = np.reshape(state, self.canvas_dim + [1])
        new_state = np.reshape(new_state, self.canvas_dim + [1])
        experience = (goal, state, action, reward, new_state, terminal)
        if self.num_experience < self.buffer_size:
            self.buffer.append(experience)
            self.num_experience += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        return self.num_experience

    def reset(self):
        self.buffer = deque()
        self.num_experience = 0
