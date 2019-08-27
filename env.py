from classifier import SketchClassifier
import numpy as np
from skimage.draw import line, bezier_curve
import utils as U


class CanvasEnvironment(object):
    def __init__(self, classifier=SketchClassifier(save_path='../trained_discr/1'), dim=[28, 28], max_stroke=40):
        self.classifier = classifier
        self.game_count = 0
        self.stroke_count = 0
        self.dim = dim
        self.canvas = np.zeros(self.dim)
        self.max_stroke = max_stroke
        self.terminal = False
        self.previous_score = None
        self.goal = None

    def draw(self, action):

        assert self.goal is not None

        if self.previous_score is None:
            self.previous_score = \
                self.classifier.get_score(self.canvas.reshape(-1, self.dim[0], self.dim[1], 1), self.goal)

        # do_nothing, q_line, q_curve, x0_line, y0_line, x1_line ,y1_line,
        # x0_c, y0_c, x1_c, y1_c, x2_c, y2_c, c

        if self.stroke_count > self.max_stroke:
            self.terminal = True

        # Parameter Validation and noises
        action_category = np.argmax(action[0:3])
        axis = np.asarray(action[3:13], dtype=np.uint8) + np.int_(np.random.normal(0, 3, action[3:13].shape[0]))
        c_p = action[13] + np.random.normal(0, 1)

        for i in range(axis.shape[0]):
            if axis[i] < 0:
                axis[i] = 1
            elif axis[i] >= self.dim[0]:
                axis[i] = self.dim[0] - 2

        if action_category == 1:
            self.stroke_count += 1
            # Draw line
            rr, cc = line(axis[0], axis[1], axis[2], axis[3])
            self.canvas[rr, cc] = 1

        if action_category == 2:
            self.stroke_count += 1
            # Draw Curve
            rr, cc = bezier_curve(axis[4], axis[5],
                                  axis[6], axis[7],
                                  axis[8], axis[9],
                                  c_p)
            try:
                self.canvas[rr, cc] = 1
            except IndexError:
                print(axis[4], axis[5],
                      axis[6], axis[7],
                      axis[8], axis[9],
                      c_p)
                print(rr)
                print(cc)
                raise

        score = self.classifier.get_score(self.canvas.reshape(-1, self.dim[0], self.dim[1], 1), self.goal)

        if score > 0.9:
            self.terminal = True

        if action_category == 0:
            self.terminal = True

        #reward = 10*score**3
        reward = 1

        if score > self.previous_score:
            pass
            # reward = 1 + score - self.previous_score
        else:
            reward = 0


        '''
        if self.terminal:
            reward = reward + np.log(score) + 0.7
        '''

        if self.terminal and self.stroke_count == 0:
            reward = -10

        self.previous_score = score

        return U.to_one_hot(self.goal, self.classifier.number_of_goals), self.canvas, reward, self.terminal

    def new_canvas(self, goal):
        self.goal = goal
        self.canvas = np.zeros(self.dim)
        self.previous_score = None
        self.stroke_count = 0
        self.terminal = False

        # print(self.goal)
        # print(self.classifier.number_of_goals)

        return U.to_one_hot(self.goal, self.classifier.number_of_goals), self.canvas, 0, self.terminal

    def get_score(self):
        if self.goal is None:
            return np.zeros(self.classifier.number_of_goals)
        else:
            return self.classifier.get_score(self.canvas.reshape(-1, self.dim[0], self.dim[1], 1), self.goal)





