import numpy as np
import tensorflow as tf


def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))

def binary_cross_entropy(x, z):
    eps = 1e-12
    return (-(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))


# Load pre-trained classifier
class SketchClassifier(object):
    def __init__(self, save_path):
        self.save_path = save_path
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='X')
        self.number_of_goals = 1
        with tf.variable_scope("discriminator", reuse=None):
            x = tf.reshape(self.X, shape=[-1, 28, 28, 1])
            x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=2, padding='same', activation=lrelu)
            x = tf.layers.dropout(x, 1.0)
            x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=lrelu)
            x = tf.layers.dropout(x, 1.0)
            x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=lrelu)
            x = tf.layers.dropout(x, 1.0)
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=128, activation=lrelu)
            self.score = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_path))

    def inference(self, X):
        X.reshape(-1, 28, 28)
        # plt.show(X.reshape(self.n_H0, self.n_W0))
        result = self.sess.run(self.score, feed_dict={self.X: X})
        return X.reshape(28, 28), result

    def get_score(self, X, goal):
        X.reshape(-1,28,28)
        _, scores = self.inference(X)
        score = scores[0][goal]

        return score
