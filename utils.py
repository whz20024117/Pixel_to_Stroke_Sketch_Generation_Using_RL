import numpy as np


def to_one_hot(x, depth):
    assert np.max(x) < depth
    x = np.reshape(x, [-1, 1])
    x_one_hot = np.zeros((x.shape[0], depth))
    for i in range(x_one_hot.shape[0]):
        for j in x[i]:
            x_one_hot[i][j] = 1

    return x_one_hot


def calc_ep(t, ep_start, ep_end, t_ep_end):
    ep = (ep_start - ep_end) * max(0, t_ep_end - t) / t_ep_end + ep_end
    return ep

