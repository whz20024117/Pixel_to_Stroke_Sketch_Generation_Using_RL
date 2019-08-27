import tensorflow as tf


def conv2d(x,
           output_dim,
           kernal_size,
           stride,
           name='conv2d',
           activation_func = tf.nn.relu,
           trainable=True):
    with tf.variable_scope(name):
        kernel_shape = [kernal_size[0], kernal_size[1], x.get_shape()[-1], output_dim]
        stride = [1, stride[0], stride[1], 1]
        w = tf.get_variable('w', kernel_shape, dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer,
                            trainable=trainable)
        b = tf.get_variable('b', [output_dim], dtype=tf.float32,
                            initializer=tf.zeros_initializer,
                            trainable=trainable)
        conv = tf.nn.conv2d(x, w, stride, padding='VALID')
        out = tf.nn.bias_add(conv, b)

        if activation_func is not None:
            out = activation_func(out)

    return out, w, b


def linear(input_,
           output_size,
           activation_func=None,
           trainable=True,
           name='linear'):
    shape = input_.get_shape().as_list()
    if len(shape)>2:
        dims = shape[1:]
        units = 1
        for i in dims:
            units = units*i
        input_= tf.reshape(input_, [-1, units])
        shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable('w', [shape[1], output_size], tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer,
                            trainable=trainable)
        b = tf.get_variable('b', [output_size], dtype=tf.float32, initializer=tf.zeros_initializer,
                            trainable=trainable)
        out = tf.nn.bias_add(tf.matmul(input_, w), b)

        if activation_func is not None:
            out = activation_func(out)

        return out, w, b


