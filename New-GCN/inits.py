import tensorflow as tf
import numpy as np


def gb_tensor(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros_tensor(shape, name=None):
    """All zeros."""
    return tf.Variable(tf.zeros(shape, dtype=tf.float32), name=name)


def ones_tensor(shape, name=None):
    """All ones."""
    return tf.Variable(tf.ones(shape, dtype=tf.float32), name=name)


