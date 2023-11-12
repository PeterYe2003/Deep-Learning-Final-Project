import tensorflow as tf
import numpy as np


def urand_tensor(shape, scale=0.05, name=None):
    """
    Initialize a TensorFlow Variable with values drawn from a uniform distribution.

    :param shape: The shape of the tensor.
    :type shape: tuple
    :param scale: A scaling factor to determine the range of values.
                  Values will be drawn from the uniform distribution [-scale, scale].
                  Defaults to 0.05.
    :type scale: float, optional
    :param name: A name for the TensorFlow Variable.
                 Defaults to None.
    :type name: str, optional

    :return: A TensorFlow Variable initialized with values from a uniform distribution.
    :rtype: tf.Variable

    :example:
    >>> urand_tensor((3, 3), scale=0.1, name='weights')
    <tf.Variable 'weights:0' shape=(3, 3) dtype=float32, numpy=
    array([[ 0.08312319, -0.03281264,  0.09327489],
           [ 02151136, -0.06447601,  0.01234598],
           [-0.01456735,  0.07887638, -0.09776568]], dtype=float32)>
    """
    initial = tf.random.uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


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


