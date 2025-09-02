# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Weight and bias initialization helpers for linear layers."""

from nerva_tensorflow.matrix_operations import Matrix
import tensorflow as tf


def set_bias_to_zero(b: Matrix):
    """Set all bias values to zero."""
    b.assign(tf.zeros_like(b))


def set_weights_xavier(W: Matrix):
    """Initialize weights using Xavier/Glorot initialization."""
    initializer = tf.initializers.GlorotUniform()
    tf_w = tf.Variable(initializer(shape=W.shape))
    W.assign(tf.cast(tf_w, dtype=W.dtype))

def set_bias_xavier(b: Matrix):
    """Set bias to zero (Xavier scheme for bias)."""
    set_bias_to_zero(b)


def set_weights_xavier_normalized(W: Matrix):
    """Initialize weights using normalized Xavier initialization."""
    initializer = tf.initializers.GlorotNormal()
    tf_w = tf.Variable(initializer(shape=W.shape))
    W.assign(tf.cast(tf_w, dtype=W.dtype))


def set_bias_xavier_normalized(b: Matrix):
    """Set bias to zero (normalized Xavier scheme)."""
    set_bias_to_zero(b)


def set_weights_he(W: Matrix):
    """Initialize weights using He initialization for ReLU networks."""
    initializer = tf.initializers.HeNormal()
    tf_w = tf.Variable(initializer(shape=W.shape))
    W.assign(tf.cast(tf_w, dtype=W.dtype))


def set_bias_he(b: Matrix):
    """Set bias to zero (He scheme for bias)."""
    set_bias_to_zero(b)


def set_layer_weights(layer, text: str):
    """Initialize a layer's parameters according to a named scheme."""
    if text == 'Xavier':
        set_weights_xavier(layer.W)
        set_bias_xavier(layer.b)
    elif text == 'XavierNormalized':
        set_weights_xavier_normalized(layer.W)
        set_bias_xavier_normalized(layer.b)
    elif text == 'He':
        set_weights_he(layer.W)
        set_bias_he(layer.b)
    else:
        raise RuntimeError(f'Could not parse weight initializer "{text}"')
