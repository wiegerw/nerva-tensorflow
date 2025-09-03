#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from pathlib import Path

import tensorflow as tf

from nerva_tensorflow.activation_functions import ActivationFunction, HyperbolicTangentActivation
from nerva_tensorflow.datasets import create_npz_dataloaders
from nerva_tensorflow.layers import ActivationLayer, LinearLayer
from nerva_tensorflow.learning_rate import TimeBasedScheduler
from nerva_tensorflow.loss_functions import LossFunction
from nerva_tensorflow.matrix_operations import elements_sum, Matrix
from nerva_tensorflow.multilayer_perceptron import MultilayerPerceptron
from nerva_tensorflow.optimizers import MomentumOptimizer, NesterovOptimizer, CompositeOptimizer
from nerva_tensorflow.training import stochastic_gradient_descent
from nerva_tensorflow.weight_initializers import set_bias_to_zero, set_weights_xavier_normalized

# ------------------------
# Custom activation function
# ------------------------


def Elu(alpha):
    return lambda X: tf.where(X > 0, X, alpha * (tf.exp(X) - 1))


def Elu_gradient(alpha):
    return lambda X: tf.where(X > 0, tf.ones_like(X), alpha * tf.exp(X))


class ELUActivation(ActivationFunction):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, X: Matrix) -> Matrix:
        return Elu(self.alpha)(X)

    def gradient(self, X: Matrix) -> Matrix:
        return Elu_gradient(self.alpha)(X)


# ------------------------
# Custom weight initializer
# ------------------------

def set_weights_lecun(W: Matrix):
    K, D = W.shape
    stddev = tf.sqrt(tf.constant(1.0 / D, dtype=tf.float32))
    W.assign(tf.random.normal((K, D), stddev=stddev))


# ------------------------
# Custom loss function
# ------------------------

class AbsoluteErrorLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return tf.reduce_sum(tf.abs(Y - T)).numpy()

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return tf.sign(Y - T)


def main():
    if not Path("../data/mnist-flattened.npz").exists():
        print("Error: MNIST dataset not found. Please provide the correct location or run the prepare_datasets.py script first.")
        return

    train_loader, test_loader = create_npz_dataloaders("../data/mnist-flattened.npz", batch_size=100)

    M = MultilayerPerceptron()

    # configure layer 1
    layer1 = ActivationLayer(784, 1024, ELUActivation(0.1))
    set_weights_xavier_normalized(layer1.W)
    set_bias_to_zero(layer1.b)
    optimizer_W = MomentumOptimizer(layer1.W, layer1.DW, 0.9)
    optimizer_b = NesterovOptimizer(layer1.b, layer1.Db, 0.75)
    layer1.optimizer = CompositeOptimizer([optimizer_W, optimizer_b])

    # configure layer 2
    layer2 = ActivationLayer(1024, 512, HyperbolicTangentActivation())
    set_weights_lecun(layer1.W)
    set_bias_to_zero(layer1.b)
    layer2.set_optimizer("Momentum(0.8)")

    # configure layer 3
    layer3 = LinearLayer(512, 10)
    layer3.set_weights("He")
    layer3.set_optimizer("GradientDescent")

    M.layers = [layer1, layer2, layer3]

    loss: LossFunction = AbsoluteErrorLossFunction()

    learning_rate = TimeBasedScheduler(lr=0.1, decay=0.09)

    epochs = 5

    stochastic_gradient_descent(M, epochs, loss, learning_rate, train_loader, test_loader)


if __name__ == '__main__':
    main()
