#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import List, Tuple

import sklearn.datasets as dt
import tensorflow as tf
from nerva_tensorflow.activation_functions import ReLUActivation
from nerva_tensorflow.datasets import DataLoader
from nerva_tensorflow.layers import ActivationLayer, LinearLayer
from nerva_tensorflow.learning_rate import MultiStepLRScheduler
from nerva_tensorflow.loss_functions import StableSoftmaxCrossEntropyLossFunction
from nerva_tensorflow.multilayer_perceptron import MultilayerPerceptron
from nerva_tensorflow.training import stochastic_gradient_descent


def generate_synthetic_dataset(num_train_samples, num_test_samples, num_features, num_classes, num_redundant=2, class_sep=0.8, random_state=None):
    X, T = dt.make_classification(
        n_samples=num_train_samples + num_test_samples,
        n_features=num_features,
        n_informative=int(0.7 * num_features),
        n_classes=num_classes,
        n_redundant=num_redundant,
        class_sep=class_sep,
        random_state=random_state
    )

    # Split the dataset into a training and test set
    train_batch = range(0, num_train_samples)
    test_batch = range(num_train_samples, num_train_samples + num_test_samples)
    Xtrain = tf.convert_to_tensor(X[train_batch], dtype=tf.float32)
    Ttrain = tf.convert_to_tensor(T[train_batch], dtype=tf.int64)
    Xtest = tf.convert_to_tensor(X[test_batch], dtype=tf.float32)
    Ttest = tf.convert_to_tensor(T[test_batch], dtype=tf.int64)
    return Xtrain, Ttrain, Xtest, Ttest


def create_mlp(sizes: List[Tuple[int, int]]):
    M = MultilayerPerceptron()

    for i, (input_size, output_size) in enumerate(sizes):
        if i == len(sizes) - 1:
            layer = LinearLayer(input_size, output_size)
        else:
            layer = ActivationLayer(input_size, output_size, ReLUActivation())
        layer.set_optimizer('Momentum(0.9)')
        layer.set_weights('XavierNormal')
        M.layers.append(layer)

    return M


def main():
    num_train_samples = 50000
    num_test_samples = 10000
    num_features = 8
    num_classes = 5
    batch_size = 100

    Xtrain, Ttrain, Xtest, Ttest = generate_synthetic_dataset(num_train_samples, num_test_samples, num_features, num_classes)
    train_loader = DataLoader(Xtrain, Ttrain, batch_size=batch_size, num_classes=num_classes)
    test_loader = DataLoader(Xtest, Ttest, batch_size=batch_size, num_classes=num_classes)

    M = create_mlp([(num_features, 200), (200, 200), (200, num_classes)])
    loss = StableSoftmaxCrossEntropyLossFunction()
    epochs = 5
    learning_rate = MultiStepLRScheduler(lr=0.1, milestones=[10, 15], gamma=0.3)
    stochastic_gradient_descent(M, epochs, loss, learning_rate, train_loader, test_loader)


if __name__ == '__main__':
    main()
