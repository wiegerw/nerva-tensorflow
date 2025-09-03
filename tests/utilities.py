# Copyright 2024 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import math
from typing import Union, Sequence
import numpy as np
import tensorflow as tf

# ------------------------
# Tensor conversion
# ------------------------


def to_tensor(array: Union[Sequence, np.ndarray, tf.Tensor], variable: bool = True) -> tf.Tensor:
    """
    Convert a Python list, NumPy array, or TensorFlow tensor to a TensorFlow Variable (mutable).
    - Float arrays become tf.float32.
    - Integer arrays become tf.int64.
    - If input is already a tensor/variable, it is returned (converted if necessary).
    """
    if isinstance(array, tf.Variable):
        return array
    if isinstance(array, tf.Tensor):
        return tf.Variable(array) if variable else array
    if isinstance(array, np.ndarray) and np.issubdtype(array.dtype, np.integer):
        t = tf.convert_to_tensor(array, dtype=tf.int64)
    else:
        t = tf.convert_to_tensor(array, dtype=tf.float32)
    return tf.Variable(t) if variable else t


def to_long(array: Union[Sequence, np.ndarray, tf.Tensor]) -> tf.Tensor:
    """Convert a Python list, NumPy array, or TensorFlow tensor to tf.int64."""
    if isinstance(array, tf.Tensor):
        return tf.cast(array, tf.int64)
    return tf.convert_to_tensor(array, dtype=tf.int64)

# ------------------------
# Tensor comparison
# ------------------------


def equal_tensors(x: tf.Tensor, y: tf.Tensor) -> bool:
    """Check if two tensors are exactly equal."""
    return bool(tf.reduce_all(tf.equal(x, y)).numpy())


def almost_equal(a: Union[float, int, tf.Tensor],
                 b: Union[float, int, tf.Tensor],
                 rel_tol: float = 1e-5,
                 abs_tol: float = 1e-8) -> bool:
    """
    Compare two numeric scalars (float, int, or 0-d TensorFlow tensor) approximately.
    Returns True if close within given relative and absolute tolerances.
    """
    # Extract scalar if tensor
    if isinstance(a, tf.Tensor):
        a = a.numpy().item()
    if isinstance(b, tf.Tensor):
        b = b.numpy().item()
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def all_close(X1: tf.Tensor, X2: tf.Tensor, rtol: float = 1e-6, atol: float = 1e-6) -> bool:
    """Compare two TensorFlow tensors approximately. Returns True if all elements are close."""
    return bool(tf.reduce_all(tf.math.abs(X1 - X2) <= (atol + rtol * tf.math.abs(X2))).numpy())


def all_true(mask: tf.Tensor) -> bool:
    """Return True if all elements of a boolean tensor are True."""
    return bool(tf.reduce_all(mask).numpy())


def all_finite(x: tf.Tensor) -> bool:
    """Return True if all elements of a tensor are finite."""
    return bool(tf.reduce_all(tf.math.is_finite(x)).numpy())


def all_positive(X: tf.Tensor) -> bool:
    """Return True if all entries of X are strictly positive."""
    return bool(tf.reduce_all(X > 0).numpy())

# ------------------------
# Random tensors
# ------------------------

def randn(*shape: int) -> tf.Tensor:
    """Return a random normal tensor of given shape."""
    return tf.random.normal(shape)


def rand(*shape: int) -> tf.Tensor:
    """Return a uniform random tensor in [0,1) of given shape."""
    return tf.random.uniform(shape)

# ------------------------
# Test helpers
# ------------------------

def assert_tensors_are_close(name1: str, X1: tf.Tensor,
                             name2: str, X2: tf.Tensor,
                             rtol: float = 1e-6, atol: float = 1e-6):
    """
    Assert that two tensors are close, with helpful diagnostics.
    Raises AssertionError if not.
    """
    if not all_close(X1, X2, rtol=rtol, atol=atol):
        diff = tf.math.abs(X1 - X2)
        max_diff = tf.reduce_max(diff).numpy().item()
        raise AssertionError(f"Tensors {name1} and {name2} are not close. Max diff: {max_diff:.8f}")


def as_float(x: tf.Tensor) -> float:
    """Convert a 0-d TensorFlow tensor to a Python float."""
    if x.shape.rank != 0:
        raise ValueError("Input must be 0-dimensional")
    return float(x.numpy())

# ------------------------
# Test generation
# ------------------------

def random_float_matrix(shape, a, b):
    """
    Generates a random numpy array with the given shape and float values in the range [a, b].

    Parameters:
    shape (tuple): The shape of the numpy array to generate.
    a (float): The minimum value in the range.
    b (float): The maximum value in the range.

    Returns:
    np.ndarray: A numpy array of the specified shape with random float values in the range [a, b].
    """
    # Generate a random array with values in the range [0, 1)
    rand_array = np.random.rand(*shape)

    # Scale and shift the array to the range [a, b]
    scaled_array = a + (b - a) * rand_array

    return scaled_array


def make_target(Y: np.ndarray) -> np.ndarray:
    """
    Creates a boolean matrix T with the same shape as Y,
    where each row of T has exactly one value set to 1.

    Parameters:
    Y (np.ndarray): The input numpy array.

    Returns:
    np.ndarray: A boolean matrix with the same shape as Y,
                with exactly one True value per row.
    """
    if Y.ndim != 2:
        raise ValueError("The input array must be two-dimensional")

    # Get the shape of the input array
    rows, cols = Y.shape

    # Initialize an array of zeros with the same shape as Y
    T = np.zeros((rows, cols), dtype=bool)

    # Set one random element in each row to True
    for i in range(rows):
        random_index = np.random.randint(0, cols)
        T[i, random_index] = True

    return T
