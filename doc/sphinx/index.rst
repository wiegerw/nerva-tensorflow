nerva_tensorflow documentation
=========================

A tiny, educational set of neural network components built on TensorFlow.

Install and build
-----------------

.. code-block:: bash

    # from repository root
    python -m pip install -U sphinx sphinx-rtd-theme
    # build HTML docs into docs_sphinx/_build/html
    sphinx-build -b html docs_sphinx docs_sphinx/_build/html

API reference
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   nerva_tensorflow
   nerva_tensorflow.activation_functions
   nerva_tensorflow.datasets
   nerva_tensorflow.layers
   nerva_tensorflow.learning_rate
   nerva_tensorflow.loss_functions
   nerva_tensorflow.matrix_operations
   nerva_tensorflow.multilayer_perceptron
   nerva_tensorflow.optimizers
   nerva_tensorflow.softmax_functions
   nerva_tensorflow.training
   nerva_tensorflow.utilities
   nerva_tensorflow.weight_initializers
