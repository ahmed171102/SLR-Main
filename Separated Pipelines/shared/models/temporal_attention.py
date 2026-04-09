"""
temporal_attention.py — Custom Keras layer for temporal attention.

Used by the BiLSTM word models.  Must be registered as a custom_object
when loading any word-level .h5 model via tf.keras.models.load_model().
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class TemporalAttention(Layer):
    """Attention mechanism over the time dimension.

    Given input ``x`` of shape ``(batch, timesteps, features)``:
      1. Alignment scores ``e = tanh(x @ W + b)``
      2. Normalise         ``a = softmax(e, axis=1)``
      3. Context vector    ``c = sum(x * a, axis=1)``  → ``(batch, features)``
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.W = self.add_weight(
            name="attention_weight",
            shape=(feature_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        # e = tanh(x @ W + b)  → (batch, timesteps, 1)
        e = K.tanh(K.dot(x, self.W) + self.b)
        # a = softmax over time axis → (batch, timesteps, 1)
        a = K.softmax(e, axis=1)
        # context = weighted sum → (batch, features)
        context = K.sum(x * a, axis=1)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super().get_config()
