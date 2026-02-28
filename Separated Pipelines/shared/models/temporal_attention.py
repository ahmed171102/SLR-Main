"""
temporal_attention.py — Custom Keras layer required to load the word BiLSTM model.

This MUST be registered before tf.keras.models.load_model() is called
on any word model that uses attention.
"""

import tensorflow as tf


class TemporalAttention(tf.keras.layers.Layer):
    """Attention mechanism over the temporal (sequence) dimension.

    Given input x of shape (batch, time_steps, features):
    1. Compute alignment scores  e = tanh(x @ W + b)
    2. Normalize via softmax     a = softmax(e, axis=1)
    3. Context vector            c = sum(x * a, axis=1)   → (batch, features)
    """

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

    def get_config(self):
        return super().get_config()
