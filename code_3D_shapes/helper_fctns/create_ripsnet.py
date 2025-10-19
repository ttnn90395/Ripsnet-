# Functions to create the RipsNet architecture.

import os
import tensorflow as tf
from tensorflow.keras import regularizers, layers

class DenseRagged(tf.keras.layers.Layer):
    def __init__(self, units, use_bias=True, activation='linear', **kwargs):
        super(DenseRagged, self).__init__(**kwargs)
        self._supports_ragged_inputs = True
        self.units = units
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
    def build(self, input_shape):
        last_dim = input_shape[-1]
        self.kernel = self.add_weight('kernel', shape=[last_dim, self.units], trainable=True)
        if self.use_bias:
            self.bias = self.add_weight('bias', shape=[self.units,], trainable=True)
        else:
            self.bias = None
        super(DenseRagged, self).build(input_shape)
    def call(self, inputs):
        outputs = tf.ragged.map_flat_values(tf.matmul, inputs, self.kernel)
        if self.use_bias:
            outputs = tf.ragged.map_flat_values(tf.nn.bias_add, outputs, self.bias)
        outputs = tf.ragged.map_flat_values(self.activation, outputs)
        return outputs

class PermopRagged(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PermopRagged, self).__init__(**kwargs)
        self._supports_ragged_inputs = True
    def build(self, input_shape):
        super(PermopRagged, self).build(input_shape)
    def call(self, inputs):
#out = tf.math.reduce_sum(inputs, axis=1)
        out = tf.math.reduce_mean(inputs, axis=1)
        return out


class DenseRagged(tf.keras.layers.Layer):
    def __init__(self, units, use_bias=True, activation='linear', **kwargs):
        super(DenseRagged, self).__init__(**kwargs)
        self._supports_ragged_inputs = True
        self.units = units
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
    def build(self, input_shape):
        last_dim = input_shape[-1]
        self.kernel = self.add_weight('kernel', shape=[last_dim, self.units], trainable=True)
        if self.use_bias:
            self.bias = self.add_weight('bias', shape=[self.units,], trainable=True)
        else:
            self.bias = None
        super(DenseRagged, self).build(input_shape)
    def call(self, inputs):
        outputs = tf.ragged.map_flat_values(tf.matmul, inputs, self.kernel)
        if self.use_bias:
            outputs = tf.ragged.map_flat_values(tf.nn.bias_add, outputs, self.bias)
        outputs = tf.ragged.map_flat_values(self.activation, outputs)
        return outputs

class PermopRagged(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PermopRagged, self).__init__(**kwargs)
        self._supports_ragged_inputs = True
    def build(self, input_shape):
        super(PermopRagged, self).build(input_shape)
    def call(self, inputs):
        #out = tf.math.reduce_sum(inputs, axis=1)
        out = tf.math.reduce_mean(inputs, axis=1)
        return out

def create_ripsnet(
                   input_dimension,
                   ragged_layers=[30,20,10],
                   dense_layers=[64,128,256],
                   output_units=2500,
                   activation_fct='relu',
                   output_activation='sigmoid',
                   dropout=0,
                   kernel_regularization=0
    ):
    dim = input_dimension
    regularization = kernel_regularization

    inputs = tf.keras.Input(shape=(None, dim), dtype="float32", ragged=True)
    x = DenseRagged(units=ragged_layers[0], use_bias=True, activation=activation_fct)(inputs)  # 30
    # x = tf.keras.layers.Dropout(dropout)(x)
    for n_units in ragged_layers[1:]:
        x = DenseRagged(units=n_units, use_bias=True, activation=activation_fct)(x)
        # x = tf.keras.layers.Dropout(dropout)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
    x = PermopRagged()(x)

    for n_units in dense_layers:
        x = tf.keras.layers.Dense(n_units, activation=activation_fct,
                                  kernel_regularizer=regularizers.l2(regularization))(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        #x = tf.keras.layers.BatchNormalization()(x)

    outputs = tf.keras.layers.Dense(output_units, activation=output_activation)(x)

    return inputs, outputs