from typing import Any, Dict, List

import tensorflow as tf
from tensorflow.keras.layers import Layer

from kgcn.types import ActivationType, InitializerType, RegularizerType


class SumAggregator(Layer):
    def __init__(
            self,
            activation: ActivationType = 'relu',
            kernel_initializer: InitializerType = 'glorot_normal',
            kernel_regularizer: RegularizerType = None,
            **kwargs) -> None:

        super(SumAggregator, self).__init__(**kwargs)
        self._activation = tf.keras.activations.get(activation)
        self._kernel_initializer = tf.keras.initializers.get(
            kernel_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(
            kernel_regularizer)

    def build(
            self,
            input_shape: List[tf.TensorShape]) -> None:
        dim = input_shape[0][-1]
        self.w = self.add_weight(
            name='kernel',
            shape=(dim, dim),
            initializer=self._kernel_initializer,
            regularizer=self._kernel_regularizer)
        self.b = self.add_weight(
            name='bias',
            shape=(dim,),
            initializer='zeros')
        super(SumAggregator, self).build(input_shape)

    def call(self, inputs: List[tf.Tensor], **kwargs) -> tf.Tensor:
        entity, neighbor = inputs
        output = tf.matmul(tf.add(entity, neighbor), self.w) + self.b
        if self._activation is None:
            return output
        return self._activation(output)

    def compute_output_shape(
            self,
            input_shape: List[tf.TensorShape]) -> tf.TensorShape:
        return input_shape[0]

    def get_config(self) -> Dict[str, Any]:
        config = {
            'activation':
                tf.keras.activations.serialize(self._activation),
            'kernel_initializer':
                tf.keras.initializers.serialize(self._kernel_initializer),
            'kernel_regularizer':
                tf.keras.regularizers.serialize(self._kernel_regularizer)
        }
        base_config = super(SumAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConcatAggregator(Layer):
    def __init__(
            self,
            activation: ActivationType = 'relu',
            kernel_initializer: InitializerType = 'glorot_normal',
            kernel_regularizer: RegularizerType = None,
            **kwargs) -> None:

        super(ConcatAggregator, self).__init__(**kwargs)
        self._activation = tf.keras.activations.get(activation)
        self._kernel_initializer = tf.keras.initializers.get(
            kernel_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(
            kernel_regularizer)

    def build(
            self,
            input_shape: List[tf.TensorShape]) -> None:
        entity_emb_dim = input_shape[0][-1]
        neighbor_emb_dim = input_shape[1][-1]
        self.w = self.add_weight(
            name='kernel',
            shape=(entity_emb_dim+neighbor_emb_dim, entity_emb_dim),
            initializer=self._kernel_initializer,
            regularizer=self._kernel_regularizer)
        self.b = self.add_weight(
            name='bias',
            shape=(entity_emb_dim,),
            initializer='zeros')
        super(ConcatAggregator, self).build(input_shape)

    def call(self, inputs: List[tf.Tensor], **kwargs) -> tf.Tensor:
        entity, neighbor = inputs
        output = tf.matmul(tf.concat([entity, neighbor]), self.w) + self.b
        if self._activation is None:
            return output
        return self._activation(output)

    def compute_output_shape(
            self,
            input_shape: List[tf.TensorShape]) -> tf.TensorShape:
        return input_shape[0]

    def get_config(self) -> Dict[str, Any]:
        config = {
            'activation':
                tf.keras.activations.serialize(self._activation),
            'kernel_initializer':
                tf.keras.initializers.serialize(self._kernel_initializer),
            'kernel_regularizer':
                tf.keras.regularizers.serialize(self._kernel_regularizer)
        }
        base_config = super(ConcatAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NeighborAggregator(Layer):
    def __init__(
            self,
            activation: ActivationType = 'relu',
            kernel_initializer: InitializerType = 'glorot_normal',
            kernel_regularizer: RegularizerType = None,
            **kwargs) -> None:

        super(NeighborAggregator, self).__init__(**kwargs)
        self._activation = tf.keras.activations.get(activation)
        self._kernel_initializer = tf.keras.initializers.get(
            kernel_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(
            kernel_regularizer)

    def build(
            self,
            input_shape: List[tf.TensorShape]) -> None:
        entity_emb_dim = input_shape[0][-1]
        neighbor_emb_dim = input_shape[1][-1]
        self.w = self.add_weight(
            name='kernel',
            shape=(neighbor_emb_dim, entity_emb_dim),
            initializer=self._kernel_initializer,
            regularizer=self._kernel_regularizer)
        self.b = self.add_weight(
            name='bias',
            shape=(entity_emb_dim,),
            initializer='zeros')
        super(NeighborAggregator, self).build(input_shape)

    def call(self, inputs: List[tf.Tensor], **kwargs) -> tf.Tensor:
        _, neighbor = inputs
        output = tf.matmul(neighbor, self.w) + self.b
        if self._activation is None:
            return output
        return self._activation(output)

    def compute_output_shape(
            self,
            input_shape: List[tf.TensorShape]) -> tf.TensorShape:
        return input_shape[0]

    def get_config(self) -> Dict[str, Any]:
        config = {
            'activation':
                tf.keras.activations.serialize(self._activation),
            'kernel_initializer':
                tf.keras.initializers.serialize(self._kernel_initializer),
            'kernel_regularizer':
                tf.keras.regularizers.serialize(self._kernel_regularizer)
        }
        base_config = super(NeighborAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
