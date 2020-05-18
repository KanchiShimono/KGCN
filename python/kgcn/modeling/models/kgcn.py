from typing import Any, Dict

import tensorflow as tf
from numpy import ndarray
from tensorflow.keras.layers import Activation, Embedding, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from kgcn.modeling.layers import (
    ConcatAggregator, NeighborAggregator,
    NeighborsCombination, ReceptiveField, SumAggregator
)


class KGCN(Model):
    def __init__(
            self,
            dim: int,
            n_user: int,
            n_entity: int,
            n_relation: int,
            adj_entity: ndarray,
            adj_relation: ndarray,
            n_iter: int,
            aggregator_type: str,
            regularizer_weight: float = 0.01,
            **kwargs) -> None:

        self.dim = dim
        self.n_user = n_user
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.n_neighbor = adj_entity.shape[1]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        self.n_iter = n_iter
        if aggregator_type == 'sum':
            self.aggregator_class = SumAggregator
        elif aggregator_type == 'concat':
            self.aggregator_class = ConcatAggregator
        elif aggregator_type == 'neighbor':
            self.aggregator_class = NeighborAggregator
        else:
            raise ValueError(
                    'aggregator type requires on of sum, concat or neighbor')

        input_user = Input(shape=(1,), name='input_user', dtype=tf.int64)
        input_item = Input(shape=(1,), name='input_item', dtype=tf.int64)

        user_embedding = Embedding(
            input_dim=self.n_user,
            output_dim=self.dim,
            embeddings_initializer='glorot_normal',
            embeddings_regularizer=l2(regularizer_weight),
            name='user_embedding')
        entity_embedding = Embedding(
            input_dim=self.n_entity,
            output_dim=self.dim,
            embeddings_initializer='glorot_normal',
            embeddings_regularizer=l2(regularizer_weight),
            name='entity_embedding')
        relation_embedding = Embedding(
            input_dim=self.n_relation,
            output_dim=self.dim,
            embeddings_initializer='glorot_normal',
            embeddings_regularizer=l2(regularizer_weight),
            name='relation_embedding')

        # [batch_size, 1, dim]
        user_embed = user_embedding(input_user)

        # [(batch_size, 1), (batch_size, n_neighbor),
        # (batch_size, n_neighbor**2), (batch_size, n_neighbor**3), ...]
        entities, relations = ReceptiveField(
            num=self.n_iter,
            adj_entity=self.adj_entity,
            adj_relation=self.adj_relation)(input_item)

        # [(batch_size, 1, dim), (batch_size, n_neighbor, dim),
        # (batch_size, n_neighbor**2, dim), ...]
        neigh_ent_embed_list = [entity_embedding(e) for e in entities]
        neigh_rel_embed_list = [relation_embedding(r) for r in relations]

        neighbor_embedding = NeighborsCombination(self.n_neighbor)

        for i in range(self.n_iter):
            # use tanh as activate function only last layer
            aggregator = self.aggregator_class(
                activation='tanh' if i == self.n_iter - 1 else 'relu',
                kernel_regularizer=l2(regularizer_weight),
                name=f'aggregator_{i}')

            next_neigh_ent_embed_list = []
            for hop in range(self.n_iter - i):
                # (batch_size, n_neighbor ** hop, dim)
                neighbor_embed = neighbor_embedding(
                    [user_embed,
                     neigh_rel_embed_list[hop],
                     neigh_ent_embed_list[hop + 1]])
                next_entity_embed = aggregator(
                    [neigh_ent_embed_list[hop], neighbor_embed])
                next_neigh_ent_embed_list.append(next_entity_embed)

            neigh_ent_embed_list = next_neigh_ent_embed_list

        # [batch_size, dim]
        user_squeeze_embed = Lambda(
            lambda x: tf.squeeze(x, axis=1))(user_embed)
        item_squeeze_embed = Lambda(
            lambda x: tf.squeeze(x, axis=1))(neigh_ent_embed_list[0])

        score = Lambda(
            lambda x: tf.reduce_sum(x[0] * x[1], axis=1, keepdims=True))(
                [user_squeeze_embed, item_squeeze_embed])
        output = Activation('sigmoid')(score)

        super(KGCN, self).__init__(
            inputs=[input_user, input_item], outputs=[output], **kwargs)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'KGCN':
        return cls(**config)
