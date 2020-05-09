from typing import List, Tuple

import tensorflow as tf
from numpy import ndarray
from tensorflow.keras.layers import Activation, Embedding, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from kgcn.layers import ConcatAggregator, NeighborAggregator, SumAggregator


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
        entities, relations = Lambda(
            lambda x: self.get_neighbors(x),
            name='receptive_field')(input_item)

        # [(batch_size, 1, dim), (batch_size, n_neighbor, dim),
        # (batch_size, n_neighbor**2, dim), ...]
        neigh_ent_embed_list = [entity_embedding(e) for e in entities]
        neigh_rel_embed_list = [relation_embedding(r) for r in relations]

        neighbor_embedding = Lambda(
            lambda x: self.get_neighbor_info(x[0], x[1], x[2]),
            name='neighbor_embedding')

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

    # TODO: get_neighbors may be better to implement as keras layer
    def get_neighbors(
            self,
            entity: tf.Tensor) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        """Get entity and relation id of fixed size receptive field

        Args:
            entity: Tensor of item indices that you want to get
                receiptive field.
                Tensor shape (batch_size, 1, 1)

        Returns:
            A list of tensor shape [(batch_size, 1), (batch_size, n_neighbor),
            (batch_size, n_neighbor**2), (batch_size, n_neighbor**3), ...]
            Values of tensor is the receptive field entity/relation id for
            given entity as function arg.
        """

        entities = [entity]
        relations = []

        for i in range(self.n_iter):
            neighbor_eintities = tf.gather(self.adj_entity, entities[-1])
            neighbor_relations = tf.gather(self.adj_relation, entities[-1])
            entities.append(
                tf.reshape(
                    neighbor_eintities, shape=(-1, self.n_neighbor ** (i+1))))
            relations.append(
                tf.reshape(
                    neighbor_relations, shape=(-1, self.n_neighbor ** (i+1))))

        return entities, relations

    # TODO: get_neighbor_info may be better to implement as keras layer
    def get_neighbor_info(
            self,
            user_emb: tf.Tensor,
            rel_emb: tf.Tensor,
            ent_emb: tf.Tensor) -> tf.Tensor:
        """Get user personalized neighbor representation

        Obrain user personalized neighbor representation by computing the
        linear compination of forcused item entity v.
        The personalized mechanism is Attention.
        Query, Key and Value of attention mechanism correspond to
            query: user embedding
            key: relation embedding
            value: entities embedding of neighbor entities of forcused entity

        Args:
            user_emb: User embedding tensor shape (bath_size, 1, dim)
            rel_emb: Neighbor relation tensor
                shape (batch_size, n_neighbor ** hop, dim)
            ent_emb: Neighbor entity embedding tensor
                shape (batch_size, n_neighbor ** hop, dim)

        Returns:
            A tensor shape (bathch_size, n_neighbor ** (hop - 1), dim).
            Tensor is the neighborhood representation of item entity v.
            v is personalized for each user.
        """

        # batch_size is None when KGCN model is building.
        # So bath_size can not get by user_emb.shape[0].
        batch_size = tf.shape(user_emb)[0]
        # dim is be determined when KGCN model is building so can get by
        # user_emb.shape[-1]. If use tf.shape, fllowing tf.reshape will
        # be error.
        dim = user_emb.shape[-1]
        user_rel_score = tf.reduce_sum(
            tf.multiply(user_emb, rel_emb), axis=-1, keepdims=True)
        user_rel_score = tf.reshape(
            user_rel_score, shape=(batch_size, -1, self.n_neighbor))
        normalize_user_rel_score = tf.nn.softmax(user_rel_score, axis=-1)
        normalize_user_rel_score = tf.expand_dims(
            normalize_user_rel_score, axis=-1)

        ent_emb = tf.reshape(
            ent_emb, shape=(batch_size, -1, self.n_neighbor, dim))
        neighbors_aggregated = tf.reduce_sum(
            tf.multiply(normalize_user_rel_score, ent_emb), axis=2)
        return neighbors_aggregated
