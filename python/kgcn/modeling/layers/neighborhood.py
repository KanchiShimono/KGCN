from typing import Any, Dict, List, Tuple

import tensorflow as tf
from numpy import ndarray
from tensorflow.keras.layers import Layer


class ReceptiveField(Layer):
    def __init__(
            self,
            num: int,
            adj_entity: ndarray,
            adj_relation: ndarray) -> None:
        """Initialize ReceptiveField

        Args:
            num (int): Number of depth for receptive field
            adj_entity (ndarray): Adjusted entity array.
                The shape is [num_entity, num_neighbor]
            adj_relation (ndarray): Adjusted relation array.
                The shape is [num_entity, num_neighbor]
        """
        super(ReceptiveField, self).__init__()
        self.num = num
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        self.num_neighbor = adj_entity.shape[-1]

    def call(self, inputs: tf.Tensor) -> Tuple[
            List[tf.Tensor], List[tf.Tensor]]:
        """Get entity and relation id of fixed size receptive field

        Args:
            inputs: Tensor of item indices that you want to get
                receiptive field.
                Tensor shape (batch_size, 1, 1)

        Returns:
            A list of tensor shape [(batch_size, 1), (batch_size, n_neighbor),
            (batch_size, n_neighbor**2), (batch_size, n_neighbor**3), ...]
            Values of tensor is the receptive field entity/relation id for
            given entity as function arg.
        """
        entities = [inputs]
        relations = []

        for i in range(self.num):
            neighbor_eintities = tf.gather(self.adj_entity, entities[-1])
            neighbor_relations = tf.gather(self.adj_relation, entities[-1])
            entities.append(
                tf.reshape(
                    neighbor_eintities,
                    shape=(-1, self.num_neighbor ** (i+1))))
            relations.append(
                tf.reshape(
                    neighbor_relations,
                    shape=(-1, self.num_neighbor ** (i+1))))

        return entities, relations

    def compute_output_shape(self, input_shape: Tuple[int, int]) -> Tuple[
            List[Tuple[int, int]], List[Tuple[int, int]]]:
        common_shape = [(input_shape[0], self.num_neighbor**(i + 1))
                        for i in range(self.num)]
        entity_shape = [input_shape] + common_shape
        relation_shape = common_shape
        return (entity_shape, relation_shape)

    def get_config(self) -> Dict[str, Any]:
        config = {
            'num': self.num,
            'adj_entity': self.adj_entity,
            'adj_relation': self.adj_relation
        }
        base_config = super(ReceptiveField, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NeighborsCombination(Layer):
    def __init__(self, num_neighbor: int) -> None:
        """Initialize NeighborsCombination

        Args:
            num_neighbor (int): Number of neighbor after adjusted.
        """
        super(NeighborsCombination, self).__init__()
        self.num_neighbor = num_neighbor

    def call(
            self,
            inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Get user personalized neighbor representation

        Obrain user personalized neighbor representation by computing the
        linear compination of forcused item entity v.
        The personalized mechanism is Attention.
        Query, Key and Value of attention mechanism correspond to
            query: user embedding
            key: relation embedding
            value: entities embedding of neighbor entities of forcused entity

        Args:
            inputs: assumed three tensors.
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

        user_emb, rel_emb, ent_emb = inputs
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
            user_rel_score, shape=(batch_size, -1, self.num_neighbor))
        normalize_user_rel_score = tf.nn.softmax(user_rel_score, axis=-1)
        normalize_user_rel_score = tf.expand_dims(
            normalize_user_rel_score, axis=-1)

        ent_emb = tf.reshape(
            ent_emb, shape=(batch_size, -1, self.num_neighbor, dim))
        neighbors_combination = tf.reduce_sum(
            tf.multiply(normalize_user_rel_score, ent_emb), axis=2)
        return neighbors_combination

    def compute_output_shape(
            self,
            input_shape: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
        _, _, shape = input_shape
        return (shape[0], shape[1] // self.num_neighbor, shape[-1])

    def get_config(self) -> Dict[str, Any]:
        config = {'num_neighbor': self.num_neighbor}
        base_config = super(NeighborsCombination, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
