from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import tensorflow as tf

from kgcn.types import IDVocab, RatingTriplet


class TrainingInstance:
    """A single training instance"""

    def __init__(self, input_user: int, input_item: int, label: int) -> None:
        self.input_user = input_user
        self.input_item = input_item
        self.label = label

    def __str__(self) -> str:
        return (
            f'input_user: {self.input_user}, '
            f'input_item: {self.input_item}, '
            f'label: {self.label}'
        )

    def __repr__(self) -> str:
        return self.__str__()


def create_int_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def convert_instance_to_example(
        instance: TrainingInstance) -> tf.train.Example:
    """convert a single TrainigInstance to Tensorflow Example"""
    features = {
        'input_user': create_int_feature(instance.input_user),
        'input_item': create_int_feature(instance.input_item),
        'label': create_int_feature(instance.label)
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def create_instances_from_iterable(
        itr: Iterable[RatingTriplet]) -> List[TrainingInstance]:
    """Create A TrainingInstance from Iterable of RatingTriplet"""

    instances: List[TrainingInstance] = []
    for i in itr:
        assert len(i) == 3
        instances.append(TrainingInstance(
            input_user=i[0],
            input_item=i[1],
            label=i[2]))
    return instances


def write_instance_to_example_files(
        instances: List[TrainingInstance],
        output_file: str) -> None:
    """Create TF example files from `TrainingInstance`s."""

    writer = tf.io.TFRecordWriter(output_file)
    for instance in instances:
        tf_example = convert_instance_to_example(instance)
        writer.write(tf_example.SerializeToString())
    writer.close()


def read_item_index_to_entity_id_file(
        path: str,
        sep: str = '\t') -> Tuple[IDVocab, IDVocab]:
    """Read

    xxx_vocab is id mapping between original ids on domain and sequencial id
    for used on KGCN.

    Args:
        path (str): Path to id correspondence table file original item index
            and knowlege graph original entity id
        sep (str, optional): Separator charactor. Defaults to '\t'.

    Returns:
        Tuple[IDVocab, IDVocab]: Return item vocabulary and entity vocabulary.
            item vocabulary
                key: original item index on domain.
                value: sequencial id used in KGCN.
                       Same as value of entity vocabulary.
            entity vocabulary
                key: original entity index on knowlege graph.
                value: sequencial id used in KGCN.
                       Same as value of item vocabulary.
    """

    # key: original id, value: sequencial id
    item_vocab: IDVocab = dict()
    entity_vocab: IDVocab = dict()

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            # item is original item id on target domain
            # entity is original entity id on knowlege graph
            item, entity = line.strip().split(sep)
            item_vocab[item] = len(item_vocab)
            entity_vocab[entity] = len(entity_vocab)

    return item_vocab, entity_vocab


def read_rating_file(
        path: str,
        item_vocab: IDVocab,
        sep: str = ',',
        skip_header: bool = True,
        threshold: float = 4.0) -> Tuple[IDVocab, List[RatingTriplet]]:
    """Read file interactions between user and item

    Args:
        path (str): Path to user item interactions file. Generally csv format.
        item_vocab (IDVocab): Dictionary of convert from original item index
            to sequencial entity id used in KGCN.
        sep (str, optional): Separator charactor. Defaults to ','.
        skip_header (bool, optional): Skip csv header. Defaults to True.
        threshold (float, optional): Threshold for regarding positive reaction.
            Defaults to 4.0.

    Returns:
        Tuple[IDVocab, List[RatingTriplet]]:
            Return user vocabulary and transformed interaction data.
    """

    assert len(item_vocab) > 0

    user_pos_rating: Dict[str, Set[int]] = defaultdict(set)
    user_neg_rating: Dict[str, Set[int]] = defaultdict(set)

    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            # skip header
            if skip_header and idx == 0:
                continue

            # assume file in columns starts order that user, item, rating
            user, item, rating = line.strip().split(sep)[:3]
            # only consider items that has corresponding entities
            if item not in item_vocab:
                continue

            if float(rating) >= threshold:
                user_pos_rating[user].add(item_vocab[item])
            else:
                user_neg_rating[user].add(item_vocab[item])

    all_item_id_set = set(item_vocab.values())
    rating_data: List[RatingTriplet] = []
    user_vocab: IDVocab = dict()

    for user, pos_item_id_set in user_pos_rating.items():
        user_vocab[user] = len(user_vocab)
        user_id = user_vocab[user]

        for item_id in pos_item_id_set:
            rating_data.append((user_id, item_id, 1))

        unwatched_set = all_item_id_set - pos_item_id_set
        if user in user_neg_rating:
            unwatched_set -= user_neg_rating[user]

        for item_id in np.random.choice(
                list(unwatched_set),
                size=len(pos_item_id_set),
                replace=False
                if len(list(unwatched_set)) >= len(pos_item_id_set) else True):
            rating_data.append((user_id, item_id, 0))

    return user_vocab, rating_data


def read_kg_file(
        path: str,
        entity_vocab: IDVocab,
        neighbor_sample_size: int,
        sep: str = '\t') -> Tuple[IDVocab, np.ndarray, np.ndarray]:
    """Read knowlege graph file

    Args:
        path (str): Path to knowlege graph.
        entity_vocab (IDVocab): Dictionary of convert from original entity id
            in knowlege graph to sequencial entity id used in KGCN.
        neighbor_sample_size (int): Adjusting number of sample neighbors
            for convolution.
        sep (str, optional): Separator charactor. Defaults to '\t'.

    Returns:
        Tuple[IDVocab, np.ndarray, np.ndarray]: Return relation vocabulary,
            adjusted entity array and adjusted relation array.
    """

    relation_vocab: IDVocab = dict()
    # kg is {'sequencial head entity id',
    #   [(sequencial tail entitiy id, sequencial relation id), ...]}
    kg: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            head, relation, tail = line.strip().split(sep)

            # add entity and relation id only appeared in knowledge graph
            # (not contains in rating file) to xxx_vocab
            if head not in entity_vocab:
                entity_vocab[head] = len(entity_vocab)
            if tail not in entity_vocab:
                entity_vocab[tail] = len(entity_vocab)
            if relation not in relation_vocab:
                relation_vocab[relation] = len(relation_vocab)

            # undirected graph
            kg[entity_vocab[head]].append(
                (entity_vocab[tail], relation_vocab[relation]))
            kg[entity_vocab[tail]].append(
                (entity_vocab[head], relation_vocab[relation]))

    n_entity = len(entity_vocab)
    # each line of adj_entity stores
    # the sampled neighbor entities for a given entity
    # each line of adj_relation stores
    # the corresponding sampled neighbor relations
    adj_entity = np.zeros(
        shape=(n_entity, neighbor_sample_size), dtype=np.int64)
    adj_relation = np.zeros(
        shape=(n_entity, neighbor_sample_size), dtype=np.int64)

    for entitiy_id in range(n_entity):
        all_neighbors = kg[entitiy_id]
        n_neighbor = len(all_neighbors)

        sample_indices = np.random.choice(
            n_neighbor,
            neighbor_sample_size,
            replace=False if n_neighbor >= neighbor_sample_size else True)

        adj_entity[entitiy_id] = np.array(
            [all_neighbors[i][0] for i in sample_indices])
        adj_relation[entitiy_id] = np.array(
            [all_neighbors[i][1] for i in sample_indices])

    return relation_vocab, adj_entity, adj_relation
