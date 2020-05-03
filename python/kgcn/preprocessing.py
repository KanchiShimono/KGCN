import os
import pickle
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from kgcn.types import IDVocab, RatingTriplet


def read_item_index_to_entity_id_file(
        path: str,
        sep: str = '\t') -> Tuple[IDVocab, IDVocab]:
    # xxx_vocab is id mapping between original ids on domain and sequencial id
    # for using on kgcn.
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


def process_data(
        item_id_to_entity_path: str,
        kg_path: str,
        rating_path: str,
        neighbor_sample_size: int,
        output_dir: str) -> None:
    item_vocab, entity_vocab = read_item_index_to_entity_id_file(
        item_id_to_entity_path)

    user_vocab, rating_data = read_rating_file(rating_path, item_vocab)

    relation_vocab, adj_entity, adj_relation = read_kg_file(
        kg_path, entity_vocab, neighbor_sample_size)

    # TODO: propotion of train, dev and test
    # should to be passed as function argument
    # train : dev : test = 6 : 2 : 2
    train_data, valid_data = train_test_split(rating_data, test_size=0.4)
    valid_data, test_data = train_test_split(valid_data, test_size=0.5)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for data, path in zip(
            (item_vocab, entity_vocab, user_vocab, relation_vocab),
            ('item_vocab', 'entity_vocab', 'user_vocab', 'relation_vocab')):
        with open(os.path.join(output_dir, f'{path}.pickle'), 'wb') as f:
            pickle.dump(data, f)

    for data, path in zip(
            (adj_entity, adj_relation, train_data, valid_data, test_data),
            ('adj_entity', 'adj_relation',
             'train_data', 'valid_data', 'test_data')):
        np.save(os.path.join(output_dir, f'{path}.npy'), data)
