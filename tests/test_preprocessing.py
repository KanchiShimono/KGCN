from pathlib import Path

import numpy as np

from kgcn.preprocessing import (
    read_item_index_to_entity_id_file, read_kg_file, read_rating_file
)

EXPECTED_ITEM_VOCAB = {
    '10000': 0,
    '30000': 1,
    '50000': 2,
    '70000': 3,
    '90000': 4,
}

EXPECTED_ENTITY_VOCAB = {
    '1': 0,
    '3': 1,
    '5': 2,
    '7': 3,
    '9': 4,
}

# e is not appered because rate low rating only
EXPECTED_USER_VOCAB = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'f': 4,
}

EXPECTED_RELATION_VOCAB = {
    'item.store': 0,
    'item.genre': 1,
    'item.production_company': 2,
    'item.country': 3,
    'item.language': 4,
}

EXPECTED_RATINGS = np.array([
    # a
    (0, 0, 1),
    (0, 1, 1),
    (0, 4, 1),
    (0, 2, 0),
    (0, 2, 0),
    (0, 2, 0),
    # b
    (1, 0, 1),
    (1, 1, 1),
    (1, 4, 0),
    (1, 2, 0),
    # c
    (2, 0, 1),
    (2, 1, 1),
    (2, 2, 0),
    (2, 4, 0),
    # d
    (3, 0, 1),
    (3, 1, 1),
    (3, 4, 0),
    (3, 2, 0),
    # e is not appeared
    # f
    (4, 0, 1),
    (4, 1, 1),
    (4, 2, 0),
    (4, 4, 0),
])

EXPECTED_ADJ_ENTITY = np.array([
    # neighbor entities of item
    # item 1
    [14, 5, 8, 16, 17],
    # item 3
    [6, 14, 9, 17, 16],
    # item 5
    [10, 16, 17, 6, 14],
    # item 7
    [16, 11, 14, 17, 7],
    # item 9
    [17, 13, 15, 16, 12],
    # neighbors of entities only be appeared in knowlege graph
    # store 100
    [0, 4, 0, 4, 0],
    # store 101
    [2, 2, 1, 2, 2],
    # store 102
    [3, 3, 3, 3, 3],
    # genre 205
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3],
    [4, 4, 4, 4, 4],
    [4, 4, 4, 4, 4],
    # production_company 300
    [0, 2, 3, 0, 1],
    [4, 4, 4, 4, 4],
    # country 400
    [0, 4, 2, 1, 3],
    # language 500
    [4, 2, 3, 1, 0]
])

EXPECTED_ADJ_RELATION = np.array([
    # relation item entity and neighbors of entity
    [2, 0, 1, 3, 4],
    [0, 2, 1, 4, 3],
    [1, 3, 4, 0, 2],
    [3, 1, 2, 4, 0],
    [4, 1, 2, 3, 1],
    # relations of entities only be appeared in knowlege graph
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3],
    [4, 4, 4, 4, 4]
])


def test_read_item_index_to_entity_id_file(shared_datadir: Path) -> None:
    item_vocab, entity_vocab = read_item_index_to_entity_id_file(
        shared_datadir / 'test_item_index2entity_id.txt', sep='\t')
    assert item_vocab == EXPECTED_ITEM_VOCAB
    assert entity_vocab == EXPECTED_ENTITY_VOCAB


def test_read_rating_file(shared_datadir: Path) -> None:
    np.random.seed(0)
    user_vocab, rating_data = read_rating_file(
        shared_datadir / 'test_ratings.csv',
        EXPECTED_ITEM_VOCAB,
        sep=',',
        skip_header=True,
        threshold=4.0)
    assert user_vocab == EXPECTED_USER_VOCAB
    assert np.array_equal(np.array(rating_data), EXPECTED_RATINGS)


def test_read_kg_file(shared_datadir: Path) -> None:
    np.random.seed(0)
    relation_vocab, adj_entity, adj_relation = read_kg_file(
        shared_datadir / 'test_kg.txt', EXPECTED_ENTITY_VOCAB, 5, sep='\t')
    assert relation_vocab == EXPECTED_RELATION_VOCAB
    assert np.array_equal(adj_entity, EXPECTED_ADJ_ENTITY)
    assert np.array_equal(adj_relation, EXPECTED_ADJ_RELATION)
