import os
import pickle

import numpy as np
from absl import app, flags, logging
from sklearn.model_selection import train_test_split

from kgcn.preprocessing import (
    create_instances_from_iterable, read_item_index_to_entity_id_file,
    read_kg_file, read_rating_file, write_instance_to_example_files
)
from kgcn.util import get_json_formatter

# logger in absl-py cause log duplicates when use module logger like
# getLogger(__name__), so I use absl loggin here
logging._absl_handler.setFormatter(get_json_formatter())
logger = logging._absl_logger


flags.DEFINE_string(
    'item_id_to_entity_file', 'item_index2entity_id.txt',
    'The dictionary file from original item id to knowlege graph entity id')
flags.DEFINE_string('kg_file', 'kg.txt', 'The knowlege graph file')
flags.DEFINE_string('rating_file', 'ratings.csv', 'User item interaction file')
flags.DEFINE_integer(
    'neighbor_sample_size', 4, 'The number of neighbors to be sampled')
flags.DEFINE_integer('batch_size', 65536, 'Batch size')
flags.DEFINE_string(
    'output_data_dir', 'output_data',
    'Path to root directory of to be wrote preprocessed data')
flags.DEFINE_bool(
    'use_tfrecord', False,
    'If ture, preprocessed data be wrote as tfrecord. '
    'If false, be wrote as numpy binary format.')

FLAGS = flags.FLAGS


def process_data(
        item_id_to_entity_path: str,
        kg_path: str,
        rating_path: str,
        neighbor_sample_size: int,
        output_dir: str,
        user_tfrecord: bool = False) -> None:

    logger.info(
        f'reading item index to entity id file {item_id_to_entity_path}')
    item_vocab, entity_vocab = read_item_index_to_entity_id_file(
        item_id_to_entity_path)

    logger.info(f'reading ratings file {rating_path}')
    user_vocab, rating_data = read_rating_file(rating_path, item_vocab)

    logger.info(f'reading knowlege graph file {kg_path}')
    relation_vocab, adj_entity, adj_relation = read_kg_file(
        kg_path, entity_vocab, neighbor_sample_size)

    logger.info(
        f'number of usres: {len(user_vocab)}, '
        f'items: {len(item_vocab)}, '
        f'entities: {len(entity_vocab)}, '
        f'relations: {len(relation_vocab)}, '
        f'interactions: {len(rating_data)}')

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
        file_name = os.path.join(output_dir, f'{path}.pickle')
        logger.info(f'writing to {file_name}')
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)

    for data, path in zip(
            (adj_entity, adj_relation), ('adj_entity', 'adj_relation')):
        file_name = os.path.join(output_dir, f'{path}.npy')
        logger.info(f'writing to {file_name}')
        np.save(file_name, data)

    if user_tfrecord:
        for data, path in zip(
                (train_data, valid_data, test_data),
                ('train', 'valid', 'test')):
            instances = create_instances_from_iterable(data)
            file_name = os.path.join(output_dir, f'{path}.tfrecords')
            logger.info(f'writing to {file_name}')
            write_instance_to_example_files(instances, file_name)
    else:
        for data, path in zip(
                (train_data, valid_data, test_data),
                ('train', 'valid', 'test')):
            file_name = os.path.join(output_dir, f'{path}.npy')
            logger.info(f'writing to {file_name}')
            np.save(file_name, data)


def main(_) -> None:
    process_data(
        item_id_to_entity_path=FLAGS.item_id_to_entity_file,
        kg_path=FLAGS.kg_file,
        rating_path=FLAGS.rating_file,
        neighbor_sample_size=FLAGS.neighbor_sample_size,
        output_dir=FLAGS.output_data_dir,
        user_tfrecord=FLAGS.use_tfrecord)


if __name__ == "__main__":
    app.run(main)
