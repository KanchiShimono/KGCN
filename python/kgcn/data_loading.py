from functools import partial
from typing import Dict, Tuple

import tensorflow as tf
import numpy as np


def decode_record(
        record: tf.Tensor,
        name_to_features: Dict[str,
            tf.io.FixedLenFeature]) -> Dict[str, tf.Tensor]:
    """Decodes a record to a TensorFlow example"""

    return tf.io.parse_single_example(record, name_to_features)


def create_dataset(
        input_path: str,
        batch_size: int,
        is_training: bool) -> tf.data.Dataset:
    """Creates input dataset from TFRecords files"""

    name_to_features = {
        'input_user': tf.io.FixedLenFeature([], tf.int64),
        'input_item': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    if not tf.io.gfile.exists(input_path):
        raise ValueError(f'specified input file not exists {input_path}')

    _decode_fn = partial(decode_record, name_to_features=name_to_features)

    def _select_data(
            record: Dict[str,
                tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        x = {
            'input_user': record['input_user'],
            'input_item': record['input_item']
        }
        y = record['label']
        return (x, y)

    return (
        tf.data.TFRecordDataset(input_path)
        .map(_decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(_select_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(batch_size, drop_remainder=is_training)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )


def load_tf_dataset(
        input_path: str,
        batch_size: int,
        is_training: bool = False) -> tf.data.Dataset:

    return create_dataset(
            input_path=input_path,
            batch_size=batch_size,
            is_training=is_training)


def load_np_dataset(
        input_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    data = np.load(input_path)
    return data[:, :1], data[:, 1:2], data[:, 2:]
