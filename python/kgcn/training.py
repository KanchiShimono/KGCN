import os

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam

from kgcn.data_loading import load_np_dataset, load_tf_dataset
from kgcn.modeling.models import KGCN
from kgcn.util import deserialize_pickle


def train(
        aggregator: str,
        n_epochs: int,
        dim: int,
        n_iter: int,
        batch_size: int,
        l2_weight: float,
        lr: float,
        output_data_dir: str,
        use_tfrecord: bool = False) -> KGCN:

    user_vocab = deserialize_pickle(os.path.join(output_data_dir, 'user_vocab.pickle'))
    relation_vocab = deserialize_pickle(os.path.join(output_data_dir, 'relation_vocab.pickle'))

    adj_entitiy = np.load(os.path.join(output_data_dir, 'adj_entity.npy'))
    adj_relation = np.load(os.path.join(output_data_dir, 'adj_relation.npy'))

    if use_tfrecord:
        train = load_tf_dataset(
            input_path=os.path.join(output_data_dir, 'train.tfrecords'),
            batch_size=batch_size,
            is_training=True)
        valid = load_tf_dataset(
            input_path=os.path.join(output_data_dir, 'valid.tfrecords'),
            batch_size=batch_size,
            is_training=False)
        test = load_tf_dataset(
            input_path=os.path.join(output_data_dir, 'test.tfrecords'),
            batch_size=batch_size,
            is_training=False)
    else:
        train = load_np_dataset(os.path.join(output_data_dir, 'train.npy'))
        valid = load_np_dataset(os.path.join(output_data_dir, 'valid.npy'))
        test = load_np_dataset(os.path.join(output_data_dir, 'test.npy'))

    model = KGCN(
        dim=dim,
        n_user=len(user_vocab),
        n_entity=adj_entitiy.shape[0],
        n_relation=len(relation_vocab),
        adj_entity=adj_entitiy,
        adj_relation=adj_relation,
        n_iter=n_iter,
        aggregator_type=aggregator,
        regularizer_weight=l2_weight)

    adam = Adam(learning_rate=lr)
    escb = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    tbcb = TensorBoard(
        log_dir='logs',
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch',
        embeddings_freq=1)
    model.compile(optimizer=adam, loss='binary_crossentropy')

    if use_tfrecord:
        model.fit(
            train,
            batch_size=batch_size,
            epochs=n_epochs,
            callbacks=[escb, tbcb],
            validation_data=valid,
            verbose=1)
    else:
        model.fit(
            [train[0], train[1]],
            train[2],
            batch_size=batch_size,
            epochs=n_epochs,
            callbacks=[escb, tbcb],
            validation_data=([valid[0], valid[1]], valid[2]),
            verbose=1)

    return model
