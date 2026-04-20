"""Data utilities for the dILP wildfire experiment."""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from src.wildfire.constants import (
    VISUAL_VARIABLES,
    NON_VISUAL_VARIABLES,
    GROUND_TRUTH_FORMULA,
)
from src.wildfire.fuzzy_compile import FormulaCompiler


def _generate_non_visual_features(n_samples, n_non_visual, seed=42):
    rng = np.random.RandomState(seed)
    return (rng.rand(n_samples, n_non_visual) > 0.5).astype(np.float32)


def _compute_ground_truth_labels(visual_features, non_visual_features):
    # visual_features: [N,2], non_visual_features: [N,7]
    env = {}
    for i, name in enumerate(VISUAL_VARIABLES):
        env[name] = tf.convert_to_tensor(visual_features[:, i], dtype=tf.float32)
    for i, name in enumerate(NON_VISUAL_VARIABLES):
        env[name] = tf.convert_to_tensor(non_visual_features[:, i], dtype=tf.float32)
    compiler = FormulaCompiler(GROUND_TRUTH_FORMULA)
    y = compiler.evaluate(env)
    y = tf.where(y >= 0.5, 1.0, 0.0)
    return y.numpy().astype(np.float32)


def _extract_train_val_indices(df):
    if 'split' not in df.columns:
        return None, None
    split = df['split'].astype(str).str.lower()
    train_mask = split == 'train'
    val_mask = split.isin(['val', 'validation', 'test'])
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        return None, None
    train_idx = np.where(train_mask.to_numpy())[0].astype(np.int32)
    val_idx = np.where(val_mask.to_numpy())[0].astype(np.int32)
    return train_idx, val_idx


def load_wildfire_table(dataset_dir, non_visual_seed=42):
    csv_path = os.path.join(dataset_dir, 'labels.csv')
    img_dir = os.path.join(dataset_dir, 'images')
    df = pd.read_csv(csv_path)

    visual = df[VISUAL_VARIABLES].to_numpy(dtype=np.float32)
    if all(col in df.columns for col in NON_VISUAL_VARIABLES):
        non_visual = df[NON_VISUAL_VARIABLES].to_numpy(dtype=np.float32)
    else:
        print('No non-visual columns found in CSV, generating random features...')
        non_visual = _generate_non_visual_features(len(df), len(NON_VISUAL_VARIABLES), seed=non_visual_seed)

    if 'wildfire_risk' in df.columns:
        labels = df['wildfire_risk'].to_numpy(dtype=np.float32).reshape(-1)
    else:
        print('No wildfire_risk column found in CSV, computing ground truth labels from features...')
        labels = _compute_ground_truth_labels(visual, non_visual)

    train_indices, val_indices = _extract_train_val_indices(df)
    filepaths = [os.path.join(img_dir, fn) for fn in df['filename'].tolist()]

    return {
        'dataset_dir': dataset_dir,
        'csv_path': csv_path,
        'img_dir': img_dir,
        'filepaths': np.array(filepaths),
        'visual_features': visual,
        'non_visual_features': non_visual,
        'labels': labels,
        'train_indices': train_indices,
        'val_indices': val_indices,
    }


def train_test_split_indices(n_samples, train_ratio=0.75, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    train_size = int(train_ratio * n_samples)
    train_idx = np.sort(idx[:train_size])
    test_idx = np.sort(idx[train_size:])
    return train_idx, test_idx


def get_train_val_indices(table, train_ratio=0.75, seed=42):
    train_idx = table.get('train_indices')
    val_idx = table.get('val_indices')
    if train_idx is not None and val_idx is not None:
        return train_idx, val_idx
    print('No train/val split found in CSV, performing random split...')
    return train_test_split_indices(len(table['labels']), train_ratio=train_ratio, seed=seed)


def build_tf_dataset(filepaths, non_visual, labels, example_indices, batch_size=256, shuffle=False, drop_remainder=False):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, non_visual, labels, example_indices))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(example_indices), reshuffle_each_iteration=True)

    def _load(path, nv, y, idx):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_png(img_bytes, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        # Keep size consistent with original notebook assumptions (256x256).
        img = tf.image.resize(img, [256, 256])
        # Match torchvision.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).
        img = (img - 0.5) / 0.5
        return img, tf.cast(nv, tf.float32), tf.cast(y, tf.float32), tf.cast(idx, tf.int32)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
