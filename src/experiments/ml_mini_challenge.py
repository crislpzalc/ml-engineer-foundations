"""
Mini-ML Challenge — Noisy and Small Data Scenario

This script simulates a realistic machine learning workflow with
imperfect data (small sample size, noisy features, reduced signal).
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

def load_data():
    """Load the breast cancer dataset and return (X, y, feature_names)."""
    ds = load_breast_cancer()
    return ds.data, ds.target, ds.feature_names

def make_small_dataset(X, y, keep_frac=0.25):
    """
    Subsample the dataset while preserving class proportions.
    This simulates a low-data regime commonly found in real-world problems.
    """
    X_small, y_small = train_test_split(X, y, train_size=keep_frac, random_state=RANDOM_SEED, stratify=y)

    return X_small, y_small

def add_noise_features(X, noise_std=0.8):
    """
    Add Gaussian noise to input features to simulate measurement noise.
    Noise is sampled from N(0, noise_std) and applied element-wise.
    """
    noise = rng.normal(loc=0.0, scale=noise_std, size=X.shape)
    X_noisy = X + noise
    return X_noisy

def drop_features(X, feature_names, drop_frac=0.4):
    """
    Randomly drop a fraction of features to simulate loss of useful signal.
    This mimics scenarios such as missing sensors or unavailable variables.
    """
    n_features = X.shape[1]
    n_drop = int(n_features * drop_frac)
    drop_indices = rng.choice(n_features, size=n_drop, replace=False)

    keep_mask = np.ones(n_features, dtype=bool)
    keep_mask[drop_indices] = False

    X_new = X[:, keep_mask]
    names_new = feature_names[keep_mask]
    return X_new, names_new

def build_bad_dataset(X, y, feature_names):
    """
    Construct a degraded dataset by combining:
    - subsampling (small data),
    - noisy features,
    - random feature removal.

    This simulates a realistic low-quality data scenario.
    """
    X_bad, y_bad = make_small_dataset(X, y, keep_frac=0.25)
    X_bad = add_noise_features(X_bad, noise_std=0.8)
    X_bad, names_bad = drop_features(X_bad, feature_names, drop_frac=0.4)
    return X_bad, y_bad, names_bad

