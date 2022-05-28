import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def load_mitbih(path="../data/"):
    """Load MIT-BIH dataset.

    Args:
        path (str): Path to dataset directory.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray): X, Y, X_test, Y_test.
    """
    train_path = os.path.join(path, "mitbih_train.csv")
    test_path = os.path.join(path, "mitbih_test.csv")

    df_train = pd.read_csv(train_path, header=None)
    df_train = df_train.sample(frac=1)
    df_test = pd.read_csv(test_path, header=None)

    Y = np.array(df_train[187].values).astype(np.int8)
    X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

    Y_test = np.array(df_test[187].values).astype(np.int8)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]
    return X, Y, X_test, Y_test


def load_ptbdb(path="../data/"):
    """Load PTBDB dataset..

    Args:
        path (str): Path to dataset directory.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray): X, Y, X_test, Y_test.
    """
    train_path = os.path.join(path, "ptbdb_normal.csv")
    test_path = os.path.join(path, "ptbdb_abnormal.csv")

    df_1 = pd.read_csv(train_path, header=None)
    df_2 = pd.read_csv(test_path, header=None)
    df = pd.concat([df_1, df_2])

    df_train, df_test = train_test_split(df,
                                         test_size=0.2,
                                         random_state=1337,
                                         stratify=df[187])

    Y = np.array(df_train[187].values).astype(np.int8)
    X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

    Y_test = np.array(df_test[187].values).astype(np.int8)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

    return X, Y, X_test, Y_test
