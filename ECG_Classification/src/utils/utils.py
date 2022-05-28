import datetime
import os
import pathlib
import random
import math
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras import losses, activations


def set_seed(seed):
    """Set random seed for reproducibility.

    Args:
        seed (int): random seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_logs(log_dir, model_name, seed):
    """Create log directory and return paths to log files.

    Args:
        log_dir (str): Path to log directory.
        model_name (str): Name of model.
        seed (int): Random seed.

    Returns:
        (str, str, str, str, str): Path to checkpoint, path to history,
            path to predictions, path to unnormalized predictions, path to metrics.
    """
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = os.path.join(log_dir, model_name, str(seed), time)
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

    checkpoint_filename = os.path.join(dir, "model.h5")
    history_filename = os.path.join(dir, "history.csv")
    pred_filename = os.path.join(dir, "pred.npy")
    pred_proba_filename = os.path.join(dir, "pred_proba.npy")
    metrics_filename = os.path.join(dir, "metrics.txt")

    return (
        checkpoint_filename,
        history_filename,
        pred_filename,
        pred_proba_filename,
        metrics_filename,
    )


def get_metrics(path):
    """Return metrics from specified metrics file.

    Args:
        path (str): Path to metrics file.

    Returns:
        dict: Dictionary of metrics.
    """
    metrics = {}

    with open(os.path.join(path, "metrics.txt"), "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.split(":")
                if "seconds" in value:
                    value = value.replace("seconds", "")
                metrics[key.strip()] = float(value.strip())

    return metrics


def get_infos(log_dir, dataset, exclusion_list=[]):
    """Return execution info from specified log directory.

    Args:
        log_dir (str): Path to logs.
        dataset (str): Dataset name.
        exclusion_list (list): List of model names to exclude.

    Returns:
        dict: Dictionary of models and their execution info.
    """
    # Ensure directory exists and ends with /
    log_dir = os.path.join(log_dir, "")

    infos = {}

    pred_paths = [f for f in os.walk(log_dir) if "pred.npy" in f[2]]

    for p in pred_paths:
        path = p[0]
        model_info = path.replace(log_dir, "").split("/")
        model, seed, timestamp = model_info[0], model_info[1], model_info[2]
        pred = np.load(os.path.join(path, "pred.npy"))
        pred_proba = np.load(os.path.join(path, "pred_proba.npy"))
        pred = pred.reshape(pred.shape[0], -1)
        pred_proba = pred_proba.reshape(pred_proba.shape[0], -1)
        if os.path.exists(os.path.join(path, "history.csv")):
            history = pd.read_csv(os.path.join(path, "history.csv"))
        else:
            history = None
        metrics = get_metrics(path)

        info = {
            "path": path,
            "seed": seed,
            "timestamp": timestamp,
            "history": history,
            "metrics": metrics,
            "pred": pred,
            "pred_proba": pred_proba,
        }
        if dataset in model and all([excl not in model for excl in exclusion_list]):
            infos.setdefault(model, []).append(info)

    return infos


def get_act_loss_logits(n_classes):
    if n_classes == 2:
        final_activation = activations.sigmoid
        loss = losses.binary_crossentropy
        n_logits = 1
    elif n_classes > 2:
        final_activation = activations.softmax
        loss = losses.sparse_categorical_crossentropy
        n_logits = n_classes
    else:
        raise ValueError("Need to predict at least 2 classes")

    return final_activation, loss, n_logits


def get_mean_std_losses(
    dataset, log_dir="../logs/", exclusion_list=["ensemble", "extra_trees"]
):
    tf_infos = get_infos(log_dir, dataset, exclusion_list=exclusion_list)
    mean_std_losses = []

    for model, info_list in tf_infos.items():
        min_epoch = math.inf
        train_losses = []
        val_losses = []
        epochs = []
        for info in info_list:
            epoch = info["history"]["epoch"]
            train_loss = info["history"]["loss"]
            val_loss = info["history"]["val_loss"]
            epochs.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            min_epoch = min(min_epoch, len(epoch))

        # Ensure all runs have the same number of epochs by cropping
        # to the shortest epoch number.
        train_losses = np.array([train_loss[:min_epoch] for train_loss in train_losses])
        val_losses = np.array([val_loss[:min_epoch] for val_loss in val_losses])
        epochs = np.array([epoch[:min_epoch] for epoch in epochs])

        # Compute mean and std.
        train_losses_mean = np.mean(train_losses, axis=0)
        train_losses_std = np.std(train_losses, axis=0)
        val_losses_mean = np.mean(val_losses, axis=0)
        val_losses_std = np.std(val_losses, axis=0)
        epochs = epochs[0]

        mean_std_loss = {
            "model": model,
            "train_losses_mean": train_losses_mean,
            "train_losses_std": train_losses_std,
            "val_losses_mean": val_losses_mean,
            "val_losses_std": val_losses_std,
            "epochs": epochs,
        }
        mean_std_losses.append(mean_std_loss)

    return mean_std_losses
