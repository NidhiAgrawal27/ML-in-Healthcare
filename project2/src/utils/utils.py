import datetime
import os
import pathlib
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from collections import Counter


def set_seed(seed):
    """Set random seed for reproducibility.

    Args:
        seed (int): random seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)


def get_logs(log_dir, model_name, seed, torch_model=False):
    """Create log directory and return paths to log files.

    Args:
        log_dir (str): Path to log directory.
        model_name (str): Name of model.
        seed (int): Random seed.

    Returns:
        (str, str, str, str, str): Path to checkpoint, path to history,
            path to predictions, path to unnormalized predictions,
            path to metrics.
    """
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = os.path.join(log_dir, model_name, str(seed), time)
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

    checkpoint_filename = os.path.join(
        dir, "model.h5" if not torch_model else "model.pt"
    )
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


def log_training_time(elapsed_time, metrics_filename):
    """Log training time to metrics_filename.

    Args:
        elapsed_time (float): Elapsed time in seconds.
        metrics_filename (str): Path to file to write metrics to.
    """
    time_str = f"Training time: {elapsed_time:.4f} seconds"
    print(time_str)

    with open(metrics_filename, "a+", encoding="utf-8") as f:
        f.write(time_str + "\n")


def get_metrics(path):
    """Return metrics from specified metrics file.

    Args:
        path (str): Path to metrics file.

    Returns:
        dict: Dictionary of metrics.
    """
    metrics = {}

    with open(os.path.join(path, "metrics.txt"), "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                key, value = line.split(":")
                if "seconds" in value:
                    value = value.replace("seconds", "")
                metrics[key.strip()] = float(value.strip())

    return metrics


def get_infos(log_dir, exclusion_list=[]):
    """Return execution info from specified log directory.

    Args:
        log_dir (str): Path to logs.
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
        if all(excl not in model for excl in exclusion_list):
            infos.setdefault(model, []).append(info)

    return infos


def get_class_weights(y):
    num_samples = Counter(y)
    max_sample = max(num_samples.values())
    return {cls: max_sample / n for cls, n in num_samples.items()}
