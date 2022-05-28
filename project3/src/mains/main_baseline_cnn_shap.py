
from fileinput import filename
import time
import pathlib
import pytorch_lightning as pl
from torchvision import transforms
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import numpy as np
import os
import random
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.baseline_cnn import BaselineClf
from models.train_cnn import CNNModel
from utilities import utils
from utilities.data import get_img_dataset
from visualization import shap_func

import warnings
# training runs faster without additional processes on m1 mac
warnings.filterwarnings("ignore", ".*does not have many workers.*")


CONFIG = {"log_dir": "../logs/", "model_name": "baseline_cnn_shap", "es_patience": 10,
          "max_epochs": 50, "batch_size": 64, "train_hparams": {"lr": 0.001, "lr_patience": 5}, "figure_dir": "../logs/figures/shap_baseline_cnn/"}


def main():

    seed = 0  # TODO

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random seed", required=True)
    args = parser.parse_args()

    utils.set_seed(args.seed)

    checkpoint_filename, _, pred_filename, pred_proba_filename, metrics_filename = utils.get_logs(log_dir=CONFIG["log_dir"],
                                                                                                  model_name=CONFIG["model_name"],
                                                                                                  seed=args.seed,
                                                                                                  model_ext=".ckpt")
    CONFIG["train_hparams"] = {**CONFIG["train_hparams"], "log_params": {"pred_filename": pred_filename,
                               "pred_proba_filename": pred_proba_filename, "metrics_filename": metrics_filename}}

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename=checkpoint_filename, monitor="val_loss")
    es_callback = pl.callbacks.EarlyStopping(
        "val_loss", patience=CONFIG["es_patience"])

    trainer = pl.Trainer(callbacks=[checkpoint_callback, es_callback],
                         max_epochs=CONFIG["max_epochs"], log_every_n_steps=1)

    # Setup data
    train_dataset, val_dataset, test_dataset = get_img_dataset(transform=[
        transforms.RandomHorizontalFlip(0.2),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomAutocontrast(0.2),
        transforms.RandomAdjustSharpness(0.2)
    ])
    train_dataloader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    # Setup model
    model = CNNModel(model=BaselineClf(), **CONFIG["train_hparams"])

    # Train
    start_time = time.time()
    trainer.fit(model, train_dataloader, val_dataloader)
    utils.log_training_time(time.time() - start_time, metrics_filename)

    # Test
    model = CNNModel.load_from_checkpoint(
        checkpoint_callback.best_model_path, model=BaselineClf(), **CONFIG["train_hparams"])
    print("Validation accruacy of best model: ",
          checkpoint_callback.best_model_score)
    trainer.test(model=model, dataloaders=[test_dataloader])

    # Load complete test data for SHAP values and visualization
    test_dataloader = DataLoader(
        test_dataset, batch_size=len(test_dataset), shuffle=True)
    test_features, test_labels = next(iter(test_dataloader))

    # Create figure directory if it does not exist.
    pathlib.Path(CONFIG["figure_dir"]).mkdir(parents=True, exist_ok=True)

    classes = {'0': 'No Tumor', '1': 'Tumor'}
    shap_func.shap_tensor(
        model, test_features, test_labels, classes, CONFIG["figure_dir"]+'baseline_cnn_')

if __name__ == "__main__":
    main()
