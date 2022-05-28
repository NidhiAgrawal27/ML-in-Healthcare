# parts adadpted from https://github.com/alain-ryser/interpretability-project/blob/main/model.py

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import numpy as np
import os
import random

from utilities import evaluation


class CNNModel(pl.LightningModule):
    def __init__(self, model, lr, lr_patience, log_params):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lr_patience = lr_patience
        self.log_params = log_params

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def training_step(self, batch, batch_idx):
        images, labels = batch
        # Get predictions
        out = self(images)
        # Get loss
        loss = F.cross_entropy(out, labels)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        # Get predictions
        out = self(images)
        # Get loss
        loss = F.cross_entropy(out, labels)
        return {'preds': out[:, 1].numpy().tolist(), "labels": labels.view(-1).numpy().tolist(), "loss": loss}

    def test_step(self, batch, batch_idx):
        images, labels = batch
        # Get predictions
        out = self(images)
        return {'preds': out[:, 1].numpy().tolist(), "labels": labels.view(-1).numpy().tolist()}

    @staticmethod
    def _aggregate_preds(outputs):
        pred_proba, labels = [], []
        for x in outputs:
            pred_proba += x['preds']
            labels += x['labels']
        pred_proba, labels = np.array(pred_proba), np.array(labels)
        pred = pred_proba > 0.5
        return pred_proba, pred, labels

    def test_epoch_end(self, outputs):
        pred_proba, pred, labels = self._aggregate_preds(outputs)

        # save log_files
        np.save(self.log_params["pred_proba_filename"], pred_proba)
        np.save(self.log_params["pred_filename"], pred)
        evaluation.evaluate(labels, pred, pred_proba,
                            self.log_params["metrics_filename"])

        # log acc
        test_acc = accuracy_score(labels, pred)
        self.log("test_acc", test_acc)

    def validation_epoch_end(self, outputs):
        pred_proba, pred, labels = self._aggregate_preds(outputs)
        val_loss = np.mean([x['loss'] for x in outputs])
        val_acc = accuracy_score(labels, pred)
        self.log("val_loss", val_loss)
        self.log("val_acc", val_acc)
        print("Epoch: {} done. Val loss: {:.4f}, Val acc: {:.4f}".format(
            self.current_epoch, val_loss, val_acc))
