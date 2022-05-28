import os
import joblib
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

from utils.utils import get_infos

PARAMS = {
    "exclusion_list": ["ensemble", "transfer"],
    "penalty": "l2",
    "C": 1,
    "tol": 0.0001,
    "max_iter": 200,
    "solver": "lbfgs",
}


def get_model(params=PARAMS, log_dir=None, dataset=None):
    return StackedEnsemble(params, log_dir, dataset)


class StackedEnsemble:
    def __init__(self, params, log_dir, dataset):
        if log_dir is None:
            raise ValueError("log_dir must be specified.")
        if dataset is None:
            raise ValueError("dataset must be specified.")
        self.params = params
        self.log_dir = log_dir
        self.dataset = dataset
        self.exclusion_list = params.get("exclusion_list", [])
        self.tf_models = None
        self.skl_models = None
        self.clf = LogisticRegression(
            penalty=PARAMS["penalty"],
            C=PARAMS["C"],
            tol=PARAMS["tol"],
            max_iter=PARAMS["max_iter"],
            solver=PARAMS["solver"],
            n_jobs=-1,
        )

    def _load_models(self):
        tf_models = []
        sklearn_models = []
        infos = get_infos(self.log_dir, self.dataset, self.exclusion_list)

        for model, info_list in infos.items():
            for info in info_list:
                model_path = os.path.join(info["path"], "model.h5")
                if "extra_trees" in model:
                    sklearn_models.append(joblib.load(model_path))
                else:
                    tf_models.append(tf.keras.models.load_model(model_path))
        self.tf_models = tf_models
        self.skl_models = sklearn_models

    def _get_stacked_pred_probas(self, X):
        if self.tf_models is None or self.skl_models is None:
            raise ValueError("Models must be loaded before prediction.")

        stacked_pred_probas = None
        for tf_model in self.tf_models:
            pred_proba = tf_model.predict(X)
            if stacked_pred_probas is None:
                stacked_pred_probas = pred_proba
            else:
                stacked_pred_probas = np.dstack((stacked_pred_probas, pred_proba))

        # Reshape X to be 2D for sklearn models.
        X = X.reshape(X.shape[0], X.shape[1])

        for skl_model in self.skl_models:
            if self.dataset == "ptbdb":
                pred_proba = skl_model.predict_proba(X)[:, 1].reshape(-1, 1)
            else:
                pred_proba = skl_model.predict_proba(X)
            if stacked_pred_probas is None:
                stacked_pred_probas = pred_proba
            else:
                stacked_pred_probas = np.dstack((stacked_pred_probas, pred_proba))

        # Replace NaN and infinity to ensure classifier can train.
        stacked_pred_probas = np.nan_to_num(stacked_pred_probas)

        stacked_pred_probas = stacked_pred_probas.reshape(
            (
                stacked_pred_probas.shape[0],
                stacked_pred_probas.shape[1] * stacked_pred_probas.shape[2],
            )
        )
        return stacked_pred_probas

    def fit(self, X, Y):
        self._load_models()
        stacked_pred_probas = self._get_stacked_pred_probas(X)
        self.clf.fit(stacked_pred_probas, Y)

    def predict_proba(self, X):
        stacked_pred_probas = self._get_stacked_pred_probas(X)
        predict_proba = self.clf.predict_proba(stacked_pred_probas)
        # Self.clf is from sklearn and outputs all class probabilities.
        if self.dataset == "ptbdb":
            predict_proba = predict_proba[:, 1]
        return predict_proba
