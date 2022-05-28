import numpy as np

from utils.utils import get_infos

PARAMS = {"exclusion_list": ["ensemble", "transfer"]}


def get_model(params=PARAMS, log_dir=None, dataset=None):
    return ModelAveraging(log_dir, dataset, PARAMS["exclusion_list"])


class ModelAveraging:
    def __init__(self, log_dir, dataset, exclusion_list=[]):
        if log_dir is None:
            raise ValueError("log_dir must be specified.")
        if dataset is None:
            raise ValueError("dataset must be specified.")
        self.log_dir = log_dir
        self.dataset = dataset
        self.exclusion_list = exclusion_list

    def predict_proba(self):
        infos = get_infos(self.log_dir, self.dataset, self.exclusion_list)
        pred_probas = []

        for _, info_list in infos.items():
            for info in info_list:
                pred_probas.append(info["pred_proba"])

        return np.mean(pred_probas, axis=0)
