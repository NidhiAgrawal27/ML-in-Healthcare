import pickle
import argparse
import time
import numpy as np
import pandas as pd

from models import decision_tree
from utilities import utils, evaluation, data

CONFIG = {"log_dir": "../logs/", "model_name": "decision_tree"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random seed", required=True)
    args = parser.parse_args()

    utils.set_seed(args.seed)

    (checkpoint_filename, _, pred_filename, pred_proba_filename,
     metrics_filename) = (utils.get_logs(log_dir=CONFIG["log_dir"],
                                         model_name=CONFIG["model_name"],
                                         seed=args.seed,
                                         model_ext=".pkl"))

    # Load data
    train_data, train_labels, val_data, val_labels, test_data, test_labels = (
        data.get_radiomics_dataset())

    # Retrieve model.
    model = decision_tree.get_model()

    # Train final model on both trainset and devset.
    final_train_data = pd.concat([train_data, val_data], axis=0)
    final_train_labels = np.concatenate([train_labels, val_labels], axis=0)
    start_time = time.time()
    model.fit(final_train_data, final_train_labels)
    elapsed_time = time.time() - start_time

    # Predict and evaluate on testset.
    pred_proba = model.predict_proba(test_data)
    pred = model.predict(test_data)
    np.save(pred_proba_filename, pred_proba)
    np.save(pred_filename, pred)

    pickle.dump(model, open(checkpoint_filename, "wb"))
    utils.log_training_time(elapsed_time, metrics_filename)
    evaluation.evaluate(test_labels, pred, pred_proba, metrics_filename)


if __name__ == "__main__":
    main()
