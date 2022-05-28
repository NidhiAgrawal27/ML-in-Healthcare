import pickle
import argparse
import time
import numpy as np
from sklearn.preprocessing import StandardScaler

from models import logistic_regression
from utilities import utils, evaluation, data

CONFIG = {"log_dir": "../logs/", "model_name": "logistic_regression"}


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
    X_train, y_train = train_data.to_numpy(), train_labels
    X_val, y_val = val_data.to_numpy(), val_labels
    X_test, y_test = test_data.to_numpy(), test_labels

    # Normalize data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Retrieve model.
    model = logistic_regression.get_model()

    # Train final model on both trainset and devset.
    final_train_data = np.concatenate([X_train, X_val], axis=0)
    final_train_labels = np.concatenate([y_train, y_val], axis=0)
    start_time = time.time()
    model.fit(final_train_data, final_train_labels)
    elapsed_time = time.time() - start_time

    # Predict and evaluate on testset.
    pred_proba = model.predict_proba(X_test)
    pred = model.predict(X_test)
    np.save(pred_proba_filename, pred_proba)
    np.save(pred_filename, pred)

    pickle.dump(model, open(checkpoint_filename, "wb"))
    utils.log_training_time(elapsed_time, metrics_filename)
    evaluation.evaluate(y_test, pred, pred_proba, metrics_filename)

    # Explanation.
    named_coeffs = list(zip(list(train_data.columns), model.coef_[0]))
    named_coeffs = sorted(named_coeffs, key=lambda x: abs(x[1]), reverse=True)


if __name__ == "__main__":
    main()
