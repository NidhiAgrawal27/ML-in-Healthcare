import time
import joblib
import numpy as np

from utils import data_preparation, evaluation, utils


def train_and_evaluate(model, dataset, params, seed):
    """Train and evaluate scikit-learn model with logging.

    Args:
        model (obj): scikit-learn model to train.
        dataset (str): Either "mitbih" or "ptbdb".
        params (dict): Training parameters containing model_name, log_dir 
         (NOT model params).
        seed (int): Random seed.

    Returns:
        (str): Path to checkpoint file with highest val score
    """
    # Check dataset validity.
    if dataset == "mitbih":
        dataset_loader = data_preparation.load_mitbih
        run_evaluation = evaluation.evaluate_mitbih
        get_preds_proba = lambda X: model.predict_proba(X)
    elif dataset == "ptbdb":
        dataset_loader = data_preparation.load_ptbdb
        run_evaluation = evaluation.evaluate_ptbdb
        get_preds_proba = lambda X: model.predict_proba(X)[:, 1]
    else:
        raise ValueError("Dataset must be either mitbih or ptdbd")

    checkpoint_filename, history_filename, pred_filename, pred_proba_filename, metrics_filename = utils.get_logs(
        params["log_dir"], params["model_name"], seed)

    # Prepare data.
    X, Y, X_test, Y_test = dataset_loader()

    # Reshape X and X_test to be 2D for scikit-learn classifier.
    X = X.reshape(X.shape[0], X.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

    # Train model.
    start_time = time.time()
    history = model.fit(X, Y)
    elapsed_time = time.time() - start_time
    joblib.dump(model, checkpoint_filename)

    # Generate predictions and evaluate model.
    pred_proba_test = get_preds_proba(X_test)
    pred_test = model.predict(X_test)
    np.save(pred_proba_filename, pred_proba_test)
    np.save(pred_filename, pred_test)

    evaluation.log_training_time(elapsed_time, metrics_filename)
    run_evaluation(Y_test, pred_test, pred_proba_test, metrics_filename)

    return checkpoint_filename
