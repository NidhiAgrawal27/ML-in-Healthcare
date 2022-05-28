import time
import numpy as np

from utils import data_preparation, evaluation, utils


def train_and_evaluate(model, dataset, params, seed):
    """Train and evaluate stacked ensemble with logging.

    Args:
        model (obj): Stacked ensemble.
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
        get_preds = lambda pred_proba_test: np.argmax(pred_proba_test, axis=-1)
    elif dataset == "ptbdb":
        dataset_loader = data_preparation.load_ptbdb
        run_evaluation = evaluation.evaluate_ptbdb
        get_preds = lambda pred_proba_test: (pred_proba_test > 0.5).astype(
            np.int8)
    else:
        raise ValueError("Dataset must be either mitbih or ptdbd")

    checkpoint_filename, history_filename, pred_filename, pred_proba_filename, metrics_filename = utils.get_logs(
        params["log_dir"], params["model_name"], seed)

    # Prepare data.
    X, Y, X_test, Y_test = dataset_loader()

    # No training required as model takes average of all generated preds.
    start_time = time.time()
    model.fit(X, Y)
    elapsed_time = time.time() - start_time

    # Generate predictions and evaluate model.
    pred_proba_test = model.predict_proba(X_test)
    pred_test = get_preds(pred_proba_test)
    np.save(pred_proba_filename, pred_proba_test)
    np.save(pred_filename, pred_test)

    evaluation.log_training_time(elapsed_time, metrics_filename)
    run_evaluation(Y_test, pred_test, pred_proba_test, metrics_filename)

    return checkpoint_filename
