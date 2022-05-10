from collections import Counter
import time
import numpy as np
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
)

from utils import data_preparation, evaluation, utils


def train_and_evaluate(
    model, dataset, params, seed, weighted_loss=False, fit_params={}
):
    """Train and evaluate Tensorflow model with logging.

    Args:
        model (keras.Model): Keras model to train.
        dataset (str): Either "mitbih" or "ptbdb".
        params (dict): Training parameters containing model_name, log_dir,
         epochs, es_patience, lr_patience (NOT model params).
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
        get_preds = lambda pred_proba_test: (pred_proba_test > 0.5).astype(np.int8)
    else:
        raise ValueError("Dataset must be either mitbih or ptdbd")

    (
        checkpoint_filename,
        history_filename,
        pred_filename,
        pred_proba_filename,
        metrics_filename,
    ) = utils.get_logs(params["log_dir"], params["model_name"], seed)

    # Prepare data and callbacks.
    X, Y, X_test, Y_test = dataset_loader()
    checkpoint = ModelCheckpoint(
        checkpoint_filename,
        monitor="val_acc",
        verbose=1,
        save_best_only=True,
        mode="max",
    )
    early = EarlyStopping(
        monitor="val_acc", mode="max", patience=params["es_patience"], verbose=1
    )
    redonplat = ReduceLROnPlateau(
        monitor="val_acc", mode="max", patience=params["lr_patience"], verbose=2
    )
    csv_logger = CSVLogger(history_filename, separator=",", append=False)
    callbacks_list = [checkpoint, early, redonplat, csv_logger]

    # get class weights
    if weighted_loss:
        num_samples = Counter(Y)
        max_sample = max(num_samples.values())
        class_weight = {cls: max_sample / n for cls, n in num_samples.items()}
    else:
        class_weight = None

    # Train model.
    start_time = time.time()
    history = model.fit(
        X,
        Y,
        epochs=params["epochs"],
        verbose=2,
        callbacks=callbacks_list,
        validation_split=0.1,
        class_weight=class_weight,
        **fit_params
    )
    elapsed_time = time.time() - start_time
    model.load_weights(checkpoint_filename)

    # Generate predictions and evaluate model.
    pred_proba_test = model.predict(X_test)
    pred_test = get_preds(pred_proba_test)
    np.save(pred_proba_filename, pred_proba_test)
    np.save(pred_filename, pred_test)

    evaluation.log_training_time(elapsed_time, metrics_filename)
    run_evaluation(Y_test, pred_test, pred_proba_test, metrics_filename)

    return checkpoint_filename
