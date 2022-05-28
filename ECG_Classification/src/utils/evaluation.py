from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score


def evaluate_mitbih(Y_test, pred_test, pred_proba_test, metrics_filename):
    """Report accuracy, F1-score, and write to metrics_filename.

    Args:
        Y_test (np.ndarray): True labels.
        pred_test (np.ndarray): Predicted labels.
        metrics_filename (str): Path to file to write metrics to.

    Returns:
        (float, float): Accuracy, F1-score.
    """
    acc = accuracy_score(Y_test, pred_test)
    f1 = f1_score(Y_test, pred_test, average="macro")
    acc_str = f"Test accuracy score: {acc}"
    f1_str = f"Test F1-score: {f1}"
    print(acc_str)
    print(f1_str)

    with open(metrics_filename, "a", encoding="utf-8") as f:
        f.write(acc_str + "\n")
        f.write(f1_str + "\n")

    return acc


def evaluate_ptbdb(Y_test, pred_test, pred_proba_test, metrics_filename):
    """Report accuracy, F1-score, AUROC, and AUPRC and write to
    metrics_filename.
    
    Args:
        Y_test (np.ndarray): True labels.
        pred_test (np.ndarray): Predicted labels.
        pred_probas_test (np.ndarray): Unnormalized target values.
        metrics_filename (str): Path to file to write metrics to.
    
    Returns:
        (float, float, float, float): Accuracy, F1-score, AUROC, AUPRC.
    """
    acc = accuracy_score(Y_test, pred_test)
    f1 = f1_score(Y_test, pred_test, average="macro")
    auroc = roc_auc_score(Y_test, pred_proba_test)
    auprc = average_precision_score(Y_test, pred_proba_test)
    acc_str = f"Test accuracy score: {acc}"
    f1_str = f"Test F1-score: {f1}"
    auroc_str = f"Test AUROC: {auroc}"
    auprc_str = f"Test AUPRC: {auprc}"
    print(acc_str)
    print(f1_str)
    print(auroc_str)
    print(auprc_str)

    with open(metrics_filename, "a", encoding="utf-8") as f:
        f.write(acc_str + "\n")
        f.write(f1_str + "\n")
        f.write(auroc_str + "\n")
        f.write(auprc_str + "\n")

    return acc, f1, auroc, auprc


def log_training_time(elapsed_time, metrics_filename):
    """Log training time to metrics_filename.

    Args:
        elapsed_time (float): Elapsed time in seconds.
        metrics_filename (str): Path to file to write metrics to.
    """
    time_str = f"Training time: {elapsed_time:.4f} seconds"
    print(time_str)

    with open(metrics_filename, "a", encoding="utf-8") as f:
        f.write(time_str + "\n")
