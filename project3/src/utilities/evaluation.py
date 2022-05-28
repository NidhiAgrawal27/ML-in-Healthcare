from sklearn.metrics import accuracy_score, f1_score, classification_report


def evaluate(y, pred, pred_proba, metrics_filename):
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    class_report = classification_report(y, pred, digits=4)
    acc_str = f"Test accuracy: {acc}"
    f1_str = f"Test F1-score: {f1}"

    print(acc_str)
    print(f1_str)
    print(class_report)

    with open(metrics_filename, "a+", encoding="utf-8") as f:
        f.write(acc_str + "\n")
        f.write(f1_str + "\n")
        f.write(class_report)

    return acc
