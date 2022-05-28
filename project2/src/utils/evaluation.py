from sklearn.metrics import f1_score, classification_report


def evaluate(y, pred, pred_proba, metrics_filename):
    f1 = f1_score(y, pred, average="weighted")
    class_report = classification_report(y, pred, digits=4)
    f1_str = f"Test F1-score: {f1}"

    print(f1_str)
    print(class_report)

    with open(metrics_filename, "a+", encoding="utf-8") as f:
        f.write(f1_str + "\n")
        f.write(class_report)

    return f1
