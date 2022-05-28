import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.inspection import permutation_importance
# from dtreeviz.trees import dtreeviz # UNCOMMENT TO RUN TREE VISUALIZATION.
from sklearn.tree import plot_tree
from utilities.data import get_radiomics_dataset

from utilities import utils

MARKERSIZE = 14
LINEWIDTH = 4
MAP = {
    "random_forest": {
        "name": "Random Forest"
    },
    "decision_tree": {
        "name": "Decision Tree"
    },
    "baseline_cnn_lime": {
        "name": "Baseline CNN"
    },
    "logistic_regression": {
        "name": "Logistic Regression"
    },
    "transfer_learning": {
        "name": "VGG16 Transfer Learning"
    }
}


def get_name(model):
    return MAP[model]["name"]


def plot_decision_tree(model_name,
                       x,
                       y,
                       class_names,
                       figsize,
                       filename,
                       figure_dir="../logs/figures/",
                       log_dir="../logs/",
                       measure="Test accuracy",
                       higher_is_better=True):
    infos = utils.get_infos(log_dir)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i, (model, info_list) in enumerate(infos.items()):
        if model != model_name:
            continue

        best_measure = -math.inf if higher_is_better else math.inf
        best_info = None
        for info in info_list:
            if higher_is_better and info["metrics"][measure] > best_measure:
                best_measure = info["metrics"][measure]
                best_info = info
            elif not higher_is_better and info["metrics"][
                    measure] < best_measure:
                best_measure = info["metrics"][measure]
                best_info = info

        # Plot best decision tree
        with open(os.path.join(best_info["path"], "model.pkl"), "rb") as f:
            clf = pickle.load(f)

        viz = dtreeviz(clf,
                       x,
                       y,
                       feature_names=x.columns,
                       class_names=class_names)

    viz.save(os.path.join(figure_dir, filename))
    return viz


def plot_logistic_regression_feature_importance(figure_dir="../logs/figures", log_dir="../logs"):

    infos = utils.get_infos(log_dir)
    best_measure = -math.inf
    best_info = None
    for info in infos["logistic_regression"]:
        if info["metrics"]["Test accuracy"] > best_measure:
            best_measure = info["metrics"]["Test accuracy"]
            best_info = info

    # Load model
    with open(os.path.join(best_info["path"], "model.pkl"), "rb") as f:
        model = pickle.load(f)

    train_data = get_radiomics_dataset()[0]

    # Interpret
    named_coeffs = list(zip(list(train_data.columns), model.coef_[0]))
    named_coeffs = sorted(named_coeffs, key=lambda x: abs(x[1]))
    names, coeffs = zip(*named_coeffs)

    plt.figure(figsize=(20, 20))
    plt.bar(x=list(range(len(coeffs))),
            height=np.abs(np.array(list(reversed(coeffs)))))
    plt.tight_layout()
    plt.savefig(os.path.join(
        figure_dir, "logistic_regression_feature_importance.png"))
    plt.close()

    plt.figure(figsize=(40, 20))
    plt.barh(y=names[-20:], width=coeffs[-20:])
    plt.tight_layout()
    plt.savefig(os.path.join(
        figure_dir, "logistic_regression_top20.png"))


def plot_permutation_based_feature_importance(
    model_name,
    x,
    y,
    figsize,
    filename,
    figure_dir="../logs/figures/",
    log_dir="../logs/",
    measure="Test accuracy",
    higher_is_better=True,
    top_features=15,
):
    infos = utils.get_infos(log_dir)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i, (model, info_list) in enumerate(infos.items()):
        if model != model_name:
            continue

        best_measure = -math.inf if higher_is_better else math.inf
        best_info = None
        for info in info_list:
            if higher_is_better and info["metrics"][measure] > best_measure:
                best_measure = info["metrics"][measure]
                best_info = info
            elif not higher_is_better and info["metrics"][
                    measure] < best_measure:
                best_measure = info["metrics"][measure]
                best_info = info

        # Plot top 10 features for best model.
        with open(os.path.join(best_info["path"], "model.pkl"), "rb") as f:
            clf = pickle.load(f)

        result = permutation_importance(clf,
                                        x,
                                        y,
                                        n_repeats=50,
                                        random_state=42,
                                        n_jobs=-1)

        res_dict = {
            "importances_mean": result.importances_mean,
            "importances_std": result.importances_std
        }
        forest_importance = pd.DataFrame(res_dict, index=x.columns)
        forest_importance.sort_values(by=["importances_mean"],
                                      ascending=False,
                                      inplace=True)
        forest_importance = forest_importance.head(top_features)
        forest_importance["importances_mean"].plot(
            kind="bar", ax=ax, yerr=forest_importance["importances_std"])

    ax.set_title(
        f"PyRadiomics Feature Importances Using Permutation (Top {top_features})"
    )
    ax.set_ylabel("Mean accuracy decrease")
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=45,
                       ha="right",
                       rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=22)
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, filename))
    return fig, ax


def compute_scores(filename,
                   sort_by_measure="Test accuracy",
                   figure_dir="../logs/figures/",
                   log_dir="../logs/",
                   exclusion_list=[]):
    infos = utils.get_infos(log_dir, exclusion_list)

    scores = {}

    sort_by_metric = sort_by_measure + " (mean)"

    for model, info_list in infos.items():
        score = {}
        mets = {}
        # Add metrics for all model runs.
        for info in info_list:
            metrics = info["metrics"]
            for metric, value in metrics.items():
                if metric not in mets:
                    mets[metric] = []
                mets[metric].append(value)
        # Compute mean and std
        for metric, values in mets.items():
            score[metric + " (mean)"] = np.mean(values)
            score[metric + " (std)"] = np.std(values)
        scores[get_name(model)] = score

    df = pd.DataFrame.from_dict(scores, orient="index")
    df = df.sort_values(by=sort_by_metric, ascending=False)
    df.to_csv(os.path.join(figure_dir, filename + ".csv"))

    cols = list(df)
    new_cols_map = {}
    for col in cols:
        new_col = (col.replace(" (mean)", "").replace(" (std)", "").replace(
            "Test ", "").replace("-score", "").replace(" score", ""))
        new_col = new_col[0].upper() + new_col[1:]

        new_cols_map.setdefault(new_col, {
            "mean": None,
            "std": None
        })["mean" if "(mean)" in col else "std"] = col

    for new_col, value in new_cols_map.items():
        mean_name = value["mean"]
        std_name = value["std"]
        df[new_col] = (df[mean_name].apply(lambda x: f"{x:.4f}" if new_col !=
                                           "Training time" else f"{x:.0f}") +
                       r" \pm " +
                       df[std_name].apply(lambda x: f"{x:.4f}" if new_col !=
                                          "Training time" else f"{x:.0f}"))

    df = df.drop(columns=cols)

    try:
        df.style.to_latex(
            os.path.join(figure_dir, filename + ".tex"),
            column_format="l" + "r" * len(df.columns),
        )
    except Exception as e:
        df.to_latex(
            os.path.join(figure_dir, filename + ".tex"),
            column_format="l" + "r" * len(df.columns),
        )

    return df
