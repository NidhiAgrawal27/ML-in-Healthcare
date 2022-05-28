import os
import math
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score

from utils import utils

MARKERSIZE = 14
LINEWIDTH = 4
MAP = {
    "baseline": {
        "name": "Baseline",
        "color": "#1f77b4",
        "marker": "D",
        "index": (0, 0)
    },
    "our_cnn": {
        "name": "Vanilla CNN",
        "color": "slategrey",
        "marker": "P",
        "index": (0, 1)
    },
    "our_cnn_weightedloss": {
        "name": "Vanilla CNN w. weighted loss",
        "color": "maroon",
        "marker": "x",
        "index": (0, 2)
    },
    "rnn": {
        "name": "Vanilla RNN",
        "color": "#2ca02c",
        "marker": "o",
        "index": (0, 3)
    },
    "rnn_relu_clip": {
        "name": "Vanilla RNN w. ReLU + clip",
        "color": "#ff7f0e",
        "marker": "s",
        "index": (0, 4)
    },
    "paper_cnn": {
        "name": "Deep ResCNN",
        "color": "#9467bd",
        "marker": "v",
        "index": (0, 5)
    },
    "transfer_our_cnn": {
        "name": "Vanilla CNN (transfer)",
        "color": "#17becf",
        "marker": "<",
        "index": (0, 6)
    },
    "res_cnn": {
        "name": "Deep++ ResCNN",
        "color": "#8c564b",
        "marker": "*",
        "index": (1, 0)
    },
    "attention_cnn": {
        "name": "AttCNN",
        "color": "#e377c2",
        "marker": "X",
        "index": (1, 1)
    },
    "gru_clip": {
        "name": "GRU",
        "color": "#d62728",
        "marker": "^",
        "index": (1, 2)
    },
    "extra_trees": {
        "name": "Extra Trees",
        "color": "#17becf",
        "marker": ">",
        "index": (1, 3)
    },
    "ensemble_stacking": {
        "name": "Stacking ensemble",
        "color": "#7f7f7f",
        "marker": "2",
        "index": (1, 4)
    },
    "ensemble_model_averaging": {
        "name": "Model avg. ensemble",
        "color": "#bcbd22",
        "marker": "H",
        "index": (1, 5)
    },
    "transfer_gru_clip": {
        "name": "GRU (transfer)",
        "color": "navy",
        "marker": ">",
        "index": (1, 6)
    },
    # Unused after here
    "gru": {
        "name": "GRU",
    },
    "gru_relu_clip": {
        "name": "GRU w. ReLU + clip"
    },
    "rnn_clip": {
        "name": "Vanilla RNN w. clip"
    }
}


def get_prefix(model):
    return ("_").join(model.split("_")[:-1])


def get_name(model):
    return MAP[get_prefix(model)]["name"]


def get_color(model):
    return MAP[get_prefix(model)]["color"]


def get_marker(model):
    return MAP[get_prefix(model)]["marker"]


def get_index(model):
    return MAP[get_prefix(model)]["index"]


def plot_learning_curves(filename,
                         figure_dir="../logs/figures/",
                         log_dir="../logs/",
                         exclusion_list=["ensemble", "extra_trees"]):
    datasets = ["ptbdb", "mitbih"]

    # Plot learning curves.
    fig, axes = plt.subplots(2, 2, sharex="col", figsize=(38, 18))

    axes[0, 0].set_ylabel("Training loss")
    axes[0, 1].set_ylabel("Training loss")

    axes[1, 0].set_ylabel("Validation loss")
    axes[1, 0].set_xlabel("Epoch")

    axes[1, 1].set_ylabel("Validation loss")
    axes[1, 1].set_xlabel("Epoch")

    ax2_is_set = False

    for i, dataset in enumerate(datasets):
        mean_std_histories = utils.get_mean_std_losses(dataset, log_dir,
                                                       exclusion_list)

        axes[0, i].set_title(dataset.upper())

        for mean_std_history in mean_std_histories:
            model = mean_std_history["model"]
            train_losses_mean = mean_std_history["train_losses_mean"]
            train_losses_std = mean_std_history["train_losses_std"]
            val_losses_mean = mean_std_history["val_losses_mean"]
            val_losses_std = mean_std_history["val_losses_std"]
            epochs = mean_std_history["epochs"]

            # If model is rnn_relu_clip then plot train loss on diff axis.
            if get_prefix(model) == "rnn_relu_clip":
                ax2_is_set = True
                ax2 = axes[0, i].twinx()
                ax2.plot(epochs,
                         train_losses_mean,
                         label=get_name(model),
                         color=get_color(model),
                         marker=get_marker(model),
                         markersize=MARKERSIZE,
                         linewidth=LINEWIDTH)
                ax2.fill_between(epochs,
                                 train_losses_mean - train_losses_std,
                                 train_losses_mean + train_losses_std,
                                 color=get_color(model),
                                 alpha=0.15)
                ax2.tick_params(axis="y", labelcolor=get_color(model))
            else:
                axes[0, i].plot(epochs,
                                train_losses_mean,
                                label=get_name(model),
                                color=get_color(model),
                                marker=get_marker(model),
                                markersize=MARKERSIZE,
                                linewidth=LINEWIDTH)
                axes[0, i].fill_between(epochs,
                                        train_losses_mean - train_losses_std,
                                        train_losses_mean + train_losses_std,
                                        color=get_color(model),
                                        alpha=0.15)

            axes[1, i].plot(epochs,
                            val_losses_mean,
                            label=get_name(model),
                            color=get_color(model),
                            marker=get_marker(model),
                            markersize=MARKERSIZE,
                            linewidth=LINEWIDTH)
            axes[1, i].fill_between(epochs,
                                    val_losses_mean - val_losses_std,
                                    val_losses_mean + val_losses_std,
                                    color=get_color(model),
                                    alpha=0.15)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if ax2_is_set:
        handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
        handles += handles_ax2
        labels += labels_ax2

    # Sort labels and handles:
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

    legend = fig.legend(handles,
                        labels,
                        loc="upper center",
                        ncol=6,
                        bbox_to_anchor=(0, 1.1, 1, 0))
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, filename),
                bbox_extra_artists=(legend, ),
                bbox_inches="tight")

    return fig, axes


def plot_conf_matrix(axis,
                     conf_matrix,
                     class_list,
                     model_name,
                     normalize,
                     set_x=True,
                     set_y=True):
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(
            axis=1)[:, np.newaxis]
        axis.set_title(model_name)
        format_data = ".2f"
    else:
        axis.set_title(model_name)
        format_data = "d"

    treshold = conf_matrix.max() / 2
    for i, j in itertools.product(range(conf_matrix.shape[0]),
                                  range(conf_matrix.shape[1])):
        axis.text(j,
                  i,
                  format(conf_matrix[i, j], format_data),
                  horizontalalignment="center",
                  color="white" if conf_matrix[i, j] > treshold else "black")

    im = axis.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.BuPu)
    ticks = np.arange(len(class_list))
    if set_x:
        axis.set_xlabel('True labels')
        axis.set_xticks(ticks, class_list)
    else:
        axis.set_xticks([])
    if set_y:
        axis.set_ylabel('Predicted labels')
        axis.set_yticks(ticks, class_list)
    else:
        axis.set_yticks([])

    return im


def plot_confusion_matrices(Y,
                            dataset,
                            subplots,
                            figsize,
                            class_list,
                            filename,
                            figure_dir="../logs/figures/",
                            log_dir="../logs/",
                            measure="Test accuracy score",
                            higher_is_better=True,
                            exclusion_list=[]):
    infos = utils.get_infos(log_dir, dataset, exclusion_list)

    fig, axes = plt.subplots(subplots[0], subplots[1], figsize=figsize)

    for i, (model, info_list) in enumerate(infos.items()):
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
        # Plot confusion matrix for best scoring model
        Y_pred = best_info["pred"]
        conf_matrix = confusion_matrix(Y, Y_pred)

        index = get_index(model)
        axis = axes[index[0], index[1]]

        im = plot_conf_matrix(axis,
                              conf_matrix,
                              class_list,
                              get_name(model),
                              normalize=True,
                              set_x=index[0] == 1,
                              set_y=index[1] == 0)

    fig.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist())
    fig.savefig(os.path.join(figure_dir, filename))
    return fig, axes


def plot_prc(Y,
             filename,
             figure_dir="../logs/figures/",
             log_dir="../logs/",
             exclusion_list=[]):
    infos = utils.get_infos(log_dir, "ptbdb", exclusion_list)

    fig, ax = plt.subplots(1, 1, figsize=(18, 18))

    for i, (model, info_list) in enumerate(infos.items()):
        precisions = []
        aucs = []
        mean_recall = np.linspace(0, 1, 100)
        for info in info_list:
            pred_proba = info["pred_proba"]
            auprc = average_precision_score(Y, pred_proba)
            precision, recall, tresholds = precision_recall_curve(
                Y, pred_proba)
            interp_precision = np.interp(mean_recall, precision, recall)
            interp_precision[0] = 1.0
            precisions.append(interp_precision)
            aucs.append(auprc)
        mean_precision = np.mean(precisions, axis=0)
        mean_precision[-1] = 0.0
        mean_auc = auc(mean_precision, mean_recall)
        std_auc = np.std(aucs)
        ax.plot(
            mean_recall,
            mean_precision,
            color=get_color(model),
            label=fr"{get_name(model)} (AP={mean_auc:.2f}$\pm${std_auc:.2f})",
            alpha=0.8,
            linewidth=5,
            marker=get_marker(model),
            markersize=12)
        std_precision = np.std(precisions, axis=0)
        tprs_upper = np.minimum(mean_precision + std_precision, 1)
        tprs_lower = np.maximum(mean_precision - std_precision, 0)
        ax.fill_between(mean_recall,
                        tprs_lower,
                        tprs_upper,
                        color=get_color(model),
                        alpha=0.1)

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Precision recall curve",
    )

    ax.axes.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='best', fontsize=24)

    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, filename))
    return fig, ax


def plot_roc(Y,
             filename,
             figure_dir="../logs/figures/",
             log_dir="../logs/",
             exclusion_list=[]):
    infos = utils.get_infos(log_dir, "ptbdb", exclusion_list)

    fig, ax = plt.subplots(1, 1, figsize=(18, 18))

    ax.plot([0, 1], [0, 1], linestyle="--", color="r", alpha=0.8, linewidth=5)

    for i, (model, info_list) in enumerate(infos.items()):
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        for info in info_list:
            pred_proba = info["pred_proba"]
            auroc = roc_auc_score(Y, pred_proba)
            fpr, tpr, thresholds = roc_curve(Y, pred_proba)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auroc)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color=get_color(model),
            label=fr"{get_name(model)} (AUC={mean_auc:.2f}$\pm${std_auc:.2f})",
            alpha=0.8,
            linewidth=5,
            marker=get_marker(model),
            markersize=12)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr,
                        tprs_lower,
                        tprs_upper,
                        color=get_color(model),
                        alpha=0.1)

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Receiver operating characteristic",
    )

    ax.axes.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='best', fontsize=24)

    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, filename))
    return fig, ax


def compute_scores(dataset,
                   filename,
                   figure_dir="../logs/figures/",
                   log_dir="../logs/",
                   exclusion_list=[]):
    infos = utils.get_infos(log_dir, dataset, exclusion_list)

    scores = {}

    sort_by_metric = "Test accuracy score" + " (mean)"

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
            #score[metric] = {"mean": np.mean(values), "std": np.std(values)}
        scores[get_name(model)] = score

    df = pd.DataFrame.from_dict(scores, orient="index")
    df = df.sort_values(by=sort_by_metric, ascending=False)
    df.to_csv(os.path.join(figure_dir, filename + ".csv"))

    cols = list(df)
    new_cols_map = {}
    for col in cols:
        new_col = col.replace(" (mean)", "").replace(" (std)", "").replace(
            "Test ", "").replace("-score", "").replace(" score", "")
        new_col = new_col[0].upper() + new_col[1:]

        new_cols_map.setdefault(new_col, {
            "mean": None,
            "std": None
        })["mean" if "(mean)" in col else "std"] = col

    for new_col, value in new_cols_map.items():
        mean_name = value["mean"]
        std_name = value["std"]
        df[new_col] = df[mean_name].apply(
            lambda x: f"{x:.4f}" if new_col != "Training time" else f"{x:.0f}"
        ) + r" \pm " + df[std_name].apply(
            lambda x: f"{x:.4f}" if new_col != "Training time" else f"{x:.0f}")

    df = df.drop(columns=cols)

    try:
        df.style.to_latex(os.path.join(figure_dir, filename + ".tex"),
                          column_format="l" + "r" * len(df.columns))
    except Exception as e:
        df.to_latex(os.path.join(figure_dir, filename + ".tex"),
                    column_format="l" + "r" * len(df.columns))

    return df
