import os
import math
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Levenshtein
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from utils import data_preparation, utils
from utils.tokenizers.simple_tokenizer import SimpleTokenizer

MARKERSIZE = 14
LINEWIDTH = 4
MAP = {
    "baseline": {
        "name": "Baseline",
        "color": "#1f77b4",
        "marker": "D",
        "index": (0, 0),
    },
    "baseline_lemmatize": {
        "name": "Baseline lemm.",
        "color": "#2ca02c",
        "marker": "o",
        "index": (1, 0),
    },
    "word2vec_mlp": {
        "name": "Word2Vec MLP",
        "color": "slategrey",
        "marker": "p",
        "index": (0, 1),
    },
    "word2vec_mlp_lemmatize": {
        "name": "Word2Vec lemm. MLP",
        "color": "maroon",
        "marker": "x",
        "index": (1, 1),
    },
    "word2vec_lr": {
        "name": "Word2Vec LR",
        "color": "#ff7f0e",
        "marker": "s",
        "index": (0, 2),
    },
    "word2vec_lr_lemmatize": {
        "name": "Word2Vec lemm. LR",
        "color": "#9467bd",
        "marker": "v",
        "index": (1, 2),
    },
    "one_hot_encoding": {
        "name": "One Hot",
        "color": "#17becf",
        "marker": "<",
        "index": (0, 3),
    },
    "one_hot_encoding_lemmatize": {
        "name": "One Hot lemm.",
        "color": "#8c564b",
        "marker": "*",
        "index": (1, 3),
    },
    "glove_mlp": {
        "name": "GloVe MLP",
        "color": "slategrey",  # TODO
        "marker": "p",  # TODO
        "index": (0, 4),
    },
    "emilyalsentzer-Bio_ClinicalBERT__False__20__0-1__3e-05__32__5__3__False__False": {
        "name": "BERT finetuned",
        "color": "slategrey",  # TODO
        "marker": "p",  # TODO
        "index": (1, 4),
    },
    "emilyalsentzer-Bio_ClinicalBERT__False__20__0-1__3e-05__32__5__3__False__True": {
        "name": "BERT finetuned + index",
        "color": "slategrey",  # TODO
        "marker": "p",  # TODO
        "index": (0, 5),
    },
    "emilyalsentzer-Bio_ClinicalBERT__True__20__0-1__0-001__32__5__3__True__False": {
        "name": "BERT frozen",
        "color": "slategrey",  # TODO
        "marker": "p",  # TODO
        "index": (1, 5),
    },
}


def get_name(model):
    return MAP[model]["name"]


def get_color(model):
    return MAP[model]["color"]


def get_marker(model):
    return MAP[model]["marker"]


def get_index(model):
    return MAP[model]["index"]


def plot_conf_matrix(
    axis, conf_matrix, class_list, model_name, normalize, set_x=True, set_y=True
):
    if normalize:
        conf_matrix = (
            conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
        )
        axis.set_title(model_name)
        format_data = ".2f"
    else:
        axis.set_title(model_name)
        format_data = "d"

    treshold = conf_matrix.max() / 2
    for i, j in itertools.product(
        range(conf_matrix.shape[0]), range(conf_matrix.shape[1])
    ):
        axis.text(
            j,
            i,
            format(conf_matrix[i, j], format_data),
            horizontalalignment="center",
            color="white" if conf_matrix[i, j] > treshold else "black",
        )

    im = axis.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.BuPu)
    ticks = np.arange(len(class_list))
    if set_x:
        axis.set_xlabel("True labels")
        axis.set_xticks(ticks, class_list)
    else:
        axis.set_xticks([])
    if set_y:
        axis.set_ylabel("Predicted labels")
        axis.set_yticks(ticks, class_list)
    else:
        axis.set_yticks([])

    return im


def plot_confusion_matrices(
    Y,
    subplots,
    figsize,
    class_list,
    filename,
    figure_dir="../logs/figures/",
    log_dir="../logs/",
    measure="Test F1-score",
    higher_is_better=True,
    exclusion_list=[],
):
    infos = utils.get_infos(log_dir, exclusion_list)

    fig, axes = plt.subplots(subplots[0], subplots[1], figsize=figsize)

    for i, (model, info_list) in enumerate(infos.items()):
        best_measure = -math.inf if higher_is_better else math.inf
        best_info = None
        for info in info_list:
            if higher_is_better and info["metrics"][measure] > best_measure:
                best_measure = info["metrics"][measure]
                best_info = info
            elif not higher_is_better and info["metrics"][measure] < best_measure:
                best_measure = info["metrics"][measure]
                best_info = info
        # Plot confusion matrix for best scoring model
        Y_pred = best_info["pred"]
        conf_matrix = confusion_matrix(Y, Y_pred)
        index = get_index(model)
        axis = axes[index[0], index[1]]

        im = plot_conf_matrix(
            axis,
            conf_matrix,
            class_list,
            get_name(model),
            normalize=True,
            set_x=index[0] == 1,
            set_y=index[1] == 0,
        )

    fig.tight_layout()
    # fig.colorbar(im, ax=axes.ravel().tolist())
    fig.savefig(os.path.join(figure_dir, filename))
    return fig, axes


def print_example_sentences(
    indices,
    data_path,
    filename,
    figure_dir="../logs/figures/",
    length_treshold=10,
    find_most_different=True,
):
    tokenizer_no_lemmatize = SimpleTokenizer(lemmatization=False)
    trainset_no_lemmatize, _, _ = data_preparation.load_data(
        data_path, tokenizer=tokenizer_no_lemmatize
    )

    tokenizer_lemmatize = SimpleTokenizer(lemmatization=True)
    trainset_lemmatize, _, _ = data_preparation.load_data(
        data_path, tokenizer=tokenizer_lemmatize
    )

    if find_most_different:
        best_lev = 0
        best_index = None

        for i, tokens in enumerate(trainset_no_lemmatize.X):
            token_len = len(tokens)
            if token_len < length_treshold:
                continue
            lev = 0
            for token, token_lemma in zip(tokens, trainset_lemmatize.X[i]):
                lev += Levenshtein.distance(token, token_lemma)
            if token_len > 0:
                lev /= token_len
            if lev > best_lev:
                best_lev = lev
                best_index = i

            indices = [best_index]

    for index in indices:
        sentence_str = f"Sentence:\n{trainset_no_lemmatize.data[index]['sentence']}"
        token_no_lemma_str = (
            f"Tokenized w. no lemmatization:\n{trainset_no_lemmatize.X[index]}"
        )
        token_lemma_str = f"Tokenized w. lemmatization:\n{trainset_lemmatize.X[index]}"

        with open(os.path.join(figure_dir, filename), "w", encoding="utf-8") as f:
            f.write(sentence_str + "\n")
            f.write(token_no_lemma_str + "\n")
            f.write(token_lemma_str + "\n")


def plot_classification_report(
    Y,
    subplots,
    figsize,
    class_list,
    filename,
    figure_dir="../logs/figures/",
    log_dir="../logs/",
    measure="Test F1-score",
    higher_is_better=True,
    exclusion_list=[],
):
    infos = utils.get_infos(log_dir, exclusion_list)

    fig, axes = plt.subplots(subplots[0], subplots[1], figsize=figsize)

    for i, (model, info_list) in enumerate(infos.items()):
        best_measure = -math.inf if higher_is_better else math.inf
        best_info = None
        for info in info_list:
            if higher_is_better and info["metrics"][measure] > best_measure:
                best_measure = info["metrics"][measure]
                best_info = info
            elif not higher_is_better and info["metrics"][measure] < best_measure:
                best_measure = info["metrics"][measure]
                best_info = info

        Y_pred = best_info["pred"]
        report = classification_report(
            Y, Y_pred, target_names=class_list, output_dict=True, digits=2
        )
        index = get_index(model)
        axis = axes[index[0], index[1]]
        axis.set_title(get_name(model))
        # Plot per class metrics.
        ax = sns.heatmap(
            pd.DataFrame(report).iloc[:-1, :].T,
            annot=True,
            ax=axis,
            cbar=False,
            cmap="BuPu",
            xticklabels=index[0] == 1,
            yticklabels=index[1] == 0,
            fmt=".2f",
        )
        # Add some whitespace
        ax.axhline(5, color="white", lw=15)

    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, filename))
    return fig, axes


def get_len_df(datasets, key, class_list, treshold=None):
    lens = []
    data = []
    for dataset in datasets:
        data += dataset.data
    for datum in data:
        lens.append([len(datum[key]), datum["label"]])
    df = pd.DataFrame(lens, columns=["len", "label"])
    if treshold is not None:
        df = df[df["len"] < treshold]
    return df


def plot_distributions(
    datasets,
    figsize,
    filename,
    class_list,
    figure_dir="../logs/figures/",
    treshold=True,
):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    get_len_df(datasets, "sentence", class_list)

    # Class distribution
    Y = np.concatenate([s.y for s in datasets])
    _, counts = np.unique(Y, return_counts=True)
    axes[0].set_title("Class distribution")
    axes[0].set_xlabel("Number of sentences")
    axis = sns.barplot(
        x=counts, y=class_list, ax=axes[0], linewidth=5, edgecolor=".2", palette="deep"
    )
    axis.bar_label(axis.containers[0])

    # Token distribution
    token_df = get_len_df(
        datasets, "tokens", class_list, treshold=(None if not treshold else 40)
    )
    axes[1].set_title("Token distribution")
    axis = sns.violinplot(
        x="len",
        y="label",
        data=token_df,
        ax=axes[1],
        orient="h",
        palette="deep",
        linewidth=5,
    ).set(xlabel="Number of tokens per sentence", ylabel="", yticks=[])

    # Character distribution
    char_df = get_len_df(
        datasets, "sentence", class_list, treshold=(None if not treshold else 400)
    )
    axes[2].set_title("Character distribution")
    axis = sns.violinplot(
        x="len",
        y="label",
        data=char_df,
        ax=axes[2],
        orient="h",
        palette="deep",
        linewidth=5,
    ).set(xlabel="Number of characters per sentence", ylabel="", yticks=[])

    fig.savefig(os.path.join(figure_dir, filename), bbox_inches="tight")
    return fig, axes


def compute_scores(
    filename, figure_dir="../logs/figures/", log_dir="../logs/", exclusion_list=[]
):
    infos = utils.get_infos(log_dir, exclusion_list)

    scores = {}

    sort_by_metric = "Test F1-score" + " (mean)"

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
        new_col = (
            col.replace(" (mean)", "")
            .replace(" (std)", "")
            .replace("Test ", "")
            .replace("-score", "")
            .replace(" score", "")
        )
        new_col = new_col[0].upper() + new_col[1:]

        new_cols_map.setdefault(new_col, {"mean": None, "std": None})[
            "mean" if "(mean)" in col else "std"
        ] = col

    for new_col, value in new_cols_map.items():
        mean_name = value["mean"]
        std_name = value["std"]
        df[new_col] = (
            df[mean_name].apply(
                lambda x: f"{x:.4f}" if new_col != "Training time" else f"{x:.0f}"
            )
            + r" \pm "
            + df[std_name].apply(
                lambda x: f"{x:.4f}" if new_col != "Training time" else f"{x:.0f}"
            )
        )

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
