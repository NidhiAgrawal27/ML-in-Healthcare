import pathlib
import matplotlib.pyplot as plt

from utils import visualization, data_preparation, utils

LOG_DIR = "../logs/"
FIGURE_DIR = "../logs/figures/"
FONT_SIZE = 30


def main():
    # Create figure directory if it does not exist.
    pathlib.Path(FIGURE_DIR).mkdir(parents=True, exist_ok=True)

    # Update font size.
    plt.rcParams.update({'font.size': FONT_SIZE})

    _, _, _, Y_mitbih_test = data_preparation.load_mitbih()
    _, _, _, Y_ptbdb_test = data_preparation.load_ptbdb()

    fig, axes = visualization.plot_learning_curves(
        filename="learning_curves.png",
        figure_dir=FIGURE_DIR,
        log_dir=LOG_DIR,
        exclusion_list=[
            "ensemble_model_averaging", "ensemble_stacking", "extra_trees",
            "gru", "gru_relu_clip", "rnn_clip"
        ])
    fig, axes = visualization.plot_confusion_matrices(
        Y=Y_mitbih_test,
        dataset="mitbih",
        subplots=(2, 6),  # No transfer learning so 1 col smaller.
        figsize=(54, 18),
        filename="confusion_matrices_mitbih.png",
        class_list=['N', 'S', 'V', 'F', 'Q'],
        figure_dir=FIGURE_DIR,
        log_dir=LOG_DIR,
        exclusion_list=["gru", "gru_relu_clip", "rnn_clip"])
    fig, axes = visualization.plot_confusion_matrices(
        Y=Y_ptbdb_test,
        dataset="ptbdb",
        subplots=(2, 7),
        figsize=(66, 18),
        class_list=["Normal", "Abnormal"],
        filename="confusion_matrices_ptbdb.png",
        figure_dir=FIGURE_DIR,
        log_dir=LOG_DIR,
        exclusion_list=["gru", "gru_relu_clip", "rnn_clip"])

    fig, ax = visualization.plot_roc(
        Y=Y_ptbdb_test,
        filename="roc_ptbdb.png",
        figure_dir=FIGURE_DIR,
        log_dir=LOG_DIR,
        exclusion_list=["gru", "gru_relu_clip", "rnn_clip"])
    fig, ax = visualization.plot_prc(
        Y=Y_ptbdb_test,
        filename="prc_ptbdb.png",
        figure_dir=FIGURE_DIR,
        log_dir=LOG_DIR,
        exclusion_list=["gru", "gru_relu_clip", "rnn_clip"])
    visualization.compute_scores(
        dataset="mitbih",
        filename="scores_mitbih",
        figure_dir=FIGURE_DIR,
        log_dir=LOG_DIR,
        exclusion_list=["gru", "gru_relu_clip", "rnn_clip"])
    visualization.compute_scores(
        dataset="ptbdb",
        filename="scores_ptbdb",
        figure_dir=FIGURE_DIR,
        log_dir=LOG_DIR,
        exclusion_list=["gru", "gru_relu_clip", "rnn_clip"])


if __name__ == '__main__':
    main()
