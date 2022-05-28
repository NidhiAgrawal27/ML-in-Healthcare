import pathlib
import matplotlib.pyplot as plt

from visualization import plots
from utilities import data

CONFIG = {
    "log_dir": "../logs/",
    "figure_dir": "../logs/figures/",
    "font_size": 30,
    "exclusion_list": ["baseline_cnn_shap"]
}


def main():
    # Create figure directory if it does not exist.
    pathlib.Path(CONFIG["figure_dir"]).mkdir(parents=True, exist_ok=True)

    # Update font size.
    plt.rcParams.update({"font.size": CONFIG["font_size"]})

    train_data, train_labels, val_data, val_labels, test_data, test_labels = (
        data.get_radiomics_dataset())

    # Outcommented and not run by default since it uses dtreeviz requiring
    # graphviz installation.
    # Follow installation guide on https://github.com/parrt/dtreeviz
    # Generated svg file also needs to be converted to pdf, e.g. using inkscape
    # or https://cloudconvert.com/svg-to-pdf.
    """
    plots.plot_decision_tree(model_name="decision_tree",
                             x=train_data,
                             y=train_labels,
                             class_names=["No Tumor", "Tumor"],
                             figsize=(50, 50),
                             filename="decision_tree.svg",
                             figure_dir=CONFIG["figure_dir"],
                             log_dir=CONFIG["log_dir"],
                             measure="Test accuracy",
                             higher_is_better=True)
    """

    plots.plot_permutation_based_feature_importance(
        model_name="random_forest",
        x=test_data,
        y=test_labels,
        figsize=(20, 20),
        filename="random_forest_permutation_based_feature_importance.png",
        figure_dir=CONFIG["figure_dir"],
        log_dir=CONFIG["log_dir"],
        measure="Test accuracy",
        higher_is_better=True,
        top_features=20)

    plots.plot_logistic_regression_feature_importance(
        figure_dir=CONFIG["figure_dir"],
        log_dir=CONFIG["log_dir"],
    )

    plots.compute_scores(
        filename="scores",
        figure_dir=CONFIG["figure_dir"],
        log_dir=CONFIG["log_dir"],
        exclusion_list=CONFIG["exclusion_list"],
    )


if __name__ == "__main__":
    main()
