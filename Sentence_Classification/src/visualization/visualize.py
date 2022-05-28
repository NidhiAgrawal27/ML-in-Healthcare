import pathlib
import matplotlib.pyplot as plt
import numpy as np

from utils import data_preparation, visualization, utils
from utils.tokenizers.simple_tokenizer import SimpleTokenizer

CONFIG = {
    "data_path": "../data/pubmed-rct/PubMed_200k_RCT/",
    "log_dir": "../logs/",
    "figure_dir": "../logs/figures/",
    "font_size": 30,
    "exclusion_list": [
        "emilyalsentzer-Bio_ClinicalBERT__True__20__0-1__0-001__32__5__3__False__False"
    ],
}


def main():
    # Create figure directory if it does not exist.
    pathlib.Path(CONFIG["figure_dir"]).mkdir(parents=True, exist_ok=True)

    # Update font size.
    plt.rcParams.update({"font.size": CONFIG["font_size"]})

    tokenizer = SimpleTokenizer(lemmatization=False)
    trainset, devset, testset = data_preparation.load_data(
        CONFIG["data_path"], tokenizer=tokenizer
    )
    class_list = ["B", "O", "M", "R", "C"]
    class_list_full = ["BACKGROUND", "OBJECTIVE", "METHOD", "RESULT", "CONCLUSION"]

    visualization.plot_confusion_matrices(
        Y=testset.y,
        subplots=(2, 6),
        figsize=(54, 18),
        filename="confusion_matrices.png",
        class_list=class_list,
        figure_dir=CONFIG["figure_dir"],
        log_dir=CONFIG["log_dir"],
        exclusion_list=CONFIG["exclusion_list"],
    )

    visualization.print_example_sentences(
        indices=[140182],
        data_path=CONFIG["data_path"],
        filename="example_sentences.txt",
        figure_dir=CONFIG["figure_dir"],
        length_treshold=10,
        find_most_different=False,
    )

    visualization.plot_classification_report(
        Y=testset.y,
        subplots=(2, 6),
        figsize=(54, 20),
        filename="classification_report.png",
        class_list=class_list,
        figure_dir=CONFIG["figure_dir"],
        log_dir=CONFIG["log_dir"],
        exclusion_list=CONFIG["exclusion_list"],
    )

    visualization.plot_distributions(
        datasets=[trainset, devset, testset],
        figsize=(40, 10),
        filename="distributions_no_treshold.png",
        class_list=class_list_full,
        figure_dir=CONFIG["figure_dir"],
        treshold=False,
    )

    visualization.plot_distributions(
        datasets=[trainset, devset, testset],
        figsize=(40, 10),
        filename="distributions_tresholded.png",
        class_list=class_list_full,
        figure_dir=CONFIG["figure_dir"],
        treshold=True,
    )

    visualization.compute_scores(
        filename="scores",
        figure_dir=CONFIG["figure_dir"],
        log_dir=CONFIG["log_dir"],
        exclusion_list=CONFIG["exclusion_list"],
    )


if __name__ == "__main__":
    main()
