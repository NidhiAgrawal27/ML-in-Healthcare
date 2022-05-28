import os
import argparse
import numpy as np

from models import word_embedding
from models import tsne_visualization
from utils import utils, data_preparation
from utils.tokenizers.simple_tokenizer import SimpleTokenizer

CONFIG = {
    "data_path": "../data/pubmed-rct/PubMed_200k_RCT/",
    "figure_dir": "../logs/figures/",
    "model_name": "tsne_visualization",
    "verbose": 1,
    "vector_size": 100,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random seed", required=True)
    parser.add_argument("--lemmatize", action="store_true", help="lemmatize tokens")
    parser.add_argument(
        "--no-lemmatize", action="store_false", help="do not lemmatize tokens"
    )
    parser.set_defaults(lemmatize=False)
    args = parser.parse_args()

    utils.set_seed(args.seed)
    if args.lemmatize:
        CONFIG["model_name"] += "_lemmatize"

    # Load data and tokenize
    tokenizer = SimpleTokenizer(lemmatization=args.lemmatize, verbose=CONFIG["verbose"])
    trainset, devset, testset = data_preparation.load_data(
        CONFIG["data_path"], tokenizer=tokenizer, verbose=CONFIG["verbose"]
    )

    embedding_train = np.concatenate([trainset.X, devset.X, testset.X], axis=0)

    # Word Embedding
    vector_size = CONFIG["vector_size"]
    wv_model = word_embedding.train_word2vec(embedding_train, vector_size)

    # TSNE Visualization
    disease_treatment_dict = {
        "disease1": ["appendicitis"],
        "treatment1": ["appendectomy"],
        "disease2": ["gallstones"],
        "treatment2": ["cholecystectomy"],
        "disease3": ["osteoarthritis"],
        "treatment3": ["prednisolone"],
        "disease4": ["colon", "cancer"],
        "treatment4": ["folfox", "folfiri"],
        "disease5": ["radiation", "dermatitis"],
        "treatment5": ["mometasone", "furoate"],
        "disease6": ["hpv", "infections"],
        "treatment6": ["hpv", "vaccine"],
        "disease7": ["obesity"],
        "treatment7": ["probiotics"],
    }

    fig = tsne_visualization.visualize_word_embedding(
        wv_model, disease_treatment_dict, "Treatment(T) - Disease(D)", vector_size
    )
    filename = "treatment_disease.png"
    fig.savefig(os.path.join(CONFIG["figure_dir"], filename), bbox_inches="tight")

    disease_cause_dict = {
        "disease1": ["diabetic", "polyneuropathy"],
        "cause1": ["chronic", "compression", "nerves"],
        "disease2": ["obesity"],
        "cause2": ["emotional", "eating"],
        "disease3": ["diarrhoea"],
        "cause3": ["clostridium", "difficile", "infection"],
        "disease4": ["cardiac", "arrest"],
        "cause4": ["ischaemic", "brain", "injury"],
        "disease5": ["obesity"],
        "cause5": ["excessive", "intake", "protein"],
    }

    fig = tsne_visualization.visualize_word_embedding(
        wv_model, disease_cause_dict, "Cause(C) - Disease(D)", vector_size
    )
    filename = "cause_disease.png"
    fig.savefig(os.path.join(CONFIG["figure_dir"], filename), bbox_inches="tight")


if __name__ == "__main__":
    main()
