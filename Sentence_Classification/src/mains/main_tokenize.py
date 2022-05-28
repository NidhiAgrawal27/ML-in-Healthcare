import numpy as np
import argparse

from utils import data_preparation
from utils.tokenizers.simple_tokenizer import SimpleTokenizer

CONFIG = {"data_path": "../data/pubmed-rct/PubMed_200k_RCT/", "verbose": 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lemmatize", action="store_true", help="lemmatize tokens")
    parser.add_argument(
        "--no-lemmatize", action="store_false", help="do not lemmatize tokens"
    )
    parser.set_defaults(lemmatize=False)
    args = parser.parse_args()

    # Load data and tokenize. Saves tokenized data
    data_preparation.load_data(
        CONFIG["data_path"],
        tokenizer=SimpleTokenizer(
            lemmatization=args.lemmatize, verbose=CONFIG["verbose"]
        ),
        verbose=CONFIG["verbose"],
    )


if __name__ == "__main__":
    main()
