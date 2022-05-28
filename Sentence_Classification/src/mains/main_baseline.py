import pickle
import os
import argparse
import time
import numpy as np

from models import baseline
from utils import utils, evaluation, data_preparation
from utils.tokenizers.simple_tokenizer import SimpleTokenizer

CONFIG = {
    "data_path": "../data/pubmed-rct/PubMed_200k_RCT/",
    "log_dir": "../logs/",
    "model_name": "baseline",
    "verbose": 1
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random seed", required=True)
    parser.add_argument("--lemmatize",
                        action="store_true",
                        help="lemmatize tokens")
    parser.add_argument("--no-lemmatize",
                        action="store_false",
                        help="do not lemmatize tokens")
    parser.set_defaults(lemmatize=False)
    args = parser.parse_args()

    utils.set_seed(args.seed)
    if args.lemmatize:
        CONFIG["model_name"] += "_lemmatize"

    _, _, pred_filename, pred_proba_filename, metrics_filename = (
        utils.get_logs(CONFIG["log_dir"], CONFIG["model_name"], args.seed))

    # Load data and tokenize.
    tokenizer = SimpleTokenizer(lemmatization=args.lemmatize,
                                verbose=CONFIG["verbose"])
    trainset, devset, testset = data_preparation.load_data(
        CONFIG["data_path"], tokenizer=tokenizer, verbose=CONFIG["verbose"])

    # Retrieve model.
    model = baseline.get_model()

    # Train final model on both trainset and devset.
    final_train_X = np.concatenate([trainset.X, devset.X], axis=0)
    final_train_y = np.concatenate([trainset.y, devset.y], axis=0)
    start_time = time.time()
    model.fit(final_train_X, final_train_y)
    elapsed_time = time.time() - start_time

    # Predict and evaluate on testset.
    pred_proba = model.predict_proba(testset.X)
    pred = model.predict(testset.X)
    np.save(pred_proba_filename, pred_proba)
    np.save(pred_filename, pred)

    utils.log_training_time(elapsed_time, metrics_filename)
    evaluation.evaluate(testset.y, pred, pred_proba, metrics_filename)


if __name__ == "__main__":
    main()
