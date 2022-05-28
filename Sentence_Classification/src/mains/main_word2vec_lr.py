import argparse
import time
import numpy as np

from models import word_embedding
from utils import utils, evaluation, data_preparation
from utils.tokenizers.simple_tokenizer import SimpleTokenizer

CONFIG = {
    "data_path": "../data/pubmed-rct/PubMed_200k_RCT/",
    "log_dir": "../logs/",
    "model_name": "word2vec_lr",
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

    _, _, pred_filename, pred_proba_filename, metrics_filename = utils.get_logs(
        CONFIG["log_dir"], CONFIG["model_name"], args.seed
    )

    # Load data and tokenize.
    tokenizer = SimpleTokenizer(lemmatization=args.lemmatize, verbose=CONFIG["verbose"])
    trainset, devset, testset = data_preparation.load_data(
        CONFIG["data_path"], tokenizer=tokenizer, verbose=CONFIG["verbose"]
    )

    embedding_train = np.concatenate([trainset.X, devset.X, testset.X], axis=0)

    # Word Embedding
    vector_size = CONFIG["vector_size"]
    wv_model = word_embedding.train_word2vec(embedding_train, vector_size)

    # Sentence embedding
    X_train = word_embedding.embed_sentences(trainset.X, wv_model, vector_size)
    X_dev = word_embedding.embed_sentences(devset.X, wv_model, vector_size)
    X_test = word_embedding.embed_sentences(testset.X, wv_model, vector_size)

    X_train = np.asarray(X_train).astype(np.float32)
    X_dev = np.array(X_dev).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)

    # Train final model on both X_train and X_dev
    final_train_X = np.concatenate([X_train, X_dev], axis=0)
    final_train_y = np.concatenate([trainset.y, devset.y], axis=0)
    start_time = time.time()
    lr_model = word_embedding.logistic_regression(final_train_X, final_train_y)
    elapsed_time = time.time() - start_time

    # Predict and evaluate on testset
    pred_proba = lr_model.predict_proba(X_test)
    pred = lr_model.predict(X_test)
    np.save(pred_proba_filename, pred_proba)
    np.save(pred_filename, pred)
    utils.log_training_time(elapsed_time, metrics_filename)
    evaluation.evaluate(testset.y, pred, pred_proba, metrics_filename)


if __name__ == "__main__":
    main()
