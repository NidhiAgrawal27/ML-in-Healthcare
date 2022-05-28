import argparse
import time
import numpy as np

from models import word_embedding
from utils import utils, evaluation, data_preparation
from utils.glove import load_glove_embeddings, embed_glove

CONFIG = {
    "data_path": "../data/pubmed-rct/PubMed_200k_RCT/",
    "glove_file": "../data/glove/glove.6B.100d.txt",
    "log_dir": "../logs/",
    "model_name": "glove_mlp",
    "verbose": 1,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random seed", required=True)
    args = parser.parse_args()

    utils.set_seed(args.seed)

    _, _, pred_filename, pred_proba_filename, metrics_filename = utils.get_logs(
        CONFIG["log_dir"], CONFIG["model_name"], args.seed
    )

    # Load data and tokenize.
    trainset, devset, testset = data_preparation.load_data(
        CONFIG["data_path"], verbose=CONFIG["verbose"]
    )

    # Load embeddings
    glove_emb_idx = load_glove_embeddings(CONFIG["glove_file"])

    # Embed sentences with GloVe
    X_train = embed_glove(trainset.X, glove_emb_idx)
    X_dev = embed_glove(devset.X, glove_emb_idx)
    X_test = embed_glove(testset.X, glove_emb_idx)

    X_train = np.asarray(X_train).astype(np.float32)
    X_dev = np.array(X_dev).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)

    # Train final model on both X_train and X_dev
    final_train_X = np.concatenate([X_train, X_dev], axis=0)
    final_train_y = np.concatenate([trainset.y, devset.y], axis=0)
    start_time = time.time()
    model = word_embedding.train_mlp(final_train_X, final_train_y)
    elapsed_time = time.time() - start_time

    # Predict and evaluate on testset
    pred_proba = model.predict(X_test)
    pred = np.argmax(pred_proba, axis=-1)
    np.save(pred_proba_filename, pred_proba)
    np.save(pred_filename, pred)

    utils.log_training_time(elapsed_time, metrics_filename)
    evaluation.evaluate(testset.y, pred, pred_proba, metrics_filename)


if __name__ == "__main__":
    main()
