import numpy as np
from tqdm import tqdm


def load_glove_embeddings(glove_file):
    idx = {}
    with open(glove_file, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            emb = np.asarray(values[1:], dtype="float32")
            idx[word] = emb
    return idx


def embed_glove(sentences, emb_idx, verbose=True):
    X = np.zeros((len(sentences), emb_idx["to"].shape[0]), dtype="float32")

    iterator = tqdm(
        enumerate(sentences),
        total=len(sentences),
        desc="Loading data",
        disable=not verbose,
    )

    for i, x in iterator:
        word_embeddings = [
            emb_idx[w.lower()] for w in x.split(" ") if w.lower() in emb_idx
        ]
        if len(word_embeddings) > 0:
            X[i] = np.mean(word_embeddings, axis=0)
            # otherwise leave at 0
    return X
