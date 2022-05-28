import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from utils.utils import get_class_weights


def train_word2vec(tokens, vector_size):
    wv_model = Word2Vec(tokens, window=3, vector_size=vector_size, min_count=1)
    wv_model.build_vocab(tokens)
    wv_model.train(tokens, total_examples=wv_model.corpus_count, epochs=wv_model.epochs)
    print(wv_model)

    return wv_model


def embed_sentences(sentence_tokens, wv_model, vector_size):

    words = set(wv_model.wv.index_to_key)

    X_vectors = np.array(
        [
            np.array([wv_model.wv[w] for w in sentence if w in words])
            for sentence in sentence_tokens
        ]
    )
    X = []
    for vec in X_vectors:
        if vec.size:
            X.append(vec.mean(axis=0))
        else:
            X.append(np.zeros(vector_size, dtype=float))

    return X


def train_mlp(X, y):
    model = Sequential()
    model.add(Dense(512, activation="relu", input_shape=(X.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation="softmax"))
    model.build()

    model.compile(
        optimizer="adam", loss="SparseCategoricalCrossentropy", metrics=["accuracy"]
    )
    print(model.summary())
    model.fit(
        X,
        y,
        validation_split=0.2,
        epochs=50,
        batch_size=128,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        ],
        class_weight=get_class_weights(y),
    )

    return model


def logistic_regression(X_train, y):
    PARAMS = {
        "penalty": "l2",
        "tol": 1e-4,
        "C": 30,
        "solver": "saga",
        "class_weight": "balanced",
        "max_iter": 100,
        "n_jobs": -1,
    }

    lr = LogisticRegression(
        **PARAMS,
        verbose=0,
    )

    lr_model = lr.fit(X_train, y)

    return lr_model
