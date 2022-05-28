from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PARAMS = {
    "ngram_range": (1, 2),
    "penalty": "l2",
    "tol": 1e-4,
    "C": 0.9,
    "solver": "saga",
    "class_weight": "balanced",
    "max_iter": 100,
    "n_jobs": -1,
}


def get_model(params=PARAMS, class_weights=True):
    tfidf = TfidfVectorizer(
        lowercase=False, tokenizer=(lambda x: x), ngram_range=PARAMS["ngram_range"]
    )
    clf = LogisticRegression(
        penalty=PARAMS["penalty"],
        tol=PARAMS["tol"],
        C=PARAMS["C"],
        solver=PARAMS["solver"],
        class_weight=PARAMS["class_weight"],
        max_iter=PARAMS["max_iter"],
        n_jobs=PARAMS["n_jobs"],
        verbose=2,
    )
    return Pipeline([("tfidf", tfidf), ("clf", clf)])
