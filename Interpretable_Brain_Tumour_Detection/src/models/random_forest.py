from sklearn.ensemble import RandomForestClassifier

PARAMS = {
    "n_estimators": 100,
    "criterion": "entropy",
    "max_depth": None,
    "max_features": "sqrt",
    "oob_score": False,
    "n_jobs": -1,
    "verbose": 0
}


def get_model(params=PARAMS):
    return RandomForestClassifier(**params)
