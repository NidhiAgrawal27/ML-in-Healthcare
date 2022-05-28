from sklearn.tree import DecisionTreeClassifier

PARAMS = {
    "criterion": "entropy",
    "splitter": "best",
    "max_depth": None,
    "max_features": None,
    "ccp_alpha": 0.0
}


def get_model(params=PARAMS):
    return DecisionTreeClassifier(**params)
