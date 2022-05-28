from sklearn.linear_model import LogisticRegression

PARAMS = {"penalty": "l1", "solver": "saga", "max_iter": 10000}


def get_model(params=PARAMS):
    return LogisticRegression(**params)
