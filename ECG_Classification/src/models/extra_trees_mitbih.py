from sklearn.ensemble import ExtraTreesClassifier

PARAMS = {"n_estimators": 313, "criterion": "gini", "max_depth": None}


def get_model(params=PARAMS):
    return ExtraTreesClassifier(n_estimators=params["n_estimators"],
                                criterion=params["criterion"],
                                max_depth=params["max_depth"],
                                n_jobs=-1,
                                verbose=1)
